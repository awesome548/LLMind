from __future__ import annotations
from dotenv import load_dotenv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, Literal
from datetime import datetime
from data.prompts.prompts import IDEA_FIRST_PROMPT, IDEA_REFLECTION_PROMPT
import typer
import chromadb

# =============================
# Utils
# =============================

def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4))


_JSON_FENCE_RE = re.compile(r"```json\s*(?P<body>.*?)\s*```", re.IGNORECASE | re.DOTALL)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_json_between_markers(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from a markdown ```json fenced block, else best-effort braces.
    Returns a dict or None.
    """
    m = _JSON_FENCE_RE.search(text)
    candidate = m.group("body") if m else None
    if not candidate:
        m2 = _BRACE_RE.search(text)
        candidate = m2.group(0) if m2 else None
    if not candidate:
        return None

    # Try strict parse first
    try:
        return json.loads(candidate)
    except Exception:
        # Gentle cleanup: strip stray backticks/whitespace
        candidate2 = candidate.strip().strip("`")
        try:
            return json.loads(candidate2)
        except Exception:
            return None


# =============================
# Chroma helpers
# =============================

def _load_selected_ids(path: Path) -> List[str]:
    """Load list of ids from JSON file. Returns [] if not found/invalid."""
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x) for x in data]
    except Exception:
        pass
    return []


def _fetch_descs_from_chroma(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch metadatas from Chroma for given ids.
    Returns mapping id -> {"Desc": str, ...meta}
    """
    if not ids:
        return {}
    db_path = os.getenv("CHROMA_DB_PATH")
    coll_name = os.getenv("COLLECTION_NAME")
    if not db_path or not coll_name:
        return {}
    client = chromadb.PersistentClient(path=Path(db_path))
    coll = client.get_collection(name=str(coll_name))
    res = coll.get(ids=ids, include=["metadatas"])
    out: Dict[str, Dict[str, Any]] = {}
    for i, _id in enumerate(res.get("ids", []) or []):
        meta = (res.get("metadatas") or [{}])[i]
        out[str(_id)] = dict(meta)
    return out


def build_artefacts_from_farthest(
    ids_file: Path,
    *,
    mode: Literal["details_only", "both"] = "details_only",
) -> List[Dict[str, Any]]:
    """
    Load farthest-selected ids, fetch their Chroma metadatas, and build artefacts without chunking.

    Artefact schema:
    { "ID": <id>, "Description": <text> }

    Modes:
    - details_only: use only details-like fields (Desc/Details/detail[s]).
    - both: combine description-like and details-like fields if present.
    """
    ids = _load_selected_ids(ids_file)
    if not ids:
        return []
    metas_by_id = _fetch_descs_from_chroma(ids)
    artefacts: List[Dict[str, Any]] = []
    for _id in ids:
        meta = metas_by_id.get(str(_id)) or {}
        if mode == "both":
            a = meta.get("Descriptions")
            b = meta.get("Details")
            combined = "; ".join([s for s in [a, b] if s])
            text = combined 
        else:  # details_only
            text = meta.get("Details")

        if not text:
            continue
        artefacts.append({
            "ID": str(_id),
            "Description": text,
        })
    return artefacts


# =============================
# Provider-agnostic chat interface
# =============================

class ChatSession(Protocol):
    def send_message(self, content: str) -> str: ...

@dataclass
class GeminiChat(ChatSession):
    model: str
    system_message: str

    def __post_init__(self) -> None:
        from google import genai
        from google.genai import types
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        self._client = genai.Client(api_key=api_key)
        self._config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            system_instruction=self.system_message,
        )
        self._chat = self._client.chats.create(model=self.model, config=self._config)

    def send_message(self, content: str) -> str:
        resp = self._chat.send_message(content)
        return getattr(resp, "text", str(resp))


@dataclass
class OpenAIChat(ChatSession):
    model: str
    system_message: str

    def __post_init__(self) -> None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self._client = OpenAI(api_key=api_key)
        self._mode = "responses"

        self._messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_message}
        ]

    def send_message(self, content: str) -> str:
        self._messages.append({"role": "user", "content": content})
        resp = self._client.responses.create(
            model=self.model,
            # reasoning={ "effort": "low" },
            input=self._messages,
        )
        out_chunks = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", "") == "output_text":
                        out_chunks.append(getattr(c, "text", ""))
        text = "\n".join(out_chunks).strip()
        self._messages.append({"role": "assistant", "content": text})
        return text or "(empty response)"

# =============================
# Generation core (shared)
# =============================

def run_generate(
    chat: ChatSession,
    project_overview: str,
    existing_artefacts: List[Dict[str, Any]],
    num_reflections: int,
    out_file: Path,
) -> Dict[str, Any]:
    existing_artefacts_string = "\n\n".join(
        f"{art.get('ID')}: {art.get('Description')}" for art in existing_artefacts
    ) or "(none yet)"

    typer.secho("Generating initial idea...", fg=typer.colors.GREEN)
    first_text = chat.send_message(
        IDEA_FIRST_PROMPT.format(
            project_overview=project_overview,
            existing_artefacts=existing_artefacts_string,
            num_reflections=num_reflections,
        )
    )
    typer.secho(first_text, fg=typer.colors.CYAN)

    idea_json = extract_json_between_markers(first_text)
    if idea_json is None:
        raise ValueError("Could not extract a valid JSON block from the initial LLM response.")

    last_idea = idea_json  # ensure defined for reflections

    for i in range(1, num_reflections + 1):
        try:
            new_text = chat.send_message(
                IDEA_REFLECTION_PROMPT.format(current_round=i, num_reflections=num_reflections)
            )
            maybe = extract_json_between_markers(new_text)
            if maybe:
                last_idea = maybe
            typer.secho(new_text, fg=typer.colors.CYAN)
            typer.secho(
                f"Reflection {i}/{num_reflections} complete. Updated aspects: {last_idea.get('Aspects')}",
                fg=typer.colors.GREEN
            )
        except Exception as e:
            typer.secho(f"Reflection {i}/{num_reflections} failed: {e}", fg=typer.colors.RED)
            continue

    typer.secho(f"Final idea aspects: {last_idea.get('Aspects')}", fg=typer.colors.CYAN)

    return last_idea

def save_json(out_file: Path, data: Any, mode: str, model: str) -> None:
    file_name = out_file.with_name(
        out_file.stem + "_" + mode + "_" + model + "_" + datetime.now().strftime("%Y%m%d_%H%M") + ".json"
    )
    _save_json(file_name, data)
    typer.secho(f"Saved final result to: {file_name}", fg=typer.colors.BLUE)


# =============================
# Typer CLI
# =============================

app = typer.Typer(
    add_completion=False,
    help="Generate ideas with an LLM and iteratively refine them through self-review."
)

def _common_options():
    return dict(
        prompt_file=typer.Option(
            Path("./data/prompts/system_prompt.json"),
            help='Path to a JSON file containing "system" and "project" fields.'
        ),
        out_file=typer.Option(
            Path("./data/schema"),
            help="Path to the output JSON file for the final design-space schema."
        ),
        num_reflections=typer.Option(
            1,
            "--num",
            min=1,
            help="Number of self-review iterations per idea (max)."
        ),
        ids_file=typer.Option(
            Path("data/seletected_projects.json"),
            "--ids",
            help="Path to farthest-selected ids JSON from clustering."
        ),
        selected_mode=typer.Option(
            "both",
            "--mode",
            help="Use 'details' or 'both' (details + descriptions)."
        ),
    )

@app.command("gemini")
def gemini_generate(
    prompt_file: Path = _common_options()["prompt_file"],
    out_file: Path = _common_options()["out_file"],
    ids_file: Path = _common_options()["ids_file"],
    num_reflections: int = _common_options()["num_reflections"],
    selected_mode: Optional[str] = _common_options()["selected_mode"],
    model_name: str = typer.Option("gemini-2.5-flash", help="Gemini model to use."),
):
    """
    Generate using Google Gemini models.
    """
    load_dotenv()

    prompt = _load_json(prompt_file, {})
    project_overview = prompt.get("project", "")
    system_message = prompt.get("system", "You are a creative professional designer.")

    # Load selected projects from Chroma (no chunking)
    try:
        existing_artefacts = build_artefacts_from_farthest(ids_file, mode=(selected_mode or "details"))
        if existing_artefacts:
            typer.echo(f"Loaded {len(existing_artefacts)} selected artefacts from Chroma (replaced seeds).")
        else:
            typer.echo(f"No artefacts loaded from ids file ({ids_file}).")
    except Exception as e:
        typer.echo(f"Warning: failed to build artefacts from ids ({ids_file}): {e}")

    chat = GeminiChat(model=model_name, system_message=system_message)
    final = run_generate(chat, project_overview, existing_artefacts, num_reflections, out_file)
    save_json(out_file, final, str(selected_mode), model_name)


@app.command("openai")
def openai_generate(
    prompt_file: Path = _common_options()["prompt_file"],
    out_file: Path = _common_options()["out_file"],
    ids_file: Path = _common_options()["ids_file"],
    num_reflections: int = _common_options()["num_reflections"],
    selected_mode: Optional[str] = _common_options()["selected_mode"],
    model_name: str = typer.Option(
        "gpt-4o-mini",
        help="OpenAI model to use (e.g., gpt-4o, gpt-4o-mini, gpt-4.1, o3-mini)."
    ),
):
    """
    Generate using OpenAI models.
    """
    load_dotenv()

    prompt = _load_json(prompt_file, {})
    project_overview = prompt.get("project", "")
    system_message = prompt.get("system", "You are a creative professional designer.")

    # Load selected projects from Chroma (no chunking)
    try:
        existing_artefacts = build_artefacts_from_farthest(ids_file, mode=(selected_mode or "details"))
        if existing_artefacts:
            typer.echo(f"Loaded {len(existing_artefacts)} selected artefacts from Chroma (replaced seeds).")
        else:
            typer.echo(f"No artefacts loaded from ids file ({ids_file}).")
    except Exception as e:
        typer.echo(f"Warning: failed to build artefacts from ids ({ids_file}): {e}")

    chat = OpenAIChat(model=model_name, system_message=system_message)
    final = run_generate(chat, project_overview, existing_artefacts, num_reflections, out_file)
    save_json(out_file, final, str(selected_mode), model_name)


if __name__ == "__main__":
    app()
