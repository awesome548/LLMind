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

# Supabase
from supabase import create_client, Client

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

    try:
        return json.loads(candidate)
    except Exception:
        candidate2 = candidate.strip().strip("`")
        try:
            return json.loads(candidate2)
        except Exception:
            return None


# =============================
# Supabase helpers
# =============================

def _get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set.")
    return create_client(url, key)

def _table_name() -> str:
    return os.getenv("SUPABASE_TABLE") or "media_docs"

def _load_selected_ids(path: Optional[Path]) -> List[str]:
    """Load list of ids from JSON file. Returns [] if not found/invalid/None."""
    if not path:
        return []
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [str(x) for x in data]
    except Exception:
        pass
    return []

def _fetch_rows_from_supabase_by_ids(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch rows for given ids.
    Expect table schema: id (text), content (text), metadata (jsonb), embedding (vector)
    Returns mapping id -> dict(row)
    """
    if not ids:
        return {}
    sb = _get_supabase()
    tbl = _table_name()
    # Supabase PostgREST supports .in_ for filtering by a list
    res = sb.table(tbl).select("id, content, metadata").in_("id", ids).execute()
    if getattr(res, "error", None):
        raise RuntimeError(f"Supabase error: {res.error}")
    rows = res.data or []
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        out[str(r.get("id"))] = dict(r)
    return out

def _fetch_all_rows_from_supabase(limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Fetch all rows from the table with simple paging.
    """
    sb = _get_supabase()
    tbl = _table_name()
    start = 0
    page = []
    all_rows: List[Dict[str, Any]] = []
    while True:
        # range is inclusive
        res = sb.table(tbl).select("id, content, metadata").range(start, start + limit - 1).execute()
        if getattr(res, "error", None):
            raise RuntimeError(f"Supabase error: {res.error}")
        page = res.data or []
        all_rows.extend(page)
        if len(page) < limit:
            break
        start += limit
    return all_rows


def _artefacts_from_rows(
    rows: List[Dict[str, Any]],
    *,
    mode: Literal["details_only", "both"] = "details_only",
    preserve_order_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Convert supabase rows -> artefacts list.
    - Each row: { id, content, metadata{ Name, Descriptions, Details, Image } }
    - mode 'details_only' uses Details or falls back to content
    - mode 'both' combines Descriptions + Details
    - If preserve_order_ids is provided, artefacts are sorted by that order and filtered to the listed ids.
    """
    # Build mapping for ordering if provided
    order_index: Dict[str, int] = {}
    if preserve_order_ids:
        order_index = {str(v): i for i, v in enumerate(preserve_order_ids)}

    artefacts: List[Dict[str, Any]] = []
    for r in rows:
        _id = str(r.get("id"))
        meta = r.get("metadata") or {}
        details = meta.get("Details")
        descriptions = meta.get("Descriptions")
        content = r.get("content")

        if mode == "both":
            combined = "; ".join([s for s in [descriptions, details] if s])
            text = combined or content
        else:
            text = details or content

        if not text:
            continue

        artefacts.append({
            "ID": _id,
            "Description": text,
        })

    # If we need to preserve order from ids file
    if preserve_order_ids:
        # keep only ids present in preserve_order_ids and sort
        artefacts = [a for a in artefacts if a["ID"] in order_index]
        artefacts.sort(key=lambda a: order_index.get(a["ID"], 10**9))

    return artefacts


def build_artefacts(
    *,
    source: Literal["selected", "all_supabase"] = "selected",
    ids_file: Optional[Path] = None,
    mode: Literal["details_only", "both"] = "details_only",
) -> List[Dict[str, Any]]:
    """
    Build artefacts from Supabase.

    - source = "selected": load ids from file, fetch matching rows from Supabase, preserve the ids order.
    - source = "all_supabase": fetch all rows from Supabase (ignores ids_file).

    Returns list of { "ID": <id>, "Description": <text> }.
    """
    if source == "selected":
        ids = _load_selected_ids(ids_file)
        if not ids:
            return []
        rows_map = _fetch_rows_from_supabase_by_ids(ids)
        # preserve order based on ids file
        rows = [rows_map[i] for i in ids if i in rows_map]
        return _artefacts_from_rows(rows, mode=mode, preserve_order_ids=ids)

    elif source == "all_supabase":
        rows = _fetch_all_rows_from_supabase()
        return _artefacts_from_rows(rows, mode=mode)

    else:
        raise ValueError("Invalid source. Use 'selected' or 'all_supabase'.")


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
    dev_mode: bool = False,
) -> Dict[str, Any]:
    existing_artefacts_string = "\n\n".join(
        f"{art.get('ID')}: {art.get('Description')}" for art in existing_artefacts
    ) or "(none yet)"

    if dev_mode:
        typer.secho("-- Dev mode -- ", fg=typer.colors.YELLOW)
        typer.secho(
            IDEA_FIRST_PROMPT.format(
                project_overview=project_overview,
                existing_artefacts=existing_artefacts_string,
                num_reflections=num_reflections,
            ), fg=typer.colors.YELLOW)
        with open("debug_artefacts.txt", "w", encoding="utf-8") as f:
            f.write(IDEA_FIRST_PROMPT.format(
                project_overview=project_overview,
                existing_artefacts=existing_artefacts_string,
                num_reflections=num_reflections,
            ))

    # typer.secho("Generating initial idea...", fg=typer.colors.GREEN)
    # first_text = chat.send_message(
    #     IDEA_FIRST_PROMPT.format(
    #         project_overview=project_overview,
    #         existing_artefacts=existing_artefacts_string,
    #         num_reflections=num_reflections,
    #     )
    # )
    # typer.secho(first_text, fg=typer.colors.CYAN)

    # idea_json = extract_json_between_markers(first_text)
    # if idea_json is None:
    #     raise ValueError("Could not extract a valid JSON block from the initial LLM response.")

    # last_idea = idea_json  # ensure defined for reflections

    # for i in range(1, num_reflections + 1):
    #     try:
    #         new_text = chat.send_message(
    #             IDEA_REFLECTION_PROMPT.format(current_round=i, num_reflections=num_reflections)
    #         )
    #         maybe = extract_json_between_markers(new_text)
    #         if maybe:
    #             last_idea = maybe
    #         typer.secho(new_text, fg=typer.colors.CYAN)
    #         typer.secho(
    #             f"Reflection {i}/{num_reflections} complete.  {new_text}",
    #             fg=typer.colors.GREEN
    #         )
    #     except Exception as e:
    #         typer.secho(f"Reflection {i}/{num_reflections} failed: {e}", fg=typer.colors.RED)
    #         continue

    # return last_idea
    return {}

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
            Path("./taxonomy/taxonomy"),
            help="Path to the output JSON file for the final design-space schema."
        ),
        num_reflections=typer.Option(
            1,
            "--num",
            min=1,
            help="Number of self-review iterations per idea (max)."
        ),
        ids_file=typer.Option(
            None,
            "-i",
            help="Path to farthest-selected ids JSON from clustering (used when --source=selected)."
        ),
        selected_mode=typer.Option(
            "both",
            "--mode",
            help="Use 'details_only' or 'both' (details + descriptions)."
        ),
        source=typer.Option(
            "selected",
            "--source",
            help="Where to load artefacts from: 'selected' (ids_file) or 'all_supabase'."
        ),
        dev_mode=typer.Option(
            False,
            "--dev",
            help="Enable dev mode (not used currently)."
        )
    )

@app.command("gemini")
def gemini_generate(
    prompt_file: Path = _common_options()["prompt_file"],
    out_file: Path = _common_options()["out_file"],
    ids_file: Optional[Path] = _common_options()["ids_file"],
    num_reflections: int = _common_options()["num_reflections"],
    selected_mode: str = _common_options()["selected_mode"],
    source: str = _common_options()["source"],
    dev_mode: bool = _common_options()["dev_mode"],
    model_name: str = typer.Option("gemini-2.5-flash", help="Gemini model to use."),
):
    """
    Generate using Google Gemini models.
    """
    load_dotenv()

    prompt = _load_json(prompt_file, {})
    project_overview = prompt.get("project", "")
    system_message = prompt.get("system", "You are a creative professional designer.")

    # Load artefacts from Supabase
    try:
        artefacts = build_artefacts(
            source="all_supabase" if source == "all_supabase" else "selected",
            ids_file=ids_file,
            mode="both" if selected_mode == "both" else "details_only",
        )
        if artefacts:
            if source == "all_supabase":
                typer.echo(f"Loaded {len(artefacts)} artefacts from Supabase (all rows).")
            else:
                typer.echo(f"Loaded {len(artefacts)} selected artefacts from Supabase (ids file).")
        else:
            typer.echo(f"No artefacts loaded (source={source}).")
    except Exception as e:
        typer.echo(f"Warning: failed to build artefacts (source={source}): {e}")
        artefacts = []

    chat = GeminiChat(model=model_name, system_message=system_message)
    final = run_generate(chat, project_overview, artefacts, num_reflections, dev_mode)
    save_json(out_file, final, f"{source}_{selected_mode}", model_name)


@app.command("openai")
def openai_generate(
    prompt_file: Path = _common_options()["prompt_file"],
    out_file: Path = _common_options()["out_file"],
    ids_file: Optional[Path] = _common_options()["ids_file"],
    num_reflections: int = _common_options()["num_reflections"],
    selected_mode: str = _common_options()["selected_mode"],
    source: str = _common_options()["source"],
    dev_mode: bool = _common_options()["dev_mode"],
    model_name: str = typer.Option(
        "gpt-5-mini-2025-08-07",
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

    # Load artefacts from Supabase
    try:
        artefacts = build_artefacts(
            source="all_supabase" if source == "all_supabase" else "selected",
            ids_file=ids_file,
            mode="both" if selected_mode == "both" else "details_only",
        )
        if artefacts:
            if source == "all_supabase":
                typer.echo(f"Loaded {len(artefacts)} artefacts from Supabase (all rows).")
            else:
                typer.echo(f"Loaded {len(artefacts)} selected artefacts from Supabase (ids file).")
        else:
            typer.echo(f"No artefacts loaded (source={source}).")
    except Exception as e:
        typer.echo(f"Warning: failed to build artefacts (source={source}): {e}")
        artefacts = []

    chat = OpenAIChat(model=model_name, system_message=system_message)
    final = run_generate(chat, project_overview, artefacts, num_reflections, dev_mode)
    save_json(out_file, final, f"{source}_{selected_mode}", model_name)


if __name__ == "__main__":
    app()