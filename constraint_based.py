from __future__ import annotations
from dotenv import load_dotenv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from data.prompts import IDEA_FIRST_PROMPT, IDEA_REFLECTION_PROMPT
import typer

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
) -> None:
    existing_artefacts_string = "\n\n".join(
        f"{idx+1}: {i.get('Description', str(i))}" for idx, i in enumerate(existing_artefacts)
    ) or "(none yet)"

    typer.echo("Generating initial idea...")
    first_text = chat.send_message(
        IDEA_FIRST_PROMPT.format(
            project_overview=project_overview,
            existing_artefacts=existing_artefacts_string,
            num_reflections=num_reflections,
        )
    )
    print(first_text)

    idea_json = extract_json_between_markers(first_text)
    if idea_json is None:
        raise ValueError("Could not extract a valid JSON block from the initial LLM response.")

    last_idea = idea_json  # ensure defined for reflections

    for i in range(1, num_reflections + 1):
        try:
            new_text = chat.send_message(
                IDEA_REFLECTION_PROMPT.format(current_round=i, num_reflections=num_reflections)
            )
            print(new_text)
            maybe = extract_json_between_markers(new_text)
            if maybe:
                last_idea = maybe
            typer.secho(
                f"Reflection {i}/{num_reflections} complete. Updated aspects: {last_idea.get('Aspects')}",
                fg=typer.colors.GREEN
            )
        except Exception as e:
            typer.secho(f"Reflection {i}/{num_reflections} failed: {e}", fg=typer.colors.RED)
            continue

    typer.echo(f"Final idea aspects: {last_idea.get('Aspects')}")
    _save_json(out_file, last_idea)
    typer.secho(f"Saved final result to: {out_file}", fg=typer.colors.BLUE)


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
            Path("./data/system_prompt.json"),
            help='Path to a JSON file containing "system" and "project" fields.'
        ),
        seed_file=typer.Option(
            Path("./data/seed.json"),
            help="Path to a JSON file containing an array of seed artifacts."
        ),
        out_file=typer.Option(
            Path("./data/schema.json"),
            help="Path to the output JSON file for the final design-space schema."
        ),
        num_reflections=typer.Option(
            1,
            min=1,
            help="Number of self-review iterations per idea (max)."
        ),
        resume=typer.Option(
            False,
            "--resume",
            help="Resume from an existing output file, if present. (Reserved for future use.)"
        ),
    )

@app.command("gemini")
def gemini_generate(
    prompt_file: Path = _common_options()["prompt_file"],
    seed_file: Path = _common_options()["seed_file"],
    out_file: Path = _common_options()["out_file"],
    model_name: str = typer.Option("gemini-2.5-flash", help="Gemini model to use."),
    num_reflections: int = _common_options()["num_reflections"],
    resume: bool = _common_options()["resume"],
):
    """
    Generate using Google Gemini models.
    """
    load_dotenv()

    prompt = _load_json(prompt_file, {})
    project_overview = prompt.get("project", "")
    system_message = prompt.get("system", "You are a creative professional designer.")

    seeds = _load_json(seed_file, [])
    existing_artefacts: List[Dict[str, Any]] = seeds if isinstance(seeds, list) else []

    chat = GeminiChat(model=model_name, system_message=system_message)
    run_generate(chat, project_overview, existing_artefacts, num_reflections, out_file)


@app.command("openai")
def openai_generate(
    prompt_file: Path = _common_options()["prompt_file"],
    seed_file: Path = _common_options()["seed_file"],
    out_file: Path = _common_options()["out_file"],
    model_name: str = typer.Option(
        "gpt-4o-mini",
        help="OpenAI model to use (e.g., gpt-4o, gpt-4o-mini, gpt-4.1, o3-mini)."
    ),
    num_reflections: int = _common_options()["num_reflections"],
    resume: bool = _common_options()["resume"],
):
    """
    Generate using OpenAI models.
    """
    load_dotenv()

    prompt = _load_json(prompt_file, {})
    project_overview = prompt.get("project", "")
    system_message = prompt.get("system", "You are a creative professional designer.")

    seeds = _load_json(seed_file, [])
    existing_artefacts: List[Dict[str, Any]] = seeds if isinstance(seeds, list) else []

    chat = OpenAIChat(model=model_name, system_message=system_message)
    run_generate(chat, project_overview, existing_artefacts, num_reflections, out_file)


if __name__ == "__main__":
    app()