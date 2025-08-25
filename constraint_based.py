from __future__ import annotations
from dotenv import load_dotenv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from llm_service import LLMClient, get_response_from_llm, extract_json_between_markers
from prompt import IDEA_FIRST_PROMPT, IDEA_REFLECTION_PROMPT
import typer

# =============================
# Core generation logic
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


# =============================
# Typer CLI
# =============================
app = typer.Typer(
    add_completion=False,
    help="Generate ideas with an LLM and iteratively refine them through self-review."
)

@app.command("gemini")
def gemini_generate(
    prompt_file: Path = typer.Option(
        Path("./prompt.json"),
        help='Path to a JSON file containing "system" and "task_description" fields.'
    ),
    seed_file: Path = typer.Option(
        Path("./seed.json"),
        help="Path to a JSON file containing an array of seed artifacts."
    ),
    out_file: Path = typer.Option(
        Path("./schema.json"),
        help="Path to the output JSON file for the final design-space schema."
    ),
    model_name: str = typer.Option(
        "gemini-2.5-flash",
        help="LLM model to use."
    ),
    num_reflections: int = typer.Option(
        1,
        min=1,
        help="Number of self-review iterations per idea (max)."
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Resume from an existing output file, if present."
    ),
):
    from google import genai
    from google.genai import types
    load_dotenv()

    prompt = _load_json(prompt_file, {})
    project_overview = prompt.get("project", "")
    system_message = prompt.get("system", "You are a creative professional designer.")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),  # Disables thinking
        system_instruction=system_message,
    )
    chat = client.chats.create(model=model_name, config=config)

    existing_artefacts = []
    seeds = _load_json(seed_file, [])
    existing_artefacts.extend(seeds if isinstance(seeds, list) else [])

    existing_artefacts_string = "\n\n".join(
        f"{idx+1}: {i['Description']}" for idx, i in enumerate(existing_artefacts)
    ) or "(none yet)"

    typer.echo("Generating initial idea...", fg=typer.colors.BLUE)
    first_idea = chat.send_message(
        IDEA_FIRST_PROMPT.format(
            project_overview=project_overview,
            existing_artefacts=existing_artefacts_string,
            num_reflections=num_reflections,
        )
    )

    print(first_idea.text)

    idea_json = extract_json_between_markers(first_idea.text)
    if idea_json is None:
        raise ValueError("Could not extract a valid JSON block from the initial LLM response.")

    last_idea = idea_json  # ensure this is defined prior to reflections

    for i in range(1, num_reflections + 1):
        try:
            new_constraint = chat.send_message(
                IDEA_REFLECTION_PROMPT.format(current_round=i, num_reflections=num_reflections)
            )
            print(new_constraint.text)
            last_idea = extract_json_between_markers(new_constraint.text) or last_idea
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

if __name__ == "__main__":
    app()