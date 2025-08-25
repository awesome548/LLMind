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


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _generate_constraints(
    design_space: str,
    system_message: str,
    existing_artefacts: List[Dict[str, Any]],
    client: LLMClient,
    model: str,
    num_reflections: int,
) -> Dict[str, Any]:

    existing_artefacts_string = "\n\n".join(
        f"A{idx+1}: {i['Description']}" for idx, i in enumerate(existing_artefacts)
    ) or "(none yet)"

    msg_history: List[Dict[str, Any]] = []

    first_text, msg_history = get_response_from_llm(
        IDEA_FIRST_PROMPT.format(
            design_space=design_space,
            existing_artefacts=existing_artefacts_string,
            num_reflections=num_reflections,
        ),
        client=client,
        model=model,
        system_message=system_message,
        msg_history=msg_history,
    )
    print(f"First text: {first_text}")
    idea_json = extract_json_between_markers(first_text)
    if idea_json is None:
        raise ValueError("Failed to extract JSON from first LLM output")

    return idea_json

def _refine_constraints(
    *,
    system_message: str,
    client: LLMClient,
    model: str,
    num_reflections: int,
) -> Dict[str, Any]:
    # Reflection rounds
    if num_reflections > 1:
        for j in range(2, num_reflections + 1):
            refl_text, msg_history = get_response_from_llm(
                IDEA_REFLECTION_PROMPT.format(current_round=j, num_reflections=num_reflections),
                client=client,
                model=model,
                system_message=system_message,
                msg_history=msg_history,
            )
            maybe_json = extract_json_between_markers(refl_text)
            if maybe_json is not None:
                idea_json = maybe_json
            if "I am done" in refl_text:
                break



# =============================
# Typer CLI (no 'experiment' concept)
# =============================
app = typer.Typer(add_completion=False, help="Generate and iteratively self-review ideas with an LLM.")


@app.command("generate")
def generate_command(
    prompt_file: Path = typer.Option(Path("./prompt.json"), help='Path to JSON with {"system":..., "task_description":...}'),
    seed_file: Path = typer.Option(Path("./seed.json"), help="Path to JSON array of seed artefacts."),
    out_file: Path = typer.Option(Path("./schema.json"), help="Output JSON archive for design space schema."),
    model_name: str = typer.Option("gemini-2.5-flash", help="LLM model name"),
    num_reflections: int = typer.Option(1, min=1, help="Max self-review iterations per idea."),
    resume: bool = typer.Option(False, "--resume", help="Resume from existing out_file if present."),
):
    client = LLMClient()

    prompt = _load_json(prompt_file, {})
    design_space = prompt.get("design_space", "")
    system_message = prompt.get("system", "You are a creative professionaldesigner.")

    existing_artefacts: List[Dict[str, Any]] = _load_json(out_file, []) if resume else []
    if not existing_artefacts:
        seeds = _load_json(seed_file, [])
        existing_artefacts.extend(seeds if isinstance(seeds, list) else [])
    
    aspect_options = []

    new_constraint = _generate_constraints(
        design_space=design_space,
        system_message=system_message,
        existing_artefacts=existing_artefacts,
        client=client,
        model=model_name,
        num_reflections=num_reflections,
    )
    print(f"New constraint: {new_constraint}")
    # for _ in range(num_reflections):
    #     try:
    #         new_constraint = _refine_constraints(
    #             design_space=design_space,
    #             system_message=system_message,
    #             existing_artefacts=existing_artefacts,
    #             client=client,
    #             model=model_name,
    #             num_reflections=num_reflections,
    #         )
    #         aspect_options.append(new_constraint)
    #         typer.secho(f"Refined idea #{len(existing_artefacts)}: {new_constraint.get('Name', '(unnamed)')}", fg=typer.colors.GREEN)
    #     except Exception as e:
    #         typer.secho(f"Failed to generate idea: {e}", fg=typer.colors.RED)
    #         continue

    # _save_json(out_file, aspect_options)

@app.command("gemini")
def gemini_generate(
    prompt_file: Path = typer.Option(Path("./prompt.json"), help='Path to JSON with {"system":..., "task_description":...}'),
    seed_file: Path = typer.Option(Path("./seed.json"), help="Path to JSON array of seed artefacts."),
    out_file: Path = typer.Option(Path("./schema.json"), help="Output JSON archive for design space schema."),
    model_name: str = typer.Option("gemini-2.5-flash", help="LLM model name"),
    num_reflections: int = typer.Option(1, min=1, help="Max self-review iterations per idea."),
    resume: bool = typer.Option(False, "--resume", help="Resume from existing out_file if present."),
):
    from google import genai
    from google.genai import types
    load_dotenv()


    prompt = _load_json(prompt_file, {})
    design_space = prompt.get("design_space", "")
    system_message = prompt.get("system", "You are a creative professional designer.")

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0), # Disables thinking
        system_instruction=system_message,
    )
    chat = client.chats.create(model=model_name,config=config)

    existing_artefacts: List[Dict[str, Any]] = _load_json(out_file, []) if resume else []
    if not existing_artefacts:
        seeds = _load_json(seed_file, [])
        existing_artefacts.extend(seeds if isinstance(seeds, list) else [])
    
    existing_artefacts_string = "\n\n".join(
        f"{idx+1}: {i['Description']}" for idx, i in enumerate(existing_artefacts)
    ) or "(none yet)"

    first_idea = chat.send_message(
        IDEA_FIRST_PROMPT.format(
            design_space=design_space,
            existing_artefacts=existing_artefacts_string,
            num_reflections=num_reflections,
        )
    )
    print(first_idea.text)
    idea_json = extract_json_between_markers(first_idea.text)
    if idea_json is None:
        raise ValueError("Failed to extract JSON from first LLM output")

    for i in range(1, num_reflections + 1):
        try:
            new_constraint = chat.send_message(
                IDEA_REFLECTION_PROMPT.format(current_round=i, num_reflections=num_reflections)
            )
            print(new_constraint.text)
            typer.secho(f"Refined idea #{len(existing_artefacts)}: {new_constraint.get('Name', '(unnamed)')}", fg=typer.colors.GREEN)
        except Exception as e:
            typer.secho(f"Failed to generate idea: {e}", fg=typer.colors.RED)
            continue

    print(f"Last idea: {new_constraint.text}")
    last_idea = extract_json_between_markers(new_constraint.text)
    _save_json(out_file, last_idea)

if __name__ == "__main__":
    app()