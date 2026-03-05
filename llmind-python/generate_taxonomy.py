from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime

from dotenv import load_dotenv
import typer

from data.models import Taxonomy
from data.prompts import IDEA_FIRST_PROMPT, IDEA_REFLECTION_PROMPT, SYSTEM_PROMPT


# =============================
# Provider-agnostic chat interface
# =============================

class ChatSession(Protocol):
    """Stateful chat session that always returns a structured Taxonomy."""
    def send_message(self, content: str) -> Taxonomy: ...


# =============================
# Schema helpers
# =============================

def _make_strict_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively ensure every object node has ``additionalProperties: false``.

    OpenAI's structured outputs (``strict: true``) require this on every
    nested object, including those inside ``$defs``.  Pydantic's
    ``model_json_schema()`` does not add it automatically.
    """
    import copy
    schema = copy.deepcopy(schema)

    def _visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            node.setdefault("additionalProperties", False)
            node.setdefault("properties", {})
        for child in node.get("properties", {}).values():
            _visit(child)
        for child in node.get("$defs", {}).values():
            _visit(child)
        if "items" in node:
            _visit(node["items"])

    _visit(schema)
    return schema


# =============================
# OpenAI / vLLM implementation
# =============================

@dataclass
class OpenAIChat:
    """OpenAI-compatible structured-output chat session.

    Works with the official OpenAI API and any vLLM endpoint that
    supports the OpenAI ``chat.completions`` API + ``json_schema``
    response format.

    Args:
        model:            Model name (e.g. ``"gpt-4o"``, ``"meta-llama/..."``).
        system_message:   System prompt prepended to every session.
        reasoning_effort: ``"low" | "medium" | "high"`` — forwarded to
                          OpenAI reasoning models; ignored for vLLM.
        base_url:         vLLM server URL (e.g. ``"http://localhost:8000/v1"``).
                          When set, ``OPENAI_API_KEY`` is not required.
    """

    model: str
    system_message: str
    reasoning_effort: str = "medium"
    base_url: Optional[str] = None
    _client: Any = field(init=False, repr=False)
    _messages: List[Dict[str, Any]] = field(init=False, repr=False)
    _vllm_response_format: Dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key and not self.base_url:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Provide it via env var or set base_url for a vLLM endpoint."
            )

        self._client = OpenAI(
            api_key=api_key or "vllm",  # vLLM ignores the key value
            base_url=self.base_url,
        )
        self._messages = [{"role": "system", "content": self.system_message}]

        # Pre-build the strict schema for the vLLM path
        self._vllm_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "taxonomy",
                "strict": True,
                "schema": _make_strict_schema(Taxonomy.model_json_schema()),
            },
        }

    def send_message(self, content: str) -> Taxonomy:
        self._messages.append({"role": "user", "content": content})

        if self.base_url:
            # vLLM: standard chat.completions with a manually strictified schema
            completion = self._client.chat.completions.create(
                model=self.model,
                messages=self._messages,
                response_format=self._vllm_response_format,
            )
            raw: str = completion.choices[0].message.content or ""
            taxonomy = Taxonomy.model_validate_json(raw)
        else:
            # OpenAI: beta.parse handles strict schema transformation automatically
            completion = self._client.beta.chat.completions.parse(
                model=self.model,
                messages=self._messages,
                response_format=Taxonomy,
                reasoning_effort=self.reasoning_effort,
            )
            taxonomy = completion.choices[0].message.parsed

        self._messages.append({"role": "assistant", "content": taxonomy.model_dump_json()})
        return taxonomy


# =============================
# Generation core
# =============================

def run_generate(
    chat: ChatSession,
    project_overview: str,
    existing_artefacts: List[Dict[str, Any]],
    num_reflections: int,
    dev_mode: bool = False,
) -> Taxonomy:
    existing_artefacts_string = "\n\n".join(
        f"{art.get('ID')}: {art.get('Description')}" for art in existing_artefacts
    ) or "(none yet)"

    first_prompt = IDEA_FIRST_PROMPT.format(
        project_overview=project_overview,
        existing_artefacts=existing_artefacts_string,
        num_reflections=num_reflections,
    )

    if dev_mode:
        typer.secho("-- Dev mode --", fg=typer.colors.YELLOW)
        typer.secho(first_prompt, fg=typer.colors.YELLOW)
        Path("debug_artefacts.txt").write_text(first_prompt, encoding="utf-8")

    typer.secho("Generating initial taxonomy...", fg=typer.colors.GREEN)
    taxonomy = chat.send_message(first_prompt)
    typer.secho(taxonomy.model_dump_json(indent=2), fg=typer.colors.CYAN)

    # ── Self-refine loop (kept for future use) ──────────────────────────────
    # for i in range(1, num_reflections + 1):
    #     try:
    #         taxonomy = chat.send_message(
    #             IDEA_REFLECTION_PROMPT.format(
    #                 current_round=i, num_reflections=num_reflections
    #             )
    #         )
    #         typer.secho(
    #             f"Reflection {i}/{num_reflections} complete.",
    #             fg=typer.colors.GREEN,
    #         )
    #     except Exception as exc:
    #         typer.secho(
    #             f"Reflection {i}/{num_reflections} failed: {exc}",
    #             fg=typer.colors.RED,
    #         )
    # ────────────────────────────────────────────────────────────────────────

    return taxonomy


# =============================
# Persistence helpers
# =============================

def _save_taxonomy(out_file: Path, taxonomy: Taxonomy, mode: str, model: str) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    dest = out_file.with_name(f"{out_file.stem}_{mode}_{model}_{stamp}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(taxonomy.model_dump_json(indent=2), encoding="utf-8")
    typer.secho(f"Saved taxonomy to: {dest}", fg=typer.colors.BLUE)


# =============================
# Typer CLI
# =============================

app = typer.Typer(
    add_completion=False,
    help="Generate a design-space taxonomy with an LLM.",
)


def _common_options() -> Dict[str, Any]:
    return dict(
        out_file=typer.Option(
            Path("../results/taxonomy/schema"),
            help="Base path for the output JSON file (timestamp is appended).",
        ),
        num_reflections=typer.Option(
            1,
            "--num",
            min=1,
            help="Number of self-review iterations (future use).",
        ),
        ids_file=typer.Option(
            None,
            "-i",
            help="Path to farthest-selected ids JSON from clustering (source=selected).",
        ),
        selected_mode=typer.Option(
            "both",
            "--mode",
            help="'details_only' or 'both' (details + descriptions).",
        ),
        source=typer.Option(
            "selected",
            "--source",
            help="'selected' (ids_file) or 'all_supabase'.",
        ),
        dev_mode=typer.Option(
            False,
            "--dev",
            help="Print and dump the full prompt to debug_artefacts.txt.",
        ),
        reasoning_effort=typer.Option(
            "medium",
            "--reasoning",
            help="Reasoning effort for OpenAI models: 'low', 'medium', or 'high'.",
        ),
        base_url=typer.Option(
            None,
            "--base-url",
            help="vLLM server base URL (e.g. http://localhost:8000/v1). Omit for OpenAI.",
        ),
    )


@app.command("openai")
def openai_generate(
    out_file: Path = _common_options()["out_file"],
    ids_file: Optional[Path] = _common_options()["ids_file"],
    num_reflections: int = _common_options()["num_reflections"],
    selected_mode: str = _common_options()["selected_mode"],
    source: str = _common_options()["source"],
    dev_mode: bool = _common_options()["dev_mode"],
    model_name: str = typer.Option(
        "gpt-5-nano-2025-08-07",
        help="Model name (e.g. 'gpt-4o', 'o3-mini', or a vLLM model path).",
    ),
    reasoning_effort: str = _common_options()["reasoning_effort"],
    base_url: Optional[str] = _common_options()["base_url"],
) -> None:
    """Generate a taxonomy using an OpenAI-compatible model with structured output.

    Supports the official OpenAI API and self-hosted vLLM endpoints.
    """
    load_dotenv()

    from utils.supabase import build_artefacts

    prompt = SYSTEM_PROMPT
    project_overview: str = prompt.get("project", "")
    system_message: str = prompt.get("system", "You are a creative professional designer.")

    artefacts: List[Dict[str, Any]] = []
    try:
        artefacts = build_artefacts(
            source="all_supabase" if source == "all_supabase" else "selected",
            ids_file=ids_file,
            mode="both" if selected_mode == "both" else "details_only",
        )
        label = "all rows" if source == "all_supabase" else "selected rows (ids file)"
        typer.echo(f"Loaded {len(artefacts)} artefacts from Supabase ({label}).")
    except Exception as exc:
        typer.secho(f"Warning: failed to build artefacts: {exc}", fg=typer.colors.YELLOW)

    backend = f"vLLM @ {base_url}" if base_url else "OpenAI"
    typer.echo(f"Backend: {backend}  |  Model: {model_name}")

    chat = OpenAIChat(
        model=model_name,
        system_message=system_message,
        reasoning_effort=reasoning_effort,
        base_url=base_url,
    )

    taxonomy = run_generate(chat, project_overview, artefacts, num_reflections, dev_mode)
    _save_taxonomy(out_file, taxonomy, f"{source}_{selected_mode}", model_name)


if __name__ == "__main__":
    app()
