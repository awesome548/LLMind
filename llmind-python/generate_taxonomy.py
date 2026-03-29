from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime

import typer

from config import settings
from utils.models import Taxonomy
from utils.prompts import IDEA_FIRST_PROMPT, IDEA_REFLECTION_PROMPT, SYSTEM_PROMPT
from utils.modes import BackendMode, ContentMode
from utils.clients import build_openai_client, build_vllm_client

TAXONOMY_DIR = settings.taxonomy_dir
logger = logging.getLogger(__name__)


class TaxonomyGenerationError(RuntimeError):
    """Structured failure for taxonomy generation internals."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        context: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.context = context or {}


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
        if self.base_url:
            self._client = build_vllm_client(self.base_url)
        else:
            self._client = build_openai_client()
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
        import openai
        from pydantic import ValidationError

        outgoing = [*self._messages, {"role": "user", "content": content}]

        if self.base_url:
            # vLLM: standard chat.completions with a manually strictified schema
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=outgoing,
                    response_format=self._vllm_response_format,
                )
            except openai.APIError as exc:
                raise TaxonomyGenerationError(
                    "Failed to call vLLM chat completion.",
                    stage="provider_request",
                    context={
                        "provider": "vllm",
                        "model": self.model,
                        "base_url": self.base_url,
                    },
                ) from exc

            raw: str = completion.choices[0].message.content or ""
            try:
                taxonomy = Taxonomy.model_validate_json(raw)
            except ValidationError as exc:
                raise TaxonomyGenerationError(
                    "Failed to validate vLLM taxonomy JSON response.",
                    stage="provider_response_validation",
                    context={
                        "provider": "vllm",
                        "model": self.model,
                        "response_length": len(raw),
                    },
                ) from exc
        else:
            # OpenAI: beta.parse handles strict schema transformation automatically
            try:
                completion = self._client.beta.chat.completions.parse(
                    model=self.model,
                    messages=outgoing,
                    response_format=Taxonomy,
                    reasoning_effort=self.reasoning_effort,
                )
            except openai.APIError as exc:
                raise TaxonomyGenerationError(
                    "Failed to call OpenAI structured parse endpoint.",
                    stage="provider_request",
                    context={
                        "provider": "openai",
                        "model": self.model,
                        "reasoning_effort": self.reasoning_effort,
                    },
                ) from exc
            taxonomy = completion.choices[0].message.parsed
            if taxonomy is None:
                raise TaxonomyGenerationError(
                    "OpenAI response did not include a parsed taxonomy payload.",
                    stage="provider_response_validation",
                    context={"provider": "openai", "model": self.model},
                )

        # Only persist messages after both API call and validation succeed
        self._messages = [*outgoing, {"role": "assistant", "content": taxonomy.model_dump_json()}]
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

    if dev_mode:
        typer.echo(taxonomy.model_dump_json(indent=2))

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

def generate_taxonomy(
    project_overview_input: str,
    ids_file: Path | None,
    num_reflections: int,
    content_mode: ContentMode,
    model_name: str | None = None,
    reasoning_effort: str = "medium",
    mode: BackendMode = BackendMode.openai,
    base_url: str | None = None,
) -> Taxonomy:
    """Generate a taxonomy using an OpenAI-compatible model with structured output.

    Generation backends:
      openai  Use OpenAI's hosted API (requires OPENAI_API_KEY).
      vllm    Use a local vLLM server (OpenAI-compatible). Start it with:
                vllm serve <model>
    """
    from utils.supabase import build_artefacts

    prompt = SYSTEM_PROMPT
    project_overview: str = project_overview_input if project_overview_input else prompt.get("project", "")
    system_message: str = prompt.get("system", "You are a creative professional designer.")

    artefacts: List[Dict[str, Any]] = []

    if ids_file is not None and ids_file.exists():
        try:
            artefacts = build_artefacts(
                source="selected",
                ids_file=ids_file,
                content_mode=content_mode,
                max_projects=None,
            )
            typer.echo(f"Loaded {len(artefacts)} artefacts from Supabase (selected).")
        except Exception as exc:
            typer.secho(f"Warning: failed to build artefacts: {exc}", fg=typer.colors.YELLOW)
            logger.exception(
                "taxonomy.generate artefact build failed; continuing with empty artefacts",
                extra={
                    "stage": "artefact_build",
                    "ids_file": str(ids_file),
                    "content_mode": content_mode.value,
                },
            )
    elif ids_file is not None:
        raise TaxonomyGenerationError(
            f"ids_file '{ids_file}' was provided but does not exist.",
            stage="input_validation",
            context={"ids_file": str(ids_file)},
        )
    # ids_file is None → proceed with empty artefacts

    resolved_base_url = base_url if mode == BackendMode.vllm else None
    backend_label = f"vLLM @ {base_url}" if mode == BackendMode.vllm else "OpenAI"
    typer.echo(f"Backend: {backend_label}  |  Model: {model_name}")
    logger.info(
        "taxonomy.generate provider selected",
        extra={
            "stage": "provider_select",
            "mode": mode.value,
            "model_name": model_name,
            "base_url": resolved_base_url,
            "artefact_count": len(artefacts),
        },
    )

    chat = OpenAIChat(
        model=model_name,
        system_message=system_message,
        reasoning_effort=reasoning_effort,
        base_url=resolved_base_url,
    )

    taxonomy = run_generate(chat, project_overview, artefacts, num_reflections, dev_mode=False) # Always disable dev_mode for this API 
    return taxonomy

# =============================
# Persistence helpers
# =============================

def _save_taxonomy(out_file: Path, taxonomy: Taxonomy, mode: str, model: str) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    dest = out_file / f"tax_{mode}_{model}_{stamp}.json"
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
        output=typer.Option(
            TAXONOMY_DIR, 
            "--output", 
            "-o", 
            help="Write results to JSON file"
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
        content_mode=typer.Option(
            ContentMode.details,
            "--content-mode",
            "-c",
            help="Which media_doc column to use: 'description', 'details', or 'hybrid'.",
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
        mode=typer.Option(
            BackendMode.openai,
            "--mode",
            "-m",
            help="Generation backend: 'openai' or 'vllm' (local).",
        ),
        base_url=typer.Option(
            settings.vllm_base_url,
            "--base-url",
            help="vLLM server base URL (used when --mode=vllm).",
        ),
    )


@app.command()
def generate_tax(
    output: Optional[Path] = _common_options()["output"],
    ids_file: Optional[Path] = _common_options()["ids_file"],
    num_reflections: int = _common_options()["num_reflections"],
    content_mode: ContentMode = _common_options()["content_mode"],
    dev_mode: bool = _common_options()["dev_mode"],
    model_name: str = typer.Option(
        "gpt-5-nano-2025-08-07",
        help="Model name (e.g. 'gpt-4o', 'o3-mini', or a vLLM model path).",
    ),
    reasoning_effort: str = _common_options()["reasoning_effort"],
    mode: BackendMode = _common_options()["mode"],
    base_url: str = _common_options()["base_url"],
) -> None:
    """Generate a taxonomy using an OpenAI-compatible model with structured output.

    Generation backends:
      openai  Use OpenAI's hosted API (requires OPENAI_API_KEY).
      vllm    Use a local vLLM server (OpenAI-compatible). Start it with:
                vllm serve <model>
    """
    from utils.supabase import build_artefacts

    prompt = SYSTEM_PROMPT
    project_overview: str = prompt.get("project", "")
    system_message: str = prompt.get("system", "You are a creative professional designer.")

    artefacts: List[Dict[str, Any]] = []
    try:
        artefacts = build_artefacts(
            source="selected" if ids_file else "all_supabase",
            ids_file=ids_file,
            content_mode=content_mode,
            max_projects=50 if ids_file is None else None,
        )
        label = "selected" if ids_file is not None else "all-rows"
        typer.echo(f"Loaded {len(artefacts)} artefacts from Supabase ({label}).")
    except Exception as exc:
        typer.secho(f"Warning: failed to build artefacts: {exc}", fg=typer.colors.YELLOW)

    resolved_base_url = base_url if mode == BackendMode.vllm else None
    backend_label = f"vLLM @ {base_url}" if mode == BackendMode.vllm else "OpenAI"
    typer.echo(f"Backend: {backend_label}  |  Model: {model_name}")

    chat = OpenAIChat(
        model=model_name,
        system_message=system_message,
        reasoning_effort=reasoning_effort,
        base_url=resolved_base_url,
    )

    taxonomy = run_generate(chat, project_overview, artefacts, num_reflections, dev_mode)
    _save_taxonomy(output, taxonomy, f"{label}-{content_mode.value}", model_name)


if __name__ == "__main__":
    app()
