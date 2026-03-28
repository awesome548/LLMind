from __future__ import annotations

from pathlib import Path
from typing import Any

from utils.modes import BackendMode, ContentMode
from generate_taxonomy import generate_taxonomy as _generate_taxonomy


class TaxonomyServiceError(RuntimeError):
    """Raised when taxonomy generation fails."""


def generate_taxonomy(
    *,
    project_overview: str | None = None,
    num_reflections: int = 1,
    content_mode: ContentMode = ContentMode.details,
    ids_file: str | None = None,
    model_name: str = "gpt-5-nano-2025-08-07",
    reasoning_effort: str = "medium",
    mode: BackendMode = BackendMode.openai,
    base_url: str | None = None,
) -> dict[str, Any]:
    resolved_ids_file = Path(ids_file) if ids_file else None

    try:
        taxonomy = _generate_taxonomy(
            project_overview_input=project_overview,
            ids_file=resolved_ids_file,
            num_reflections=num_reflections,
            content_mode=content_mode,
            model_name=model_name,
            reasoning_effort=reasoning_effort,
            mode=mode,
            base_url=base_url,
        )
        if taxonomy is None:
            raise TaxonomyServiceError("Taxonomy generation returned no result.")
        return taxonomy.model_dump()
    except Exception as exc:
        raise TaxonomyServiceError("Failed to generate taxonomy.") from exc
