from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from config import settings
from utils.modes import BackendMode, ContentMode
from generate_taxonomy import TaxonomyGenerationError, generate_taxonomy as _generate_taxonomy

logger = logging.getLogger(__name__)


class TaxonomyServiceError(RuntimeError):
    """Raised when taxonomy generation fails."""


def generate_taxonomy(
    *,
    project_overview: str | None = None,
    num_reflections: int = 1,
    content_mode: ContentMode = ContentMode.details,
    ids_file: str | None = "50_selected_updated.json",
    reasoning_effort: str = "medium",
    mode: BackendMode = BackendMode.openai,
) -> dict[str, Any]:
    request_id = uuid4().hex[:12]

    # No model-name and base-url param from service (backend should be fixed to use config)

    if ids_file:
        candidate_ids_file = Path(ids_file)
        resolved_ids_file = (settings.data_dir / candidate_ids_file).resolve()
        if not resolved_ids_file.is_relative_to(settings.data_dir.resolve()):
            raise TaxonomyServiceError("ids_file path traversal detected")
    else:
        resolved_ids_file = None

    model_name = settings.openai_node_model if mode == BackendMode.openai else settings.vllm_model
    base_url = settings.vllm_base_url if mode == BackendMode.vllm else None

    logger.info(
        "taxonomy.generate request started request_id=%s mode=%s model=%s content_mode=%s ids_file=%s",
        request_id,
        mode.value,
        model_name,
        content_mode.value,
        str(resolved_ids_file) if resolved_ids_file is not None else None,
    )

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
        logger.info(
            "taxonomy.generate request succeeded request_id=%s aspect_count=%s",
            request_id,
            len(taxonomy.aspects),
        )
        return taxonomy.model_dump()
    except TaxonomyGenerationError as exc:
        logger.exception(
            "taxonomy.generate request failed request_id=%s stage=%s context=%s",
            request_id,
            exc.stage,
            exc.context,
        )
        raise TaxonomyServiceError(
            f"Failed to generate taxonomy. request_id={request_id} stage={exc.stage}"
        ) from exc
    except Exception as exc:
        logger.exception(
            "taxonomy.generate request failed request_id=%s stage=service_unhandled",
            request_id,
        )
        raise TaxonomyServiceError(
            f"Failed to generate taxonomy. request_id={request_id} stage=service_unhandled"
        ) from exc
