from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from backend.taxonomy.service import TaxonomyServiceError, generate_taxonomy as generate_taxonomy_service
from utils.modes import BackendMode, ContentMode


router = APIRouter(prefix="/api/taxonomy", tags=["taxonomy"])


class GenerateTaxonomyRequest(BaseModel):
    project_overview: str = Field(min_length=1, max_length=10000)
    num_reflections: int = Field(default=1, ge=1)
    content_mode: ContentMode = ContentMode.details
    ids_file: Optional[str] = Field(default=None, min_length=1)
    reasoning_effort: Literal["low", "medium", "high"] = "medium"
    mode: BackendMode = BackendMode.openai

    @field_validator("ids_file")
    @classmethod
    def reject_path_traversal(cls, v: str | None) -> str | None:
        if v and (v.startswith("/") or ".." in v):
            raise ValueError("ids_file must be a simple filename, not an absolute or traversal path")
        return v


class OptionResponse(BaseModel):
    name: str
    desc: str


class AspectResponse(BaseModel):
    name: str
    desc: str
    options: list[OptionResponse] = Field(default_factory=list)


class GenerateTaxonomyResponse(BaseModel):
    aspects: list[AspectResponse] = Field(default_factory=list)
    # Cosine similarity of the project overview to the corpus centroid (original
    # embedding metric). Lets the UI warn when the design-space background (a
    # media-architecture corpus) doesn't apply to this brief. None = unscored
    # (embedding server unavailable) — never blocks generation.
    corpus_similarity: float | None = None


def _corpus_similarity(project_overview: str) -> float | None:
    """Best-effort similarity of the brief to the corpus (None on any failure)."""
    try:
        import numpy as np

        from backend.corpus.service import load_corpus_vectors
        from config import settings
        from utils.clients import build_vllm_client

        ids, vecs = load_corpus_vectors()
        if not ids:
            return None
        client = build_vllm_client(settings.vllm_base_url)
        response = client.embeddings.create(
            model=settings.vllm_embed_model, input=[project_overview]
        )
        query = np.asarray(response.data[0].embedding, dtype=float)
        norm = float(np.linalg.norm(query)) or 1.0
        centroid = vecs.mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid)) or 1.0
        return float((query / norm) @ (centroid / centroid_norm))
    except Exception:  # noqa: BLE001 — advisory metric only
        return None


@router.post(
    "/generate",
    response_model=GenerateTaxonomyResponse,
    status_code=status.HTTP_200_OK,
)
def generate_taxonomy(payload: GenerateTaxonomyRequest) -> GenerateTaxonomyResponse:
    try:
        result = generate_taxonomy_service(
            project_overview=payload.project_overview,
            num_reflections=payload.num_reflections,
            content_mode=payload.content_mode,
            ids_file=payload.ids_file,
            reasoning_effort=payload.reasoning_effort,
            mode=payload.mode,
        )
        aspects = [
            AspectResponse(
                name=aspect["name"],
                desc=aspect["desc"],
                options=[
                    OptionResponse(name=opt["name"], desc=opt["desc"])
                    for opt in aspect.get("options", [])
                ],
            )
            for aspect in result.get("aspects", [])
        ]
        return GenerateTaxonomyResponse(
            aspects=aspects,
            corpus_similarity=_corpus_similarity(payload.project_overview),
        )
    except TaxonomyServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
