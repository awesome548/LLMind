from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend import jobs
from backend.projection.service import (
    ServiceError,
    generate_at as generate_at_service,
    load_surface,
    locate_nodes as locate_nodes_service,
)
from utils.modes import BackendMode


router = APIRouter(prefix="/api/projection", tags=["projection"])


# ── /surface ──────────────────────────────────────────────────────────────────


@router.get("/surface")
def get_surface() -> dict[str, Any]:
    """Return the precomputed corpus background (grid spec, points, density)."""
    try:
        return load_surface()
    except ServiceError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


# ── /locate ─────────────────────────────────────────────────────────────────


class LocateItem(BaseModel):
    node_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class LocateRequest(BaseModel):
    items: list[LocateItem] = Field(default_factory=list)


class LocatedPoint(BaseModel):
    node_id: str
    x: float
    y: float
    z: float | None = None


class LocateResponse(BaseModel):
    points: list[LocatedPoint] = Field(default_factory=list)


@router.post("/locate", response_model=LocateResponse)
def locate(payload: LocateRequest) -> LocateResponse:
    """Embed taxonomy-node text and place each node in the frozen design space."""
    try:
        located = locate_nodes_service([item.model_dump() for item in payload.items])
        return LocateResponse(points=[LocatedPoint(**p) for p in located])
    except ServiceError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


# ── /generate-at ──────────────────────────────────────────────────────────────


class TaxonomyNodeInput(BaseModel):
    id: str
    topic: str
    parentid: Optional[str] = None
    isroot: Optional[bool] = None


class GenerateAtRequest(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    focus_node_id: str = Field(min_length=1)
    focus_node_topic: str = Field(min_length=1)
    taxonomy_nodes: list[TaxonomyNodeInput] = Field(default_factory=list)
    lineage: list[str] = Field(default_factory=list)
    k: int = Field(default=5, ge=1, le=20)
    # None → the service derives the backend from settings.vector_store, so the
    # design space generates with the same stack it embeds/retrieves with.
    mode: BackendMode | None = None
    reasoning_effort: str = "medium"


@router.post("/generate-at", status_code=status.HTTP_202_ACCEPTED)
def generate_at(payload: GenerateAtRequest) -> dict[str, Any]:
    """Start generation seeded by the corpus around a clicked location.

    Returns a ``job_id`` immediately; poll ``GET /api/jobs/{job_id}`` for the
    result. Generation is long (local LLM), so it runs as a background job rather
    than blocking the request (see backend/jobs.py).
    """
    job_id = jobs.submit(
        generate_at_service,
        x=payload.x,
        y=payload.y,
        focus_node_id=payload.focus_node_id,
        focus_node_topic=payload.focus_node_topic,
        taxonomy_nodes=[node.model_dump() for node in payload.taxonomy_nodes],
        lineage=payload.lineage,
        k=payload.k,
        mode=payload.mode,
        reasoning_effort=payload.reasoning_effort,
    )
    return {"job_id": job_id, "status": "pending"}
