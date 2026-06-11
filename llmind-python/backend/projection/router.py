from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend import jobs
from backend.projection.service import (
    ServiceError,
    compute_axes,
    compute_metrics,
    generate_at as generate_at_service,
    load_surface,
    locate_nodes as locate_nodes_service,
    peek as peek_service,
)
from utils.modes import BackendMode


router = APIRouter(prefix="/api/projection", tags=["projection"])


# ── /surface ──────────────────────────────────────────────────────────────────


class SurfacePoint(BaseModel):
    id: str
    kind: str
    x: float
    y: float
    z: float | None = None
    cell: list[int]
    cluster: int | None = None
    name: str | None = None


class SurfaceGrid(BaseModel):
    resolution: int


class SurfaceBounds(BaseModel):
    min: float
    max: float


class SurfaceResponse(BaseModel):
    version: int
    dims: int
    grid: SurfaceGrid
    bounds: SurfaceBounds
    density: list[list[int]]
    points: list[SurfacePoint]
    meta: dict[str, Any]


@router.get("/surface", response_model=SurfaceResponse)
def get_surface() -> SurfaceResponse:
    """Return the precomputed corpus background (grid spec, points, density)."""
    try:
        return SurfaceResponse(**load_surface())
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
    # Jaccard overlap between the node's true embedding neighbourhood and its 2D
    # neighbourhood (see service._placement_confidence). None = unscorable.
    confidence: float | None = None
    # Only meaningful on the no-corpus fallback transform path (the projection
    # fell outside the corpus bounds and was soft-clipped); the primary
    # evidence-anchored placement never leaves the corpus footprint.
    clipped: bool = False
    # Corpus-support percentile: how much corpus evidence exists for this point
    # in the ORIGINAL metric, against the corpus's own self-support baseline.
    support: float | None = None


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


# ── Shared node inputs (peek + generate-at) ───────────────────────────────────


class TaxonomyNodeInput(BaseModel):
    id: str
    topic: str
    parentid: Optional[str] = None
    isroot: Optional[bool] = None


class NodeCoordInput(BaseModel):
    """A located taxonomy node — lets the backend derive the parent aspect and
    the "nearby existing ideas" from the same click that picks the seeds."""

    node_id: str
    x: float
    y: float


# ── /peek ─────────────────────────────────────────────────────────────────────


class PeekRequest(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    k: int = Field(default=5, ge=1, le=20)
    taxonomy_nodes: list[TaxonomyNodeInput] = Field(default_factory=list)
    coords: list[NodeCoordInput] = Field(default_factory=list)


class PeekSeed(BaseModel):
    id: str | None = None
    Name: str
    Descriptions: str = ""
    x: float
    y: float


class PeekResponse(BaseModel):
    """What a generation at this location would be briefed with — shown to the
    designer BEFORE any LLM time is spent."""

    seeds: list[PeekSeed] = Field(default_factory=list)
    nearby_options: list[str] = Field(default_factory=list)
    parent_aspect: str | None = None


@router.post("/peek", response_model=PeekResponse)
def peek(payload: PeekRequest) -> PeekResponse:
    """Gap preview (no LLM, no embedding server): the deterministic seed set a
    generate-at would use, nearby already-explored ideas, and the parent aspect
    the click would attach to."""
    try:
        result = peek_service(
            x=payload.x,
            y=payload.y,
            k=payload.k,
            taxonomy_nodes=[node.model_dump() for node in payload.taxonomy_nodes],
            node_coords=[coord.model_dump() for coord in payload.coords],
        )
        return PeekResponse(**result)
    except ServiceError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


# ── /axes ─────────────────────────────────────────────────────────────────────


class AxisPole(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class AxisSpec(BaseModel):
    pole_a: AxisPole
    pole_b: AxisPole


class AxesItem(BaseModel):
    node_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class AxesRequest(BaseModel):
    x: AxisSpec
    y: AxisSpec
    items: list[AxesItem] = Field(default_factory=list)


class AxesPoint(BaseModel):
    id: str
    x: float
    y: float


class AxesItemPoint(BaseModel):
    node_id: str
    x: float
    y: float
    # True when the raw score fell outside the corpus range (rendered as an
    # edge marker — "outside corpus range", same honesty rule as clipping).
    clipped: bool


class AxesMeta(BaseModel):
    # cos(pole_a, pole_b) per axis — near 1.0 means the axis collapses.
    x_pole_sim: float
    y_pole_sim: float
    # Pearson r of corpus x vs y scores — near ±1 means the axes are redundant.
    axis_corr: float


class AxesResponse(BaseModel):
    corpus: list[AxesPoint] = Field(default_factory=list)
    items: list[AxesItemPoint] = Field(default_factory=list)
    meta: AxesMeta


@router.post("/axes", response_model=AxesResponse)
def axes(payload: AxesRequest) -> AxesResponse:
    """Exact bipolar coordinates for the semantic-axes perspective: every corpus
    project (and any taxonomy/candidate items) scored against two designer-chosen
    pole pairs in the original embedding metric."""
    try:
        result = compute_axes(
            x_pole_a=payload.x.pole_a.text,
            x_pole_b=payload.x.pole_b.text,
            y_pole_a=payload.y.pole_a.text,
            y_pole_b=payload.y.pole_b.text,
            items=[item.model_dump() for item in payload.items],
        )
        return AxesResponse(**result)
    except ServiceError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


# ── /metrics ──────────────────────────────────────────────────────────────────


class MetricSpec(BaseModel):
    pole_a: AxisPole
    pole_b: AxisPole


class MetricsRequest(BaseModel):
    """A LIST of bipolar metrics — the Perspectives strips (generalises /axes)."""

    metrics: list[MetricSpec] = Field(min_length=1, max_length=12)
    items: list[AxesItem] = Field(default_factory=list)


class MetricItemPoint(BaseModel):
    node_id: str
    score: float
    # The raw score fell outside the corpus range (rendered at the strip's end).
    clipped: bool


class MetricResult(BaseModel):
    # Full corpus score distribution in [-1, 1] — the strip's rug/density and
    # the basis for percentile sentences (computed client-side).
    corpus: list[float]
    items: list[MetricItemPoint] = Field(default_factory=list)
    # cos(pole_a, pole_b) — near 1.0 means the metric collapses.
    pole_sim: float


class MetricsResponse(BaseModel):
    metrics: list[MetricResult]
    # Pairwise corpus-score correlations; |r| near 1 → redundant metrics.
    corr: list[list[float]]


@router.post("/metrics", response_model=MetricsResponse)
def metrics(payload: MetricsRequest) -> MetricsResponse:
    """Score the corpus + items along a list of bipolar metrics — the data
    behind the Perspectives profile strips (exact cosine, no projection)."""
    try:
        result = compute_metrics(
            metrics=[
                {"pole_a": m.pole_a.text, "pole_b": m.pole_b.text}
                for m in payload.metrics
            ],
            items=[item.model_dump() for item in payload.items],
        )
        return MetricsResponse(**result)
    except ServiceError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc


# ── /generate-at ──────────────────────────────────────────────────────────────


class GenerateAtRequest(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    focus_node_id: str = Field(min_length=1)
    focus_node_topic: str = Field(min_length=1)
    taxonomy_nodes: list[TaxonomyNodeInput] = Field(default_factory=list)
    lineage: list[str] = Field(default_factory=list)
    k: int = Field(default=5, ge=1, le=20)
    # Current node coordinates; when present the backend derives the parent
    # aspect spatially (the provided focus node is only the fallback).
    coords: list[NodeCoordInput] | None = None
    # None → the service derives the backend from settings.vector_store, so the
    # design space generates with the same stack it embeds/retrieves with.
    mode: BackendMode | None = None
    reasoning_effort: str = "medium"
    # The active candidate's brief — context only (squiggle hypothesis, Part 10):
    # convergence material feeding a divergence step, logged for the A/B.
    brief: str | None = None


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
        node_coords=(
            [coord.model_dump() for coord in payload.coords]
            if payload.coords is not None
            else None
        ),
        mode=payload.mode,
        reasoning_effort=payload.reasoning_effort,
        brief=payload.brief,
    )
    return {"job_id": job_id, "status": "pending"}
