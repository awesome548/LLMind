from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.corpus.service import (
    CorpusServiceError,
    get_project,
    relevance_scores,
    similar_projects,
)

router = APIRouter(prefix="/api/corpus", tags=["corpus"])


class CorpusProjectResponse(BaseModel):
    id: str
    Name: str
    Descriptions: str
    Details: str
    Image: str | None = None


class SimilarProjectsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=5, ge=1, le=20)


class SimilarProject(CorpusProjectResponse):
    score: float


class SimilarProjectsResponse(BaseModel):
    projects: list[SimilarProject] = Field(default_factory=list)


@router.post("/similar", response_model=SimilarProjectsResponse)
def post_similar_projects(payload: SimilarProjectsRequest) -> SimilarProjectsResponse:
    """Closest corpus precedents to a text, ranked by true (original-metric)
    cosine similarity — used for design candidates ("my design as a whole is
    most like these real projects")."""
    try:
        rows = similar_projects(payload.text, payload.k)
        return SimilarProjectsResponse(projects=[SimilarProject(**row) for row in rows])
    except CorpusServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc


class RelevanceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class RelevanceScore(BaseModel):
    id: str
    score: float


class RelevanceResponse(BaseModel):
    scores: list[RelevanceScore] = Field(default_factory=list)
    min: float
    max: float


@router.post("/relevance", response_model=RelevanceResponse)
def post_relevance(payload: RelevanceRequest) -> RelevanceResponse:
    """True cosine similarity of every corpus project to a text — powers the
    design-space relevance lens (scores-only; min/max included so the client
    can normalise and label the painting as RELATIVE relevance)."""
    try:
        return RelevanceResponse(**relevance_scores(payload.text))
    except CorpusServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc


@router.get("/projects/{project_id}", response_model=CorpusProjectResponse)
def get_corpus_project(project_id: str) -> CorpusProjectResponse:
    """Metadata for one corpus project — the inspectable dots on the design-space surface."""
    try:
        return CorpusProjectResponse(**get_project(project_id))
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown corpus project: {project_id}",
        ) from exc
