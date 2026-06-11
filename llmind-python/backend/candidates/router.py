from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend import jobs
from backend.candidates.service import (
    CandidateServiceError,
    alignment as alignment_service,
    draft_brief as draft_brief_service,
)
from utils.modes import BackendMode

router = APIRouter(prefix="/api/candidates", tags=["candidates"])


# ── /draft-brief ──────────────────────────────────────────────────────────────


class DraftBriefAspect(BaseModel):
    aspect: str = Field(min_length=1)
    option: str = Field(min_length=1)
    desc: str = ""


class DraftBriefRequest(BaseModel):
    aspects: list[DraftBriefAspect] = Field(min_length=1)
    project_overview: str = ""
    # None → derived from settings.vector_store, like generate-at.
    mode: BackendMode | None = None


@router.post("/draft-brief", status_code=status.HTTP_202_ACCEPTED)
def draft_brief(payload: DraftBriefRequest) -> dict[str, Any]:
    """Draft the candidate's brief from its committed choices (LLM — async job).

    Returns ``{job_id}``; poll ``GET /api/jobs/{job_id}``. The result is
    ``{brief}`` — a starting point the designer edits, never the final word.
    """
    job_id = jobs.submit(
        draft_brief_service,
        aspects=[a.model_dump() for a in payload.aspects],
        project_overview=payload.project_overview,
        mode=payload.mode,
    )
    return {"job_id": job_id, "status": "pending"}


# ── /alignment ────────────────────────────────────────────────────────────────


class AlignmentOption(BaseModel):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class AlignmentAspect(BaseModel):
    aspect_id: str = Field(min_length=1)
    chosen: AlignmentOption
    alternatives: list[AlignmentOption] = Field(default_factory=list)


class AlignmentRequest(BaseModel):
    """The candidate's two layers + the per-aspect option field to score against."""

    brief: str = Field(min_length=1)
    composition: str = Field(min_length=1)
    aspects: list[AlignmentAspect] = Field(default_factory=list)


class AlignmentTopAlternative(BaseModel):
    id: str
    score: float


class AlignmentAspectResult(BaseModel):
    aspect_id: str
    # cos(brief, chosen option) — how strongly the brief expresses the commitment.
    chosen_score: float
    # The competitor the brief is most similar to (None when no alternatives).
    top_alternative: AlignmentTopAlternative | None = None
    # True when the brief leans toward the alternative over the chosen option.
    leans_away: bool


class AlignmentResponse(BaseModel):
    # cos(brief, composition) — overall concept↔commitments agreement.
    agreement: float
    per_aspect: list[AlignmentAspectResult] = Field(default_factory=list)


@router.post("/alignment", response_model=AlignmentResponse)
def alignment(payload: AlignmentRequest) -> AlignmentResponse:
    """Measure how the candidate's brief (identity) and choices (commitments)
    agree — overall and per aspect — in the original embedding metric."""
    try:
        result = alignment_service(
            brief=payload.brief,
            composition=payload.composition,
            aspects=[a.model_dump() for a in payload.aspects],
        )
        return AlignmentResponse(**result)
    except CandidateServiceError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
