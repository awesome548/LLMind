from __future__ import annotations

import hashlib

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend import jobs
from backend.corpus.annotate import annotate_taxonomy, taxonomy_hash
from backend.corpus.cell import generate_cell
from backend.corpus.rationale import (
    draft_rationales,
    probe_missing_aspect,
    rationale_set_hash,
)
from backend.corpus.service import (
    CorpusServiceError,
    get_project,
    relevance_scores,
    similar_projects,
)

router = APIRouter(prefix="/api/corpus", tags=["corpus"])


class AnnotateOption(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    desc: str = ""


class AnnotateRequest(BaseModel):
    """The designer's option set (aspect grouping is irrelevant to membership
    judgments, so the request is flat). Part 12 A2."""

    options: list[AnnotateOption] = Field(min_length=1, max_length=200)


class AnnotateJobResponse(BaseModel):
    job_id: str
    status: str


@router.post(
    "/annotate", response_model=AnnotateJobResponse, status_code=status.HTTP_202_ACCEPTED
)
def post_annotate(payload: AnnotateRequest) -> AnnotateJobResponse:
    """Annotate the corpus against the taxonomy's options (Halskov-style
    schema population, automated — Part 12 A2). Returns a ``job_id``; poll
    ``GET /api/jobs/{job_id}``. Cached per option content, so unchanged
    options resolve instantly on re-runs."""
    options = [o.model_dump() for o in payload.options]
    # Keyed on the option-set identity: concurrent clients (several browser
    # sessions, the schema AND cross-tab views) share ONE running job instead
    # of racing the local LLM through identical per-option judgments.
    job_id = jobs.submit_keyed(
        f"annotate:{taxonomy_hash(options)}", lambda: annotate_taxonomy(options)
    )
    return AnnotateJobResponse(job_id=job_id, status="pending")


class CellOption(BaseModel):
    name: str = Field(min_length=1)
    desc: str = ""


class GenerateCellRequest(BaseModel):
    """An empty cross-tab cell (Part 12 B2): two option poles + the
    half-matching precedents the frontend already holds from annotation."""

    aspect_a: str = Field(min_length=1)
    option_a: CellOption
    aspect_b: str = Field(min_length=1)
    option_b: CellOption
    exemplar_ids: list[str] = Field(default_factory=list, max_length=12)


@router.post(
    "/generate-cell",
    response_model=AnnotateJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def post_generate_cell(payload: GenerateCellRequest) -> AnnotateJobResponse:
    """Generate ONE concept into an empty option×option cell (the
    morphological-combination → candidate-skeleton flow). Returns a
    ``job_id``; poll ``GET /api/jobs/{job_id}``."""
    job_id = jobs.submit(
        lambda: generate_cell(
            aspect_a=payload.aspect_a,
            option_a=payload.option_a.model_dump(),
            aspect_b=payload.aspect_b,
            option_b=payload.option_b.model_dump(),
            exemplar_ids=payload.exemplar_ids,
        )
    )
    return AnnotateJobResponse(job_id=job_id, status="pending")


class RationaleOption(BaseModel):
    name: str = Field(min_length=1)
    count: int = Field(ge=0)


class RationaleAspect(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    desc: str = ""
    options: list[RationaleOption] = Field(default_factory=list, max_length=40)


class RationaleRequest(BaseModel):
    """The taxonomy's aspects with their annotation counts (Part 13 L-A) —
    the rationale cites the evidence, so the counts travel with the names."""

    aspects: list[RationaleAspect] = Field(min_length=1, max_length=24)
    n_projects: int = Field(default=0, ge=0)


@router.post(
    "/rationale", response_model=AnnotateJobResponse, status_code=status.HTTP_202_ACCEPTED
)
def post_rationale(payload: RationaleRequest) -> AnnotateJobResponse:
    """One-line per-aspect rationale ("why this dimension?"), grounded in the
    annotation counts. Returns a ``job_id``; poll ``GET /api/jobs/{id}`` →
    ``{rationales: {<aspect_id>: str}}``. Cached per aspect content+evidence."""
    aspects = [a.model_dump() for a in payload.aspects]
    job_id = jobs.submit_keyed(
        f"rationale:{rationale_set_hash(aspects)}",
        lambda: draft_rationales(aspects, payload.n_projects),
    )
    return AnnotateJobResponse(job_id=job_id, status="pending")


class MissingAspectRequest(BaseModel):
    """The coverage probe (Part 13 L-A): the frontend computes which corpus
    projects the taxonomy describes poorly; this asks what dimension they
    exemplify that the taxonomy misses."""

    aspect_names: list[str] = Field(min_length=1, max_length=24)
    project_ids: list[str] = Field(min_length=1, max_length=6)


@router.post(
    "/missing-aspect",
    response_model=AnnotateJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def post_missing_aspect(payload: MissingAspectRequest) -> AnnotateJobResponse:
    """Propose missing aspect(s) from poorly-covered projects. Returns a
    ``job_id``; poll ``GET /api/jobs/{id}`` → ``{proposals: [{name, desc,
    reason}]}``. Keyed so concurrent probes over the same inputs share a job."""
    key_src = ",".join(sorted(payload.project_ids)) + "|" + ",".join(sorted(payload.aspect_names))
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:16]
    job_id = jobs.submit_keyed(
        f"missing-aspect:{key}",
        lambda: probe_missing_aspect(payload.aspect_names, payload.project_ids),
    )
    return AnnotateJobResponse(job_id=job_id, status="pending")


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
