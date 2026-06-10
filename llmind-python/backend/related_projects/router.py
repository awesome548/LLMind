from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from backend import jobs
from backend.related_projects.service import (
    search_related_projects as search_related_projects_service,
    generate_nodes_from_related_projects,
)
from utils.modes import BackendMode


router = APIRouter(prefix="/api/related-projects", tags=["related-projects"])


class RelatedProject(BaseModel):
    id: Optional[str] = None
    Id: Optional[str] = None
    Name: str = Field(min_length=1)
    Descriptions: str = ""
    Details: str = ""
    Image: Optional[str] = None


class FetchRelatedProjectsRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=400)
    lineage: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    should_query_supabase: bool = True
    limit: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    embedding_model: Optional[str] = Field(default=None, min_length=1)
    match_function: Optional[str] = Field(default=None, min_length=1)


class FetchRelatedProjectsResponse(BaseModel):
    projects: list[RelatedProject] = Field(default_factory=list)


class TaxonomyNodeInput(BaseModel):
    id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    parentid: Optional[str] = None
    isroot: bool = False


class FocusNodeInput(BaseModel):
    id: str = Field(min_length=1)
    topic: str = Field(min_length=1)


class GenerateNodesRequest(BaseModel):
    taxonomy_nodes: list[TaxonomyNodeInput] = Field(min_length=1)
    focus_node: FocusNodeInput
    lineage: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    should_query_supabase: bool = True
    related_projects: Optional[list[RelatedProject]] = None
    model: Optional[str] = Field(default=None, min_length=1)
    mode: BackendMode = BackendMode.openai
    base_url: Optional[str] = Field(default=None, min_length=1)
    reasoning_effort: str = Field(default="medium", min_length=1)


class GeneratedNode(BaseModel):
    node_id: str
    topic: str
    desc: str = ""
    parent_node: str


class GenerateNodesResponse(BaseModel):
    parent_id: str
    options: dict[str, str]
    nodes: list[GeneratedNode]
    related_projects: list[RelatedProject]


@router.post(
    "/search",
    response_model=FetchRelatedProjectsResponse,
    status_code=status.HTTP_200_OK,
)
def search_related_projects(payload: FetchRelatedProjectsRequest) -> FetchRelatedProjectsResponse:
    projects = search_related_projects_service(
        topic=payload.topic,
        lineage=payload.lineage,
        description=payload.description,
        should_query_supabase=payload.should_query_supabase,
        limit=payload.limit,
        similarity_threshold=payload.similarity_threshold,
        embedding_model=payload.embedding_model,
        match_function=payload.match_function,
    )
    validated_projects = [
        RelatedProject.model_validate(project) for project in projects
    ]
    return FetchRelatedProjectsResponse(projects=validated_projects)


@router.post("/generate-nodes", status_code=status.HTTP_202_ACCEPTED)
def generate_nodes(payload: GenerateNodesRequest) -> dict:
    """Start node generation; returns a ``job_id`` to poll at ``GET /api/jobs/{id}``.

    Generation is long (local LLM), so it runs as a background job rather than
    blocking the request. The job result matches ``GenerateNodesResponse``.
    """
    job_id = jobs.submit(
        generate_nodes_from_related_projects,
        focus_node_id=payload.focus_node.id,
        focus_node_topic=payload.focus_node.topic,
        taxonomy_nodes=[node.model_dump() for node in payload.taxonomy_nodes],
        lineage=payload.lineage,
        description=payload.description,
        should_query_supabase=payload.should_query_supabase,
        related_projects=(
            [project.model_dump() for project in payload.related_projects]
            if payload.related_projects is not None
            else None
        ),
        model=payload.model,
        mode=payload.mode,
        base_url=payload.base_url,
        reasoning_effort=payload.reasoning_effort,
    )
    return {"job_id": job_id, "status": "pending"}
