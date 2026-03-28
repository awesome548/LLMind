from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.taxonomy.service import TaxonomyServiceError, generate_taxonomy as generate_taxonomy_service
from utils.modes import BackendMode, ContentMode


router = APIRouter(prefix="/api/taxonomy", tags=["taxonomy"])


class GenerateTaxonomyRequest(BaseModel):
    project_overview: str = Field(min_length=1, max_length=10000)
    num_reflections: int = Field(default=1, ge=1)
    content_mode: ContentMode = ContentMode.details
    ids_file: Optional[str] = Field(default=None, min_length=1)
    model_name: str = Field(default="gpt-5-nano-2025-08-07", min_length=1)
    reasoning_effort: str = Field(default="medium", min_length=1)
    mode: BackendMode = BackendMode.openai
    base_url: Optional[str] = Field(default=None, min_length=1)


class OptionResponse(BaseModel):
    name: str
    desc: str


class AspectResponse(BaseModel):
    name: str
    desc: str
    options: list[OptionResponse] = Field(default_factory=list)


class GenerateTaxonomyResponse(BaseModel):
    aspects: list[AspectResponse] = Field(default_factory=list)


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
            model_name=payload.model_name,
            reasoning_effort=payload.reasoning_effort,
            mode=payload.mode,
            base_url=payload.base_url,
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
        return GenerateTaxonomyResponse(aspects=aspects)
    except TaxonomyServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
