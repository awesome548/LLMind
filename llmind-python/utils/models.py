from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ── Taxonomy schema (LLM structured output) ───────────────────────────────────

class Option(BaseModel):
    name: str
    desc: str  # concise description for embedding-based retrieval


class Aspect(BaseModel):
    name: str
    desc: str  # concise description for embedding-based retrieval
    options: list[Option] = Field(default_factory=list)


class Taxonomy(BaseModel):
    aspects: list[Aspect] = Field(default_factory=list)


# ── Scraper record ─────────────────────────────────────────────────────────────

class ProjectRecord(BaseModel):
    url: str
    Name: str
    Descriptions: str
    Details: str
    image_href: str | None = None
    html_main: str | None = None


# ── Pipeline record ────────────────────────────────────────────────────────────

class EmbedRecord(BaseModel):
    """A record ready for embedding: the source doc ID and the context text."""

    model_config = ConfigDict(frozen=True)

    media_doc_id: str
    context: str
