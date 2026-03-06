from __future__ import annotations

from enum import Enum


class BackendMode(str, Enum):
    """Selects the LLM/embedding backend used across pipeline commands."""

    openai = "openai"
    vllm = "vllm"


class ContentMode(str, Enum):
    """Selects which text field(s) are used to generate embeddings."""

    description = "description"   # Descriptions column only
    details = "details"            # Details column only
    hybrid = "hybrid"              # Descriptions + Details concatenated
    all = "all"                    # Embed all three modes (ingest only)
