"""Embedding client factories for OpenAI and vLLM backends."""

from __future__ import annotations

from openai import OpenAI

from config import settings


def build_openai_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")
    return OpenAI(api_key=settings.openai_api_key)


def build_vllm_client(base_url: str) -> OpenAI:
    """Create an OpenAI-compatible client pointed at a local vLLM server."""
    return OpenAI(api_key="vllm", base_url=base_url)
