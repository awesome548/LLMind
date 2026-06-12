"""Reflection drafting — ITERATION-PLAN Part 12 C2.

Dalsgaard & Halskov's process-reflection tool died of documentation burden;
LLMind inverts it: the system drafts the one-line rationale for what the
designer just did, and the designer accepts/edits/skips. The draft is a
STARTING POINT — `edited` is tracked client-side because the difference is
study data.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, status
from pydantic import BaseModel, Field

from backend import jobs
from backend.corpus.llm import budgeted_completion
from utils.prompts import REFLECT_PROMPT

router = APIRouter(prefix="/api/reflections", tags=["reflections"])


def parse_reflection(content: str, reasoning: str = "") -> str | None:
    """The drafted sentence from an LLM response. Pure.

    First non-empty content line, stripped of wrapping quotes; when a capped
    thinking model answered nothing, the last non-empty reasoning-tail line
    (thinking models end on their conclusion).
    """
    source = content if content.strip() else ""
    if not source:
        tail_lines = [ln.strip() for ln in (reasoning or "")[-300:].splitlines() if ln.strip()]
        source = tail_lines[-1] if tail_lines else ""
    line = next((ln.strip() for ln in source.splitlines() if ln.strip()), "")
    line = line.strip('"“”‘’\'')
    return line[:200] or None


def draft_reflection(context: str) -> Dict[str, Any]:
    prompt = REFLECT_PROMPT.replace("{{EVENT}}", context.strip())
    try:
        content, reasoning = budgeted_completion(prompt)
    except Exception:  # noqa: BLE001 — drafting is assistance, never a gate
        # An LLM failure degrades to an empty draft: the chip opens blank and
        # the designer types (or skips). Same contract as an empty answer.
        return {"draft": ""}
    return {"draft": parse_reflection(content, reasoning) or ""}


class ReflectionDraftRequest(BaseModel):
    context: str = Field(min_length=1, max_length=600)


@router.post("/draft", status_code=status.HTTP_202_ACCEPTED)
def post_draft(payload: ReflectionDraftRequest) -> dict[str, Any]:
    """Draft a one-line rationale for an exploration event (async job).
    Returns ``{job_id}``; poll ``GET /api/jobs/{id}`` → ``{draft}``."""
    job_id = jobs.submit(draft_reflection, payload.context)
    return {"job_id": job_id, "status": "pending"}
