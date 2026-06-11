"""Candidate (dual-layer design) services — ITERATION-PLAN Part 10.

A candidate has two layers: a BRIEF (the designer's project-style prose — what
the design *is*) and CHOICES (one option per aspect — what it *commits to*).

* ``draft_brief`` — LLM-drafts the brief from the choices, killing the blank
  page; the designer edits the result.
* ``alignment``   — measures how the two layers agree: cos(brief, composition)
  plus, per aspect, whether the brief actually leans toward the chosen option
  or toward its strongest competitor (defined by data: the alternative most
  similar to the brief).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from config import settings
from utils.clients import build_vllm_client, build_openai_client
from utils.modes import BackendMode
from utils.prompts import DRAFT_BRIEF_PROMPT
from backend.corpus.service import CorpusServiceError, embed_texts


class CandidateServiceError(RuntimeError):
    """Raised when a candidate operation's external dependency fails."""


def _resolve_mode(mode: BackendMode | None) -> BackendMode:
    if mode is not None:
        return mode
    return BackendMode.vllm if settings.vector_store == "local" else BackendMode.openai


def render_draft_brief_prompt(
    aspects: Sequence[Dict[str, str]], project_overview: str
) -> str:
    """Fill DRAFT_BRIEF_PROMPT. Pure — unit-testable without a model."""
    choices = "\n".join(
        f"- {a['aspect']}: {a['option']}"
        + (f" — {a['desc']}" if a.get("desc") else "")
        for a in aspects
    )
    return DRAFT_BRIEF_PROMPT.replace(
        "{{PROJECT_OVERVIEW}}", project_overview.strip() or "(none provided)"
    ).replace("{{CHOICES}}", choices or "(no choices yet)")


def draft_brief(
    *,
    aspects: List[Dict[str, str]],
    project_overview: str = "",
    mode: BackendMode | None = None,
) -> Dict[str, str]:
    """Draft a project-style brief that embodies the candidate's choices."""
    if not aspects:
        raise CandidateServiceError("Cannot draft a brief without any choices.")
    prompt = render_draft_brief_prompt(aspects, project_overview)
    resolved = _resolve_mode(mode)
    try:
        if resolved == BackendMode.vllm:
            client = build_vllm_client(settings.vllm_base_url)
            model = settings.vllm_model
        else:
            client = build_openai_client()
            model = settings.openai_node_model
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 — surfaced as 502 by the router
        raise CandidateServiceError("Failed to draft the brief with the LLM.") from exc
    if not text:
        raise CandidateServiceError("The model returned an empty brief.")
    return {"brief": text}


def score_alignment(
    brief_vec: np.ndarray,
    composition_vec: np.ndarray,
    aspect_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Pure alignment scoring over unit vectors.

    ``aspect_rows``: ``[{aspect_id, chosen: {id, vec}, alternatives: [{id, vec}]}]``.
    ``top_alternative`` is the alternative most similar to the BRIEF — the
    competitor the data says the brief flirts with, not an arbitrary pole.
    """
    per_aspect: List[Dict[str, Any]] = []
    for row in aspect_rows:
        chosen_score = float(brief_vec @ row["chosen"]["vec"])
        top: Dict[str, Any] | None = None
        for alt in row["alternatives"]:
            score = float(brief_vec @ alt["vec"])
            if top is None or score > top["score"]:
                top = {"id": alt["id"], "score": score}
        per_aspect.append(
            {
                "aspect_id": row["aspect_id"],
                "chosen_score": chosen_score,
                "top_alternative": top,
                "leans_away": bool(top is not None and top["score"] > chosen_score),
            }
        )
    return {
        "agreement": float(brief_vec @ composition_vec),
        "per_aspect": per_aspect,
    }


def alignment(
    *,
    brief: str,
    composition: str,
    aspects: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Embed both layers + every option text (one batch) and score alignment.

    ``aspects``: ``[{aspect_id, chosen: {id, text}, alternatives: [{id, text}]}]``.
    """
    texts = [brief, composition]
    slots: List[tuple[int, int]] = []  # (aspect index, alternative index; -1 = chosen)
    for i, row in enumerate(aspects):
        slots.append((i, -1))
        texts.append(row["chosen"]["text"])
        for j, alt in enumerate(row["alternatives"]):
            slots.append((i, j))
            texts.append(alt["text"])

    try:
        vecs = embed_texts(texts)
    except CorpusServiceError as exc:
        raise CandidateServiceError(str(exc)) from exc

    rows: List[Dict[str, Any]] = [
        {"aspect_id": row["aspect_id"], "chosen": None, "alternatives": []}
        for row in aspects
    ]
    for (i, j), vec in zip(slots, vecs[2:]):
        if j == -1:
            rows[i]["chosen"] = {"id": aspects[i]["chosen"]["id"], "vec": vec}
        else:
            rows[i]["alternatives"].append(
                {"id": aspects[i]["alternatives"][j]["id"], "vec": vec}
            )
    return score_alignment(vecs[0], vecs[1], rows)
