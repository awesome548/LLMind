"""Candidate (dual-layer design) services — ITERATION-PLAN Parts 10 & 12.

A candidate has two layers: a BRIEF (the designer's project-style prose — what
the design *is*) and CHOICES (one option per aspect — what it *commits to*).

* ``draft_brief`` — LLM-drafts the brief from the choices, killing the blank
  page; the designer edits the result.
* ``alignment``   — measures how the two layers agree: cos(brief, composition)
  plus, per aspect, whether the brief actually leans toward the chosen option
  or toward its strongest competitor (defined by data: the alternative most
  similar to the brief).
* ``steer``       — ONE deliberate move on the brief (Part 12 B3): along a
  bipolar metric, toward a precedent, or away from one. The move is made IN
  LANGUAGE by the LLM; embeddings only measure what happened (requested vs
  achieved, along vs orthogonal) — deltas as rulers and briefs, never
  constructors (Part 11's evidence rule).
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from config import settings
from utils.clients import build_vllm_client, build_openai_client
from utils.modes import BackendMode
from utils.prompts import DRAFT_BRIEF_PROMPT, STEER_PROMPT
from backend.corpus.llm import budgeted_completion, iter_json_objects
from backend.corpus.service import CorpusServiceError, embed_texts, load_corpus_vectors


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


# ── Steering (Part 12 B3) ────────────────────────────────────────────────────


def parse_steer(text: str) -> Dict[str, Any] | None:
    """First ``{revised_brief, named_qualities?}``-shaped JSON object. Pure."""
    for obj in iter_json_objects(text):
        revised = obj.get("revised_brief")
        if not (isinstance(revised, str) and revised.strip()):
            continue
        qualities = obj.get("named_qualities")
        names = (
            [q.strip() for q in qualities if isinstance(q, str) and q.strip()]
            if isinstance(qualities, list)
            else []
        )
        return {"revised_brief": revised.strip(), "named_qualities": names[:3]}
    return None


def steer_extent(delta: float) -> str:
    """How far the designer asked to move, in words the LLM can act on. Pure."""
    magnitude = abs(delta)
    if magnitude < 0.15:
        return "subtly"
    if magnitude < 0.4:
        return "moderately"
    return "strongly"


def decompose_displacement(
    before: np.ndarray, after: np.ndarray, direction: np.ndarray
) -> Tuple[float, float]:
    """Displacement split into along-direction and orthogonal components.

    All inputs unit vectors except the displacement itself; ``direction`` is
    the requested move's axis in raw cosine space. Pure.
    """
    disp = after - before
    along = float(disp @ direction)
    return along, float(np.linalg.norm(disp - along * direction))


def _normalised(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _log_steer(payload: Dict[str, Any]) -> None:
    """Append one JSONL row per steer (best-effort — the study dataset)."""
    try:
        path = settings.projection_dir / "steer_log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — logging must never break steering
        pass


def steer(
    *,
    text: str,
    mode: str,
    metric: Dict[str, Any] | None = None,
    reference: Dict[str, Any] | None = None,
    preserve: List[str] | None = None,
) -> Dict[str, Any]:
    """One steering move on a brief; returns the revision and its measurement.

    ``metric`` (mode='metric'): ``{pole_a_text, pole_b_text, target_score}``
    with ``target_score`` on the strips' corpus-normalised −1..1 scale.
    ``reference`` (mode='toward'/'away'): ``{text, weight}``, weight 0..1.
    """
    brief = (text or "").strip()
    if not brief:
        raise CandidateServiceError("Cannot steer an empty brief.")
    preserve_list = [p.strip() for p in (preserve or []) if p.strip()]

    # Measure BEFORE the move so the move instruction can name the gap.
    if mode == "metric":
        if not metric:
            raise CandidateServiceError("Metric steering needs the metric poles.")
        vecs = embed_texts([brief, metric["pole_a_text"], metric["pole_b_text"]])
        before_vec, pole_a, pole_b = vecs[0], vecs[1], vecs[2]
        ids, corpus = load_corpus_vectors()
        if not ids:
            raise CandidateServiceError(
                "Corpus vectors not found — build the local index with build_local_index.py."
            )
        # Same corpus min-max normalisation as /metrics, recomputed here —
        # target_score and the strip the designer dragged on share a scale as
        # long as the corpus index is not rebuilt mid-session (the frozen-map
        # assumption the whole design space already rests on).
        raw = corpus @ pole_a - corpus @ pole_b
        lo, hi = float(raw.min()), float(raw.max())
        span = (hi - lo) or 1.0
        scale = lambda value: 2.0 * (value - lo) / span - 1.0  # noqa: E731
        score_before = scale(float(before_vec @ pole_a - before_vec @ pole_b))
        requested = float(metric["target_score"])
        direction = _normalised(pole_a - pole_b)
        move = (
            f"Move the design {steer_extent(requested - score_before)} toward the quality of "
            f"\"{metric['pole_a_text']}\" and away from \"{metric['pole_b_text']}\"."
        )
    elif mode in ("toward", "away"):
        if not reference or not (reference.get("text") or "").strip():
            raise CandidateServiceError("Precedent steering needs the reference text.")
        vecs = embed_texts([brief, reference["text"]])
        before_vec, ref_vec = vecs[0], vecs[1]
        weight = float(reference.get("weight", 0.5))
        requested = weight if mode == "toward" else -weight
        # The measurement axis points the way the designer ASKED to move, so
        # a compliant revision always reads as positive `along` — for "away"
        # that is the direction leading from the reference, not toward it.
        direction = (
            _normalised(ref_vec - before_vec)
            if mode == "toward"
            else _normalised(before_vec - ref_vec)
        )
        excerpt = " ".join(reference["text"].split())[:300]
        move = (
            f"Pull the design {steer_extent(requested)} toward this precedent — adopt what "
            f"defines it, without copying it: \"{excerpt}\""
            if mode == "toward"
            else f"Push the design {steer_extent(requested)} AWAY from this precedent — reduce "
            f"what they share, sharpen what differs: \"{excerpt}\""
        )
        score_before = None
        scale = None
        pole_a = pole_b = None
    else:
        raise CandidateServiceError(f"Unknown steering mode: {mode}")

    prompt = (
        STEER_PROMPT.replace("{{BRIEF}}", brief)
        .replace("{{MOVE}}", move)
        .replace(
            "{{PRESERVE}}",
            "\n".join(f"- {p}" for p in preserve_list) or "(nothing specified)",
        )
    )
    content, reasoning = budgeted_completion(prompt)
    parsed = parse_steer(content) or (parse_steer(reasoning[-800:]) if not content.strip() else None)
    if parsed is None:
        raise CandidateServiceError("The model returned no parseable revision — try again.")

    # The revision exists even if measuring it fails (the embedding service
    # can flake AFTER the LLM call) — return it unmeasured rather than losing
    # the designer's result; the client says so instead of showing numbers.
    measurement: Dict[str, Any] | None
    try:
        after_vec = embed_texts([parsed["revised_brief"]])[0]
        along, orthogonal = decompose_displacement(before_vec, after_vec, direction)
        if mode == "metric":
            score_after = scale(float(after_vec @ pole_a - after_vec @ pole_b))
            achieved = score_after - score_before
        else:
            score_after = None
            achieved = float(after_vec @ ref_vec) - float(before_vec @ ref_vec)
            if mode == "away":
                achieved = -achieved  # positive = moved as requested
        measurement = {
            "mode": mode,
            "requested": requested,
            "achieved": achieved,
            "along": along,
            "orthogonal": orthogonal,
            "score_before": score_before,
            "score_after": score_after,
        }
    except CorpusServiceError:
        measurement = None

    _log_steer(
        {
            "ts": time.time(),
            **(measurement or {"mode": mode, "requested": requested, "achieved": None,
                               "along": None, "orthogonal": None,
                               "score_before": score_before, "score_after": None}),
            "named_qualities": parsed["named_qualities"],
            "brief_chars_before": len(brief),
            "brief_chars_after": len(parsed["revised_brief"]),
        }
    )
    return {
        "revised_text": parsed["revised_brief"],
        "named_qualities": parsed["named_qualities"],
        "measurement": measurement,
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
