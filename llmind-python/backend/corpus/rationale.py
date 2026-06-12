"""The rationale layer — ITERATION-PLAN Part 13 L-A.

The dissertation's study found the taxonomy "semantically coherent but…
why these seven? is there no more?" — options got receipts (annotation), but
the ASPECTS carried no exposed why, and nothing probed completeness. Two
moves close that:

  rationale  — one line per aspect, grounded in the annotation counts
               (cached per aspect content + evidence; assistance, never a
               gate: an LLM failure degrades to an empty line).
  probe      — the coverage probe: the frontend finds the corpus projects
               the taxonomy describes poorly (pure set arithmetic on the
               annotation); this module asks what dimension those projects
               exemplify that the taxonomy misses, and the answers ride the
               existing C1 proposals channel as accept/dismiss chips.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from config import settings
from utils.prompts import MISSING_ASPECT_PROMPT, RATIONALE_PROMPT
from pipeline import register_alignment as ra
from backend.corpus.llm import budgeted_completion, iter_json_objects
from backend.corpus.service import CorpusServiceError, load_index_meta
from backend.reflections.router import parse_reflection

RATIONALE_VERSION = 1
MAX_PROPOSALS = 2


# ── Pure helpers (unit-tested offline) ───────────────────────────────────────


def aspect_content_hash(name: str, desc: str, options: Sequence[Dict[str, Any]]) -> str:
    """Cache key for one aspect's rationale. Includes the option counts — the
    rationale cites the evidence, so new annotation evidence = new rationale."""
    lines = [f"v{RATIONALE_VERSION}", (name or "").strip(), (desc or "").strip()]
    lines += sorted(f"{o.get('name', '')} {o.get('count', 0)}" for o in options)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def rationale_set_hash(aspects: Sequence[Dict[str, Any]]) -> str:
    """Order-independent identity of the whole request (job dedup key)."""
    parts = sorted(
        aspect_content_hash(a.get("name", ""), a.get("desc", ""), a.get("options", []))
        for a in aspects
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def parse_aspect_proposals(
    content: str, reasoning: str, existing_names: Sequence[str]
) -> List[Dict[str, str]]:
    """Validated aspect proposals from an LLM response. Pure.

    JSON objects with non-empty name+desc, deduplicated against the existing
    aspect names (case-insensitive — the prompt forbids renames but thinking
    models drift), capped at MAX_PROPOSALS. Falls back to the reasoning tail
    when a capped thinking model emitted no content.
    """
    taken = {n.strip().lower() for n in existing_names}
    out: List[Dict[str, str]] = []
    source = content if (content or "").strip() else (reasoning or "")[-600:]
    for obj in iter_json_objects(source):
        name = str(obj.get("name") or "").strip()
        desc = str(obj.get("desc") or "").strip()
        if not name or not desc or name.lower() in taken:
            continue
        taken.add(name.lower())
        out.append({"name": name, "desc": desc, "reason": str(obj.get("reason") or "").strip()})
        if len(out) >= MAX_PROPOSALS:
            break
    return out


# ── Cache (one JSON per aspect content hash) ─────────────────────────────────


def _rationale_dir() -> Path:
    return settings.projection_dir / "rationales"


def _load_cached(content_hash: str) -> str | None:
    path = _rationale_dir() / f"{content_hash}.json"
    if not path.exists():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("rationale", ""))
    except (OSError, ValueError):
        return None


def _save_cached(content_hash: str, rationale: str) -> None:
    directory = _rationale_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{content_hash}.json").write_text(
        json.dumps({"rationale": rationale}, ensure_ascii=False), encoding="utf-8"
    )


# ── Jobs ─────────────────────────────────────────────────────────────────────


def draft_rationales(aspects: List[Dict[str, Any]], n_projects: int) -> Dict[str, Any]:
    """One-line per-aspect rationale, grounded in annotation counts.

    ``aspects``: ``[{id, name, desc, options: [{name, count}]}]``. Returns
    ``{rationales: {<aspect_id>: str}, meta}``. Per-aspect LLM failures
    degrade to "" — rationale is assistance, never a gate.
    """
    rationales: Dict[str, str] = {}
    for aspect in aspects:
        options = aspect.get("options") or []
        content_hash = aspect_content_hash(
            aspect.get("name", ""), aspect.get("desc", ""), options
        )
        cached = _load_cached(content_hash)
        if cached is None:
            option_lines = "\n".join(
                f"- {o.get('name', '')}: {o.get('count', 0)} projects" for o in options
            )
            prompt = (
                RATIONALE_PROMPT.replace("{{ASPECT_NAME}}", (aspect.get("name") or "").strip())
                .replace("{{ASPECT_DESC}}", (aspect.get("desc") or "").strip() or "(no description)")
                .replace("{{N_PROJECTS}}", str(n_projects or "the"))
                .replace("{{OPTIONS}}", option_lines or "(no options yet)")
            )
            try:
                content, reasoning = budgeted_completion(prompt)
                cached = parse_reflection(content, reasoning) or ""
            except Exception:  # noqa: BLE001 — explanation, never a gate
                cached = ""
            if cached:
                _save_cached(content_hash, cached)
        rationales[aspect["id"]] = cached
    return {
        "rationales": rationales,
        "meta": {"model": settings.vllm_model, "version": RATIONALE_VERSION},
    }


def probe_missing_aspect(aspect_names: List[str], project_ids: List[str]) -> Dict[str, Any]:
    """What dimension do the poorly-covered projects exemplify that the
    taxonomy misses? Returns ``{proposals: [{name, desc, reason}]}``."""
    meta = load_index_meta()
    if not meta:
        raise CorpusServiceError(
            "Corpus index not found — build the local index with build_local_index.py."
        )
    lines = []
    for pid in project_ids:
        record = meta.get(pid)
        if not record:
            continue
        desc = ra.build_short_text("", record.get("Descriptions") or "", max_chars=220)
        details = " ".join((record.get("Details") or "").split())[:160]
        summary = f"{desc} [Details: {details}]" if details else desc
        lines.append(f"- {record.get('Name') or '(untitled)'}: {summary}")
    if not lines:
        raise CorpusServiceError("None of the given project ids exist in the corpus index.")
    prompt = MISSING_ASPECT_PROMPT.replace(
        "{{ASPECTS}}", "\n".join(f"- {n}" for n in aspect_names)
    ).replace("{{PROJECTS}}", "\n".join(lines))
    content, reasoning = budgeted_completion(prompt)
    return {"proposals": parse_aspect_proposals(content, reasoning, aspect_names)}
