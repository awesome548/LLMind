"""Corpus annotation against the designer's taxonomy — ITERATION-PLAN Part 12 A2.

Halskov (2021, MAB '20) annotated the MAB corpus against a design-space schema
BY HAND: every project checked against every option, yielding per-option
counts, faceted search, and cross-tab subspaces whose empty cells are exact
gaps. This module automates that bridge for the designer's own taxonomy:

    per option:  register-corrected option vector → top-k corpus shortlist by
    true cosine → ONE local-LLM membership call over the shortlist → the
    projects that genuinely exemplify the option.

Results are cached PER OPTION by content hash (name + desc), so taxonomy edits
re-annotate only what changed. Counts are evidence with receipts, never
verdicts: the project lists are always surfaced in the UI.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from config import settings
from utils.prompts import ANNOTATE_OPTION_PROMPT
from pipeline import register_alignment as ra
from backend.corpus.llm import budgeted_completion
from backend.corpus.service import (
    CorpusServiceError,
    embed_texts,
    load_corpus_vectors,
    load_index_meta,
)

SHORTLIST_K = 30
# Projects per membership call. The local model's context window is small AND
# the serving model may be thinking-only (Qwen3.6 ignores /no_think and
# chat_template_kwargs.enable_thinking — verified live): deliberation scales
# with chunk size, so chunks must be small enough that prompt + full thinking
# + answer fit the window. Measured: 10 projects ≈ 1.1k prompt + >2.5k think
# (overruns 4096); 5 projects ≈ 0.7k prompt + ~1.5k think (finishes, answers).
JUDGE_BATCH = 5
# Cache-buster baked into the option content hash. v2: summaries gained a
# Details snippet — v1 judged from opening description sentences only, which
# carry concept talk but not tech vocabulary, producing absurd false negatives
# ("LED wall panels": 0 exemplars in an LED-saturated corpus). v3 tried to
# suppress thinking (/no_think) — Qwen3.6 is thinking-only and ignored it,
# burning every capped budget inside reasoning and answering nothing. v4:
# thinking is BUDGETED, not fought — small chunks, window-aware max_tokens,
# and a reasoning-tail salvage when generation still hits the cap. v5:
# parse_membership now coerces quoted-number arrays (["1","2"]) and rejects JSON
# booleans — v4 silently counted zero members when the local model quoted its
# numbers, understating counts and caching the wrong verdict (ITERATION-M M-E2).
ANNOTATION_VERSION = 5
# Halskov's granularity principles: an option relevant to almost every
# instance, or to at most one, carries little discriminating power.
TOO_BROAD_SHARE = 0.8
UNPRECEDENTED_MAX = 1

_JSON_ARRAY = re.compile(r"\[[^\[\]]*\]")


# ── Pure helpers (unit-tested offline) ───────────────────────────────────────


def option_content_hash(name: str, desc: str) -> str:
    """Cache key for one option's annotation: depends only on its text (and
    implicitly the corpus, which invalidates by directory wipe on reindex)."""
    payload = f"v{ANNOTATION_VERSION}\n{(name or '').strip()}\n{(desc or '').strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def taxonomy_hash(options: Sequence[Dict[str, str]]) -> str:
    """Order-independent hash of the whole option set (the response identity)."""
    parts = sorted(option_content_hash(o.get("name", ""), o.get("desc", "")) for o in options)
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _as_index(value: Any) -> int | None:
    """Coerce one parsed JSON-array element to a project index, or ``None``.

    Accepts ints, floats, and numeric strings (the local model routinely
    quotes its numbers as ``["1","2"]``); rejects JSON booleans (``bool`` is an
    ``int`` subclass in Python, so ``[true]`` would otherwise count as project
    1) and every other type. Range/dedup is applied by the caller.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def parse_membership(text: str, n: int) -> List[int]:
    """1-based exemplifying-project numbers from an LLM response.

    Prefers the first JSON array in the text; falls back to bare integers.
    Out-of-range and duplicate numbers are dropped. Pure — unit-testable.
    """
    candidates: List[int] = []
    match = _JSON_ARRAY.search(text or "")
    if match:
        try:
            parsed = json.loads(match.group(0))
            candidates = [i for v in parsed if (i := _as_index(v)) is not None]
        except (ValueError, TypeError):
            # Array-shaped but not valid JSON (e.g. trailing comma) — salvage
            # the integers from the matched span only.
            candidates = [int(v) for v in re.findall(r"\d+", match.group(0))]
    else:
        candidates = [int(v) for v in re.findall(r"\d+", text or "")]
    seen: set[int] = set()
    out: List[int] = []
    for v in candidates:
        if 1 <= v <= n and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def salvage_from_reasoning(reasoning: str, n: int) -> List[int]:
    """Last-resort verdict recovery when a thinking model hit the token cap
    before emitting content: take the LAST JSON array in the reasoning tail
    (thinking models conclude with e.g. "I will output [1, 6]"). Only the tail
    is searched — earlier arrays are mid-deliberation noise. Pure."""
    tail = (reasoning or "")[-400:]
    arrays = _JSON_ARRAY.findall(tail)
    return parse_membership(arrays[-1], n) if arrays else []


def diagnostics_for(
    counts: Dict[str, int], n_projects: int, shortlist_k: int
) -> Dict[str, List[str]]:
    """Granularity flags per Halskov: too-broad and unprecedented options.

    ``too_broad`` measures **shortlist saturation**, not corpus share. Only the
    top ``shortlist_k`` corpus projects are ever judged per option, so a count
    can never exceed ``shortlist_k`` (≪ ``n_projects``) — comparing it to a
    share of the whole corpus (the pre-M-E1 bug) made the flag unreachable. An
    option that matches most of its OWN nearest neighbours fails to discriminate
    even among the projects it is closest to: the honest, measurable reading of
    "too broad". (When ``count == shortlist_k`` the corpus-wide count is only a
    lower bound — the option saturated everything the judge was shown.)
    """
    if n_projects <= 0 or shortlist_k <= 0:
        return {"too_broad": [], "unprecedented": []}
    saturation = TOO_BROAD_SHARE * min(shortlist_k, n_projects)
    return {
        "too_broad": sorted(oid for oid, c in counts.items() if c >= saturation),
        "unprecedented": sorted(
            oid for oid, c in counts.items() if c <= UNPRECEDENTED_MAX
        ),
    }


# ── Cache (one JSON per option content hash) ─────────────────────────────────


def _annotation_dir() -> Path:
    return settings.projection_dir / "annotations"


def _load_cached(content_hash: str) -> Dict[str, Any] | None:
    path = _annotation_dir() / f"{content_hash}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save_cached(content_hash: str, record: Dict[str, Any]) -> None:
    directory = _annotation_dir()
    directory.mkdir(parents=True, exist_ok=True)
    # Stamp the annotation version so tooling (annotation-stats) can tell current
    # records from orphans left when the version bumps — a version bump changes
    # the content hash, so old files linger under their old names (ITERATION-M F5).
    stamped = {**record, "version": ANNOTATION_VERSION}
    (directory / f"{content_hash}.json").write_text(
        json.dumps(stamped, ensure_ascii=False), encoding="utf-8"
    )


# ── LLM membership judgment ──────────────────────────────────────────────────


def _judge_option(
    name: str, desc: str, shortlist: List[Dict[str, str]]
) -> List[int]:
    """Membership judgments for the shortlist, in JUDGE_BATCH-sized calls.

    Returns 1-based indices INTO THE FULL SHORTLIST of the projects the model
    accepted as genuine exemplars.
    """
    kept: List[int] = []
    for start in range(0, len(shortlist), JUDGE_BATCH):
        chunk = shortlist[start : start + JUDGE_BATCH]
        projects = "\n".join(
            f"{i + 1}. {row['name']}: {row['summary']}" for i, row in enumerate(chunk)
        )
        prompt = (
            ANNOTATE_OPTION_PROMPT.replace("{{OPTION_NAME}}", name.strip())
            .replace("{{OPTION_DESC}}", desc.strip() or "(no description)")
            .replace("{{PROJECTS}}", projects)
        )
        content, reasoning = budgeted_completion(prompt)
        members = parse_membership(content, len(chunk))
        if not members and not content.strip():
            # Cap hit mid-thinking — recover the verdict from the reasoning
            # tail if one was formed.
            members = salvage_from_reasoning(reasoning, len(chunk))
        kept.extend(start + i for i in members)
    return kept


# ── The annotation job ───────────────────────────────────────────────────────


def annotate_taxonomy(options: List[Dict[str, str]]) -> Dict[str, Any]:
    """Annotate the corpus against every option; cached per option content.

    ``options``: ``[{id, name, desc}]`` (aspect grouping is irrelevant here —
    membership is judged per option). Returns
    ``{taxonomy_hash, options: {<id>: {count, project_ids}}, diagnostics, meta}``.
    """
    valid = [o for o in options if (o.get("name") or "").strip() and o.get("id")]
    if not valid:
        raise CorpusServiceError("Annotation needs at least one named option.")

    ids, corpus_unit = load_corpus_vectors()
    meta = load_index_meta()
    if not ids:
        raise CorpusServiceError(
            "Corpus vectors not found — build the local index with build_local_index.py."
        )

    # Summary = opening description sentences (the concept) + a Details
    # snippet (the tech specs — where "LED", "sensors", "projection" actually
    # live; judging from descriptions alone produced absurd false negatives).
    def _summary(pid: str) -> str:
        record = meta.get(pid, {})
        desc = ra.build_short_text("", record.get("Descriptions") or "", max_chars=220)
        details = " ".join((record.get("Details") or "").split())[:200]
        return f"{desc} [Details: {details}]" if details else desc

    summaries = [
        {
            "id": pid,
            "name": meta.get(pid, {}).get("Name") or "(untitled)",
            "summary": _summary(pid),
        }
        for pid in ids
    ]

    # Embed all uncached option texts in one batch; register-correct (options
    # are short-register text, the corpus index is full-register).
    uncached = {
        h: o
        for o in valid
        if _load_cached(h := option_content_hash(o["name"], o.get("desc", ""))) is None
    }
    vec_by_hash: Dict[str, np.ndarray] = {}
    if uncached:
        vecs = embed_texts(
            [f"{o['name']}. {o.get('desc', '')}".strip(". ") for o in uncached.values()]
        )
        rmap = ra.load_register_map(settings.projection_dir) if settings.register_alignment else None
        if rmap is not None and rmap.weights.shape[0] == vecs.shape[1]:
            vecs = rmap.apply(vecs)
        if vecs.shape[1] != corpus_unit.shape[1]:
            raise CorpusServiceError(
                f"Embedding dim {vecs.shape[1]} != corpus dim {corpus_unit.shape[1]}."
            )
        vec_by_hash = dict(zip(uncached.keys(), vecs))

    result_options: Dict[str, Any] = {}
    accepted_shares: List[float] = []
    for option in valid:
        content_hash = option_content_hash(option["name"], option.get("desc", ""))
        record = _load_cached(content_hash)
        if record is None:
            vec = vec_by_hash[content_hash]
            k = min(SHORTLIST_K, len(ids))
            top = np.argsort(corpus_unit @ vec)[-k:][::-1]
            shortlist = [summaries[int(i)] for i in top]
            kept = _judge_option(option["name"], option.get("desc", ""), shortlist)
            project_ids = [shortlist[i - 1]["id"] for i in kept]
            record = {
                "count": len(project_ids),
                "project_ids": project_ids,
                "shortlist_k": k,
            }
            _save_cached(content_hash, record)
        result_options[option["id"]] = {
            "count": record["count"],
            "project_ids": record["project_ids"],
            # Receipts decorated at assembly time (names stay out of the cache
            # so a metadata edit never invalidates annotations).
            "projects": [
                {"id": p, "name": meta.get(p, {}).get("Name") or "(untitled)"}
                for p in record["project_ids"]
            ],
        }
        if record.get("shortlist_k"):
            accepted_shares.append(record["count"] / record["shortlist_k"])

    counts = {oid: r["count"] for oid, r in result_options.items()}
    return {
        "taxonomy_hash": taxonomy_hash(valid),
        "options": result_options,
        "diagnostics": diagnostics_for(counts, len(ids), min(SHORTLIST_K, len(ids))),
        "meta": {
            "n_projects": len(ids),
            "shortlist_k": SHORTLIST_K,
            "model": settings.vllm_model,
            # Mean fraction of the embedding shortlist the LLM accepted — the
            # gap between aboutness and exemplification, reported per K2-A2.
            "mean_shortlist_acceptance": (
                round(float(np.mean(accepted_shares)), 3) if accepted_shares else None
            ),
        },
    }
