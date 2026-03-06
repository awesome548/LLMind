"""Pure data helpers: extraction, cleaning, and embedding-record construction."""

from __future__ import annotations

import statistics
from typing import Any, Dict, List
from uuid import uuid4

from utils.modes import ContentMode
from pipeline.constants import (
    EMBED_CONTENT_FIELD,
    METADATA_NOISE_KEYS,
    MIN_WORDS,
    MAX_WORDS,
)
from utils.models import EmbedRecord


# ── Field extraction ─────────────────────────────────────────────────────────


def extract_content(record: Dict[str, Any]) -> str:
    return (record.get(EMBED_CONTENT_FIELD) or "").strip()


def extract_project_id(url: str) -> str:
    if "project/" in url:
        return url.split("project/")[-1]
    return str(uuid4())


# ── Summary stats ────────────────────────────────────────────────────────────


def summary_stats(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
    }


# ── Record cleaning ──────────────────────────────────────────────────────────


def clean_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for record in records:
        content = extract_content(record)
        if not content:
            continue
        word_count = len(content.split())
        if not (MIN_WORDS <= word_count <= MAX_WORDS):
            continue
        project_id = extract_project_id(str(record.get("url", "")))
        cleaned.append(
            {
                **{k: v for k, v in record.items() if k not in METADATA_NOISE_KEYS},
                "id": project_id,
            }
        )
    return cleaned


# ── Context building for embeddings ─────────────────────────────────────────


def build_context(record: Dict[str, Any], content_mode: ContentMode) -> str:
    desc = (record.get("Descriptions") or "").strip()
    detail = (record.get("Details") or "").strip()
    if content_mode == ContentMode.description:
        return desc
    if content_mode == ContentMode.details:
        return detail
    # hybrid
    return "; ".join(s for s in [desc, detail] if s)


def build_embed_records(
    records: List[Dict[str, Any]],
    content_mode: ContentMode = ContentMode.details,
) -> List[EmbedRecord]:
    result: List[EmbedRecord] = []
    for record in records:
        record_id = record.get("id")
        context = build_context(record, content_mode)
        if not isinstance(record_id, str) or not record_id or not context:
            continue
        result.append(EmbedRecord(media_doc_id=record_id, context=context))
    return result
