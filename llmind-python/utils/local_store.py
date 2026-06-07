"""File-based vector store for fully-offline related-project search.

No database, no network: project vectors live in an ``.npz`` next to a JSON
sidecar of metadata. Cosine search is brute-force over the in-memory matrix,
which is plenty fast for the small media-architecture corpus.

Layout (for ``data/local_index.npz``):
    data/local_index.npz            ids (str[N]), vectors (float32 N×D)
    data/local_index.npz.meta.json  {id: {Name, Descriptions, Details, Image}}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _meta_path(path: Path) -> Path:
    return Path(f"{path}.meta.json")


def save_index(
    path: Path,
    ids: list[str],
    vectors: list[list[float]],
    metadata: dict[str, dict[str, Any]],
) -> None:
    """Persist row-aligned ``ids``/``vectors`` plus a metadata sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(vectors, dtype=np.float32)
    np.savez(path, ids=np.asarray(ids), vectors=matrix)
    _meta_path(path).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


# (ids, normalized vectors, metadata) cache keyed by path, invalidated on mtime.
_CACHE: dict[str, dict[str, Any]] = {}


def _load(path: Path) -> dict[str, Any]:
    key = str(path)
    mtime = path.stat().st_mtime
    cached = _CACHE.get(key)
    if cached is None or cached["mtime"] != mtime:
        data = np.load(path)
        meta = json.loads(_meta_path(path).read_text(encoding="utf-8"))
        cached = {
            "ids": data["ids"].tolist(),
            "vectors": _normalize(data["vectors"].astype(np.float32)),
            "meta": meta,
            "mtime": mtime,
        }
        _CACHE[key] = cached
    return cached


def search(
    query_vector: list[float],
    *,
    k: int = 5,
    threshold: float = 0.0,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    """Top-``k`` metadata dicts (each carrying ``id`` and ``score``), ordered by
    descending cosine similarity and filtered by ``threshold``."""
    from config import settings

    index_path = Path(path or settings.local_index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Local index not found at {index_path}. "
            f"Build it with: uv run python build_local_index.py"
        )

    store = _load(index_path)
    if not store["ids"]:
        return []

    query = np.asarray(query_vector, dtype=np.float32)
    norm = float(np.linalg.norm(query)) or 1.0
    sims = store["vectors"] @ (query / norm)

    order = np.argsort(-sims)[: max(k, 0)]
    results: list[dict[str, Any]] = []
    for idx in order:
        score = float(sims[int(idx)])
        if score < threshold:
            continue
        pid = store["ids"][int(idx)]
        meta = dict(store["meta"].get(pid, {}))
        results.append({**meta, "id": pid, "score": score})
    return results
