"""Corpus data access shared across routers.

The corpus is the scraped real-project collection behind both the local vector
index and the design-space surface. Its metadata lives in the sidecar
``local_index.npz.meta.json`` and its embeddings in ``local_index.npz`` (both
written by ``build_local_index.py``); this module is the single cached reader
for both, so the projection service and the corpus router never load them twice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from config import settings


_META_CACHE: Dict[str, Any] = {}
_VECTOR_CACHE: Dict[str, Any] = {}
_SUPPORT_CACHE: Dict[str, Any] = {}

# Corpus support = mean cosine to the k nearest corpus projects, read as a
# percentile of the corpus's own self-support distribution (ITERATION-PLAN H3).
SUPPORT_NEIGHBORS = 5


def load_corpus_vectors() -> Tuple[List[str], np.ndarray]:
    """(ids, unit-normalised vectors) from the local index, cached by mtime."""
    path = Path(settings.local_index_path)
    if not path.exists():
        return [], np.empty((0, 0))
    mtime = path.stat().st_mtime
    if _VECTOR_CACHE.get("mtime") != mtime:
        data = np.load(path, allow_pickle=True)
        ids = [str(i) for i in data["ids"].tolist()]
        vecs = np.asarray(data["vectors"], dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _VECTOR_CACHE.update(mtime=mtime, ids=ids, vecs=vecs / norms)
    return _VECTOR_CACHE["ids"], _VECTOR_CACHE["vecs"]


def load_index_meta() -> Dict[str, Dict[str, Any]]:
    """Full corpus metadata sidecar (Name/Descriptions/Details/Image), cached by mtime."""
    path = Path(f"{settings.local_index_path}.meta.json")
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    cached = _META_CACHE.get("meta")
    if cached is None or _META_CACHE.get("mtime") != mtime:
        _META_CACHE["meta"] = json.loads(path.read_text(encoding="utf-8"))
        _META_CACHE["mtime"] = mtime
    return _META_CACHE["meta"]


def support_percentiles(baseline_sorted: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fraction of the (sorted) corpus self-support baseline each value meets.

    0 = less corpus evidence than ANY corpus project has for itself;
    1 = at or above the best-supported corpus project. Pure — unit-testable.
    """
    if baseline_sorted.size == 0:
        return np.full(np.asarray(values).shape, np.nan)
    return np.searchsorted(baseline_sorted, values, side="right") / baseline_sorted.size


def support_scores(
    vecs_unit: np.ndarray,
    corpus_unit: np.ndarray,
    exclude_rows: List[int] | None = None,
    k: int = SUPPORT_NEIGHBORS,
) -> np.ndarray:
    """Raw support per row vector: mean top-k cosine to the corpus.

    ``exclude_rows`` masks one corpus row per query (fit-time only: a corpus
    project's own full text must not support its short text, because runtime
    queries have no "self" in the corpus). Pure — unit-testable.
    """
    sims = vecs_unit @ corpus_unit.T
    if exclude_rows is not None:
        for i, row in enumerate(exclude_rows):
            sims[i, row] = -np.inf
    k = min(k, corpus_unit.shape[0] - (1 if exclude_rows is not None else 0))
    top = np.sort(sims, axis=1)[:, -k:]
    return top.mean(axis=1)


def support_baseline() -> np.ndarray:
    """Sorted corpus self-support: per project, mean top-k cosine to the REST.

    The FULL-register yardstick — meaningful for project-description-length
    text. Node-length queries are compared against the short-register baseline
    fitted by ``project-align`` instead (see ``corpus_support``). Cached by
    index mtime; empty array when the corpus is unavailable.
    """
    path = Path(settings.local_index_path)
    if not path.exists():
        return np.empty(0)
    mtime = path.stat().st_mtime
    if _SUPPORT_CACHE.get("mtime") != mtime:
        ids, vecs = load_corpus_vectors()
        if len(ids) < 2:
            return np.empty(0)
        scores = support_scores(vecs, vecs, exclude_rows=list(range(len(ids))))
        _SUPPORT_CACHE.update(mtime=mtime, baseline=np.sort(scores))
    return _SUPPORT_CACHE["baseline"]


def corpus_support(
    vecs_unit: np.ndarray, baseline: np.ndarray | None = None
) -> List[float | None]:
    """Corpus-support percentile per (unit-normalised) row vector.

    ``baseline`` selects the yardstick: pass the short-register baseline from
    the fitted register map for node-length queries (Part 10 recalibration —
    "as much evidence as a real project described at this length"); defaults
    to the full-register self-support distribution.

    Best-effort, like placement confidence: ``None`` per point when the corpus
    is unavailable or its dimensionality doesn't match.
    """
    n = int(np.asarray(vecs_unit).shape[0]) if np.asarray(vecs_unit).size else 0
    if baseline is None or baseline.size == 0:
        baseline = support_baseline()
    ids, corpus = load_corpus_vectors()
    if n == 0 or baseline.size == 0 or not ids or corpus.shape[1] != vecs_unit.shape[1]:
        return [None] * n
    scores = support_scores(vecs_unit, corpus)
    return [float(p) for p in support_percentiles(baseline, scores)]


class CorpusServiceError(RuntimeError):
    """Raised when a corpus operation's external dependency fails."""


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed texts with the local model and unit-normalise them (one batch call).

    The shared entry point into the corpus's metric space for services that
    score designer text against it (relevance, alignment, metrics).
    """
    from utils.clients import build_vllm_client

    if not texts:
        return np.empty((0, 0))
    try:
        client = build_vllm_client(settings.vllm_base_url)
        response = client.embeddings.create(
            model=settings.vllm_embed_model, input=list(texts)
        )
        vectors = [d.embedding for d in response.data]
        if len(vectors) != len(texts):
            raise CorpusServiceError("Embedding count did not match input count.")
    except CorpusServiceError:
        raise
    except Exception as exc:  # noqa: BLE001 — surfaced as 502 by the router
        raise CorpusServiceError("Failed to embed text with the local model.") from exc
    arr = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _embed_query(text: str) -> np.ndarray:
    """Embed one text with the local model and unit-normalise it."""
    return embed_texts([text])[0]


def relevance_scores(text: str) -> Dict[str, Any]:
    """True cosine similarity of EVERY corpus project to ``text``.

    Powers the design-space relevance lens: the painting is faithful (original
    768-d metric) even where the 2D layout is not. Returns all scores plus the
    min/max so the client can normalise honestly ("relative relevance").
    """
    ids, vecs = load_corpus_vectors()
    if not ids:
        raise CorpusServiceError(
            "Corpus vectors not found — build the local index with build_local_index.py."
        )
    query = _embed_query(text)
    if vecs.shape[1] != query.shape[0]:
        raise CorpusServiceError(
            f"Embedding dim {query.shape[0]} != corpus dim {vecs.shape[1]} — the runtime "
            f"embedding model differs from the one the index was built with."
        )
    sims = vecs @ query
    return {
        "scores": [
            {"id": pid, "score": float(score)} for pid, score in zip(ids, sims)
        ],
        "min": float(sims.min()),
        "max": float(sims.max()),
    }


def similar_projects(text: str, k: int = 5) -> list[Dict[str, Any]]:
    """Corpus projects most similar to ``text`` in the ORIGINAL embedding metric.

    Used for "closest real precedents to this design candidate": the candidate's
    composed option text is embedded with the local model and searched against
    the offline index (true cosine similarity — not the 2D projection, so the
    ranking is faithful even where the surface is distorted).
    """
    from config import settings as _settings
    from utils import local_store
    from utils.clients import build_vllm_client

    try:
        client = build_vllm_client(_settings.vllm_base_url)
        response = client.embeddings.create(
            model=_settings.vllm_embed_model, input=[text]
        )
        vector = response.data[0].embedding if response.data else None
        if not isinstance(vector, list):
            raise CorpusServiceError("Failed to embed the candidate text.")
        rows = local_store.search(vector, k=k, threshold=0.0)
    except CorpusServiceError:
        raise
    except Exception as exc:  # noqa: BLE001 — surfaced as 502 by the router
        raise CorpusServiceError("Corpus similarity search failed.") from exc

    return [
        {
            "id": str(row.get("id")),
            "Name": row.get("Name") or "(untitled)",
            "Descriptions": row.get("Descriptions") or "",
            "Details": row.get("Details") or "",
            "Image": row.get("Image"),
            "score": float(row.get("score") or 0.0),
        }
        for row in rows
    ]


def get_project(project_id: str) -> Dict[str, Any]:
    """One corpus project's metadata, normalised to the RelatedProject shape.

    Raises ``KeyError`` when the id is not in the corpus.
    """
    record = load_index_meta().get(project_id)
    if record is None:
        raise KeyError(project_id)
    return {
        "id": project_id,
        "Name": record.get("Name") or "(untitled)",
        "Descriptions": record.get("Descriptions") or "",
        "Details": record.get("Details") or "",
        "Image": record.get("Image"),
    }
