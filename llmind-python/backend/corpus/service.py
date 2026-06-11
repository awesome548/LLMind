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


class CorpusServiceError(RuntimeError):
    """Raised when a corpus operation's external dependency fails."""


def _embed_query(text: str) -> np.ndarray:
    """Embed one text with the local model and unit-normalise it."""
    from utils.clients import build_vllm_client

    try:
        client = build_vllm_client(settings.vllm_base_url)
        response = client.embeddings.create(
            model=settings.vllm_embed_model, input=[text]
        )
        vector = response.data[0].embedding if response.data else None
        if not isinstance(vector, list):
            raise CorpusServiceError("Failed to embed the query text.")
    except CorpusServiceError:
        raise
    except Exception as exc:  # noqa: BLE001 — surfaced as 502 by the router
        raise CorpusServiceError("Failed to embed the query text.") from exc
    query = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(query)) or 1.0
    return query / norm


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
