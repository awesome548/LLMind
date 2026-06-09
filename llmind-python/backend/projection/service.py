"""Design-space projection service.

Bridges the frozen reducer (``pipeline.projection``) to the API:

* ``load_surface``    — serve the precomputed corpus background.
* ``locate_nodes``    — embed taxonomy-node text and place it in the frozen space.
* ``generate_at``     — spatial-neighbour RAG: turn a clicked empty location into
                        new mind-map nodes, then locate them on the surface.

The surface and model are loaded from disk and cached by mtime (same pattern as
``utils.local_store``). Embedding uses the local vLLM server + the configured
embedding model, matching the offline related-project path.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from config import settings
from utils.clients import build_vllm_client
from utils.modes import BackendMode
from pipeline import projection as proj
from backend.related_projects.service import (
    ServiceError,
    generate_nodes_from_related_projects,
)


# ── Cached artifacts (invalidated on mtime, like utils.local_store) ───────────

_MODEL_CACHE: Dict[str, Any] = {}
_SURFACE_CACHE: Dict[str, Any] = {}
_META_CACHE: Dict[str, Any] = {}

# The fitted model is a shared cached object and FastAPI runs sync endpoints in a
# threadpool, so transforms can run concurrently (e.g. the page's /locate and a
# /generate-at's internal locate). UMAP/numba transform tested thread-safe here,
# but serialising is cheap (~0.2s) and removes any shared-state risk.
_TRANSFORM_LOCK = threading.Lock()


def _surface_path() -> Path:
    return settings.projection_dir / proj.SURFACE_FILENAME


def _load_model() -> proj.ProjectionModel:
    path = settings.projection_dir / proj.MODEL_FILENAME
    if not path.exists():
        raise ServiceError(
            "Design-space projection model not found. Build it with: "
            "uv run python database_pipeline.py project"
        )
    mtime = path.stat().st_mtime
    cached = _MODEL_CACHE.get("model")
    if cached is None or _MODEL_CACHE.get("mtime") != mtime:
        _MODEL_CACHE["model"] = proj.load_model(settings.projection_dir)
        _MODEL_CACHE["mtime"] = mtime
    return _MODEL_CACHE["model"]


def load_surface() -> Dict[str, Any]:
    """Return the persisted background surface (grid spec, corpus points, density)."""
    path = _surface_path()
    if not path.exists():
        raise ServiceError(
            "Design-space surface not found. Build it with: "
            "uv run python database_pipeline.py project"
        )
    mtime = path.stat().st_mtime
    cached = _SURFACE_CACHE.get("surface")
    if cached is None or _SURFACE_CACHE.get("mtime") != mtime:
        _SURFACE_CACHE["surface"] = json.loads(path.read_text(encoding="utf-8"))
        _SURFACE_CACHE["mtime"] = mtime
    return _SURFACE_CACHE["surface"]


def _load_index_meta() -> Dict[str, Dict[str, Any]]:
    """Full corpus metadata sidecar (Name/Descriptions/Details/Image), cached."""
    path = Path(f"{settings.local_index_path}.meta.json")
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    cached = _META_CACHE.get("meta")
    if cached is None or _META_CACHE.get("mtime") != mtime:
        _META_CACHE["meta"] = json.loads(path.read_text(encoding="utf-8"))
        _META_CACHE["mtime"] = mtime
    return _META_CACHE["meta"]


# ── Embedding (local model) ───────────────────────────────────────────────────


def _embed_texts(texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=float)
    try:
        client = build_vllm_client(settings.vllm_base_url)
        response = client.embeddings.create(
            model=settings.vllm_embed_model, input=list(texts)
        )
    except Exception as exc:  # noqa: BLE001 — surfaced as ServiceError below
        raise ServiceError("Failed to embed node text with the local model.") from exc
    vectors = [d.embedding for d in response.data]
    if len(vectors) != len(texts):
        raise ServiceError("Embedding count did not match input count.")
    return np.asarray(vectors, dtype=float)


# ── Public operations ─────────────────────────────────────────────────────────


def locate_nodes(items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Embed each ``{node_id, text}`` and place it in the frozen design space.

    Returns ``[{node_id, x, y[, z]}]`` aligned to the input order. Items with empty
    text are skipped.
    """
    valid = [it for it in items if (it.get("text") or "").strip() and it.get("node_id")]
    if not valid:
        return []

    model = _load_model()
    embeddings = _embed_texts([it["text"] for it in valid])

    # Guard: the runtime embedding model MUST match the one the projection was fit
    # on, or coordinates are meaningless. A dim mismatch is the catastrophic case.
    expected_dim = model.meta.get("input_dims")
    if expected_dim and embeddings.shape[1] != expected_dim:
        raise ServiceError(
            f"Embedding dim {embeddings.shape[1]} != projection's fit dim {expected_dim}. "
            f"The runtime embedding model differs from the one the index/projection was "
            f"built with. Rebuild the index and run `database_pipeline.py project`, or set "
            f"VLLM_EMBED_MODEL to the original model."
        )

    with _TRANSFORM_LOCK:
        coords = model.transform(embeddings)
    axis_names = ["x", "y", "z"][: model.dims]

    located: List[Dict[str, Any]] = []
    for it, row in zip(valid, coords):
        entry: Dict[str, Any] = {"node_id": it["node_id"]}
        for a, name in enumerate(axis_names):
            entry[name] = float(row[a])
        located.append(entry)
    return located


def _neighbour_record(pid: str, surface_point: Dict[str, Any] | None) -> Dict[str, Any]:
    meta = _load_index_meta()
    record = meta.get(pid, {})
    return {
        "id": pid,
        "Name": record.get("Name") or (surface_point or {}).get("name") or "(untitled)",
        "Descriptions": record.get("Descriptions") or "",
        "Details": record.get("Details") or "",
        "Image": record.get("Image"),
        "x": float((surface_point or {}).get("x", 0.0)),
        "y": float((surface_point or {}).get("y", 0.0)),
    }


def nearest_corpus(x: float, y: float, k: int) -> List[Dict[str, Any]]:
    """The ``k`` corpus projects nearest a location *in 2-D*, with metadata for RAG."""
    surface = load_surface()
    points = surface.get("points", [])
    if not points:
        return []

    coords = np.array([[p["x"], p["y"]] for p in points], dtype=float)
    idxs = proj.nearest_indices(coords, [x, y], k)
    return [_neighbour_record(str(points[i]["id"]), points[i]) for i in idxs]


# Cache of the corpus's original (unit-normalised) embeddings for inverse seeding.
_CORPUS_CACHE: Dict[str, Any] = {}


def _load_corpus_vectors() -> tuple[List[str], np.ndarray]:
    """(ids, unit-normalised vectors) from the local index, cached by mtime."""
    path = Path(settings.local_index_path)
    if not path.exists():
        return [], np.empty((0, 0))
    mtime = path.stat().st_mtime
    if _CORPUS_CACHE.get("mtime") != mtime:
        data = np.load(path, allow_pickle=True)
        ids = [str(i) for i in data["ids"].tolist()]
        vecs = np.asarray(data["vectors"], dtype=float)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _CORPUS_CACHE.update(mtime=mtime, ids=ids, vecs=vecs / norms)
    return _CORPUS_CACHE["ids"], _CORPUS_CACHE["vecs"]


def seed_corpus(x: float, y: float, k: int) -> List[Dict[str, Any]]:
    """The ``k`` corpus projects that conceptually belong at a clicked location.

    Realizes the intent of "inverse-transform seeding" (DESIGN-SPACE-VIZ.md §3.3,
    option 3) robustly. Pure ``UMAP.inverse_transform`` proved too lossy on a small
    corpus to recover the right neighborhood, so instead we:

      1. Anchor at the corpus project nearest the click *in 2-D* — UMAP preserves
         LOCAL structure well, so this is a genuinely nearby real point.
      2. Expand the seed set to that anchor's nearest neighbors in the ORIGINAL
         embedding metric (768-d cosine) — faithful "what is similar here", not
         2-D-adjacent dots that UMAP's global distortion may have misplaced.

    Falls back to plain 2-D nearest if the corpus vectors aren't available.
    """
    surface_points = load_surface().get("points", [])
    ids, vecs = _load_corpus_vectors()
    if not surface_points or not ids or vecs.shape[0] != len(ids):
        return nearest_corpus(x, y, k)

    # Anchor: nearest surface dot to the click.
    coords2d = np.array([[p["x"], p["y"]] for p in surface_points], dtype=float)
    anchor = proj.nearest_indices(coords2d, [x, y], 1)
    if not anchor:
        return nearest_corpus(x, y, k)
    anchor_id = str(surface_points[anchor[0]]["id"])
    try:
        ai = ids.index(anchor_id)
    except ValueError:
        return nearest_corpus(x, y, k)

    # Expand in the original (faithful) metric around the anchor.
    sims = vecs @ vecs[ai]
    order = [int(i) for i in np.argsort(-sims)[: max(k, 0)]]
    point_by_id = {str(p["id"]): p for p in surface_points}
    return [_neighbour_record(ids[i], point_by_id.get(ids[i])) for i in order]


def generate_at(
    *,
    x: float,
    y: float,
    focus_node_id: str,
    focus_node_topic: str,
    taxonomy_nodes: List[Dict[str, Any]],
    lineage: List[str] | None = None,
    k: int = 5,
    mode: BackendMode | None = None,
    base_url: str | None = None,
    reasoning_effort: str = "medium",
) -> Dict[str, Any]:
    """Generate mind-map nodes seeded by whatever real projects surround a location.

    Faithful seeding (DESIGN-SPACE-VIZ.md §3.3, option 3 — see ``seed_corpus``):
    anchor at the click's nearest real project and expand in the original embedding
    metric, so the ``k`` seed projects genuinely belong at that location. Generated
    nodes are then embedded and placed in the frozen space at their own coordinates.

    ``mode`` defaults to the local backend when ``VECTOR_STORE=local`` so the design
    space generates with the same stack it embeds and retrieves with; pass it
    explicitly to override.
    """
    if mode is None:
        mode = BackendMode.vllm if settings.vector_store == "local" else BackendMode.openai

    neighbours = seed_corpus(x, y, k)

    result = generate_nodes_from_related_projects(
        focus_node_id=focus_node_id,
        focus_node_topic=focus_node_topic,
        taxonomy_nodes=taxonomy_nodes,
        related_projects=neighbours or None,
        lineage=lineage,
        should_query_supabase=False,
        mode=mode,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
    )

    located = locate_nodes(
        [{"node_id": n["node_id"], "text": n["topic"]} for n in result.get("nodes", [])]
    )
    coord_by_id = {c["node_id"]: c for c in located}

    result["coords"] = located
    result["seed_neighbours"] = neighbours
    result["target"] = {"x": x, "y": y}
    for node in result.get("nodes", []):
        coord = coord_by_id.get(node["node_id"])
        if coord:
            node["x"] = coord.get("x")
            node["y"] = coord.get("y")
            if "z" in coord:
                node["z"] = coord["z"]
    return result
