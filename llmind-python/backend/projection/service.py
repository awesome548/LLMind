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
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from config import settings
from utils.clients import build_vllm_client
from utils.modes import BackendMode
from utils.prompts import GENERATE_AT_PROMPT, GENERATE_AT_PROMPT_VERSION
from pipeline import projection as proj
from pipeline import register_alignment as ra
from backend.corpus.service import SUPPORT_NEIGHBORS, corpus_support
from backend.corpus.service import load_corpus_vectors as _load_corpus_vectors
from backend.corpus.service import load_index_meta
from backend.related_projects.service import (
    ServiceError,
    generate_nodes_from_related_projects,
)


# ── Cached artifacts (invalidated on mtime, like utils.local_store) ───────────

_MODEL_CACHE: Dict[str, Any] = {}
_SURFACE_CACHE: Dict[str, Any] = {}
_REGISTER_CACHE: Dict[str, Any] = {}

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


def _placement_frame() -> tuple[np.ndarray, np.ndarray] | None:
    """``(corpus_unit, corpus_coords)`` aligned by project id, for
    evidence-anchored placement (Part 11). ``None`` when the corpus vectors or
    surface are unavailable — /locate then falls back to the frozen transform.
    """
    try:
        ids, vecs = _load_corpus_vectors()
        points = load_surface().get("points", [])
    except ServiceError:
        return None
    coord_by_id = {
        str(p["id"]): [p[a] for a in ("x", "y", "z") if a in p] for p in points
    }
    rows = [i for i, pid in enumerate(ids) if pid in coord_by_id]
    if len(rows) < 2:
        return None
    return vecs[rows], np.asarray([coord_by_id[ids[i]] for i in rows], dtype=float)


def placement_method() -> str:
    """Which out-of-sample placement /locate currently uses — logged per
    generation so drift stays comparable within one placement regime."""
    return "knn" if _placement_frame() is not None else "umap"


def _load_register_map() -> ra.RegisterMap | None:
    """The fitted short→long register correction, cached by mtime.

    ``None`` when the artifact is absent (run ``database_pipeline.py
    project-align`` to fit it) — /locate then places raw embeddings.
    """
    path = settings.projection_dir / ra.REGISTER_MAP_FILENAME
    if not path.exists():
        _REGISTER_CACHE.clear()
        return None
    mtime = path.stat().st_mtime
    if _REGISTER_CACHE.get("mtime") != mtime:
        _REGISTER_CACHE["map"] = ra.load_register_map(settings.projection_dir)
        _REGISTER_CACHE["mtime"] = mtime
    return _REGISTER_CACHE["map"]


def register_map_active() -> bool:
    """Whether /locate currently applies the register correction (logged per
    generation so variants stay comparable in generate_log.jsonl)."""
    return settings.register_alignment and _load_register_map() is not None


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


# ── Placement confidence ──────────────────────────────────────────────────────

CONFIDENCE_NEIGHBORS = 10


def _placement_confidence(
    embeddings_unit: np.ndarray, coords: np.ndarray, k: int = CONFIDENCE_NEIGHBORS
) -> List[float | None]:
    """How much to trust each node's 2D position, as a Jaccard overlap in [0, 1].

    For each located point, compare its ``k`` nearest corpus projects in the
    ORIGINAL embedding metric (cosine on unit vectors — ground truth similarity)
    with its ``k`` nearest corpus projects in the projected 2D space. 1.0 means
    the projection placed the node among exactly the projects it is truly similar
    to; near 0 means the 2D position is a projection artifact for this node.

    Best-effort: returns ``None`` per point when the corpus vectors or surface
    are unavailable (the coordinate is still served, just unscored).
    """
    try:
        ids, vecs = _load_corpus_vectors()
        surface_points = load_surface().get("points", [])
    except ServiceError:
        return [None] * len(coords)
    if not ids or vecs.shape[0] != len(ids) or not surface_points:
        return [None] * len(coords)
    if vecs.shape[1] != embeddings_unit.shape[1]:
        return [None] * len(coords)

    pts2d = np.array([[p["x"], p["y"]] for p in surface_points], dtype=float)
    ids2d = [str(p["id"]) for p in surface_points]
    k_eff = min(k, len(ids), len(ids2d))
    if k_eff == 0:
        return [None] * len(coords)

    out: List[float | None] = []
    for emb, coord in zip(embeddings_unit, coords):
        sims = vecs @ emb
        top_orig = {ids[int(i)] for i in np.argsort(-sims)[:k_eff]}
        dists = np.linalg.norm(pts2d - np.asarray(coord[:2], dtype=float), axis=1)
        top_2d = {ids2d[int(i)] for i in np.argsort(dists)[:k_eff]}
        union = top_orig | top_2d
        out.append(len(top_orig & top_2d) / len(union) if union else None)
    return out


# ── Public operations ─────────────────────────────────────────────────────────


def locate_nodes(items: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Embed each ``{node_id, text}`` and place it in the frozen design space.

    Returns ``[{node_id, x, y[, z], confidence, clipped, support}]`` aligned to
    the input order. Items with empty text are skipped. ``confidence`` scores
    how well the 2D neighbourhood matches the true embedding neighbourhood;
    ``support`` is the corpus-support percentile (how much corpus evidence
    exists for this point at all); both ``None`` when unscorable. ``clipped``
    can only be true on the fallback (no-corpus) transform path — the primary
    evidence-anchored placement is a convex combination of corpus positions.
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

    # Unit-normalise once; optionally close the short→long register gap. The
    # transform, the confidence score, and the support score all consume the
    # SAME (corrected) vector, so the three signals describe one placement.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embeddings / norms
    rmap = _load_register_map() if settings.register_alignment else None
    if rmap is not None and rmap.weights.shape[0] == unit.shape[1]:
        unit = rmap.apply(unit)

    # Evidence-anchored placement (Part 11): the similarity-weighted centroid
    # of the top-k corpus neighbours' frozen coordinates — the SAME anchors
    # that drive corpus support and the precedents panel, so position and
    # evidence tell one story. The frozen UMAP transform (with its soft-clip
    # band + clipped flag) remains the fallback when the corpus is unavailable.
    frame = _placement_frame()
    if frame is not None and frame[0].shape[1] == unit.shape[1]:
        coords = proj.place_by_neighbors(unit, frame[0], frame[1], k=SUPPORT_NEIGHBORS)
        clipped = np.zeros(len(coords), dtype=bool)
    else:
        with _TRANSFORM_LOCK:
            coords, clipped = model.transform_with_flags(unit)
    axis_names = ["x", "y", "z"][: model.dims]

    confidences = _placement_confidence(unit, coords)
    # Support is read against the SHORT-register baseline when the fitted map
    # provides one (Part 10 recalibration): node-length texts are compared to
    # real projects described at node length, not to full descriptions — which
    # flattened every short text to the 0th percentile.
    supports = corpus_support(
        unit, baseline=rmap.support_baseline if rmap is not None else None
    )

    located: List[Dict[str, Any]] = []
    for it, row, conf, clip, sup in zip(valid, coords, confidences, clipped, supports):
        entry: Dict[str, Any] = {"node_id": it["node_id"]}
        for a, name in enumerate(axis_names):
            entry[name] = float(row[a])
        entry["confidence"] = conf
        entry["clipped"] = bool(clip)
        entry["support"] = sup
        located.append(entry)
    return located


def _neighbour_record(pid: str, surface_point: Dict[str, Any] | None) -> Dict[str, Any]:
    meta = load_index_meta()
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


def _anchor_seed_corpus(x: float, y: float, k: int) -> List[Dict[str, Any]]:
    """Legacy seeding: the single nearest project's true-metric neighbourhood.

    Kept behind ``SEED_STRATEGY=anchor`` for A/B comparison. Its weakness (the
    reason "bracket" replaced it): the seeds form a tight cluster AROUND ONE
    project at the gap's edge, so generation drifts toward imitating that
    project instead of filling the gap between several.
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


def _bracket_seed_corpus(x: float, y: float, k: int) -> List[Dict[str, Any]]:
    """Bracketing seeds: projects on DIFFERENT sides of the clicked gap.

    "Generate in this gap" needs seeds that describe the gap's boundary, not one
    of its edges (ITERATION-PLAN B2). So:

      1. Pool the ~2k corpus points nearest the click in 2D (its visual
         neighbourhood — UMAP is locally trustworthy).
      2. Greedy max-min (farthest-point) selection of up to 3 anchors from the
         pool, so the anchors spread AROUND the click instead of clustering on
         its nearest edge.
      3. Deepen each anchor with its top neighbour in the ORIGINAL embedding
         metric (768-d cosine) — true similarity, immune to 2D distortion.

    Falls back to plain 2D-nearest if corpus vectors aren't available.
    """
    surface_points = load_surface().get("points", [])
    ids, vecs = _load_corpus_vectors()
    if not surface_points or not ids or vecs.shape[0] != len(ids):
        return nearest_corpus(x, y, k)

    coords2d = np.array([[p["x"], p["y"]] for p in surface_points], dtype=float)
    pool = proj.nearest_indices(coords2d, [x, y], max(2 * k, 6))
    if not pool:
        return nearest_corpus(x, y, k)

    # Greedy max-min anchors, seeded with the closest point (pool is
    # distance-ordered by nearest_indices).
    n_anchors = max(1, min(3, k))
    anchors: List[int] = [pool[0]]
    while len(anchors) < n_anchors and len(anchors) < len(pool):
        best_idx, best_d = None, -1.0
        for candidate in pool:
            if candidate in anchors:
                continue
            d = min(
                float(np.linalg.norm(coords2d[candidate] - coords2d[a])) for a in anchors
            )
            if d > best_d:
                best_idx, best_d = candidate, d
        if best_idx is None:
            break
        anchors.append(best_idx)

    row_of = {pid: i for i, pid in enumerate(ids)}
    surface_idx_of = {str(p["id"]): i for i, p in enumerate(surface_points)}
    selected: List[int] = []
    seen: set[str] = set()

    def _add(surface_idx: int) -> None:
        pid = str(surface_points[surface_idx]["id"])
        if pid not in seen:
            seen.add(pid)
            selected.append(surface_idx)

    for anchor in anchors:
        if len(selected) >= k:
            break
        _add(anchor)
        row = row_of.get(str(surface_points[anchor]["id"]))
        if row is None or len(selected) >= k:
            continue
        sims = vecs @ vecs[row]
        for j in np.argsort(-sims):
            neighbour_id = ids[int(j)]
            if neighbour_id in seen:
                continue
            neighbour_idx = surface_idx_of.get(neighbour_id)
            if neighbour_idx is not None:
                _add(neighbour_idx)
            break

    # Top up with plain 2D-nearest if anchors + depth didn't reach k.
    for candidate in pool:
        if len(selected) >= k:
            break
        _add(candidate)

    return [
        _neighbour_record(str(surface_points[i]["id"]), surface_points[i])
        for i in selected[:k]
    ]


def seed_corpus(x: float, y: float, k: int) -> List[Dict[str, Any]]:
    """The ``k`` corpus projects that conceptually belong at a clicked location.

    Strategy is configurable (``SEED_STRATEGY``): "bracket" (default) surrounds
    the gap; "anchor" is the legacy nearest-project expansion. Both fall back to
    plain 2D-nearest when corpus vectors are unavailable.
    """
    if settings.seed_strategy == "anchor":
        return _anchor_seed_corpus(x, y, k)
    return _bracket_seed_corpus(x, y, k)


# Located taxonomy nodes within this radius of the click count as "nearby
# existing ideas" the prompt must not duplicate (~6 lattice cells at R=48).
NEARBY_OPTIONS_RADIUS = 0.12


def compute_axes(
    *,
    x_pole_a: str,
    x_pole_b: str,
    y_pole_a: str,
    y_pole_b: str,
    items: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """Exact bipolar coordinates for the semantic-axes perspective.

    ``score_axis(v) = cos(v, pole_a) − cos(v, pole_b)`` in the ORIGINAL embedding
    metric — no UMAP, no stochasticity, no distortion ("exact by construction").
    Corpus scores are min-max normalised per axis to [-1, 1]; taxonomy/candidate
    items use the corpus range and are clipped (and flagged) when outside it.
    Diagnostics expose degenerate axes (poles too similar) and redundant axis
    pairs (high |corr| → points hug the diagonal).
    """
    ids, vecs = _load_corpus_vectors()
    if not ids:
        raise ServiceError(
            "Corpus vectors not found — build the local index with build_local_index.py."
        )

    valid_items = [
        it for it in (items or [])
        if (it.get("text") or "").strip() and it.get("node_id")
    ]
    texts = [x_pole_a, x_pole_b, y_pole_a, y_pole_b] + [it["text"] for it in valid_items]
    embeddings = _embed_texts(texts)
    if embeddings.shape[1] != vecs.shape[1]:
        raise ServiceError(
            f"Embedding dim {embeddings.shape[1]} != corpus dim {vecs.shape[1]} — the "
            f"runtime embedding model differs from the one the index was built with."
        )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embeddings / norms
    pole_ax, pole_bx, pole_ay, pole_by = unit[0], unit[1], unit[2], unit[3]
    item_vecs = unit[4:]

    raw_x = vecs @ pole_ax - vecs @ pole_bx
    raw_y = vecs @ pole_ay - vecs @ pole_by
    x_lo, x_hi = float(raw_x.min()), float(raw_x.max())
    y_lo, y_hi = float(raw_y.min()), float(raw_y.max())

    def _norm(value: float, lo: float, hi: float) -> float:
        if math.isclose(hi, lo):
            return 0.0
        return 2.0 * (value - lo) / (hi - lo) - 1.0

    corpus = [
        {"id": pid, "x": _norm(float(rx), x_lo, x_hi), "y": _norm(float(ry), y_lo, y_hi)}
        for pid, rx, ry in zip(ids, raw_x, raw_y)
    ]

    located_items: List[Dict[str, Any]] = []
    for it, vec in zip(valid_items, item_vecs):
        ix = _norm(float(vec @ pole_ax - vec @ pole_bx), x_lo, x_hi)
        iy = _norm(float(vec @ pole_ay - vec @ pole_by), y_lo, y_hi)
        clipped = abs(ix) > 1.0 or abs(iy) > 1.0
        located_items.append(
            {
                "node_id": it["node_id"],
                "x": max(-1.0, min(1.0, ix)),
                "y": max(-1.0, min(1.0, iy)),
                "clipped": clipped,
            }
        )

    if float(raw_x.std()) > 0 and float(raw_y.std()) > 0:
        axis_corr = float(np.corrcoef(raw_x, raw_y)[0, 1])
    else:
        axis_corr = 0.0

    return {
        "corpus": corpus,
        "items": located_items,
        "meta": {
            "x_pole_sim": float(pole_ax @ pole_bx),
            "y_pole_sim": float(pole_ay @ pole_by),
            "axis_corr": axis_corr,
        },
    }


def compute_metrics(
    *,
    metrics: List[Dict[str, str]],
    items: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """N bipolar metrics scored over the corpus and the given items (Part 10 I3).

    Generalises the axes scoring to a metric LIST — the Perspectives strips:
    ``score_k(v) = cos(v, pole_a_k) − cos(v, pole_b_k)``, min-max normalised over
    the corpus to [-1, 1]. Returns the FULL corpus score array per metric (the
    strip's distribution — percentiles are computed client-side from it),
    clip-flagged item scores, per-metric pole similarity, and the pairwise
    corpus-score correlation matrix (rubric-redundancy diagnostic).
    """
    ids, vecs = _load_corpus_vectors()
    if not ids:
        raise ServiceError(
            "Corpus vectors not found — build the local index with build_local_index.py."
        )
    valid_items = [
        it for it in (items or [])
        if (it.get("text") or "").strip() and it.get("node_id")
    ]
    texts = [t for m in metrics for t in (m["pole_a"], m["pole_b"])]
    texts += [it["text"] for it in valid_items]
    embeddings = _embed_texts(texts)
    if embeddings.shape[1] != vecs.shape[1]:
        raise ServiceError(
            f"Embedding dim {embeddings.shape[1]} != corpus dim {vecs.shape[1]} — the "
            f"runtime embedding model differs from the one the index was built with."
        )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = embeddings / norms
    item_vecs = unit[2 * len(metrics):]

    def _norm(value: float, lo: float, hi: float) -> float:
        if math.isclose(hi, lo):
            return 0.0
        return 2.0 * (value - lo) / (hi - lo) - 1.0

    results: List[Dict[str, Any]] = []
    raw_rows: List[np.ndarray] = []
    for k in range(len(metrics)):
        pole_a, pole_b = unit[2 * k], unit[2 * k + 1]
        raw = vecs @ pole_a - vecs @ pole_b
        raw_rows.append(raw)
        lo, hi = float(raw.min()), float(raw.max())
        scored_items: List[Dict[str, Any]] = []
        for it, vec in zip(valid_items, item_vecs):
            score = _norm(float(vec @ pole_a - vec @ pole_b), lo, hi)
            scored_items.append(
                {
                    "node_id": it["node_id"],
                    "score": max(-1.0, min(1.0, score)),
                    "clipped": abs(score) > 1.0,
                }
            )
        results.append(
            {
                "corpus": [_norm(float(r), lo, hi) for r in raw],
                "items": scored_items,
                "pole_sim": float(pole_a @ pole_b),
            }
        )

    # Pairwise corpus-score correlations: |r| near 1 → redundant rubric metrics.
    corr: List[List[float]] = [[0.0] * len(metrics) for _ in range(len(metrics))]
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            if i == j:
                corr[i][j] = 1.0
            elif j > i:
                a, b = raw_rows[i], raw_rows[j]
                value = (
                    float(np.corrcoef(a, b)[0, 1])
                    if float(a.std()) > 0 and float(b.std()) > 0
                    else 0.0
                )
                corr[i][j] = corr[j][i] = value
    return {"metrics": results, "corr": corr}


def _derive_parent_aspect(
    taxonomy_nodes: List[Dict[str, Any]],
    node_coords: List[Dict[str, Any]],
    x: float,
    y: float,
) -> Dict[str, Any] | None:
    """The aspect whose located nodes' centroid is nearest the click.

    One spatial notion of "near here", owned by the backend: the same click that
    chooses the seed projects also chooses the parent aspect (ITERATION-PLAN B4).
    Returns ``{id, topic, lineage}`` or ``None`` when underivable.
    """
    if not node_coords:
        return None
    by_id = {str(n.get("id")): n for n in taxonomy_nodes if n.get("id")}
    root = next((n for n in taxonomy_nodes if n.get("isroot")), None) or next(
        (n for n in taxonomy_nodes if not n.get("parentid")), None
    )
    if not root:
        return None
    root_id = str(root.get("id"))

    def aspect_of(node_id: str) -> str | None:
        cur = by_id.get(node_id)
        hops = 0
        while cur is not None and hops < 50:
            parent = str(cur.get("parentid") or "")
            if not parent:
                return None  # reached the root itself
            if parent == root_id:
                return str(cur.get("id"))
            cur = by_id.get(parent)
            hops += 1
        return None

    sums: Dict[str, List[float]] = {}
    for c in node_coords:
        aspect_id = aspect_of(str(c.get("node_id") or ""))
        if not aspect_id:
            continue
        acc = sums.setdefault(aspect_id, [0.0, 0.0, 0.0])
        acc[0] += float(c["x"])
        acc[1] += float(c["y"])
        acc[2] += 1.0

    best_id, best_d = None, float("inf")
    for aspect_id, (sx, sy, n) in sums.items():
        d = (sx / n - x) ** 2 + (sy / n - y) ** 2
        if d < best_d:
            best_id, best_d = aspect_id, d
    if best_id is None:
        return None
    aspect = by_id[best_id]
    return {
        "id": best_id,
        "topic": str(aspect.get("topic") or ""),
        "lineage": [str(root.get("topic") or ""), str(aspect.get("topic") or "")],
    }


def _nearby_options(
    taxonomy_nodes: List[Dict[str, Any]],
    node_coords: List[Dict[str, Any]],
    x: float,
    y: float,
    limit: int = 8,
) -> List[str]:
    """Topics of located taxonomy nodes within ``NEARBY_OPTIONS_RADIUS`` of the
    click, nearest first — the "already explored here" set."""
    by_id = {str(n.get("id")): n for n in taxonomy_nodes if n.get("id")}
    rows: List[tuple[float, str]] = []
    for c in node_coords:
        node = by_id.get(str(c.get("node_id") or ""))
        if not node:
            continue
        d = math.hypot(float(c["x"]) - x, float(c["y"]) - y)
        if d <= NEARBY_OPTIONS_RADIUS:
            topic = str(node.get("topic") or "").strip()
            if topic:
                rows.append((d, topic))
    rows.sort()
    return [topic for _, topic in rows[:limit]]


def _format_nearby_options(
    taxonomy_nodes: List[Dict[str, Any]],
    node_coords: List[Dict[str, Any]],
    x: float,
    y: float,
) -> str:
    """Prompt rendering of ``_nearby_options`` ("do not duplicate" section)."""
    topics = _nearby_options(taxonomy_nodes, node_coords, x, y)
    if not topics:
        return "(none yet — this region of the map is unexplored)"
    return "\n".join(f"- {topic}" for topic in topics)


def peek(
    *,
    x: float,
    y: float,
    k: int = 5,
    taxonomy_nodes: List[Dict[str, Any]] | None = None,
    node_coords: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Gap preview — everything a generation at ``(x, y)`` would be briefed with,
    WITHOUT calling the LLM (and without the embedding server: seeds come from
    precomputed corpus vectors).

    ``seed_corpus`` is deterministic for a given click, so a subsequent
    ``generate_at`` uses exactly the seeds shown here. The preview also names the
    parent aspect the click would attach new options to (same derivation).
    """
    nodes_list = taxonomy_nodes or []
    coords_list = node_coords or []
    derived = _derive_parent_aspect(nodes_list, coords_list, x, y)
    return {
        "seeds": seed_corpus(x, y, k),
        "nearby_options": _nearby_options(nodes_list, coords_list, x, y),
        "parent_aspect": derived["topic"] if derived else None,
    }


def _log_generation(payload: Dict[str, Any]) -> None:
    """Append one JSONL row per generate-at call (best-effort).

    This is the evaluation dataset for comparing prompt/seeding variants
    (ITERATION-PLAN B5): drift before/after is the concrete before/after metric.
    """
    try:
        path = settings.projection_dir / "generate_log.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 — logging must never break generation
        pass


def generate_at(
    *,
    x: float,
    y: float,
    focus_node_id: str,
    focus_node_topic: str,
    taxonomy_nodes: List[Dict[str, Any]],
    lineage: List[str] | None = None,
    k: int = 5,
    node_coords: List[Dict[str, Any]] | None = None,
    mode: BackendMode | None = None,
    base_url: str | None = None,
    reasoning_effort: str = "medium",
    brief: str | None = None,
) -> Dict[str, Any]:
    """Generate mind-map nodes that fill the gap at a clicked location.

    Location-conditioned (ITERATION-PLAN B1-B5): the seeds bracket the gap
    (``seed_corpus``), the prompt states the spatial intent and lists nearby
    already-explored options, the parent aspect is derived from the same click
    (``_derive_parent_aspect``, falling back to the caller's focus node), and
    every generated node reports its drift from the click. Each call is logged
    to ``data/projection/generate_log.jsonl``.

    ``mode`` defaults to the local backend when ``VECTOR_STORE=local`` so the
    design space generates with the same stack it embeds and retrieves with.
    """
    if mode is None:
        mode = BackendMode.vllm if settings.vector_store == "local" else BackendMode.openai

    neighbours = seed_corpus(x, y, k)
    coords_list = node_coords or []

    # One click, one notion of "here": the parent aspect comes from the same
    # location as the seeds. The caller's focus node is only the fallback.
    derived = _derive_parent_aspect(taxonomy_nodes, coords_list, x, y)
    if derived and derived["topic"]:
        focus_node_id = derived["id"]
        focus_node_topic = derived["topic"]
        lineage = derived["lineage"]

    nearby_options = _format_nearby_options(taxonomy_nodes, coords_list, x, y)

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
        prompt_template=GENERATE_AT_PROMPT,
        extra_template_fields={
            "NEARBY_OPTIONS": nearby_options,
            # Squiggle hypothesis (Part 10): the designer's work-in-progress
            # concept conditions gap-filling; logged so the effect is measurable.
            "DESIGNER_BRIEF": (brief or "").strip() or "(none provided)",
        },
    )

    # Place each generated option by the same text composition the page uses
    # (topic + one-line desc) — the desc is what makes short labels locatable.
    located = locate_nodes(
        [
            {
                "node_id": n["node_id"],
                "text": (
                    f"{n['topic']}. {n['desc']}" if n.get("desc") else n["topic"]
                ),
            }
            for n in result.get("nodes", [])
        ]
    )
    coord_by_id = {c["node_id"]: c for c in located}

    result["coords"] = located
    result["seed_neighbours"] = neighbours
    result["target"] = {"x": x, "y": y}
    # Clipped placements are directions, not locations — they are flagged and
    # excluded from the drift aggregate so the A/B metric stays meaningful.
    drifts: List[float] = []
    for node in result.get("nodes", []):
        coord = coord_by_id.get(node["node_id"])
        if coord:
            node["x"] = coord.get("x")
            node["y"] = coord.get("y")
            if "z" in coord:
                node["z"] = coord["z"]
            node["clipped"] = bool(coord.get("clipped", False))
            node["support"] = coord.get("support")
            node["drift"] = math.hypot(float(node["x"]) - x, float(node["y"]) - y)
            if not node["clipped"]:
                drifts.append(node["drift"])
    result["mean_drift"] = (sum(drifts) / len(drifts)) if drifts else None

    _log_generation(
        {
            "ts": time.time(),
            "prompt_version": GENERATE_AT_PROMPT_VERSION,
            "seed_strategy": settings.seed_strategy,
            "register_aligned": register_map_active(),
            "brief_context": bool((brief or "").strip()),
            # Drift means different things under different placements — never
            # aggregate across regimes (Part 11).
            "placement": placement_method(),
            "target": {"x": x, "y": y},
            "parent_id": result.get("parent_id"),
            "parent_derived": bool(derived),
            "mode": str(mode),
            "reasoning_effort": reasoning_effort,
            "seed_ids": [n.get("id") for n in neighbours],
            "mean_drift": result["mean_drift"],
            "nodes": [
                {
                    "id": n.get("node_id"),
                    "topic": n.get("topic"),
                    # The desc is part of the located text — kept so later
                    # diagnostics can re-embed exactly what was placed.
                    "desc": n.get("desc"),
                    "x": n.get("x"),
                    "y": n.get("y"),
                    "drift": n.get("drift"),
                    "clipped": n.get("clipped", False),
                    "support": n.get("support"),
                }
                for n in result.get("nodes", [])
            ],
        }
    )
    return result
