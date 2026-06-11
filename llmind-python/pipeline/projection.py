"""Frozen design-space projection: fit once, transform forever.

The *design space* is the 2D (or 3D) manifold of the real project corpus. We fit
a reducer (PCA → UMAP, cosine) **once** on the corpus embeddings and persist it.
Every later point — a new taxonomy node, a clicked location — is mapped into that
*same frozen space* via ``.transform()``, so coordinates are stable across
sessions and incremental mind-map growth.

This is the keystone of the design-space visualisation: without a persisted
reducer + persisted normalisation bounds, coordinates would relayout on every
call and the mind-map ↔ surface mapping would break.

Artifacts (under ``data/projection/``):
    model.joblib    fitted PCA + UMAP + bounds + meta (for transforming new points)
    surface.json    precomputed corpus background: grid spec, points, density

Nothing here does I/O to Supabase or an LLM; it is pure numpy/sklearn/umap.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_RESOLUTION = 48
DEFAULT_PRE_PCA = 64
DEFAULT_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1
DEFAULT_RANDOM_STATE = 42
DEFAULT_TRUST_NEIGHBORS = 15
MODEL_FILENAME = "model.joblib"
SURFACE_FILENAME = "surface.json"
# Out-of-hull points are squashed into a band this wide OUTSIDE [0, 1] instead of
# being pinned to the edge (≈3 lattice cells at R=48). Direction and ordering
# survive. Fallback-path + diagnostics only since Part 11 — the primary
# evidence-anchored placement cannot leave [0, 1].
SOFT_MARGIN = 0.06


# ── Fitted model ──────────────────────────────────────────────────────────────


@dataclass
class ProjectionModel:
    """A frozen reducer plus the normalisation bounds learned at fit time.

    ``pca`` may be ``None`` when the input dimensionality is already ``<= pre_pca``.
    ``bounds`` are the per-axis (min, max) of the *reference* projection; new points
    are normalised against these. Points outside the corpus hull are soft-clipped
    into a ``SOFT_MARGIN`` band around ``[0, 1]`` (and flagged) rather than pinned
    to the edge, so direction and ordering survive.
    """

    reducer: Any                       # fitted umap.UMAP
    pca: Any                           # fitted sklearn PCA or None
    bounds: List[Tuple[float, float]]  # one (min, max) per output axis
    dims: int
    meta: Dict[str, Any] = field(default_factory=dict)
    # Inputs are L2-normalised before reduction (Euclidean on unit vectors ≈
    # cosine).
    normalized: bool = True

    def _prep(self, arr: np.ndarray) -> np.ndarray:
        if not self.normalized:
            return arr
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def transform(self, X: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Map raw high-dim embeddings into the frozen, [0, 1]-normalised space."""
        coords, _ = self.transform_with_flags(X)
        return coords

    def transform_with_flags(
        self, X: np.ndarray | Sequence[Sequence[float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """``transform`` plus a per-point boolean mask of points that fell OUTSIDE
        the reference bounds and were soft-clipped into the margin band.

        A clipped point's position is an extrapolation, not corpus-supported.
        Used by the no-corpus fallback path and by diagnostics (`project-align`
        round-trips) — runtime placement is ``place_by_neighbors``.
        """
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = self._prep(arr)
        reduced = self.pca.transform(arr) if self.pca is not None else arr
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"umap(\.|$)")
            coords = np.asarray(self.reducer.transform(reduced), dtype=float)
        return self._normalize(coords)

    def _normalize(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalise to [0, 1] against the fit bounds; also return the per-point
        mask of coordinates that fell outside (out-of-hull points).

        Inside [0, 1] the mapping is the identity. Outside, the overshoot is
        tanh-compressed into ``(0, SOFT_MARGIN)`` so out-of-hull points keep
        their direction and relative ordering instead of stacking at the edge.
        """
        out = np.empty_like(coords, dtype=float)
        for axis, (lo, hi) in enumerate(self.bounds):
            if math.isclose(hi, lo):
                out[:, axis] = 0.5
            else:
                out[:, axis] = (coords[:, axis] - lo) / (hi - lo)
        clipped = ((out < 0.0) | (out > 1.0)).any(axis=1)
        return _soft_clip(out, SOFT_MARGIN), clipped


def _soft_clip(values: np.ndarray, margin: float) -> np.ndarray:
    """Identity inside [0, 1]; tanh-compressed overshoot outside, asymptoting at
    ``±margin``. Monotonic, so out-of-hull ordering is preserved."""
    out = np.asarray(values, dtype=float).copy()
    under = out < 0.0
    over = out > 1.0
    out[under] = -margin * np.tanh(-out[under] / margin)
    out[over] = 1.0 + margin * np.tanh((out[over] - 1.0) / margin)
    return out


# ── Out-of-sample placement (evidence-anchored, Part 11) ─────────────────────


def place_by_neighbors(
    vecs_unit: np.ndarray,
    corpus_unit: np.ndarray,
    corpus_coords: np.ndarray,
    k: int,
    exclude_rows: Sequence[int] | None = None,
) -> np.ndarray:
    """Place each query at the similarity-weighted centroid of its top-``k``
    corpus neighbours' frozen coordinates.

    UMAP has no principled out-of-sample extension — ``.transform()`` is itself
    neighbour-anchored plus stochastic optimisation, and on corpus round-trips
    it displaced points further than this interpolation on every statistic
    (ITERATION-PLAN Part 11). A convex combination of precedents' positions
    cannot leave the corpus footprint by construction; outsideness is corpus
    support's job (768-d, faithful), not the 2D layout's.

    Weights ∝ positive cosine similarity (uniform when degenerate).
    ``exclude_rows`` masks one corpus row per query — fit-time diagnostics
    only, mirroring ``support_scores`` (runtime queries have no "self").
    """
    sims = np.asarray(vecs_unit, dtype=float) @ np.asarray(corpus_unit, dtype=float).T
    if exclude_rows is not None:
        for i, row in enumerate(exclude_rows):
            sims[i, row] = -np.inf
    coords = np.asarray(corpus_coords, dtype=float)
    k = max(1, min(k, sims.shape[1] - (1 if exclude_rows is not None else 0)))
    out = np.empty((sims.shape[0], coords.shape[1]), dtype=float)
    for i, row in enumerate(sims):
        top = np.argsort(row)[-k:]
        weights = np.clip(row[top], 0.0, None)
        total = weights.sum()
        weights = weights / total if total > 0 else np.full(k, 1.0 / k)
        out[i] = (coords[top] * weights[:, None]).sum(axis=0)
    return out


# ── Fitting ─────────────────────────────────────────────────────────────────


def fit_projection(
    X: np.ndarray | Sequence[Sequence[float]],
    *,
    dims: int = 2,
    n_neighbors: int = DEFAULT_NEIGHBORS,
    min_dist: float = DEFAULT_MIN_DIST,
    pre_pca: Optional[int] = DEFAULT_PRE_PCA,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ProjectionModel:
    """Fit PCA → UMAP on the L2-normalised reference corpus and record [0,1] bounds.

    Vectors are unit-normalised and UMAP uses the Euclidean metric — equivalent to
    cosine on unit vectors. The returned model's ``transform`` reproduces these
    coordinates for the same inputs and places new inputs consistently in the same
    frame (runtime /locate placement is evidence-anchored instead — see
    ``place_by_neighbors``; ``transform`` is the no-corpus fallback and the
    diagnostics baseline).
    """
    import umap  # type: ignore

    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("fit_projection expects a non-empty 2D array of embeddings.")

    # L2-normalise so Euclidean ≈ cosine and the inverse map is well-defined.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr = arr / norms

    n_samples = arr.shape[0]
    pca = None
    reduced = arr
    if pre_pca and arr.shape[1] > pre_pca:
        from sklearn.decomposition import PCA  # type: ignore

        n_components = min(pre_pca, n_samples, arr.shape[1])
        pca = PCA(n_components=n_components, random_state=random_state).fit(arr)
        reduced = pca.transform(arr)

    # UMAP needs n_neighbors < n_samples; clamp for tiny corpora.
    safe_neighbors = max(2, min(n_neighbors, n_samples - 1))
    reducer = umap.UMAP(
        n_components=dims,
        n_neighbors=safe_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"n_jobs value .* overridden to 1 by setting random_state.*",
            category=UserWarning,
            module=r"umap(\.|$)",
        )
        coords = np.asarray(reducer.fit_transform(reduced), dtype=float)

    # Layout fidelity: sklearn trustworthiness of the 2D embedding w.r.t. the
    # (unit-normalised) input space. Reported to the UI so the surface is honest
    # about how much of the high-dim neighbourhood structure survived projection.
    trust: Optional[float] = None
    trust_k = min(DEFAULT_TRUST_NEIGHBORS, max(1, (n_samples - 1) // 2))
    if n_samples > 4:
        try:
            from sklearn.manifold import trustworthiness as _trustworthiness

            trust = float(_trustworthiness(arr, coords, n_neighbors=trust_k))
        except Exception:  # noqa: BLE001 — fidelity score is best-effort metadata
            trust = None

    bounds = [(float(coords[:, a].min()), float(coords[:, a].max())) for a in range(dims)]
    return ProjectionModel(
        reducer=reducer,
        pca=pca,
        bounds=bounds,
        dims=dims,
        meta={
            "n_reference": n_samples,
            "input_dims": int(arr.shape[1]),
            "n_neighbors": safe_neighbors,
            "min_dist": min_dist,
            "pre_pca": (pca.n_components_ if pca is not None else None),
            "metric": "euclidean (unit-normalized ≈ cosine)",
            "random_state": random_state,
            "trustworthiness": trust,
            "trust_neighbors": trust_k,
        },
        normalized=True,
    )


# ── Persistence ─────────────────────────────────────────────────────────────


def save_model(model: ProjectionModel, projection_dir: Path) -> Path:
    import joblib  # type: ignore

    projection_dir.mkdir(parents=True, exist_ok=True)
    path = projection_dir / MODEL_FILENAME
    joblib.dump(model, path)
    return path


def load_model(projection_dir: Path) -> ProjectionModel:
    import joblib  # type: ignore

    path = projection_dir / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Projection model not found at {path}. "
            f"Build it with: uv run python database_pipeline.py project"
        )
    return joblib.load(path)


# ── Grid quantisation (presentation layer; never mutates true coords) ─────────


def to_cell(x: float, y: float, resolution: int) -> Tuple[int, int]:
    gx = min(resolution - 1, max(0, int(x * resolution)))
    gy = min(resolution - 1, max(0, int(y * resolution)))
    return gx, gy


def cell_center(gx: int, gy: int, resolution: int) -> Tuple[float, float]:
    return (gx + 0.5) / resolution, (gy + 0.5) / resolution


def density_grid(points_xy: np.ndarray, resolution: int) -> List[List[int]]:
    """Per-cell occupancy counts of the corpus — the heat field on empty cells."""
    grid = [[0] * resolution for _ in range(resolution)]
    for x, y in points_xy[:, :2]:
        gx, gy = to_cell(float(x), float(y), resolution)
        grid[gy][gx] += 1
    return grid


def nearest_indices(points_xy: np.ndarray, query_xy: Sequence[float], k: int) -> List[int]:
    """Indices of the ``k`` corpus points nearest a location, by Euclidean 2D/3D distance.

    Distance is in the projected space — a presentation-faithful seed for "what is
    around this spot", not a claim about the original metric (see
    documentations/DESIGN-SPACE-VIZ.md §3.3).
    """
    if points_xy.shape[0] == 0 or k <= 0:
        return []
    q = np.asarray(query_xy, dtype=float)[: points_xy.shape[1]]
    dists = np.linalg.norm(points_xy - q, axis=1)
    order = np.argsort(dists)[: min(k, points_xy.shape[0])]
    return [int(i) for i in order]


# ── Surface payload (what the GET /surface endpoint serves) ───────────────────


def build_surface_payload(
    *,
    ids: Sequence[str],
    coords: np.ndarray,
    dims: int,
    resolution: int,
    clusters: Optional[Sequence[int]] = None,
    model_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the persisted background surface: grid spec, corpus points, density.

    Coordinates are continuous floats in ``[0, 1]``; the cell is a render-time
    convenience. The frontend reads this verbatim for the read-only background.
    """
    coords = np.asarray(coords, dtype=float)
    axis_names = ["x", "y", "z"][:dims]
    points: List[Dict[str, Any]] = []
    for i, pid in enumerate(ids):
        point: Dict[str, Any] = {"id": str(pid), "kind": "project"}
        for a, name in enumerate(axis_names):
            point[name] = float(coords[i, a])
        gx, gy = to_cell(float(coords[i, 0]), float(coords[i, 1]), resolution)
        point["cell"] = [gx, gy]
        if clusters is not None:
            point["cluster"] = int(clusters[i])
        points.append(point)

    return {
        "version": 1,
        "dims": dims,
        "grid": {"resolution": resolution},
        "bounds": {"min": 0.0, "max": 1.0},  # coords are pre-normalised
        "density": density_grid(coords, resolution),
        "points": points,
        "meta": dict(model_meta or {}),
    }
