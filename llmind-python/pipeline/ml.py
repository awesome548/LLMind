"""Math and ML helpers: normalisation, UMAP, KMeans, farthest-point selection."""

from __future__ import annotations

import math
import warnings
from typing import Any, List, Optional

import numpy as np


def unit_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def normalize_to_unit_interval(arr: np.ndarray) -> List[float]:
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if math.isclose(hi, lo):
        return [0.5] * len(arr)
    return [float((v - lo) / (hi - lo)) for v in arr]


def numpy_json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def umap_reduce(
    X: np.ndarray,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    pre_pca: Optional[int] = 64,
) -> np.ndarray:
    """Reduce high-dimensional embeddings to 2D with UMAP (cosine metric)."""
    import umap  # type: ignore

    X_in = X
    if pre_pca and X.shape[1] > pre_pca:
        from sklearn.decomposition import PCA  # type: ignore
        X_in = PCA(n_components=pre_pca, random_state=random_state).fit_transform(X)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"n_jobs value .* overridden to 1 by setting random_state.*",
            category=UserWarning,
            module=r"umap(\.|$)",
        )
        return reducer.fit_transform(X_in)


def kmeans_cluster(X: np.ndarray, k: int) -> List[int]:
    try:
        from sklearn.cluster import KMeans  # type: ignore
        return KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(X).tolist()
    except Exception:
        return [0] * len(X)


def select_farthest(
    embeddings: np.ndarray | List[List[float]],
    k: int = 20,
    seed: int = 42,
) -> List[int]:
    """Greedy farthest-point subset selection using cosine distance."""
    X = np.asarray(embeddings, dtype=float)
    n = X.shape[0]
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))

    Xn = unit_normalize(X)
    rng = np.random.RandomState(seed)
    start = int(rng.randint(0, n))
    selected = [start]
    min_distances = 1.0 - (Xn @ Xn[start])
    min_distances[start] = -np.inf

    while len(selected) < k:
        nxt = int(np.argmax(min_distances))
        selected.append(nxt)
        new_distances = 1.0 - (Xn @ Xn[nxt])
        min_distances = np.minimum(min_distances, new_distances)
        min_distances[nxt] = -np.inf

    return selected
