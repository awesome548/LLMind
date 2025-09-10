# cluster_from_chroma.py
# Fetch items (ids, metadatas, embeddings) from a Chroma collection,
# reduce to 2D, cluster them, and output JSON.
# Optionally, shows a scatter plot when executed directly.
#
# Usage:
#   pip install chromadb scikit-learn numpy matplotlib umap-learn
#   python cluster_from_chroma.py --db-path ./chroma_db --collection mab_projects --algo auto --clusters 8 --plot
#   python cluster_from_chroma.py --db-path ./chroma_db --collection mab_projects --save-json output.json
#
# Without --plot or --save-json, it prints JSON to stdout.

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import chromadb


# ---------- Helpers ----------
def safe_imports(algo: str = "auto"):
    """
    Return (reducer_name, ReducerClassOrCallable) based on availability & preference.
    - "auto": prefer UMAP -> TSNE -> PCA
    """
    algo = (algo or "auto").lower()

    def try_umap():
        try:
            import umap
            return "umap", umap.UMAP
        except Exception:
            return None

    def try_tsne():
        try:
            from sklearn.manifold import TSNE
            return "tsne", TSNE
        except Exception:
            return None

    def try_pca():
        try:
            from sklearn.decomposition import PCA
            return "pca", PCA
        except Exception:
            return None

    if algo == "auto":
        for chooser in (try_umap, try_tsne, try_pca):
            got = chooser()
            if got:
                return got
        raise RuntimeError("No reducer available. Install `umap-learn` or `scikit-learn`.")

    if algo == "umap":
        got = try_umap()
        if got:
            return got
        # graceful fallback
        got = try_tsne() or try_pca()
        if got:
            sys.stderr.write("umap not available, falling back.\n")
            return got
        raise RuntimeError("UMAP requested but not available. Install `umap-learn`.")

    if algo == "tsne":
        got = try_tsne()
        if got:
            return got
        got = try_pca()
        if got:
            sys.stderr.write("t-SNE not available, falling back to PCA.\n")
            return got
        raise RuntimeError("t-SNE requested but not available. Install `scikit-learn`.")

    if algo == "pca":
        got = try_pca()
        if got:
            return got
        raise RuntimeError("PCA requested but not available. Install `scikit-learn`.")

    raise ValueError(f"Unknown algo: {algo}")


def maybe_unit_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def reduce_2d(
    X: np.ndarray,
    reducer_name: str,
    Reducer,
    *,
    metric: str = "euclidean",
    perplexity: int = 30,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    pre_pca: int | None = None
):
    X_in = X

    # Optional pre-PCA to speed up t-SNE/UMAP on large dims
    if pre_pca and X.shape[1] > pre_pca:
        try:
            from sklearn.decomposition import PCA
            X_in = PCA(n_components=pre_pca, random_state=random_state).fit_transform(X)
        except Exception:
            # if PCA unavailable, just skip pre-PCA
            X_in = X

    if reducer_name == "umap":
        # UMAP supports cosine directly
        reducer = Reducer(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        return reducer.fit_transform(X_in)

    if reducer_name == "tsne":
        # sklearn TSNE has limited metric support; for cosine we unit-normalize first
        X_tsne = maybe_unit_normalize(X_in) if metric == "cosine" else X_in
        reducer = Reducer(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            # If your sklearn supports it, you can add: metric=("cosine" if metric=="cosine" else "euclidean")
        )
        return reducer.fit_transform(X_tsne)

    if reducer_name == "pca":
        reducer = Reducer(n_components=2, random_state=random_state)
        return reducer.fit_transform(X_in)

    raise ValueError(f"Unsupported reducer: {reducer_name}")


def kmeans_labels(X2d: np.ndarray, k: int) -> List[int]:
    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        return km.fit_predict(X2d).tolist()
    except Exception:
        return [0] * len(X2d)


def np_default(o):
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.integer, np.int32, np.int64)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def normalize01(arr: np.ndarray) -> List[float]:
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if math.isclose(hi, lo):
        return [0.5] * len(arr)
    return [float((v - lo) / (hi - lo)) for v in arr]


def fetch_all_from_chroma(client: chromadb.PersistentClient, collection_name: str, batch_size: int = 500, where: Dict[str, Any] | None = None):
    coll = client.get_collection(name=collection_name)
    ids_all, metas_all, embs_all = [], [], []
    offset = 0
    while True:
        res = coll.get(limit=batch_size, offset=offset, where=where, include=["metadatas", "embeddings"])
        got = len(res.get("ids", []))
        if got == 0:
            break
        for i, _id in enumerate(res["ids"]):
            emb = res["embeddings"][i]
            meta = res["metadatas"][i]
            if emb is None:
                continue
            ids_all.append(_id)
            metas_all.append(meta or {})
            embs_all.append(emb)
        offset += got
    return ids_all, metas_all, embs_all


def plot_clusters(out: List[Dict[str, Any]]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.stderr.write("matplotlib not installed, skipping plot.\n")
        return

    x = [o["x"] for o in out]
    y = [o["y"] for o in out]
    c = [o["cluster"] for o in out]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y, c=c, cmap="tab10", alpha=0.7, s=50)
    plt.colorbar(scatter, label="Cluster")
    plt.title("Chroma Embeddings Clusters")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# ---------- Main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", default=os.getenv("CHROMA_DB_PATH", "./chroma_db"))
    p.add_argument("--collection", default=os.getenv("COLLECTION_NAME", "mab_projects"))

    # Reducer + hyperparams
    p.add_argument("--algo", choices=["auto", "umap", "tsne", "pca"], default="auto")
    p.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    p.add_argument("--perplexity", type=float, default=None, help="t-SNE perplexity (auto if omitted)")
    p.add_argument("--neighbors", type=int, default=15, help="UMAP n_neighbors")
    p.add_argument("--min-dist", type=float, default=0.1, dest="min_dist", help="UMAP min_dist")
    p.add_argument("--pre-pca", type=int, default=64, help="If >0, run PCA to this many dims before UMAP/t-SNE")

    # Clustering / IO
    p.add_argument("--clusters", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--plot", action="store_true", help="Show a test plot instead of JSON output")
    p.add_argument("--save-json", type=str, help="Save JSON output to specified file path")
    args = p.parse_args()

    client = chromadb.PersistentClient(path=args.db_path)
    ids, metas, embs = fetch_all_from_chroma(client, collection_name=args.collection, batch_size=args.batch_size)

    if not ids:
        print("[]")
        return

    X = np.array(embs, dtype=float)

    reducer_name, Reducer = safe_imports(algo=args.algo)

    # Reasonable auto-perplexity for t-SNE: ~N/100 bounded
    if args.perplexity is None:
        # t-SNE requires 5 < perplexity < N; clamp safely
        auto_perp = max(5, min(50, int(len(ids) / 100) or 5))
        perplexity = min(auto_perp, max(5, len(ids) - 1))
    else:
        perplexity = max(5, min(int(args.perplexity), max(5, len(ids) - 1)))

    X2d = reduce_2d(
        X,
        reducer_name,
        Reducer,
        metric=args.metric,
        perplexity=perplexity,
        n_neighbors=args.neighbors,
        min_dist=args.min_dist,
        random_state=args.random_state,
        pre_pca=(args.pre_pca if args.pre_pca and args.pre_pca > 0 else None),
    )

    k = max(1, min(args.clusters, len(ids)))
    clusters = kmeans_labels(X2d, k)

    x_n = normalize01(X2d[:, 0])
    y_n = normalize01(X2d[:, 1])

    out = []
    for i, _id in enumerate(ids):
        meta = metas[i] or {}
        out.append({
            "id": _id,
            "name": meta.get("Name"),
            "description": meta.get("Description"),
            "x": x_n[i],
            "y": y_n[i],
            "cluster": int(clusters[i]),
        })

    if args.plot:
        plot_clusters(out)
    elif args.save_json:
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, default=np_default, indent=2)
        print(f"JSON output saved to: {args.save_json}")
    else:
        json.dump(out, sys.stdout, default=np_default)


if __name__ == "__main__":
    main()