# Simple CLI for clustering and farthest selection on Chroma embeddings.
# Examples:
#   python clustering.py cluster --db-path ./chroma_db --collection mab_projects --plot
#   python clustering.py select-farthest --db-path ./chroma_db --collection mab_projects --k 20 --json-file data/cleaned_media_arch.json

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv
from pathlib import Path

import numpy as np
import chromadb
import typer

load_dotenv()

# Quiet noisy runtime messages during clustering:
# - OMP: Info #276 from Intel OpenMP
# - UMAP n_jobs override warning when a seed is set
os.environ.setdefault("KMP_WARNINGS", "0")      # suppress Intel OpenMP info/warnings
os.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")
CHROMA_DB_PATH = Path(str(os.getenv("CHROMA_DB_PATH")))
COLLECTION_NAME = str(os.getenv("COLLECTION_NAME"))

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
        # Suppress specific UMAP warning about forcing n_jobs=1 when random_state is set.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"n_jobs value .* overridden to 1 by setting random_state.*",
                category=UserWarning,
                module=r"umap(\.|$)"
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


# ---------- Diverse subset selection ----------
def select_farthest(
    embs: np.ndarray | List[List[float]],
    k: int = 20,
    seed: int = 42,
) -> List[int]:
    """Greedy farthest-point subset (cosine distance)."""
    X = np.asarray(embs, dtype=float)
    n = X.shape[0]
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))

    Xn = maybe_unit_normalize(X)
    rng = np.random.RandomState(seed)
    start = int(rng.randint(0, n))
    sel = [start]
    d_min = 1.0 - (Xn @ Xn[start])
    d_min[start] = -np.inf
    while len(sel) < k:
        nxt = int(np.argmax(d_min))
        sel.append(nxt)
        d_new = 1.0 - (Xn @ Xn[nxt])
        d_min = np.minimum(d_min, d_new)
        d_min[nxt] = -np.inf
    return sel


# ---------- Typer CLI ----------
app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("cluster")
def cmd_cluster(
    algo: str = typer.Option("auto", help="Dimensionality reducer: auto|umap|tsne|pca"),
    metric: str = typer.Option("euclidean", help="Distance metric for reducer: euclidean|cosine"),
    perplexity: float | None = typer.Option(None, help="t-SNE perplexity; auto if omitted"),
    neighbors: int = typer.Option(15, help="UMAP n_neighbors"),
    min_dist: float = typer.Option(0.1, help="UMAP min_dist"),
    pre_pca: int = typer.Option(64, help="If >0, run PCA to this many dims before UMAP/t-SNE"),
    clusters: int = typer.Option(8, help="Number of KMeans clusters"),
    batch_size: int = typer.Option(500, help="Batch size for Chroma fetch"),
    random_state: int = typer.Option(42, help="Random seed"),
    plot: bool = typer.Option(False, help="Show a scatter plot instead of JSON"),
    output_json: str | None = typer.Option(None,"--output","-o",help="Save JSON output to file"),
):
    """Reduce to 2D, KMeans cluster, and output JSON or plot."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ids, metas, embs = fetch_all_from_chroma(client, collection_name=COLLECTION_NAME, batch_size=batch_size)
    typer.secho(f"Fetched {len(ids)} items from Chroma collection '{COLLECTION_NAME}'", fg=typer.colors.GREEN)

    if not ids:
        typer.echo("[]")
        raise typer.Exit()

    X = np.array(embs, dtype=float)
    reducer_name, Reducer = safe_imports(algo=algo)

    # Reasonable auto-perplexity for t-SNE: ~N/100 bounded
    if perplexity is None:
        auto_perp = max(5, min(50, int(len(ids) / 100) or 5))
        perplexity_val = min(auto_perp, max(5, len(ids) - 1))
    else:
        perplexity_val = max(5, min(int(perplexity), max(5, len(ids) - 1)))

    X2d = reduce_2d(
        X,
        reducer_name,
        Reducer,
        metric=metric,
        perplexity=perplexity_val,
        n_neighbors=neighbors,
        min_dist=min_dist,
        random_state=random_state,
        pre_pca=(pre_pca if pre_pca and pre_pca > 0 else None),
    )

    k = max(1, min(clusters, len(ids)))
    clabels = kmeans_labels(X2d, k)

    x_n = normalize01(X2d[:, 0])
    y_n = normalize01(X2d[:, 1])

    out: List[Dict[str, Any]] = []
    for i, _id in enumerate(ids):
        meta = metas[i] or {}
        out.append({
            "id": _id,
            "x": x_n[i],
            "y": y_n[i],
            "cluster": int(clabels[i]),
        })

    if plot:
        plot_clusters(out)
    elif output_json:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, default=np_default, indent=2)
        typer.secho(f"JSON output saved to: {output_json}", fg=typer.colors.BLUE)
    else:
        json.dump(out, sys.stdout, default=np_default)


@app.command("farthest")
def cmd_select_farthest(
    batch_size: int = typer.Option(500, help="Batch size for Chroma fetch"),
    k: int = typer.Option(20, help="Number of items to select"),
    seed: int = typer.Option(42, help="Random seed"),
    json_file: str = typer.Option("data/20_seletected_projects.json","--json","-j", help="Write selected ids to this JSON file"),
):
    """Select k farthest (cosine) and write only ids to JSON file."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ids, metas, embs = fetch_all_from_chroma(client, collection_name=COLLECTION_NAME, batch_size=batch_size)

    if not ids:
        typer.echo("[]")
        raise typer.Exit()

    idx = select_farthest(embs, k=k, seed=seed)
    out_ids = [ids[i] for i in idx]
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(out_ids, f, indent=2)
    typer.echo(f"Wrote {len(out_ids)} ids to: {json_file}")


if __name__ == "__main__":
    app()
