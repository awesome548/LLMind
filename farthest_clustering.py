# Simple CLI for clustering and farthest selection on Supabase (pgvector) embeddings.
# Examples:
#   python clustering.py cluster --table media_docs --plot
#   python clustering.py select-farthest --table media_docs --k 20 --json-file data/cleaned_media_arch.json

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
import typer

# --- Supabase client ---
from supabase import create_client, Client

load_dotenv()

# Quiet noisy runtime messages during clustering:
os.environ.setdefault("KMP_WARNINGS", "0")      # suppress Intel OpenMP info/warnings
os.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")

# Supabase env
SUPABASE_URL = os.getenv("SUPABASE_URL") or ""
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or ""
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE") or "media_docs"

if not SUPABASE_URL or not SUPABASE_KEY:
    sys.stderr.write("WARNING: SUPABASE_URL or SUPABASE_KEY not set. Commands will fail to fetch data.\n")

sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
            X_in = X

    if reducer_name == "umap":
        reducer = Reducer(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"n_jobs value .* overridden to 1 by setting random_state.*",
                category=UserWarning,
                module=r"umap(\.|$)"
            )
            return reducer.fit_transform(X_in)

    if reducer_name == "tsne":
        X_tsne = maybe_unit_normalize(X_in) if metric == "cosine" else X_in
        reducer = Reducer(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
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
    import numpy as _np
    if isinstance(o, (_np.floating, _np.float32, _np.float64)):
        return float(o)
    if isinstance(o, (_np.integer, _np.int32, _np.int64)):
        return int(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def normalize01(arr: np.ndarray) -> List[float]:
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if math.isclose(hi, lo):
        return [0.5] * len(arr)
    return [float((v - lo) / (hi - lo)) for v in arr]


def fetch_all_from_supabase(
    client: Client,
    table: str,
    *,
    batch_size: int = 1000,
    where: Dict[str, Any] | None = None
) -> Tuple[List[str], List[Dict[str, Any]], List[List[float]]]:
    """
    Fetch all rows from Supabase table with columns: id (text), metadata (jsonb), embedding (vector).
    Pagination uses PostgREST .range(offset, end).
    """
    ids_all: List[str] = []
    metas_all: List[Dict[str, Any]] = []
    embs_all: List[List[float]] = []

    offset = 0
    while True:
        q = client.table(table).select("id, metadata, embedding")

        # Minimal JSONB filter support (exact equals) if you pass where={"metadata->>Name":"foo"} etc.
        # Keep simple—most users won't need client-side filtering here.
        if where:
            for key, value in where.items():
                # naive: treat key as a column name or expression
                q = q.eq(key, value)

        resp = q.range(offset, offset + batch_size - 1).execute()
        rows = resp.data or []
        if not rows:
            break

        for r in rows:
            emb = r.get("embedding")
            if emb is None:
                continue
            ids_all.append(r.get("id"))
            metas_all.append(r.get("metadata") or {})
            embs_all.append(emb)

        offset += len(rows)

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
    plt.title("Supabase Embeddings Clusters")
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
    table: str = typer.Option(SUPABASE_TABLE, help="Supabase table containing id, metadata, embedding"),
    algo: str = typer.Option("auto", help="Dimensionality reducer: auto|umap|tsne|pca"),
    metric: str = typer.Option("euclidean", help="Distance metric for reducer: euclidean|cosine"),
    perplexity: float | None = typer.Option(None, help="t-SNE perplexity; auto if omitted"),
    neighbors: int = typer.Option(15, help="UMAP n_neighbors"),
    min_dist: float = typer.Option(0.1, help="UMAP min_dist"),
    pre_pca: int = typer.Option(64, help="If >0, run PCA to this many dims before UMAP/t-SNE"),
    clusters: int = typer.Option(8, help="Number of KMeans clusters"),
    batch_size: int = typer.Option(1000, help="Batch size for Supabase fetch"),
    random_state: int = typer.Option(42, help="Random seed"),
    plot: bool = typer.Option(False, help="Show a scatter plot instead of JSON"),
    output_json: str | None = typer.Option(None,"--output","-o",help="Save JSON output to file"),
):
    """Reduce to 2D, KMeans cluster, and output JSON or plot."""
    ids, metas, embs = fetch_all_from_supabase(sb, table=table, batch_size=batch_size)
    typer.secho(f"Fetched {len(ids)} items from Supabase table '{table}'", fg=typer.colors.GREEN)

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
        # You can include metadata if you want—kept minimal here for plotting/JSON size.
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
    table: str = typer.Option(SUPABASE_TABLE, help="Supabase table containing id, metadata, embedding"),
    batch_size: int = typer.Option(1000, help="Batch size for Supabase fetch"),
    k: int = typer.Option(20, help="Number of items to select"),
    seed: int = typer.Option(42, help="Random seed"),
    json_file: str = typer.Option("data/selected_projects.json","--json","-j", help="Write selected ids to this JSON file"),
):
    """Select k farthest (cosine) and write only ids to JSON file."""
    ids, metas, embs = fetch_all_from_supabase(sb, table=table, batch_size=batch_size)

    if not ids:
        typer.echo("[]")
        raise typer.Exit()

    idx = select_farthest(embs, k=k, seed=seed)
    out_ids = [ids[i] for i in idx]
    Path(json_file).parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(out_ids, f, indent=2)
    typer.echo(f"Wrote {len(out_ids)} ids to: {json_file}")


if __name__ == "__main__":
    app()