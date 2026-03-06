"""
CLI pipeline for media dataset: analyze, ingest, cluster, and select.

Commands:
  init      Scrape projects and upsert into raw_projects
  analyze   EDA on the Details field with an optional word-count histogram
  ingest    Clean raw records → upsert to media_doc → embed into embedding table(s)
  cluster   Reduce embeddings to 2D via UMAP, KMeans cluster, output JSON or plot
  farthest  Select k maximally diverse items (greedy cosine farthest-point)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import typer

from utils.iter import chunked
from utils.json import save_json
from utils.modes import BackendMode, ContentMode
from utils.models import EmbedRecord, ProjectRecord
from utils.clients import build_openai_client, build_vllm_client

from pipeline.constants import (
    ANALYSIS_DIR,
    DATA_DIR,
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_FETCH_BATCH_SIZE,
    DEFAULT_HIST_BINS,
    EMBED_MODEL,
    EMB_COLUMN_MAP,
    EMB_TABLE_MAP,
    MAX_EXAMPLES,
    MIN_EXAMPLES_PATH,
    ANALYSIS_PATH,
    OUTPUT_PATH,
    PLOTS_DIR,
    SUPABASE_MEDIA_DOC_TABLE,
    SUPABASE_RAW_TABLE,
    VLLM_BASE_URL,
    VLLM_EMBED_MODEL,
)
from pipeline.data_ops import build_embed_records, clean_records, extract_content, summary_stats
from pipeline.ml import (
    kmeans_cluster,
    normalize_to_unit_interval,
    numpy_json_default,
    select_farthest,
    umap_reduce,
)
from pipeline.viz import plot_clusters
from utils.supabase import (
    fetch_embeddings,
    fetch_projects_by_ids,
    fetch_raw_records,
    get_supabase_client,
    upsert_media_doc,
    upsert_raw_to_supabase,
    upsert_rows,
)

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

os.environ.setdefault("KMP_WARNINGS", "0")
os.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Analyze, ingest, cluster, and select media dataset records.",
)


@app.command("init")
def init_scrape(
    scraped_file: Optional[Path] = typer.Option(
        None, "--scraped-file", "-s",
        help="Path to a pre-scraped JSON file. If omitted, scraping runs automatically.",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", min=0,
        help="Max projects to scrape (0 = all). Ignored when --scraped-file is provided.",
    ),
) -> None:
    """Populate Supabase from a scraped JSON file, or run the scraper automatically."""
    from scrape_projects import run_scrape

    if scraped_file and scraped_file.exists():
        typer.secho(f"Loading scraped data from {scraped_file}", fg=typer.colors.BLUE)
        with scraped_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        results = [ProjectRecord(**item) for item in data]
    else:
        typer.secho("No scraped file provided — running scraper now.", fg=typer.colors.BLUE)
        results = run_scrape(limit=limit)
        if not results:
            typer.secho("Scraper returned no results.", fg=typer.colors.RED)
            raise typer.Exit(code=2)

    upsert_raw_to_supabase(results)
    typer.secho("Done.", fg=typer.colors.GREEN)


@app.command("analyze")
def analyze(
    table: str = typer.Option(SUPABASE_RAW_TABLE, "--table", "-t", help="Supabase raw projects table"),
    batch_size: int = typer.Option(DEFAULT_FETCH_BATCH_SIZE, help="Supabase fetch batch size"),
    save_plot: bool = True,
    bins: int = typer.Option(DEFAULT_HIST_BINS, help="Histogram bins"),
) -> None:
    """Analyze raw dataset (Details field) from Supabase and print summary metrics."""
    typer.secho(f"Fetching raw records from '{table}'...", fg=typer.colors.BLUE)
    records = fetch_raw_records(table, batch_size=batch_size)
    if not records:
        typer.secho("No records found in table.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    details = [extract_content(record) for record in records]
    word_counts = [len(text.split()) if text else 0 for text in details]
    char_counts = [len(text) for text in details]

    non_empty_words = [c for c in word_counts if c > 0]
    non_empty_chars = [c for c in char_counts if c > 0]

    result: Dict[str, Any] = {
        "total_items": len(details),
        "details_non_empty": len(non_empty_words),
        "details_empty": len(details) - len(non_empty_words),
        "details_ge_min_words": sum(1 for c in word_counts if c >= 50),
        "word_count_stats": summary_stats(non_empty_words),
        "char_count_stats": summary_stats(non_empty_chars),
    }

    if non_empty_words:
        min_wc = min(non_empty_words)
        examples = [
            {
                "id": record.get("id") or record.get("url"),
                "Name": record.get("Name"),
                "word_count": count,
                "Detail": detail,
            }
            for record, detail, count in zip(records, details, word_counts)
            if count == min_wc
        ][:MAX_EXAMPLES]

        result["min_word_count"] = min_wc
        result["min_word_count_example_count"] = len(examples)
        MIN_EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
        MIN_EXAMPLES_PATH.write_text(
            json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        typer.echo(f"Min word count: {min_wc}. Examples saved to {MIN_EXAMPLES_PATH}")
        for ex in examples:
            typer.echo(f"  id={ex.get('id')} name={ex.get('Name')} wc={ex.get('word_count')}")
            typer.echo(f"  {ex.get('Detail') or ''}")

    save_json(ANALYSIS_PATH, result)
    typer.echo(json.dumps(result, indent=2))

    if save_plot:
        if plt is None:
            typer.echo("matplotlib not available; skipping plot.")
        else:
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist([c for c in word_counts if c > 0], bins=bins, color="#4e79a7", edgecolor="white")
            ax.set_title("Distribution of 'Detail' word counts")
            ax.set_xlabel("Word count")
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            out_path = PLOTS_DIR / "details_word_count_hist.png"
            fig.savefig(out_path)
            plt.close(fig)
            typer.echo(f"Saved histogram to {out_path}")


@app.command("ingest")
def ingest(
    table: str = typer.Option(SUPABASE_RAW_TABLE, "--table", "-t", help="Supabase raw projects table"),
    output_path: Path = typer.Option(OUTPUT_PATH, "--output", "-o", help="Path to write cleaned JSON"),
    save_cleaned: bool = typer.Option(False, "--save-cleaned/--no-save-cleaned", help="Persist cleaned JSON to disk"),
    batch_size: int = typer.Option(DEFAULT_EMBED_BATCH_SIZE, help="Batch size for embedding + upsert"),
    embed_mode: BackendMode = typer.Option(BackendMode.openai, "--embed-mode", "-m", help="Embedding backend: openai or vllm"),
    content_mode: ContentMode = typer.Option(ContentMode.details, "--content-mode", "-c", help="Text field(s) to embed: description, details, hybrid, or all"),
    vllm_base_url: str = typer.Option(VLLM_BASE_URL, "--vllm-base-url", help="Base URL for local vLLM server"),
    vllm_model: str = typer.Option(VLLM_EMBED_MODEL, "--vllm-model", help="Model name served by vLLM"),
) -> None:
    """Fetch raw records, clean, upsert to media_doc, then embed into embedding table(s).

    Embedding backends:
      openai  Use OpenAI's hosted embedding API (requires OPENAI_API_KEY).
      vllm    Use a local vLLM server (OpenAI-compatible). Start with:
                vllm serve <model> --task embed

    Content modes:
      description  Embed Descriptions only  → media_emb_description
      details      Embed Details only        → media_emb_details
      hybrid       Embed both concatenated   → media_emb_hybrid
      all          Run all three modes
    """
    typer.secho(f"Fetching raw records from '{table}'...", fg=typer.colors.BLUE)
    raw = fetch_raw_records(table)
    cleaned = clean_records(raw)
    typer.secho(f"Cleaned {len(cleaned)} records.", fg=typer.colors.BLUE)

    if save_cleaned:
        save_json(output_path, cleaned)
        typer.secho(f"Saved cleaned data to {output_path}", fg=typer.colors.BLUE)

    # ── Step 1: upsert cleaned records into central media_doc ─────────────────
    upsert_media_doc(cleaned)

    # ── Step 2: embedding client ───────────────────────────────────────────────
    if embed_mode == BackendMode.vllm:
        embed_client = build_vllm_client(vllm_base_url)
        active_model = vllm_model
        typer.secho(f"Using vLLM at {vllm_base_url} with model '{active_model}'", fg=typer.colors.BLUE)
    else:
        embed_client = build_openai_client()
        active_model = EMBED_MODEL

    supabase = get_supabase_client()

    modes_to_run: List[ContentMode] = (
        [ContentMode.description, ContentMode.details, ContentMode.hybrid]
        if content_mode == ContentMode.all
        else [content_mode]
    )

    # ── Step 3: embed + upsert per content mode ────────────────────────────────
    for mode in modes_to_run:
        emb_table = EMB_TABLE_MAP[mode]
        embed_records = build_embed_records(cleaned, content_mode=mode)
        skipped = len(cleaned) - len(embed_records)
        if skipped:
            typer.secho(f"[{mode}] Skipped {skipped} record(s) with empty context.", fg=typer.colors.YELLOW)
        if not embed_records:
            typer.secho(f"[{mode}] No valid documents to embed.", fg=typer.colors.YELLOW)
            continue

        total = len(embed_records)
        upserted = 0
        typer.echo(f"[{mode}] Embedding {total} doc(s) with {active_model} -> '{emb_table}'")

        emb_col = EMB_COLUMN_MAP[embed_mode]
        for batch in chunked(embed_records, batch_size):
            response = embed_client.embeddings.create(model=active_model, input=[r.context for r in batch])
            embeddings = [datum.embedding for datum in response.data]
            rows = [
                {"media_doc_id": rec.media_doc_id, "context": rec.context, emb_col: emb}
                for rec, emb in zip(batch, embeddings)
            ]
            try:
                upsert_rows(emb_table, rows, on_conflict="media_doc_id", client=supabase)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(f"[{mode}] Supabase upsert failed at offset {upserted + 1}") from exc
            upserted += len(rows)
            typer.echo(f"  {upserted}/{total}")

        typer.secho(f"[{mode}] Done. Upserted {upserted} items into '{emb_table}'.", fg=typer.colors.GREEN)


@app.command("cluster")
def cluster(
    content_mode: ContentMode = typer.Option(ContentMode.details, "--content-mode", "-c", help="Embedding table: description, details, or hybrid"),
    embed_mode: BackendMode = typer.Option(BackendMode.openai, "--embed-mode", "-m", help="Embedding column: openai (cloud) or vllm (local)"),
    table: Optional[str] = typer.Option(None, help="Override embedding table name"),
    neighbors: int = typer.Option(15, help="UMAP n_neighbors"),
    min_dist: float = typer.Option(0.1, help="UMAP min_dist"),
    pre_pca: int = typer.Option(64, help="Pre-PCA dims before UMAP (0 to disable)"),
    clusters: int = typer.Option(8, help="Number of KMeans clusters"),
    batch_size: int = typer.Option(DEFAULT_FETCH_BATCH_SIZE, help="Supabase fetch batch size"),
    random_state: int = typer.Option(42, help="Random seed"),
    plot: bool = typer.Option(False, help="Save scatter plot instead of emitting JSON"),
) -> None:
    """Reduce embeddings to 2D via UMAP, KMeans cluster, and output JSON or plot."""
    emb_table = table or EMB_TABLE_MAP[content_mode]
    typer.secho(
        f"Fetching embeddings from '{emb_table}' ({EMB_COLUMN_MAP[embed_mode]})...",
        fg=typer.colors.BLUE,
    )
    ids, _, embs = fetch_embeddings(emb_table, embed_mode=embed_mode, batch_size=batch_size)
    typer.secho(f"Fetched {len(ids)} items.", fg=typer.colors.GREEN)

    if not ids:
        typer.echo("[]")
        raise typer.Exit()

    X = np.array(embs, dtype=float)
    X2d = umap_reduce(
        X,
        n_neighbors=neighbors,
        min_dist=min_dist,
        random_state=random_state,
        pre_pca=(pre_pca if pre_pca > 0 else None),
    )

    n = len(ids)
    labels = kmeans_cluster(X2d, max(1, min(clusters, n)))
    x_norm = normalize_to_unit_interval(X2d[:, 0])
    y_norm = normalize_to_unit_interval(X2d[:, 1])

    points: List[Dict[str, Any]] = [
        {"id": id_, "x": x_norm[i], "y": y_norm[i], "cluster": labels[i]}
        for i, id_ in enumerate(ids)
    ]

    if plot:
        plot_path = PLOTS_DIR / f"clusters_{emb_table}_{clusters}.png"
        plot_clusters(points, save_path=plot_path)
        typer.secho(f"Saved cluster plot to {plot_path}", fg=typer.colors.BLUE)
    else:
        json.dump(points, sys.stdout, default=numpy_json_default)

    cluster_groups: Dict[int, List[str]] = {}
    for p in points:
        cluster_groups.setdefault(p["cluster"], []).append(p["id"])
    groups_path = ANALYSIS_DIR / f"cluster_groups_{emb_table}_{clusters}.json"
    save_json(groups_path, cluster_groups)
    typer.secho(f"Saved cluster groups to {groups_path}", fg=typer.colors.BLUE)


@app.command("fetch-cluster")
def fetch_cluster(
    group: int = typer.Argument(..., help="Cluster group number to fetch"),
    groups_file: Path = typer.Option(
        ANALYSIS_DIR / "cluster_groups.json", "--groups-file", "-g",
        help="Path to cluster_groups.json",
    ),
    table: str = typer.Option(SUPABASE_MEDIA_DOC_TABLE, "--table", "-t", help="Table to fetch project data from"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write results to JSON file"),
) -> None:
    """Fetch all projects belonging to a specific cluster group from Supabase."""
    if not groups_file.exists():
        typer.secho(f"Cluster groups file not found: {groups_file}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    cluster_groups: Dict[str, List[str]] = json.loads(groups_file.read_text(encoding="utf-8"))
    ids = cluster_groups.get(str(group)) or cluster_groups.get(group)  # type: ignore[call-overload]
    if not ids:
        typer.secho(f"No projects found for cluster group {group}.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=0)

    typer.secho(f"Fetching {len(ids)} project(s) for cluster {group} from '{table}'...", fg=typer.colors.BLUE)
    projects = fetch_projects_by_ids(ids, table=table)
    typer.secho(f"Fetched {len(projects)} project(s).", fg=typer.colors.GREEN)

    if output:
        save_json(output, projects)
        typer.secho(f"Saved to {output}", fg=typer.colors.BLUE)
    else:
        json.dump(projects, sys.stdout, indent=2, ensure_ascii=False)


@app.command()
def farthest(
    content_mode: ContentMode = typer.Option(ContentMode.details, "--content-mode", "-c", help="Embedding table: description, details, or hybrid"),
    embed_mode: BackendMode = typer.Option(BackendMode.openai, "--embed-mode", "-m", help="Embedding column: openai (cloud) or vllm (local)"),
    table: Optional[str] = typer.Option(None, help="Override embedding table name"),
    k: int = typer.Option(20, help="Number of items to select"),
    seed: int = typer.Option(42, help="Random seed"),
    batch_size: int = typer.Option(DEFAULT_FETCH_BATCH_SIZE, help="Supabase fetch batch size"),
    output: Path = typer.Option(
        DATA_DIR / "selected_projects.json", "--output", "-o", help="Write selected ids to file"
    ),
) -> None:
    """Select k maximally diverse items by greedy cosine farthest-point and write ids to JSON."""
    emb_table = table or EMB_TABLE_MAP[content_mode]
    ids, _metas, embs = fetch_embeddings(emb_table, embed_mode=embed_mode, batch_size=batch_size)

    if not ids:
        typer.echo("No embeddings found.")
        raise typer.Exit()

    indices = select_farthest(embs, k=k, seed=seed)
    selected_ids = [ids[i] for i in indices]
    save_json(output, selected_ids)
    typer.echo(f"Selected {len(selected_ids)} ids -> {output}")


if __name__ == "__main__":
    app()
