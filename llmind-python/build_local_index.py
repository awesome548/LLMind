#!/usr/bin/env python3
"""Build a fully-local vector index for related-project search.

Scrapes the Media Architecture corpus (or loads a saved JSON), embeds each
project with the local OpenAI-compatible server (LM Studio / vLLM), and writes
``data/local_index.npz`` + a metadata sidecar. No Supabase, no OpenAI.

    uv run python build_local_index.py                       # scrape all + embed
    uv run python build_local_index.py --limit 20            # quick subset
    uv run python build_local_index.py --scraped-file data/scraped.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from config import settings
from pipeline.data_ops import build_context, extract_project_id
from scrape_projects import run_scrape
from utils.clients import build_vllm_client
from utils.json import read_json_array, save_json
from utils.local_store import save_index
from utils.modes import ContentMode

app = typer.Typer(add_completion=False)


def _embed(client, model: str, texts: list[str], batch_size: int) -> list[list[float]]:
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
        typer.echo(f"  embedded {min(start + batch_size, total)}/{total}")
    return vectors


@app.command()
def build(
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max projects to scrape (omit = all)"),
    content_mode: ContentMode = typer.Option(ContentMode.hybrid, "--content-mode", "-c", help="Which text to embed"),
    scraped_file: Optional[Path] = typer.Option(None, "--scraped-file", help="Load this JSON instead of scraping"),
    save_scrape: Optional[Path] = typer.Option(None, "--save-scrape", help="Also write the scraped records here"),
    base_url: str = typer.Option(settings.vllm_base_url, "--base-url", help="Local embedding server (OpenAI-compatible)"),
    model: str = typer.Option(settings.vllm_embed_model, "--model", help="Embedding model name"),
    out: Path = typer.Option(settings.local_index_path, "--out", "-o", help="Index output path (.npz)"),
    batch_size: int = typer.Option(32, "--batch-size"),
) -> None:
    # 1. Obtain records
    if scraped_file is not None:
        records = read_json_array(scraped_file)
        typer.echo(f"Loaded {len(records)} records from {scraped_file}")
    else:
        records = [r.model_dump() for r in run_scrape(limit=limit)]
        if save_scrape is not None:
            save_json(save_scrape, records)
            typer.echo(f"Saved scraped records -> {save_scrape}")

    # 2. Build row-aligned ids / texts / metadata
    ids: list[str] = []
    texts: list[str] = []
    metadata: dict[str, dict] = {}
    seen: set[str] = set()
    for rec in records:
        pid = extract_project_id(str(rec.get("url", ""))) or str(rec.get("id", ""))
        context = build_context(rec, content_mode).strip()
        if not pid or not context or pid in seen:
            continue
        seen.add(pid)
        ids.append(pid)
        texts.append(context)
        metadata[pid] = {
            "Name": rec.get("Name") or "",
            "Descriptions": rec.get("Descriptions") or "",
            "Details": rec.get("Details") or "",
            "Image": rec.get("image_href") or None,
        }
    if not ids:
        typer.secho("No embeddable projects found.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.echo(f"{len(ids)} projects to embed ({content_mode.value} context)")

    # 3. Embed locally
    client = build_vllm_client(base_url)
    typer.echo(f"Embedding with '{model}' @ {base_url}")
    vectors = _embed(client, model, texts, batch_size)

    # 4. Persist
    save_index(out, ids, vectors, metadata)
    dims = len(vectors[0]) if vectors else 0
    typer.secho(f"OK - saved {len(ids)} vectors ({dims} dims) -> {out}", fg=typer.colors.GREEN)
    typer.echo("Set VECTOR_STORE=local in .env to use it for search.")


if __name__ == "__main__":
    app()
