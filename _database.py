"""
CLI to analyze, clean, and embed dataset into Chroma.

Commands:
  - analyze: Basic EDA on Detail field + histogram plot
  - clean:   Produce cleaned JSON with ids extracted from url
  - embed:   Upsert cleaned docs into Chroma with OpenAI embeddings
"""

from __future__ import annotations
import json, os, statistics
from pathlib import Path
from uuid import uuid4
from typing import List, Dict, Any
from supabase import create_client, Client
from openai import OpenAI

from dotenv import load_dotenv
import typer

# Optional plotting
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - plotting optional at runtime
    plt = None

load_dotenv()

COLLECTION_NAME = str(os.getenv("COLLECTION_NAME"))
OPENAI_MODEL = str(os.getenv("OPENAI_MODEL"))
CHROMA_DB_PATH = Path(str(os.getenv("CHROMA_DB_PATH")))
PLOTS_DIR = Path(str(os.getenv("PLOTS_DIR")))
DATA_DIR = Path(str(os.getenv("DATA_DIR")))
ANALYSIS_DIR = Path(str(os.getenv("ANALYSIS_DIR")))
SUPABASE_URL = str(os.getenv("SUPABASE_URL"))
SUPABASE_KEY = str(os.getenv("SUPABASE_KEY"))
SUPABASE_TABLE = str(os.getenv("SUPABASE_TABLE") or "media_docs")
EMBED_MODEL = str(os.getenv("OPENAI_MODEL") or "text-embedding-3-small")

INPUT_PATH = DATA_DIR / "media_architecture.json"
OUTPUT_PATH = DATA_DIR / "cleaned_media_architecture.json"
ANALYSIS_PATH = ANALYSIS_DIR / "analysis_summary.json"
MIN_EXAMPLES_PATH = ANALYSIS_DIR / "min_detail_examples.json"

app = typer.Typer(add_completion=False, help="Analyze, clean, and embed dataset")
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def to_doc_text(item: Dict[str, Any]) -> str:
    return (item.get("Descriptions") or "").strip()


def extract_project_id(url: str) -> str:
    """Extract project ID from URL after 'project/'"""
    if "project/" in url:
        return url.split("project/")[-1]
    # Fallback to UUID if URL doesn't match expected format
    return str(uuid4())

def embed_batch(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a batch of texts."""
    # OpenAI SDK v1 supports batching directly
    resp = oa.embeddings.create(model=EMBED_MODEL, input=texts)
    # Preserve order
    return [d.embedding for d in resp.data]

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]


def load_raw(path: Path = INPUT_PATH) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be an array of objects.")
    return [dict(x) for x in data]


def clean_items(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for item in data:
        desc_raw = item.get("Details")
        if not isinstance(desc_raw, str):
            continue
        desc = desc_raw.strip()
        if not desc:
            continue
        word_count = len(desc.split())
        if word_count < 10 or word_count > 300:
            continue
        item_copy = dict(item)
        project_id = extract_project_id(item_copy.get("url", ""))
        item_copy.pop("url", None)
        item_copy.pop("html_main", None)
        cleaned.append({"id": project_id, **item_copy})
    return cleaned


def ensure_cleaned_exists() -> List[Dict[str, Any]]:
    if OUTPUT_PATH.exists():
        data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    # Fallback: generate from raw
    raw = load_raw()
    cleaned = clean_items(raw)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"Saved cleaned data to {OUTPUT_PATH}")
    return cleaned


@app.command()
def analyze(
    file: Path = typer.Option(INPUT_PATH, "--file", "-f", help="Path to input JSON file"),
    save_plot: bool = True,
    bins: int = typer.Option(50, help="Histogram bins"),
):
    """
    Analyze dataset Details: counts and histogram plot.
    Pass a custom input JSON via --file/-f.
    """
    data = load_raw(file)
    # Word & char counts for Detail
    details = [to_doc_text(it) for it in data]
    word_counts = [len(s.split()) if s else 0 for s in details]
    char_counts = [len(s) for s in details]

    total = len(data)
    non_empty = sum(1 for s in details if s)
    empty = total - non_empty
    gt_10 = sum(1 for wc in word_counts if wc > 10)

    # Stats on non-empty only
    nonzero_words = [wc for wc in word_counts if wc > 0]
    nonzero_chars = [cc for cc in char_counts if cc > 0]

    def stats(nums: List[int]) -> Dict[str, Any]:
        if not nums:
            return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
        return {
            "min": int(min(nums)),
            "max": int(max(nums)),
            "mean": float(statistics.fmean(nums)),
            "median": float(statistics.median(nums)),
        }

    summary = {
        "total_items": total,
        "details_non_empty": non_empty,
        "details_empty": empty,
        "details_gt_10_words": gt_10,
        "word_count_stats_non_empty": stats(nonzero_words),
        "char_count_stats_non_empty": stats(nonzero_chars),
    }

    # Extract and print examples with minimum non-empty word count
    if nonzero_words:
        min_wc = min(nonzero_words)
        examples = []
        for idx, (item, s, wc) in enumerate(zip(data, details, word_counts)):
            if wc == min_wc and wc > 0:
                examples.append({
                    "id": item.get("id") or item.get("url"),
                    "Name": item.get("Name"),
                    "word_count": wc,
                    "Detail": s,
                })
            if len(examples) >= 5:
                break
        summary["min_non_empty_word_count"] = min_wc
        summary["min_non_empty_examples_count"] = len(examples)
        # Save examples to file for inspection
        MIN_EXAMPLES_PATH.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
        typer.echo(f"Min non-empty word count: {min_wc}. Saved examples to {MIN_EXAMPLES_PATH}")
        # Also print to console for a quick look
        for ex in examples:
            typer.echo("--- Min word count example ---")
            typer.echo(f"id={ex.get('id')} name={ex.get('Name')} wc={ex.get('word_count')}")
            typer.echo(ex.get("Detail") or "")

    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    typer.echo("Dataset analysis summary:")
    typer.echo(json.dumps(summary, indent=2))

    if save_plot:
        if plt is None:
            typer.echo("matplotlib not available; skipping plot.")
        else:
            PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            # Use non-empty for more informative plot
            ax.hist([wc for wc in word_counts if wc > 0], bins=bins, color="#4e79a7", edgecolor="white")
            ax.set_title("Distribution of 'Detail' word counts")
            ax.set_xlabel("Word count")
            ax.set_ylabel("Frequency")
            fig.tight_layout()
            out_path = PLOTS_DIR / "details_word_count_hist.png"
            fig.savefig(out_path)
            plt.close(fig)
            typer.echo(f"Saved histogram to {out_path}")


@app.command()
def clean(
    input: Path = typer.Option(INPUT_PATH, "--input", "-i", help="Path to raw input JSON"),
    output: Path = typer.Option(OUTPUT_PATH, "--output", "-o", help="Path to write cleaned JSON"),
):
    """
    Clean raw data and save to output JSON.
    Remove url/html_main and add id.
    Remove entries that
    - does not have "Details"
    - have more than 300 words "Details"
    """
    data = load_raw(input)
    cleaned = clean_items(data)
    typer.echo(f"length of cleaned data: {len(cleaned)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"Saved cleaned data to {output}")


@app.command()
def embed(
    file: Path = typer.Option(OUTPUT_PATH, "--file", "-f", help="Path to cleaned JSON file"),
    batch_size: int = typer.Option(180, help="Batch size for embedding + upsert"),
):
    """
    Embed cleaned data and save into Chroma(after clanning)
    """
    if file.exists():
        cleaned = json.loads(file.read_text(encoding="utf-8"))
    else:
        # Fallback for default only; if custom path missing, error
        if file == OUTPUT_PATH:
            cleaned = ensure_cleaned_exists()
        else:
            raise typer.BadParameter(f"File not found: {file}")

    ids_all = [it.get("id") for it in cleaned]
    docs_all = [to_doc_text(it) for it in cleaned]
    metas_all = [{
        "ID": it.get("id"),
        "Name": it.get("Name"),
        "Details": it.get("Details"),
        "Descriptions": it.get("Descriptions"),
        "Image": it.get("image_href"),
    } for it in cleaned]

    # Filter out empty docs
    filtered = [(i, d, m) for i, d, m in zip(ids_all, docs_all, metas_all) if isinstance(d, str) and d.strip()]
    if not filtered:
        typer.echo("No non-empty Details to embed. Nothing to upsert.")
        raise typer.Exit(code=0)

    ids, documents, metadatas = map(list, zip(*filtered))
    skipped = len(ids_all) - len(ids)
    if skipped:
        typer.echo(f"Skipped {skipped} item(s) with empty or invalid Description.")

    total = len(documents)
    typer.echo(f"Embedding {total} document(s) with {EMBED_MODEL} and upserting into Supabase table '{SUPABASE_TABLE}'")

    upserted = 0
    for chunk_ids, chunk_docs, chunk_meta in zip(
        chunked(ids, batch_size),
        chunked(documents, batch_size),
        chunked(metadatas, batch_size),
    ):
        # 1) get embeddings
        embeddings = embed_batch(chunk_docs)

        # 2) shape rows for Supabase
        rows = []
        for _id, _doc, _meta, _emb in zip(chunk_ids, chunk_docs, chunk_meta, embeddings):
            rows.append({
                "id": _id,
                "content": _doc,
                "metadata": _meta,       # jsonb
                "embedding": _emb,       # vector
            })

        # 3) upsert
        # on_conflict='id' ensures idempotent reruns
        sb.table(SUPABASE_TABLE).upsert(rows, on_conflict="id").execute()

        upserted += len(rows)
        typer.echo(f"Upserted {upserted}/{total}...")

    typer.echo(f"Done. Upserted {upserted} items into '{SUPABASE_TABLE}'.")


if __name__ == "__main__":
    app()
