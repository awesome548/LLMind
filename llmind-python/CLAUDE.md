# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python 3.13+, managed with **uv**. No pip/venv directly.

```bash
uv sync                          # install dependencies
uv run python database_pipeline.py --help
uv run python generate_taxonomy.py --help
uv run python scrape_projects.py --help
```

All settings are loaded via `config.py` (`pydantic-settings`). Every `.env` key maps 1-to-1 to a field on the `Settings` class. Override any default by setting the env var.

## Pipeline flow

**Live local path (what the app actually runs on — `VECTOR_STORE=local`):**

```
scrape_projects.py  →  data/scraped.json (210 records)
        ↓
build_local_index.py  →  data/local_index.npz (209 × 768-d, nomic-embed) + .meta.json
        ↓
database_pipeline.py project        →  data/projection/{model.joblib, surface.json}  (frozen PCA→UMAP)
database_pipeline.py project-align  →  data/projection/register_map.npz  (short→long register correction
                                        + short-register support baseline)
```

**Supabase path (largely dormant; kept for the cloud/pgvector variant):**

```
scrape_projects.py  →  raw_projects (Supabase JSONB)
        ↓
database_pipeline.py ingest   →  media_doc (flat) + media_emb_* (vectors)
        ↓
database_pipeline.py cluster / farthest   →  cluster_groups_*.json / selected_projects.json
        ↓
generate_taxonomy.py   →  Taxonomy JSON
```

### Typical end-to-end

```bash
uv run python database_pipeline.py init --scraped-file data/scraped.json
uv run python database_pipeline.py ingest --content-mode all          # fills all 3 emb tables
uv run python database_pipeline.py farthest --k 30
uv run python generate_taxonomy.py openai --model-name gpt-4o --source selected -i data/selected_projects.json
```

## Supabase schema

Three tiers, all in `migrations/`:

| Table | Purpose |
|---|---|
| `raw_projects` | Scraped records as `{id, metadata JSONB}` |
| `media_doc` | Cleaned flat: `id, name, description, detail, image` |
| `media_emb_description/details/hybrid` | `media_doc_id FK, context, embedding_cloud VECTOR(1536), embedding_local VECTOR(384)` |

Run `migrations/migrate_media_emb_columns.sql` (idempotent) to set up or upgrade. The `media_doc_tables.sql` is a clean-slate version.

## Key design decisions

### Two embedding axes

- **Content mode** (`--content-mode`): which text field → which table
  `description → media_emb_description`, `details → media_emb_details`, `hybrid → media_emb_hybrid`
- **Backend mode** (`--embed-mode`): which API → which column
  `openai → embedding_cloud (1536d)`, `vllm → embedding_local (384d)`

Both axes are independent: any content mode × any backend mode combination is valid.

> **⚠ Dimensionality — do not confuse the two storage paths.** The `VECTOR(384)`
> column reflects the *old* `BAAI/bge-small-en-v1.5` default and only describes the
> (dormant) Supabase local column. The **live local pipeline is 768-d**:
> `.env` sets `VLLM_EMBED_MODEL=text-embedding-nomic-embed-text-v1.5` (768-d), which
> built `data/local_index.npz` (209×768), the frozen projection
> (`surface.json` `input_dims: 768`), and the register map (768×768). `/locate`
> hard-fails on a dim mismatch, so switching the embed model requires rebuilding the
> index AND rerunning `project` + `project-align`.

## Module map

| Path | Responsibility |
|---|---|
| `config.py` | Single `Settings` instance (`settings`); all other modules import from here |
| `utils/models.py` | All Pydantic models: `Taxonomy`, `Aspect`, `Option`, `ProjectRecord`, `EmbedRecord` |
| `utils/modes.py` | `BackendMode` (openai/vllm), `ContentMode` (description/details/hybrid/all) |
| `utils/clients.py` | `build_openai_client`, `build_vllm_client` |
| `utils/supabase.py` | **All** Supabase operations: client, upsert, fetch, `build_artefacts` |
| `pipeline/constants.py` | Settings-derived constants + `EMB_TABLE_MAP`, `EMB_COLUMN_MAP` |
| `pipeline/data_ops.py` | Pure transforms: `clean_records`, `build_embed_records`, `build_context` |
| `pipeline/ml.py` | UMAP, KMeans, farthest-point, numpy helpers (no I/O) |
| `pipeline/projection.py` | Frozen PCA→UMAP reducer, persistence, 48×48 grid/density, trustworthiness, evidence-anchored `place_by_neighbors` |
| `pipeline/register_alignment.py` | Short→long register correction: CV fit (translation vs ridge), apply, persistence, short-register support baseline |
| `pipeline/log_stats.py` | Pure aggregation of `generate_log.jsonl` (drift/clip per prompt × seeding × alignment × placement variant) |
| `pipeline/viz.py` | `plot_clusters` (matplotlib) |
| `database_pipeline.py` | Typer CLI entry point — imports from pipeline/, delegates all logic |
| `generate_taxonomy.py` | Typer CLI for taxonomy generation; `OpenAIChat` dataclass |
| `scrape_projects.py` | Scraper → `raw_projects` |
| `build_local_index.py` | Builds the live 768-d local index (`data/local_index.npz` + meta) from `scraped.json` |

### Projection CLI commands (design-space subsystem — see BACKEND.md for details)

```bash
uv run python database_pipeline.py project             # fit + freeze the 2D surface
uv run python database_pipeline.py project-align       # fit register map + support baseline; prints 3-way transform/kNN comparison
uv run python database_pipeline.py project-calibrate   # short-text placement displacement report
uv run python database_pipeline.py project-diagnose    # reproducible validity report (--offline supported)
uv run python database_pipeline.py project-log-stats   # drift/clip stats per generation variant
```
