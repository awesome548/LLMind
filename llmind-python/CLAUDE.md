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
| `pipeline/viz.py` | `plot_clusters` (matplotlib) |
| `database_pipeline.py` | Typer CLI entry point — imports from pipeline/, delegates all logic |
| `generate_taxonomy.py` | Typer CLI for taxonomy generation; `OpenAIChat` dataclass |
| `scrape_projects.py` | Scraper → `raw_projects` |
