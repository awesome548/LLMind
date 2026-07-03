# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Thin hub** (2026-07-03 doc consolidation): environment, flow diagrams, and the
> one trap worth a warning box. Everything else — API endpoints, env-var table,
> module map, CLI reference, Supabase schema, error patterns — has ONE home:
> [`BACKEND.md`](BACKEND.md).

## Environment

Python 3.13+, managed with **uv**. No pip/venv directly.

```bash
uv sync                          # install dependencies
uv run fastapi dev backend/main.py            # dev server → :8000
uv run python test_projection.py              # offline test suite
uv run python database_pipeline.py --help     # all pipeline/projection CLIs
```

All settings are loaded via `config.py` (`pydantic-settings`). Every `.env` key maps
1-to-1 to a field on the `Settings` class. **`.env` overrides the defaults — trust
`.env` + artifact metadata, never `config.py` defaults, when reasoning about the
live stack** (see the warning box below).

Backend runs **without `--reload`** — kill and restart the `:8000` process after
every backend code change.

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

Command reference, Supabase schema, and the two embedding axes: **BACKEND.md →
"Data pipeline & corpus CLIs"**.

## ⚠ Critical: two storage paths, two dimensionalities

The Supabase schema's `VECTOR(384)` column reflects the *old* `BAAI/bge-small-en-v1.5`
default and only describes the (dormant) Supabase local column. The **live local
pipeline is 768-d**: `.env` sets `VLLM_EMBED_MODEL=text-embedding-nomic-embed-text-v1.5`
(768-d), which built `data/local_index.npz` (209×768), the frozen projection
(`surface.json` `input_dims: 768`), and the register map (768×768). `/locate`
hard-fails on a dim mismatch, so switching the embed model requires rebuilding the
index AND rerunning `project` + `project-align`.
