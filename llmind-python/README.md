# LLMind — Taxonomy Generator

Generates a structured **design-space taxonomy** from a corpus of project artefacts stored in Supabase, using an LLM with enforced structured output.

The taxonomy models the design space as a set of **Aspects** (key dimensions) and **Options** (concrete alternatives per dimension), ready for downstream embedding, clustering, and visualisation.

## How it works

```
scrape_projects.py
        │
        ▼
  raw_projects (Supabase)     ← scraped records stored as {id, metadata}
        │
        ▼
  database_pipeline.py
    analyze                   ← EDA on raw records from Supabase
    ingest                    ← clean → upsert media_doc → embed → upsert media_emb_*
    cluster                   ← UMAP + KMeans on embeddings
    farthest                  ← greedy cosine farthest-point selection
        │
        ▼
  generate_taxonomy.py        ← format prompt → call LLM → Taxonomy JSON
```

### Data model

```
Supabase tables:
  raw_projects          id (text), metadata (jsonb)         ← scraped records
  media_doc             id, name, description, detail, image ← cleaned flat data
  media_emb_description media_doc_id (FK), context, embedding_cloud, embedding_local
  media_emb_details     media_doc_id (FK), context, embedding_cloud, embedding_local
  media_emb_hybrid      media_doc_id (FK), context, embedding_cloud, embedding_local

Taxonomy (data/models.py):
  Taxonomy
  └── aspects: list[Aspect]
          ├── name: str
          ├── desc: str
          └── options: list[Option]
                  ├── name: str
                  └── desc: str
```

## Backends

### Embedding backend (`--embed-mode`)

| Value | Column written | Behaviour |
|---|---|---|
| `openai` (default) | `embedding_cloud` | Calls OpenAI's hosted API. Requires `OPENAI_API_KEY`. |
| `vllm` | `embedding_local` | Points the OpenAI-compatible client at a local vLLM server. |

### Content mode (`--content-mode`)

| Value | Context built from | Target table |
|---|---|---|
| `details` (default) | Details field only | `media_emb_details` |
| `description` | Descriptions field only | `media_emb_description` |
| `hybrid` | Descriptions + Details | `media_emb_hybrid` |
| `all` | All three (ingest only) | all three tables |

## Setup

```bash
uv sync
cp .env.example .env   # fill in OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY
```

Run the migration in the Supabase SQL editor:

```bash
# Apply schema (creates media_doc + media_emb_* tables)
# paste contents of migrations/migrate_media_emb_columns.sql
```

`.env` keys (all resolved via `config.py` / `pydantic-settings`):

| Key | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key (not required when using `--embed-mode vllm`) |
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_KEY` | — | Supabase service role or anon key |
| `SUPABASE_MEDIA_DOC_TABLE` | `media_doc` | Central flat data table |
| `SUPABASE_EMB_DESCRIPTION_TABLE` | `media_emb_description` | Description embeddings |
| `SUPABASE_EMB_DETAILS_TABLE` | `media_emb_details` | Detail embeddings |
| `SUPABASE_EMB_HYBRID_TABLE` | `media_emb_hybrid` | Hybrid embeddings |
| `SUPABASE_RAW_TABLE` | `raw_projects` | Table for raw scraped records |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | OpenAI embedding model (1536 dims) |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | Default local vLLM server URL |
| `VLLM_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Default vLLM embedding model (384 dims) |
| `BASE_URL` | `https://awards.mediaarchitecture.org` | Scraper base URL |
| `DATA_DIR` | `data/` | Local data directory |
| `ANALYSIS_DIR` | `analysis/` | Analysis output directory |
| `PLOTS_DIR` | `plots/` | Plot output directory |
| `TAXONOMY_DIR` | `taxonomy` | Taxonomy output directory |

## Scraper (`scrape_projects.py`)

Discovers project URLs from the listing page, scrapes each project, and upserts records into `raw_projects` on Supabase.

```bash
uv run scrape_projects.py scrape --limit 20
uv run scrape_projects.py scrape              # scrape all
```

| Option | Default | Description |
|---|---|---|
| `--limit` / `-n` | `None` (all) | Max projects to scrape (`0` = all) |
| `--out` / `-o` | `media_architecture_projects.json` | Local JSON filename under `DATA_DIR` |
| `--delay` / `-d` | `0.8` | Polite delay (seconds) between requests |
| `--retries` / `-r` | `4` | Max retry attempts per request |
| `--backoff` | `1.0` | Initial exponential backoff (seconds) |
| `--max-backoff` | `10.0` | Maximum backoff (seconds) |

## Database pipeline (`database_pipeline.py`)

```
raw_projects (Supabase)
        │
        ▼
  analyze      ← EDA: word counts, char counts, histogram
        │
        ▼
  ingest       ← clean → upsert media_doc → embed → upsert media_emb_*
        │
        ▼
  cluster      ← fetch embeddings → UMAP (cosine) → KMeans labels
        │
        ▼
  farthest     ← greedy cosine farthest-point selection → ids JSON
        │
        ▼
generate_taxonomy.py --source selected -i data/selected_projects.json
```

### `init`

```bash
uv run database_pipeline.py init --scraped-file data/scraped.json
uv run database_pipeline.py init --limit 50   # runs scraper automatically
```

| Option | Default | Description |
|---|---|---|
| `--scraped-file` / `-s` | `None` | Pre-scraped JSON file; if omitted, scraper runs |
| `--limit` / `-n` | `None` | Max projects to scrape |

### `analyze`

```bash
uv run database_pipeline.py analyze
uv run database_pipeline.py analyze --table raw_projects
```

| Option | Default | Description |
|---|---|---|
| `--table` / `-t` | `raw_projects` | Supabase raw projects table |
| `--batch-size` | `1000` | Fetch batch size |
| `--bins` | `50` | Histogram bins |

### `ingest`

```bash
# Cloud embeddings (OpenAI), details field only (default)
uv run database_pipeline.py ingest

# Embed all content modes with OpenAI
uv run database_pipeline.py ingest --content-mode all

# Local embeddings with vLLM (start server first)
vllm serve BAAI/bge-small-en-v1.5 --task embed
uv run database_pipeline.py ingest --embed-mode vllm --content-mode all

# Custom vLLM model / URL
uv run database_pipeline.py ingest --embed-mode vllm \
  --vllm-base-url http://localhost:8000/v1 \
  --vllm-model intfloat/multilingual-e5-large \
  --content-mode hybrid
```

| Option | Default | Description |
|---|---|---|
| `--table` / `-t` | `raw_projects` | Source raw projects table |
| `--embed-mode` / `-m` | `openai` | Embedding backend: `openai` or `vllm` |
| `--content-mode` / `-c` | `details` | Text field: `description`, `details`, `hybrid`, `all` |
| `--save-cleaned / --no-save-cleaned` | `False` | Persist cleaned JSON to disk |
| `--output` / `-o` | `data/cleaned_media_architecture.json` | Path for cleaned JSON |
| `--batch-size` | `180` | Records per embedding + upsert batch |
| `--vllm-base-url` | `$VLLM_BASE_URL` | vLLM server URL |
| `--vllm-model` | `$VLLM_EMBED_MODEL` | Model served by vLLM |

### `cluster`

```bash
uv run database_pipeline.py cluster --clusters 12
uv run database_pipeline.py cluster --content-mode hybrid --embed-mode vllm --plot

# Print clustered projects to stdout
uv run database_pipeline.py fetch-cluster 2
uv run database_pipeline.py fetch-cluster 2 --groups-file analysis/cluster_groups_media_emb_details_8.json
```

| Option | Default | Description |
|---|---|---|
| `--content-mode` / `-c` | `details` | Which embedding table to cluster |
| `--embed-mode` / `-m` | `openai` | Which embedding column to load |
| `--table` | — | Override embedding table name |
| `--neighbors` | `15` | UMAP `n_neighbors` |
| `--min-dist` | `0.1` | UMAP `min_dist` |
| `--pre-pca` | `64` | Pre-PCA dims before UMAP (`0` to disable) |
| `--clusters` | `8` | Number of KMeans clusters |
| `--batch-size` | `1000` | Supabase fetch batch size |
| `--random-state` | `42` | Random seed |
| `--plot` | `False` | Save scatter plot instead of emitting JSON |

### `farthest`

```bash
uv run database_pipeline.py farthest --k 30
uv run database_pipeline.py farthest --content-mode hybrid --embed-mode vllm --k 20
```

| Option | Default | Description |
|---|---|---|
| `--content-mode` / `-c` | `details` | Which embedding table to use |
| `--embed-mode` / `-m` | `openai` | Which embedding column to load |
| `--table` | — | Override embedding table name |
| `--k` | `20` | Number of items to select |
| `--seed` | `42` | Random seed |
| `--batch-size` | `1000` | Supabase fetch batch size |
| `--output` / `-o` | `data/selected_projects.json` | Path to write selected ids JSON |

### Typical end-to-end run

```bash
# 1. Scrape projects and store raw data in Supabase
uv run scrape_projects.py scrape

# 2. Analyse raw data
uv run database_pipeline.py analyze

# 3. Clean → upsert media_doc → embed all content modes (cloud)
uv run database_pipeline.py ingest --content-mode all

# 4. Select 30 diverse artefacts (using detail embeddings)
uv run database_pipeline.py farthest --k 30

# 5. Generate taxonomy from selected artefacts
uv run generate_taxonomy.py openai \
  --model-name gpt-4o \
  --source selected \
  -i data/selected_projects.json
```

## Taxonomy generator (`generate_taxonomy.py`)

```bash
# OpenAI (default)
uv run generate_taxonomy.py openai --model-name gpt-4o --source all_supabase

# OpenAI with pre-filtered artefact IDs
uv run generate_taxonomy.py openai \
  --model-name gpt-4o \
  -i data/selected_ids.json \
  --source selected

# vLLM — start server first, then run:
vllm serve meta-llama/Llama-3.1-8B-Instruct
uv run generate_taxonomy.py openai \
  --mode vllm \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://localhost:8000/v1

# Dev mode (inspect prompt without calling the LLM)
uv run generate_taxonomy.py openai --dev --source all_supabase
```

| Option | Default | Description |
|---|---|---|
| `--model-name` | `gpt-4o` | LLM model name |
| `--mode` / `-m` | `openai` | Generation backend: `openai` or `vllm` |
| `--base-url` | `$VLLM_BASE_URL` | vLLM server URL (when `--mode vllm`) |
| `--source` | `selected` | `selected` (ids file) or `all_supabase` |
| `-i` | `None` | Path to ids JSON (required when `--source selected`) |
| `--content-mode` | `details` | Text field for artefact context |
| `--reasoning` | `medium` | Reasoning effort: `low`, `medium`, `high` (OpenAI only) |
| `--out-file` | `../results/taxonomy/schema` | Base path; mode, model, timestamp appended |
| `--dev` | `False` | Print prompt and write `debug_artefacts.txt` |

## Project layout

```
llmind-python/
├── database_pipeline.py      # CLI entry point (thin — delegates to pipeline/)
├── generate_taxonomy.py      # LLM orchestration and taxonomy generation
├── scrape_projects.py        # Scrape projects → raw_projects in Supabase
├── config.py                 # Centralised settings via pydantic-settings
│
├── pipeline/                 # Modular pipeline logic
│   ├── constants.py          # All settings-derived constants and table/column maps
│   ├── models.py             # EmbedRecord dataclass
│   ├── data.py               # Data helpers: extract, clean, build_embed_records
│   ├── ml.py                 # Math/ML: UMAP, KMeans, farthest-point, normalisation
│   ├── storage.py            # Supabase I/O: fetch and upsert helpers
│   ├── clients.py            # OpenAI / vLLM client factories
│   └── viz.py                # Cluster scatter plot (matplotlib)
│
├── data/
│   ├── models.py             # Pydantic schema: Taxonomy, Aspect, Option, ProjectRecord
│   └── prompts.py            # System prompt and generation templates
│
├── utils/
│   ├── modes.py              # BackendMode + ContentMode enums
│   ├── iter.py               # Generic iteration utilities (chunked)
│   ├── json.py               # JSON load/save utilities
│   └── supabase.py           # Supabase client and artefact fetch helpers
│
└── migrations/
    ├── supabase_raw_table.sql        # raw_projects table
    ├── media_doc_tables.sql          # media_doc + embedding tables (clean-slate)
    └── migrate_media_emb_columns.sql # idempotent migration (handles upgrades)
```

## Self-refine loop

The iterative self-review loop (`IDEA_REFLECTION_PROMPT`) is implemented but commented out in `run_generate()`. To enable it, uncomment the loop block and pass `--num <rounds>`. Each round sends the current taxonomy back to the model as context and asks it to consolidate and refine the aspects and options.
