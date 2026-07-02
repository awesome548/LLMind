# LLMind — A Designer's Learning Guide

> **Audience:** Designers with basic knowledge of AI concepts and some exposure to Next.js.
> This guide walks you through the entire LLMind codebase from the simplest ideas to the most advanced patterns, building your understanding one layer at a time.
> **Scope:** This guide covers only the two active codebases — `llmind-python/` and `llmind-web/`.

> ⚠️ **Currency note (2026-06):** this guide predates the design-space exploration
> features and describes the foundations only. The tool has since gained: the
> **Design Space** view (frozen UMAP surface, gap preview → generate-at, relevance
> lens), the **Perspectives** view (semantic axes), **design candidates**
> (compose / compare / reject / export), provenance, fidelity metrics, session
> save/load, and an evaluation log. For the current state read
> [`DESIGN-SPACE-ITERATION-PLAN.md`](DESIGN-SPACE-ITERATION-PLAN.md) (concepts &
> history), [`llmind-web/FRONTEND.md`](../llmind-web/FRONTEND.md) and
> [`llmind-python/BACKEND.md`](../llmind-python/BACKEND.md) (architecture &
> endpoints). The foundations below (pipeline, embeddings, taxonomy generation,
> router/service pattern) are still accurate.

> ⚠️ **Staleness corrections (2026-07-03 code-verification sweep — the body below is
> archived unmodified; trust these notes over the text where they conflict):**
> 1. **The API-connection chapter (§9.2) teaches the reversed architecture.** The
>    frontend NO LONGER calls the backend through the Next.js rewrites proxy with
>    `baseURL '/'`. It calls the backend **directly** (`baseURL =
>    NEXT_PUBLIC_API_BASE_URL ?? http://localhost:8000`, CORS enabled) because the
>    dev proxy silently drops long local-LLM responses (50 s+). See the root
>    `CLAUDE.md` "API connection" section.
> 2. **Embedding model/dimensions:** any mention of `BAAI/bge-small-en-v1.5` /
>    384-d as the local embedding describes an early configuration. The live stack
>    is `text-embedding-nomic-embed-text-v1.5` (**768-d**), served by LM Studio at
>    `localhost:1234`, and every live artifact (index, projection, register map)
>    is 768-d.
> 3. The system's canonical representation is now the **living design-space
>    schema** with the map/tree/cross-tab/axes as lenses (PROJECT-REPORT §2.11,
>    §3 addendum) — a re-centering this guide predates entirely.

---

## 🚀 Quick Launch

Two terminals — **open them fresh** so newly installed tools (`uv`, `bun`) are on your PATH. If a command says *"not recognized,"* the terminal was opened before the tool was installed → just open a new one.

### Prerequisites (one-time)
- **uv** (Python runner): `python -m pip install --upgrade uv`
- **Bun** (frontend): `winget install Oven-sh.Bun` — *or* skip Bun and use Node (`npx`, see below)
- For the **fully-local / offline** setup (no OpenAI), have **LM Studio** running with a chat model and an embedding model loaded, and follow [Section 11](#11-connecting-a-local-llm-replacing-the-openai-api). Then build the search index once:
  ```powershell
  cd llmind-python
  uv run python build_local_index.py      # scrape + embed corpus locally → data/local_index.npz
  ```

### Terminal 1 — Backend (FastAPI)
```powershell
cd llmind-python
uv run fastapi dev backend/main.py        # → http://localhost:8000  (API docs at /docs)
```

### Terminal 2 — Frontend (Next.js)
```powershell
cd llmind-web
bun install                               # first time only
bun dev                                   # → http://localhost:3000/mindmap
```

**If `bun` errors or isn't recognized** (and you have Node installed), use Node directly — this bypasses the Bun-only npm script:
```powershell
cd llmind-web
npm install                               # first time only
npx next dev                              # → http://localhost:3000/mindmap
```

> **Common gotchas**
> - *"Another next dev server is already running"* → port 3000 is taken by a previous run. Stop it (the message prints the `taskkill /PID … /F` command) or close that terminal.
> - Backend `502` errors → check the uvicorn logs / `e.__cause__`; the 502 masks the real error (see [§8.6](#86-error-handling--the-502-pattern)).
> - In **local mode**, keep LM Studio open; the first request to each model is slow while it loads, then fast. The Generate dialogs default to **vLLM**.

---

## Table of Contents

1. [What Is LLMind?](#1-what-is-llmind)
2. [The Big Picture — How Everything Connects](#2-the-big-picture--how-everything-connects)
3. [Key Vocabulary](#3-key-vocabulary)
4. [Repository Map](#4-repository-map)
5. **Layer 1 — Foundations (Start Here)**
   - [5.1 Configuration & Environment Variables](#51-configuration--environment-variables)
   - [5.2 Data Models — The Shapes of Data](#52-data-models--the-shapes-of-data)
   - [5.3 Enums — Mode Selectors](#53-enums--mode-selectors)
   - [5.4 Shared Utilities](#54-shared-utilities)
6. **Layer 2 — The Data Pipeline (Python CLI)**
   - [6.1 Scraping Projects](#61-scraping-projects)
   - [6.2 Pipeline Constants — Derived Settings](#62-pipeline-constants--derived-settings)
   - [6.3 Analyzing Raw Data](#63-analyzing-raw-data)
   - [6.4 Cleaning & Ingesting Data](#64-cleaning--ingesting-data)
   - [6.5 Embeddings — Turning Text Into Numbers](#65-embeddings--turning-text-into-numbers)
   - [6.6 Clustering — Grouping Similar Projects](#66-clustering--grouping-similar-projects)
   - [6.7 Farthest-Point Selection — Picking Diverse Samples](#67-farthest-point-selection--picking-diverse-samples)
   - [6.8 Visualization — Plotting Clusters](#68-visualization--plotting-clusters)
7. **Layer 3 — AI-Powered Taxonomy Generation (Python)**
   - [7.1 What Is a Taxonomy?](#71-what-is-a-taxonomy)
   - [7.2 Prompt Engineering — Talking to the LLM](#72-prompt-engineering--talking-to-the-llm)
   - [7.3 Structured Outputs — Forcing JSON Responses](#73-structured-outputs--forcing-json-responses)
   - [7.4 The Generation Flow](#74-the-generation-flow)
   - [7.5 Building Artefacts for the LLM](#75-building-artefacts-for-the-llm)
8. **Layer 4 — The FastAPI Backend**
   - [8.1 App Entry Point](#81-app-entry-point)
   - [8.2 Router → Service Pattern](#82-router--service-pattern)
   - [8.3 The Search Endpoint](#83-the-search-endpoint)
   - [8.4 The Generate-Nodes Endpoint](#84-the-generate-nodes-endpoint)
   - [8.5 The Taxonomy Generation Endpoint](#85-the-taxonomy-generation-endpoint)
   - [8.6 Error Handling — The 502 Pattern](#86-error-handling--the-502-pattern)
9. **Layer 5 — The Next.js Frontend (`llmind-web`)**
   - [9.1 App Router & Layout](#91-app-router--layout)
   - [9.2 API Proxy — How Frontend Talks to Backend](#92-api-proxy--how-frontend-talks-to-backend)
   - [9.3 React Query — Server State Management](#93-react-query--server-state-management)
   - [9.4 Zustand Store — Client State](#94-zustand-store--client-state)
   - [9.5 Mind Elixir Mind Map Component](#95-mind-elixir-mind-map-component)
   - [9.6 The Project Panel Component](#96-the-project-panel-component)
   - [9.7 The Mindmap Page — Putting It All Together](#97-the-mindmap-page--putting-it-all-together)
   - [9.8 Auto-Generated Types From OpenAPI](#98-auto-generated-types-from-openapi)
10. **Layer 6 — Advanced Topics**
    - [10.1 Supabase Vector Search (pgvector)](#101-supabase-vector-search-pgvector)
    - [10.2 Database Schema & Migrations](#102-database-schema--migrations)
    - [10.3 OpenAI Structured Outputs Constraints](#103-openai-structured-outputs-constraints)
    - [10.4 vLLM & OpenAI-Compatible Servers](#104-vllm--openai-compatible-servers)
    - [10.5 UMAP + KMeans Dimensionality Reduction](#105-umap--kmeans-dimensionality-reduction)
    - [10.6 Immutable Tree Updates in React](#106-immutable-tree-updates-in-react)
11. **[Connecting a Local LLM (Replacing the OpenAI API)](#11-connecting-a-local-llm-replacing-the-openai-api)**
    - [11.1 How LLMind Talks to Models](#111-how-llmind-talks-to-models)
    - [11.2 What `mode = vllm` Actually Switches](#112-what-mode--vllm-actually-switches)
    - [11.3 Case A — A Local LLM on Windows](#113-case-a--a-local-llm-on-windows)
    - [11.4 Case B — A Remote Linux vLLM Server over SSH](#114-case-b--a-remote-linux-vllm-server-over-ssh)
    - [11.5 Embedding Dimensions — The 384 vs 1536 Trap](#115-embedding-dimensions--the-384-vs-1536-trap)
    - [11.6 Verifying & Troubleshooting](#116-verifying--troubleshooting)
12. [Hands-On Exercises](#12-hands-on-exercises)
13. [Further Reading](#13-further-reading)

---

## 1. What Is LLMind?

LLMind is an **LLM-assisted design-space exploration tool**. Think of it as an AI-powered brainstorming partner that:

1. **Scrapes** real-world design projects (media architecture installations) from the web.
2. **Embeds** their descriptions into numerical vectors (arrays of numbers that capture meaning).
3. **Clusters** similar projects together and picks diverse representatives.
4. **Generates a taxonomy** — a structured breakdown of the "design space" into **Aspects** (dimensions like "Display Medium" or "Interaction Mode") and **Options** (concrete alternatives like "LED panels" or "Projection mapping").
5. **Displays** that taxonomy as an interactive **mind map** in the browser, where clicking a node fetches related projects and an AI can suggest new sub-topics.

**As a designer, you can think of it as:** an AI that analyzes a collection of design precedents, extracts the key design dimensions, and then lets you interactively explore and expand those dimensions in a visual mind map — with real project references alongside.

---

## 2. The Big Picture — How Everything Connects

```
┌──────────────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE  (Python CLI)                      │
│                                                                      │
│  scrape_projects.py ──► raw_projects (Supabase)                      │
│         │                                                            │
│  database_pipeline.py                                                │
│    init      ──► scrape + upsert raw records                         │
│    analyze   ──► word-count stats, histograms                        │
│    ingest    ──► clean records → media_doc → embed → media_emb_*     │
│    cluster   ──► UMAP 2D reduction → KMeans labels                   │
│    farthest  ──► pick k most-diverse project IDs                     │
│         │                                                            │
│  generate_taxonomy.py ──► Taxonomy JSON (Aspects + Options)          │
└──────────────────────────────────────────────────────────────────────┘
                              │
                    taxonomy JSON file
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      RUNTIME (live application)                      │
│                                                                      │
│  ┌─────────────────────┐          ┌─────────────────────────┐        │
│  │  FastAPI Backend     │◄────────│  Next.js Frontend        │       │
│  │  (llmind-python/    │  HTTP    │  (llmind-web/)           │       │
│  │   backend/)         │  JSON    │                          │       │
│  │                     │─────────►│  Mind Map + Project Panel │       │
│  │  /api/related-      │          │  • Mind Elixir            │       │
│  │    projects/search  │          │  • React Query            │       │
│  │  /api/related-      │          │  • Zustand Store          │       │
│  │    projects/        │          │  • shadcn/ui              │       │
│  │    generate-nodes   │          │                           │       │
│  │  /api/taxonomy/     │          │                           │       │
│  │    generate         │          │                           │       │
│  └─────────────────────┘          └─────────────────────────┘        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Vocabulary

| Term | What It Means | Why It Matters |
|------|---------------|----------------|
| **Embedding** | A list of numbers (e.g. 1536 floats) that represents the "meaning" of a piece of text. Similar texts produce similar number lists. | This is how the system finds "related projects" — by comparing embeddings. |
| **Vector search** | Finding items whose embeddings are closest to a query embedding. | Powers the "Related Projects" panel. |
| **Taxonomy** | A hierarchical classification: `Aspects → Options`. | The core output of the AI — the mind map's structure. |
| **Design space** | The conceptual space of all possible design outcomes, defined by what you *can* and *cannot* choose. | This is what LLMind helps you *explore*. |
| **Structured output** | Forcing an LLM to reply in a specific JSON schema instead of free-form text. | Ensures the AI's response can be parsed and rendered reliably. |
| **Supabase** | An open-source Firebase alternative that provides a Postgres database, auth, and storage. | The database backend — stores projects, embeddings, and does vector search. |
| **Zustand** | A tiny React state management library (like a simpler Redux). | Manages shared UI state (selected topic, projects, etc.). |
| **React Query** | A library for managing server-state (API calls, caching, loading/error states). | Handles fetching related projects and generating nodes. |
| **FastAPI** | A modern Python web framework for building APIs with automatic documentation. | The backend server that the Next.js frontend talks to. |
| **Mind Elixir** | A JavaScript library for rendering interactive mind maps. | The visual core of the interface — the tree diagram you interact with. |
| **Pydantic** | A Python data validation library that defines data shapes with type hints. | Used everywhere — config, API models, LLM structured outputs. |
| **Typer** | A Python CLI framework built on top of Click. | Powers all the `database_pipeline.py` and `scrape_projects.py` CLI commands. |

---

## 4. Repository Map

```
LLMind/
├── LEARN.md                     ← This guide
│
├── llmind-python/               ← 🐍 ALL Python code
│   ├── pyproject.toml              Project metadata + dependencies (uv/pip)
│   ├── config.py                   Centralised settings (env vars, defaults)
│   ├── scrape_projects.py          Web scraper CLI (Typer)
│   ├── database_pipeline.py        CLI: init, analyze, ingest, cluster, farthest
│   ├── generate_taxonomy.py        LLM taxonomy generation CLI
│   ├── backend/                    FastAPI web server
│   │   ├── main.py                    App entry point (mounts routers)
│   │   ├── related_projects/          /api/related-projects/* routes
│   │   │   ├── router.py                Pydantic request/response models + endpoints
│   │   │   └── service.py               Business logic (search, generate nodes)
│   │   └── taxonomy/                  /api/taxonomy/* routes
│   │       ├── router.py                Endpoint + models for taxonomy generation
│   │       └── service.py               Thin wrapper calling generate_taxonomy
│   ├── pipeline/                   Modular data-processing logic
│   │   ├── constants.py               Derived settings from config.py
│   │   ├── data_ops.py                Clean, extract, build embedding records
│   │   ├── ml.py                      UMAP, KMeans, farthest-point, normalisation
│   │   └── viz.py                     Matplotlib cluster scatter plots
│   ├── utils/                      Shared utilities
│   │   ├── clients.py                 OpenAI/vLLM client factories
│   │   ├── prompts.py                 System + user prompt templates
│   │   ├── models.py                  Pydantic data models (Taxonomy, ProjectRecord, EmbedRecord)
│   │   ├── modes.py                   BackendMode + ContentMode enums
│   │   ├── supabase.py                All Supabase operations (upsert, fetch, artefacts)
│   │   ├── iter.py                    Chunked iteration helper
│   │   ├── json.py                    JSON load/save/extract-from-markdown utilities
│   │   └── _transcribe.py            Audio transcription utility (Whisper/diarization)
│   └── migrations/                 SQL schema files for Supabase
│       ├── supabase_raw_table.sql     raw_projects table
│       ├── media_doc_tables.sql       media_doc + 3 embedding tables + pgvector
│       └── migrate_media_emb_columns.sql  Cloud/local column migration
│
└── llmind-web/                  ← ⚛️ Next.js frontend
    ├── next.config.ts              API proxy rewrite rules + remote image patterns
    ├── package.json                Dependencies (Next 16, React 19, mind-elixir, etc.)
    ├── components.json             shadcn/ui configuration
    ├── src/
    │   ├── app/                    Next.js App Router
    │   │   ├── layout.tsx             Root layout (Geist fonts, Providers wrapper)
    │   │   ├── providers.tsx          React Query QueryClientProvider + Sonner Toaster
    │   │   ├── page.tsx               Home page (navigation cards)
    │   │   ├── globals.css            Tailwind 4 + shadcn design tokens (light/dark)
    │   │   └── mindmap/
    │   │       └── page.tsx           THE MAIN PAGE (mind map + dialogs + panels)
    │   ├── components/
    │   │   ├── mindmap/
    │   │   │   ├── simple-mindmap.tsx    Mind Elixir wrapper component
    │   │   │   └── simple-project-panel.tsx  Project list + detail split view
    │   │   └── ui/                    shadcn/ui primitives (button, card, dialog, etc.)
    │   ├── features/
    │   │   └── mindmap/
    │   │       ├── types.ts             MindmapNode, MindmapSelection, FlattenedMindmapNode, TopicProjectsMap
    │   │       ├── components/
    │   │       │   ├── generate-nodes-dialog.tsx       Two-step dialog for node generation
    │   │       │   └── generate-taxonomy-dialog.tsx    Two-step dialog for full taxonomy generation
    │   │       ├── data/
    │   │       │   └── schema-mindmap-data.ts  Converts taxonomy JSON → MindmapNode tree
    │   │       └── hooks/
    │   │           ├── use-related-projects-query.ts       React Query hook (search)
    │   │           ├── use-generate-nodes-mutation.ts      Mutation hook (generate + flatten)
    │   │           └── use-generate-taxonomy-mutation.ts   Mutation hook (full taxonomy generation)
    │   ├── store/
    │   │   └── mindmap-store.ts       Zustand: topic, projects, taxonomy, context (persisted)
    │   ├── lib/
    │   │   ├── api-client.ts          Axios instance (baseURL: '/')
    │   │   └── utils.ts               cn() helper for Tailwind class merging
    │   └── types/
    │       └── openapi.ts            Auto-generated TypeScript types from backend OpenAPI spec
    └── public/
        ├── schema_selected.json    Initial taxonomy data (farthest-selected projects)
        └── schema_all-rows.json    Full taxonomy data (all projects)
```

---

## 5. Layer 1 — Foundations (Start Here)

### 5.1 Configuration & Environment Variables

**File:** `llmind-python/config.py`

Every application needs settings — API keys, database URLs, model names. LLMind centralizes all of these in one place using **Pydantic Settings**, which automatically reads from environment variables or a `.env` file.

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # silently ignore unknown env vars
    )

    # ── Supabase ─────────────────────────────────────
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_table: str = "media_docs"
    supabase_raw_table: str = "raw_projects"
    supabase_media_doc_table: str = "media_doc"
    supabase_emb_description_table: str = "media_emb_description"
    supabase_emb_details_table: str = "media_emb_details"
    supabase_emb_hybrid_table: str = "media_emb_hybrid"

    # ── OpenAI ───────────────────────────────────────
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    openai_node_model: str = "gpt-5-mini-2025-08-07"

    # ── Related-project retrieval ────────────────────
    supabase_match_function: str = "match_media_docs"
    supabase_match_count: int = 5
    supabase_similarity_threshold: float = 0.0

    # ── vLLM (local models) ──────────────────────────
    vllm_base_url: str = "http://100.73.44.12:8001/v1"
    vllm_model: str = "qwen"
    vllm_embed_model: str = "BAAI/bge-small-en-v1.5"

    # ── Scraper ──────────────────────────────────────
    base_url: str = "https://awards.mediaarchitecture.org"
    default_listing_url: str = "https://awards.mediaarchitecture.org/mab/projects/"

    # ── Paths ────────────────────────────────────────
    data_dir: Path = Path("data")
    analysis_dir: Path = Path("analysis")
    plots_dir: Path = Path("plots")
    taxonomy_dir: Path = Path("taxonomy")

    # ── Pipeline tuning ──────────────────────────────
    min_words: int = 20
    max_words: int = 600
    default_hist_bins: int = 50
    default_embed_batch_size: int = 180
    default_fetch_batch_size: int = 1000
    max_examples: int = 5

settings = Settings()   # singleton used everywhere
```

**Key design decisions:**
- Required secrets (`supabase_url`, `supabase_key`, `openai_api_key`) default to `""` so the app can start without them — validation happens at the call site.
- `case_sensitive=False` means `OPENAI_API_KEY` and `openai_api_key` both work.
- The singleton `settings = Settings()` is imported directly by every module.

**Designer's mental model:** Think of this as a "control panel" where all the knobs and dials for the system live in one place.

---

### 5.2 Data Models — The Shapes of Data

**File:** `llmind-python/utils/models.py`

Before data can flow through the system, we define its **shape** — what fields it has and what types they are. LLMind defines three core Pydantic models:

```python
# ── Taxonomy schema (used as LLM structured output) ──────────

class Option(BaseModel):
    name: str
    desc: str  # concise description for embedding-based retrieval

class Aspect(BaseModel):
    name: str
    desc: str  # concise description for embedding-based retrieval
    options: list[Option] = Field(default_factory=list)

class Taxonomy(BaseModel):
    aspects: list[Aspect] = Field(default_factory=list)

# ── Scraper record ───────────────────────────────────────────

class ProjectRecord(BaseModel):
    url: str
    Name: str
    Descriptions: str
    Details: str
    image_href: str | None = None
    html_main: str | None = None

# ── Pipeline record ──────────────────────────────────────────

class EmbedRecord(BaseModel):
    """A record ready for embedding: the source doc ID and the context text."""
    model_config = ConfigDict(frozen=True)   # immutable once created
    media_doc_id: str
    context: str
```

**How these connect:**
- `ProjectRecord` — the raw shape of a scraped project, used in `scrape_projects.py`.
- `Taxonomy` with `Aspect` and `Option` — the LLM's structured output schema and the mind map's data structure.
- `EmbedRecord` — an intermediate record linking a document ID to its text, ready to be sent to an embedding model. `frozen=True` makes it immutable (hashable, safe to deduplicate).

**Designer's mental model:** These are like *templates* or *blueprints*. A `ProjectRecord` is the blueprint for one scraped project. A `Taxonomy` is the blueprint for the AI-generated design space.

---

### 5.3 Enums — Mode Selectors

**File:** `llmind-python/utils/modes.py`

Enums are named choices. The system can switch between different backends and content modes:

```python
class BackendMode(str, Enum):
    """Selects the LLM/embedding backend used across pipeline commands."""
    openai = "openai"    # Use OpenAI's cloud API
    vllm = "vllm"        # Use a local model server

class ContentMode(str, Enum):
    """Selects which text field(s) are used to generate embeddings."""
    description = "description"   # Descriptions column only
    details = "details"            # Details column only
    hybrid = "hybrid"              # Descriptions + Details concatenated
    all = "all"                    # Run all three modes (ingest only)
```

These enums inherit from `str` so they can be used directly as Typer CLI options and FastAPI query parameters. When a user runs `uv run database_pipeline.py ingest --embed-mode vllm --content-mode hybrid`, Typer parses those strings directly into enum values.

**Designer's mental model:** These are like *radio button groups* in a form. You pick one mode, and it changes how the pipeline behaves downstream.

---

### 5.4 Shared Utilities

#### Client Factories — `utils/clients.py`

Two factory functions create OpenAI-compatible clients:

```python
def build_openai_client() -> OpenAI:
    if not settings.openai_api_key:
        raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")
    return OpenAI(api_key=settings.openai_api_key)

def build_vllm_client(base_url: str) -> OpenAI:
    """Create an OpenAI-compatible client pointed at a local vLLM server."""
    return OpenAI(api_key="vllm", base_url=base_url)
```

The vLLM client reuses the `OpenAI` class because vLLM exposes an OpenAI-compatible API. The API key `"vllm"` is a dummy — local servers don't need authentication.

#### Chunked Iteration — `utils/iter.py`

A simple generator that breaks a sequence into batches:

```python
def chunked(sequence: Sequence[T], size: int) -> Generator[Sequence[T], None, None]:
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]
```

Used when embedding hundreds of records — the OpenAI API has a per-request token limit, so records are sent in batches of 180.

#### JSON Utilities — `utils/json.py`

Three key functions:
- `read_json_array(path)` — loads and validates a JSON array from a file.
- `save_json(path, data)` — saves data to JSON, automatically calling `.model_dump()` on Pydantic models.
- `extract_json_between_markers(text)` — extracts a JSON object from an LLM response that may contain markdown \`\`\`json fences or free-form text around the JSON. Uses regex to find either a fenced block or the first `{...}` pair.

#### Audio Transcription — `utils/_transcribe.py`

A standalone utility (prefixed with `_` to indicate it's not part of the core pipeline) that transcribes audio files using OpenAI's Whisper API. Supports:
- Standard transcription via `whisper-1`
- Speaker diarization via `gpt-4o-transcribe-diarize` with optional known-speaker references
- Converting existing JSON transcriptions back to formatted text

---

## 6. Layer 2 — The Data Pipeline (Python CLI)

The data pipeline prepares raw data for AI consumption. Think of it as the "kitchen prep" before cooking. All commands are run via `uv run database_pipeline.py <command>`.

### 6.1 Scraping Projects

**File:** `llmind-python/scrape_projects.py`

This script visits the [Media Architecture Biennale](https://awards.mediaarchitecture.org) website and extracts project data. It's built as a Typer CLI app.

**What it does, step by step:**

1. **Fetches the listing page** — the gallery at `/mab/projects/`.
2. **Extracts project URLs** — finds `div.mab-card > a[href="/mab/project/{id}"]` elements using regex `^/mab/project/(\d+)/?$`. De-duplicates and preserves page order.
3. **Visits each project page** and extracts:
   - **Name** — from `.titlepro` elements (excluding `<small>` tags), joined with ` | `.
   - **Descriptions** — from `.col-sm-6` elements, collecting `<p>` and `<h5>` text while filtering out `"None"` values and URLs via regex `http\S+|www\.\S+`.
   - **Details** — from `.col-sm-4` elements, specifically the `<p>` tags that follow a `<h5 class="mediumkur">` header with text "Descriptions", stopping at the next `<h5>`.
   - **Image** — the gallery image URL from `img.gallery.img-fluid.img-responsive`, resolved to absolute URL.
4. **Returns** a list of `ProjectRecord` Pydantic models.

**Key design decisions:**
- Uses **`tenacity`** for exponential backoff retries (up to 4 attempts, 1–10s backoff) on 5xx errors and network failures. Non-5xx HTTP errors are not retried.
- Has a **polite delay** between requests (0.8s default) to avoid overloading the server.
- URL validation ensures only `awards.mediaarchitecture.org` URLs with `http`/`https` schemes are fetched.
- The `SAVE_HTML_SNAPSHOT` flag (default `False`) can optionally capture the raw HTML of each page's main container.

```bash
# Scrape 20 projects
uv run scrape_projects.py scrape --limit 20

# Scrape all projects with custom output
uv run scrape_projects.py scrape --limit 0 -o all_projects.json
```

---

### 6.2 Pipeline Constants — Derived Settings

**File:** `llmind-python/pipeline/constants.py`

This module re-exports values from `config.py` as module-level constants, and defines two critical mapping dictionaries:

```python
# Which Supabase table stores embeddings for each content mode
EMB_TABLE_MAP: Dict[ContentMode, str] = {
    ContentMode.description: settings.supabase_emb_description_table,   # "media_emb_description"
    ContentMode.details:     settings.supabase_emb_details_table,       # "media_emb_details"
    ContentMode.hybrid:      settings.supabase_emb_hybrid_table,        # "media_emb_hybrid"
}

# Which column stores the embedding for each backend
EMB_COLUMN_MAP: Dict[BackendMode, str] = {
    BackendMode.openai: "embedding_cloud",    # 1536 dims
    BackendMode.vllm:   "embedding_local",    # 384 dims
}
```

Also defines:
- `METADATA_NOISE_KEYS = frozenset({"url", "html_main"})` — fields stripped during cleaning.
- Output paths like `DATA_DIR / "cleaned_media_architecture.json"` and `ANALYSIS_DIR / "analysis_summary.json"`.

---

### 6.3 Analyzing Raw Data

**File:** `llmind-python/database_pipeline.py` → `analyze` command

Before processing data, you need to understand what you have. The `analyze` command fetches all records from `raw_projects` and produces:

- Total items, how many have non-empty Details
- Word count and character count statistics (min, max, mean, median) via `summary_stats()` from `data_ops.py`
- A histogram plot of word counts saved to `plots/details_word_count_hist.png`
- Examples of the shortest entries saved to `analysis/min_detail_examples.json`
- A full summary saved to `analysis/analysis_summary.json`

```bash
uv run database_pipeline.py analyze
```

**Designer's mental model:** This is like doing a **content audit** before redesigning a website — you need to know the shape and quality of your content.

---

### 6.4 Cleaning & Ingesting Data

**File:** `llmind-python/pipeline/data_ops.py` + `database_pipeline.py` → `ingest` command

The `ingest` command is the main data processing pipeline. It performs three steps:

**Step 1 — Clean raw records** (`clean_records()` in `data_ops.py`):
1. Extracts the `Details` field from each record.
2. **Filters** by word count — must have between 20 and 600 words (configured in `config.py`).
3. **Strips** noise fields (`url`, `html_main`) from the metadata.
4. **Extracts** a clean project ID from the URL (everything after `project/`), or generates a UUID fallback.

**Step 2 — Upsert to `media_doc`** (`upsert_media_doc()` in `supabase.py`):
Maps scraped field names to flat database columns:
```python
{
    "id": r["id"],
    "name": r.get("Name"),
    "description": r.get("Descriptions"),
    "detail": r.get("Details"),
    "image": r.get("image_href"),
}
```

**Step 3 — Embed and upsert** per content mode:
For each content mode (`description`, `details`, `hybrid`, or all three if `--content-mode all`):
1. `build_embed_records()` creates `EmbedRecord` instances with the appropriate text (`build_context()` selects the field based on mode).
2. Records are sent to the embedding API in batches of 180 via `chunked()`.
3. Embedding vectors are upserted into the matching embedding table (e.g., `media_emb_details`) under the correct column (`embedding_cloud` for OpenAI, `embedding_local` for vLLM).

```bash
# Ingest with OpenAI embeddings, details field only
uv run database_pipeline.py ingest

# Ingest with vLLM, all content modes
uv run database_pipeline.py ingest --embed-mode vllm --content-mode all
```

---

### 6.5 Embeddings — Turning Text Into Numbers

**Embeddings** convert human-readable text into a list of numbers (a "vector") that captures the *meaning* of the text.

**How it works in the ingest flow:**
1. For each cleaned record, extract the relevant text field based on `ContentMode`.
2. Send that text to an embedding model — OpenAI's `text-embedding-3-small` (1536 dimensions) or a local vLLM model like `BAAI/bge-small-en-v1.5` (384 dimensions).
3. Get back a list of floating-point numbers.
4. Store that number list in the appropriate embedding table column.

**Why this matters:**
- Two projects about "LED installations in public squares" will have *similar* number lists.
- Two projects about completely different topics will have *different* number lists.
- This similarity can be measured mathematically (**cosine similarity**).

**Designer's mental model:** Embeddings are like GPS coordinates for concepts. Just as nearby GPS coordinates mean physically close locations, nearby embedding vectors mean semantically similar ideas.

---

### 6.6 Clustering — Grouping Similar Projects

**File:** `llmind-python/pipeline/ml.py` + `database_pipeline.py` → `cluster` command

Once you have embeddings, you can group similar projects together.

**The process:**
1. `fetch_embeddings()` retrieves all embeddings from a Supabase table, selecting the column based on backend mode (`embedding_cloud` or `embedding_local`). Handles JSON string deserialization for embeddings stored as strings.
2. `umap_reduce()` reduces the high-dimensional embeddings to 2D:
   - **Optional PCA pre-step** (`pre_pca=64`): Reduces from 1536 → 64 dimensions before UMAP, dramatically speeding it up.
   - **UMAP** with `n_neighbors=15`, `min_dist=0.1`, `metric="cosine"` projects to 2D while preserving neighborhood relationships.
3. `kmeans_cluster()` assigns each 2D point to one of k clusters (default: 8).
4. `normalize_to_unit_interval()` scales x/y coordinates to `[0, 1]` for consistent visualization.

**Output:** Each project gets `{id, x, y, cluster}`. Can be output as JSON to stdout or as a scatter plot via `--plot`. Cluster groups (mapping cluster label → list of project IDs) are saved to `analysis/cluster_groups_{table}_{k}.json`.

```bash
# Cluster and output JSON
uv run database_pipeline.py cluster

# Cluster and save plot
uv run database_pipeline.py cluster --plot --clusters 10
```

---

### 6.7 Farthest-Point Selection — Picking Diverse Samples

**File:** `llmind-python/pipeline/ml.py` → `select_farthest()` + `database_pipeline.py` → `farthest` command

When you have hundreds of projects but only want to show 20–30 to an LLM, you want **maximum diversity**. The greedy farthest-point algorithm:

1. **Normalize** all embeddings to unit length (`unit_normalize()`).
2. **Start** with a random project (seeded for reproducibility).
3. Compute **cosine distance** from all points to the selected one: `1.0 - (Xn @ Xn[start])`.
4. **Pick** the project with the maximum minimum distance from all already-selected points.
5. **Update** the minimum-distance array: `min_distances = np.minimum(min_distances, new_distances)`.
6. Mark selected points with `-np.inf` to exclude them.
7. **Repeat** until k projects are selected.

```bash
# Select 20 diverse projects
uv run database_pipeline.py farthest --k 20 -o data/selected_projects.json
```

**Designer's mental model:** It's like curating a gallery show — you don't want 20 similar pieces, you want each one to represent a different corner of the artistic landscape.

---

### 6.8 Visualization — Plotting Clusters

**File:** `llmind-python/pipeline/viz.py`

A simple matplotlib plotting function `plot_clusters()` that:
- Groups points by cluster label.
- Uses the `tab10` colormap, resampled to the number of clusters.
- Plots each cluster as a colored scatter group with alpha=0.7 and size=50.
- Adds a legend outside the chart area.
- Saves to a file or displays interactively.

---

## 7. Layer 3 — AI-Powered Taxonomy Generation (Python)

### 7.1 What Is a Taxonomy?

In LLMind, a **taxonomy** is a structured map of the design space:

```json
{
  "aspects": [
    {
      "name": "Display Medium",
      "desc": "Technology through which light or image becomes visible",
      "options": [
        { "name": "LED systems", "desc": "..." },
        { "name": "Projection mapping", "desc": "..." }
      ]
    },
    {
      "name": "Interaction Mode",
      "desc": "How users influence the experience",
      "options": [
        { "name": "Passive display", "desc": "..." },
        { "name": "Sensor-reactive", "desc": "..." }
      ]
    }
  ]
}
```

**Aspects** are the *dimensions* of the design space (like axes on a chart).
**Options** are the *choices* along each dimension.

---

### 7.2 Prompt Engineering — Talking to the LLM

**File:** `llmind-python/utils/prompts.py`

LLMind defines three prompt templates:

**`SYSTEM_PROMPT`** — A dictionary with two keys:
- `"project"` — a detailed description of the target project (an interactive media installation for Aarhus 2017).
- `"system"` — the AI persona: *"You are a creative professional designer and analytical thinker. You use concise expression with maximum information density that are highly based on facts and data."*

**`IDEA_FIRST_PROMPT`** — The initial taxonomy generation prompt that:
1. Injects existing artefacts (scraped project descriptions).
2. Defines what "design space," "aspects," and "options" mean.
3. Asks for a `THOUGHT` section (reasoning) followed by a `NEW IDEA JSON` code block.
4. Specifies the JSON structure with `Aspect` and `Option` fields.
5. Explicitly states "No further nesting is allowed."

**`IDEA_REFLECTION_PROMPT`** — A self-refine prompt (currently commented out in the generation flow) that asks the AI to review and consolidate its taxonomy over multiple rounds.

**`USER_PROMPT_TEMPLATE`** — Used by the backend's generate-nodes endpoint. Template variables use `{{DOUBLE_BRACES}}` and are replaced via `.replace()`:
- `{{TAXONOMY}}` — the current mind map tree formatted as indented text.
- `{{SELECTED_NODE_ID}}` and `{{SELECTED_NODE_TOPIC}}` — the node the user clicked.
- `{{RELATED_PROJECTS}}` — numbered list of related projects found via vector search.

The prompt asks the LLM to return a specific JSON structure: `{ "parent_id": "...", "options": [{ "id": "...", "topic": "..." }] }`.

---

### 7.3 Structured Outputs — Forcing JSON Responses

**File:** `llmind-python/generate_taxonomy.py` → `OpenAIChat` class

The `OpenAIChat` dataclass implements the `ChatSession` protocol and manages a stateful conversation with message history. It supports both OpenAI and vLLM backends:

**OpenAI path** — uses `client.beta.chat.completions.parse()` which automatically enforces the Pydantic schema:
```python
completion = client.beta.chat.completions.parse(
    model=self.model,
    messages=self._messages,
    response_format=Taxonomy,          # ← Pydantic model as schema
    reasoning_effort=self.reasoning_effort,  # "low" | "medium" | "high"
)
taxonomy = completion.choices[0].message.parsed  # ← Already a Taxonomy object!
```

**vLLM path** — manually builds a strict JSON schema because vLLM doesn't support the beta parse endpoint:
```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "taxonomy",
        "strict": True,
        "schema": _make_strict_schema(Taxonomy.model_json_schema()),
    },
}
```

The `_make_strict_schema()` helper recursively walks the JSON schema and adds `"additionalProperties": false` to every object node — a requirement for OpenAI's strict mode that Pydantic doesn't add automatically.

After each response, the assistant's message is appended to `self._messages`, maintaining conversational context for potential multi-round refinement.

---

### 7.4 The Generation Flow

**Function:** `run_generate()` in `generate_taxonomy.py`

```
1. Format existing artefacts as "ID: Description" pairs
2. Fill in IDEA_FIRST_PROMPT with artefacts
3. Send to LLM via chat.send_message() — gets structured Taxonomy back
4. (Optional) Self-refine loop (currently commented out):
   - Send IDEA_REFLECTION_PROMPT for each round
   - Each round consolidates and improves the taxonomy
5. Return final Taxonomy object
```

The CLI (`generate_tax` command) then saves the result as a timestamped JSON file: `taxonomy/tax_{mode}_{model}_{YYYYMMDD_HHMM}.json`.

---

### 7.5 Building Artefacts for the LLM

**File:** `llmind-python/utils/supabase.py` → `build_artefacts()`

This function prepares the project descriptions that get injected into the LLM prompt. It supports two source modes:

- **`"selected"`** — Loads a list of project IDs from a JSON file (typically the output of the `farthest` command), fetches matching rows from `media_doc`, and preserves the original order.
- **`"all_supabase"`** — Fetches all rows from `media_doc` (with optional `max_projects` cap), paginating in batches of 1000.

The inner `_artefacts_from_rows()` function selects the text field based on `ContentMode` and returns `[{ "ID": id, "Description": text }]` — the format expected by the prompt template.

---

## 8. Layer 4 — The FastAPI Backend

### 8.1 App Entry Point

**File:** `llmind-python/backend/main.py`

The entire backend is just 14 lines:

```python
from fastapi import FastAPI
from backend.related_projects.router import router as related_projects_router
from backend.taxonomy.router import router as taxonomy_router

app = FastAPI()
app.include_router(related_projects_router)   # /api/related-projects/*
app.include_router(taxonomy_router)           # /api/taxonomy/*

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

FastAPI auto-generates interactive API documentation at `/docs` (Swagger UI) and `/redoc`.

---

### 8.2 Router → Service Pattern

The backend is organized in two layers per feature:

```
Request → Router (validates input via Pydantic) → Service (business logic) → Response
                                                        ↕
                                                 External APIs
                                                 (OpenAI, Supabase)
```

| Layer | File | Responsibility |
|-------|------|----------------|
| **Router** | `router.py` | Defines URL endpoints, Pydantic request/response models, HTTP status codes |
| **Service** | `service.py` | Business logic, external API calls, error wrapping |

---

### 8.3 The Search Endpoint

**`POST /api/related-projects/search`**

**Router models** (`related_projects/router.py`):
- `FetchRelatedProjectsRequest` — requires `topic` (1–400 chars), optional `lineage`, `description`, `limit` (1–20), `similarity_threshold` (0–1), `embedding_model`, `match_function`.
- `FetchRelatedProjectsResponse` — contains `projects: list[RelatedProject]`.
- `RelatedProject` — has `id`, `Id`, `Name`, `Descriptions`, `Details`, `Image`.

**Service flow** (`related_projects/service.py`):
1. `build_related_query_text()` constructs a search string from `lineage > description | topic` — combining the node's hierarchy path with its description for richer semantic search.
2. `fetch_related_projects()`:
   - Creates an embedding of the query text via `build_openai_client()`.
   - Calls a Supabase RPC function (`match_media_docs`) with the query embedding, match count, and similarity threshold.
   - `_extract_related_project()` normalizes the response rows, handling both flat columns and nested `metadata` JSONB.
3. **Fallback:** If Supabase returns nothing or errors, returns a placeholder: `[{ "Name": "Relevant projects will appear here" }]`.

---

### 8.4 The Generate-Nodes Endpoint

**`POST /api/related-projects/generate-nodes`**

**Request model** (`GenerateNodesRequest`):
- `taxonomy_nodes` — the entire mind map tree as a flat array of `TaxonomyNodeInput` (id, topic, parentid, isroot).
- `focus_node` — the selected node (id + topic).
- Optional: `lineage`, `description`, `related_projects` (pre-fetched), `model`, `mode`, `base_url`, `reasoning_effort`.

**Service flow** (`generate_nodes_from_related_projects()`):
1. **Resolve projects** — uses provided `related_projects` or fetches them via `search_related_projects()`.
2. **Format taxonomy** — `_format_taxonomy()` converts the flat node array into an indented tree string by building a node map, finding the root, and recursively formatting: `  - Topic (id)\n`.
3. **Format projects** — `_format_projects_for_prompt()` creates a numbered list: `1. Project Name\n  Description: ...`.
4. **Build prompt** — fills `USER_PROMPT_TEMPLATE` with taxonomy, selected node, and related projects via `.replace()`.
5. **Generate** — `_generate_node_payload()` calls the LLM:
   - **OpenAI**: uses `client.beta.chat.completions.parse()` with `NodeGenerationPayload` as the response format. Falls back to `_extract_json_from_markdown()` if structured parsing returns `None`.
   - **vLLM**: uses `_make_strict_schema()` on `NodeGenerationPayload.model_json_schema()`.
6. **Return** — `{ parent_id, options: {id: topic}, nodes: [{node_id, topic, parent_node}], related_projects }`.

---

### 8.5 The Taxonomy Generation Endpoint

**`POST /api/taxonomy/generate`**

**Router** (`taxonomy/router.py`):
- `GenerateTaxonomyRequest` — requires `project_overview` (1–10000 chars), plus optional `num_reflections`, `content_mode`, `ids_file`, `model_name`, `reasoning_effort`, `mode`, `base_url`.
- `GenerateTaxonomyResponse` — contains `aspects: list[AspectResponse]`, each with `options: list[OptionResponse]`.

**Service** (`taxonomy/service.py`):
A thin wrapper that calls `generate_taxonomy()` from `generate_taxonomy.py`, converts the `Path` for `ids_file`, and wraps errors in `TaxonomyServiceError` → HTTP 502.

---

### 8.6 Error Handling — The 502 Pattern

Both feature modules define their own error class (`ServiceError`, `TaxonomyServiceError`). Every external call (OpenAI, Supabase) is wrapped in try/except:

```python
try:
    result = external_api_call()
except ServiceError:
    raise                           # re-raise known errors
except Exception as exc:
    raise ServiceError("What went wrong") from exc   # preserves original error
```

The router catches the service error and returns HTTP 502 (Bad Gateway):

```python
except ServiceError as exc:
    raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
```

> **Debugging tip:** The 502 masks the real error. Use `e.__cause__` or check the uvicorn server logs to find the actual problem.

---

## 9. Layer 5 — The Next.js Frontend (`llmind-web`)

### 9.1 App Router & Layout

**File:** `llmind-web/src/app/layout.tsx`

Next.js uses the **App Router** pattern where file paths map to URL routes:
- `src/app/page.tsx` → `/` (home page with navigation cards)
- `src/app/mindmap/page.tsx` → `/mindmap` (the main interactive page)

The root layout wraps everything in:
1. **Fonts** — Geist Sans and Geist Mono loaded via `next/font/google` as CSS variables (`--font-geist-sans`, `--font-geist-mono`).
2. **Providers** — the `<Providers>` wrapper component.

**Home page** (`page.tsx`) renders a simple card grid with links to `/mindmap` and `/projects` using shadcn/ui `Card` components.

**Providers** (`providers.tsx`) wraps children in React Query's `QueryClientProvider` with a `QueryClient` configured with:
- `staleTime: 5 minutes` — cached data is considered fresh for 5 minutes.
- `retry: 1` — only retry failed queries once.

It also renders a `<Toaster />` from **sonner** for toast notifications (used by the taxonomy generation dialog to show success/error messages).

The `QueryClient` is created inside `useState` to ensure it's only created once per React lifecycle (avoids recreating on re-renders).

**Global CSS** (`globals.css`) imports Tailwind CSS 4, `tw-animate-css`, and `shadcn/tailwind.css`. It defines a comprehensive design token system using CSS custom properties in OKLCH color space, with full light and dark mode variants for all semantic colors (background, foreground, card, primary, secondary, muted, destructive, etc.).

---

### 9.2 API Proxy — How Frontend Talks to Backend

**File:** `llmind-web/next.config.ts`

```typescript
const rawBackendUrl =
  process.env.BACKEND_URL ??
  process.env.NEXT_PUBLIC_BACKEND_URL ??
  'http://0.0.0.0:8000';

const BACKEND_URL = rawBackendUrl.replace(/\/$/, '');

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      { protocol: 'https', hostname: 'images.unsplash.com' },
    ],
  },
  async rewrites() {
    return [{
      source: '/api/:path*',
      destination: `${BACKEND_URL}/api/:path*`,
    }];
  },
};
```

This is a **proxy rewrite**: when the frontend makes a request to `/api/related-projects/search`, Next.js silently forwards it to `http://0.0.0.0:8000/api/related-projects/search` (the Python backend). The `images.remotePatterns` config allows Next.js Image optimization for external image hosts.

**Why?** This avoids CORS issues and keeps API keys server-side. The frontend never needs to know the backend's real address.

The **API client** (`lib/api-client.ts`) is an Axios instance with `baseURL: '/'` — all requests go to the same origin, hitting the Next.js proxy:

```typescript
const api = axios.create({
    baseURL: '/',
    headers: { 'Content-Type': 'application/json' },
});
```

---

### 9.3 React Query — Server State Management

**Files:**
- `features/mindmap/hooks/use-related-projects-query.ts` — Fetches related projects
- `features/mindmap/hooks/use-generate-nodes-mutation.ts` — Triggers node generation
- `features/mindmap/hooks/use-generate-taxonomy-mutation.ts` — Triggers full taxonomy generation

**Query (read operation — `useRelatedProjectsQuery`):**

This hook wraps a POST to `/api/related-projects/search` with:
- **Input validation** — checks topic is non-empty, limit is 1–20, similarity threshold is 0–1.
- **Query key** — includes every request parameter so React Query caches and deduplicates correctly: `['mindmap-related-projects', topic, lineage, description, ...]`.
- **Auto-fires** when the topic changes (via the query key).
- **Caches** results for 5 minutes (`staleTime: 1000 * 60 * 5`).
- **Enabled** only when `topic?.trim()` is truthy.

```typescript
const { data, isFetching } = useRelatedProjectsQuery({ request });
// data.projects = RelatedProject[]
```

**Mutation (write operation — `useGenerateNodesMutation`):**

This hook handles the "Generate Nodes" action:
1. `flattenMindmapNodes()` converts the entire nested `MindmapNode[]` tree into a flat `TaxonomyNodeInputSchema[]` array by traversing recursively. Each node gets `{ id, topic, parentid, isroot }`.
2. `deriveLineage()` walks `parentid` links upward from the focus node to build the ancestor chain (e.g. `["Design Aspects", "Display Medium"]`).
3. Sends a POST to `/api/related-projects/generate-nodes` with the full taxonomy, focus node, lineage, related projects, and model configuration.
4. Detailed Axios error handling extracts the `detail` field from 502 error responses.

```typescript
const { mutateAsync: generateNodes, isPending } = useGenerateNodesMutation();
// Call: generateNodes({ allNodes, focusNode, description, ... })
// Returns: GenerateNodesResponseSchema
```

The helper `generatedNodesToMindmapNodes()` converts the backend's flat `{ node_id, topic, parent_node }[]` response into `MindmapNode[]` — a simple map since all returned nodes are direct children of the focus node.

**Taxonomy Mutation (write operation — `useGenerateTaxonomyMutation`):**

This hook handles full taxonomy generation from a project overview:
1. Sends a POST to `/api/taxonomy/generate` with `project_overview`, `reasoning_effort` (default: `"medium"`), and `mode` (default: `"openai"`).
2. Detailed Axios error handling extracts `detail` from error responses, with fallback to the raw error message.
3. Returns `GenerateTaxonomyResponse` containing `aspects: TaxonomyAspect[]`, each with `name`, `desc`, and `options`.

```typescript
const { mutate, isPending, error, reset } = useGenerateTaxonomyMutation();
// Call: mutate({ project_overview: "...", reasoning_effort: "medium", mode: "openai" })
```

**Dialog Components — `features/mindmap/components/`:**

Two multi-step dialog components provide the UI for the generation actions:

- **`GenerateNodesDialog`** — A two-step dialog (`form` → `confirm`) for generating child nodes under a selected topic. Accepts `additionalContext` (optional textarea), `reasoningEffort` (low/medium/high select), and `mode` (openai/vllm select). On confirm, calls back with the dialog parameters.

- **`GenerateTaxonomyDialog`** — A two-step dialog for generating an entire taxonomy. Requires a `project_overview` (textarea, validated as non-empty), plus optional `reasoningEffort` and `mode` selects. Integrates directly with `useGenerateTaxonomyMutation` — on success, shows a toast via sonner and passes the result to `onSuccess`.

Both dialogs share a consistent design pattern: shadcn/ui `Dialog` primitives, a `fieldClass` constant for styled form inputs, and disabled states during generation (`isPending`).

---

### 9.4 Zustand Store — Client State

**File:** `llmind-web/src/store/mindmap-store.ts`

Manages persistent UI state using Zustand with two middleware layers:

```typescript
export const useMindmapStore = create<MindmapStoreState>()(
  devtools(                    // Enables Redux DevTools debugging
    persist(                   // Saves to localStorage
      (set) => ({
        ...createInitialState(),
        setJmRef: (ref) => set(() => ({ jmRef: ref })),
        selectTopic: ({ topic, lineage = [], contextDescription = '' }) =>
          set(() => ({
            contextText: buildContextText(topic, lineage),
            contextDescription,
            selectedTopic: topic,
          })),
        setContext: ({ contextText, contextDescription }) =>
          set(() => ({ contextText, contextDescription })),
        setProjects: (projects) =>
          set(() => ({
            projects: projects.map((project) => ({ ...project })),
          })),
        setProjectsLoading: (isLoading) =>
          set(() => ({ projectsLoading: isLoading })),
        setTaxonomy: (taxonomy) =>
          set(() => ({ taxonomy })),
        setMindmapData: (payload) =>
          set((state) => ({
            contextText: payload.contextText ?? state.contextText,
            contextDescription: payload.contextDescription ?? state.contextDescription,
            projects: payload.projects
              ? payload.projects.map((project) => ({ ...project }))
              : state.projects,
            projectsLoading: payload.projectsLoading ?? state.projectsLoading,
          })),
        resetMindmapStore: () => set(() => createInitialState()),
      }),
      {
        name: 'mindmap-store',
        partialize: (state) => ({       // Only persist these fields
          contextText: state.contextText,
          contextDescription: state.contextDescription,
          selectedTopic: state.selectedTopic,
          projects: state.projects,
          taxonomy: state.taxonomy,
        }),
      }
    ),
    { name: 'mindmap-store' }         // DevTools label
  )
);
```

The `buildContextText()` helper creates a breadcrumb string from the lineage: `lineage.slice(1).join(' > ')` — skipping the root node.

**What gets persisted to localStorage:** `contextText`, `contextDescription`, `selectedTopic`, `projects`, **and `taxonomy`** (the generated taxonomy data).

**What is NOT persisted:** `jmRef` (the Mind Elixir instance reference) and `projectsLoading` — transient UI state.

**Key actions:**
- `selectTopic()` — updates context text, description, and selected topic from a node click.
- `setTaxonomy()` — stores a generated taxonomy (from the GenerateTaxonomyDialog).
- `setContext()` — directly sets context text and description.
- `setMindmapData()` — batch-updates multiple fields with null-coalescing to preserve existing values.
- `resetMindmapStore()` — resets all state to initial values.

---

### 9.5 Mind Elixir Mind Map Component

**File:** `llmind-web/src/components/mindmap/simple-mindmap.tsx`

The mind map uses the **Mind Elixir** library (v5.9.2). Here's how the component works:

**Data conversion — `buildModel()`:**
Takes `MindmapNode[]` and produces a `MindElixirModel` containing:
- `data` — Mind Elixir's required format (`{ nodeData: { id, topic, children } }`).
- `lineageById` — maps each node ID to its full path (e.g., `"led-panels" → ["Design Aspects", "Display Medium", "LED panels"]`).
- `topicToId` — reverse lookup from topic text to node ID.

The recursive `convertNode()` function builds all three structures in a single pass. If there's a single root node, it's used directly; multiple roots get wrapped in a synthetic `__root__` node.

**Initialization (runs once):**
```typescript
const mind = new MindElixir({
  el: container,
  direction: MindElixir.SIDE,      // branches flow left/right
  editable: false,                  // read-only
  contextMenu: false,
  toolBar: false,
  keypress: false,
  allowUndo: false,
});
mind.init(model.data);
```

**Click handling:**
A click listener on `mind.map` walks up the DOM to find the `ME-TPC` element (Mind Elixir's topic element), extracts `nodeObj`, and resolves the lineage:
```typescript
const lineage = modelRef.current.lineageById[nodeObj.id] ?? buildLineageFromParent(nodeObj);
onSelectRef.current({ topic: nodeObj.topic, lineage: [...lineage] });
```

The `buildLineageFromParent()` fallback recursively walks `node.parent` links — used when a node was generated after the initial model was built.

**Syncing and updates:**
- When `nodes` prop changes (after generation), `mind.refresh(model.data)` updates the visual tree.
- When `activeTopic` changes externally, `mind.selectNode(mind.findEle(nodeId))` highlights the corresponding node.
- `isSyncingRef` prevents infinite loops when programmatic selection triggers the click handler.

---

### 9.6 The Project Panel Component

**File:** `llmind-web/src/components/mindmap/simple-project-panel.tsx`

A split-view component with a **project list** on the left and **project detail** on the right.

**Data normalization** — `toProjectListItem()` handles inconsistent field naming from the API by checking both cases (e.g., `project.Name || project.name`, `project.Image || project.image`). Image URLs are validated to ensure they start with `/` or have `http:`/`https:` protocol.

**Key sub-components:**
- `Skeleton` — an animated loading placeholder (4 pulsing bars).
- `ProjectList` — renders clickable buttons for each project. The active project gets `border-primary bg-primary/10`. Shows "No projects found." with a dashed border when empty.
- `ProjectDetail` — shows the selected project's name, image (lazy-loaded with `referrerPolicy="no-referrer"`), description, and detail text.

Auto-selects the first project when the list changes via: `activeId = items.some(item => item.id === selectedId) ? selectedId : items[0]?.id ?? null`.

---

### 9.7 The Mindmap Page — Putting It All Together

**File:** `llmind-web/src/app/mindmap/page.tsx`

This is the main interactive page. Here's the complete flow:

**Initialization:**
1. Loads initial mind map data from `schema-mindmap-data.ts`, which imports `public/schema_selected.json` and converts it into `MindmapNode[]` using `slugify()` for IDs and building a `descriptionByTopic` lookup.
2. Checks the Zustand store for a previously generated `taxonomy`. If one exists, uses `taxonomyToMindmapNodes()` to rebuild the tree from it; otherwise falls back to the static schema data.
3. Sets initial selection to the root node ("Design Aspects").
4. **Opens the `GenerateTaxonomyDialog` on mount** if no taxonomy exists in the store (`useState(() => !taxonomy)`), prompting the user to generate one.

**User generates a new taxonomy:**
```
1. "Generate Taxonomy" button → opens GenerateTaxonomyDialog
2. User fills in project overview, reasoning effort, backend mode
3. Dialog submit → useGenerateTaxonomyMutation → POST /api/taxonomy/generate
4. onSuccess → useMindmapStore.setTaxonomy(result)
5. page.tsx useEffect on taxonomy → taxonomyToMindmapNodes(taxonomy)
6. setNodes(nextNodes)  — replaces entire tree
7. setSelection(INITIAL_SELECTION) — resets to root
8. Mind Elixir refreshes with the new tree
```

**User clicks a node:**
```
1. SimpleMindMap fires onSelect({ topic, lineage })
2. setSelection() updates local state
3. useEffect syncs to Zustand store via selectTopic()
4. buildRequest() creates API payload with topic, lineage, description
5. useRelatedProjectsQuery auto-fires with new request
6. Backend embeds query → searches Supabase → returns projects
7. SimpleProjectPanel renders the project list
```

**User clicks "Generate Nodes":**
```
1. "Generate Nodes" button → opens GenerateNodesDialog
2. User optionally adds additional context, picks reasoning effort and backend
3. Dialog confirm → handleGenerateNodes():
   a. findNodeByLineage() locates the selected node in the tree
   b. Checks if fetched projects are real (not placeholder)
   c. Sends allNodes, focusNode, description, relatedProjects, mode, reasoningEffort
4. Backend formats prompt → calls LLM → returns new options
5. generatedNodesToMindmapNodes() converts response to MindmapNode[]
6. insertChildrenAtNode() immutably inserts children (see §10.6)
7. setNodes(updatedTree) → React re-renders → Mind Elixir refreshes
```

**UI layout** — uses absolute positioning for floating panels:
- **Bottom center** — floating navigator bar with LLMind branding, "Generate Taxonomy" button (Sparkles icon), "Generate Nodes" button (Zap icon, loading spinner during generation), and Home link. Rounded-full with backdrop blur.
- **Top left** — collapsible context panel showing lineage breadcrumbs (as Badges with ChevronRight separators), description text, and any generation error with retry button.
- **Top right** — collapsible related projects panel with project count badge, wrapping `SimpleProjectPanel`.
- **Background** — the full-screen `SimpleMindMap` component.
- **Dialogs** — `GenerateTaxonomyDialog` and `GenerateNodesDialog` render as modal overlays.

---

### 9.8 Auto-Generated Types From OpenAPI

**File:** `llmind-web/src/types/openapi.ts`

This file is **auto-generated** from the backend's OpenAPI specification. Never edit it manually!

```bash
# Regenerate when backend models change:
cd llmind-web
bunx openapi-typescript http://localhost:8000/openapi.json -o src/types/openapi.ts
```

This gives you **end-to-end type safety**: the Python Pydantic models define the shape, FastAPI generates an OpenAPI spec, and `openapi-typescript` creates TypeScript interfaces.

The file defines the full API surface including `paths`, `components.schemas`, and `operations`. Convenience type aliases at the bottom simplify usage:

```typescript
export type MindmapProjectSchema = components["schemas"]["RelatedProject"];
export type FetchRelatedProjectsRequestSchema = components["schemas"]["FetchRelatedProjectsRequest"];
export type FetchRelatedProjectsResponseSchema = components["schemas"]["FetchRelatedProjectsResponse"];
export type GenerateNodesRequestSchema = components["schemas"]["GenerateNodesRequest"];
export type GenerateNodesResponseSchema = components["schemas"]["GenerateNodesResponse"];
export type TaxonomyNodeInputSchema = components["schemas"]["TaxonomyNodeInput"];
export type FocusNodeInputSchema = components["schemas"]["FocusNodeInput"];
export type GeneratedNodeSchema = components["schemas"]["GeneratedNode"];
export type BackendModeSchema = components["schemas"]["BackendMode"];  // "openai" | "vllm"
```

> **Note:** The taxonomy generation types (`GenerateTaxonomyResponse`, etc.) are defined locally in `use-generate-taxonomy-mutation.ts` rather than coming from the OpenAPI spec. If the backend taxonomy router models change, you may want to regenerate the OpenAPI types and migrate these local types to use the generated ones.

---

## 10. Layer 6 — Advanced Topics

### 10.1 Supabase Vector Search (pgvector)

Supabase uses PostgreSQL with the **pgvector** extension for similarity search. The flow:

1. **Store embeddings** in a `VECTOR` column — `VECTOR(1536)` for OpenAI cloud embeddings, `VECTOR(384)` for local vLLM models.
2. **Create an RPC function** (`match_media_docs`) that:
   - Takes a `query_embedding`, `match_count`, and `similarity_threshold`.
   - Computes cosine similarity between the query and all stored embeddings.
   - Returns the top matches above the threshold.
3. **Call the function** from Python via the Supabase client:
```python
rpc_response = supabase_client.rpc(
    "match_media_docs",
    {
        "query_embedding": query_embedding,
        "match_count": 5,
        "similarity_threshold": 0.0,
    },
).execute()
```

---

### 10.2 Database Schema & Migrations

**Files:** `llmind-python/migrations/`

Three SQL files define the complete database schema:

**`supabase_raw_table.sql`** — The raw scraper output:
```sql
CREATE TABLE IF NOT EXISTS raw_projects (
    id TEXT PRIMARY KEY,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```
Includes GIN index on the `metadata` JSONB column and an expression index on `metadata->>'url'`.

**`media_doc_tables.sql`** — The core schema:
- `media_doc` — flat table with `id`, `name`, `description`, `detail`, `image`, `created_at`, `updated_at`.
- Three embedding tables (`media_emb_description`, `media_emb_details`, `media_emb_hybrid`), each with:
  - `media_doc_id TEXT PRIMARY KEY REFERENCES media_doc(id) ON DELETE CASCADE`
  - `context TEXT NOT NULL` (the text that was embedded)
  - `embedding_cloud VECTOR(1536)` (OpenAI)
  - `embedding_local VECTOR(384)` (vLLM)
- IVFFlat index creation commands (commented out — require data to train).

**`migrate_media_emb_columns.sql`** — Idempotent migration for upgrading from a legacy single-`embedding` column schema to the cloud/local split. Uses `DO $$ ... $$` blocks with `information_schema` checks.

---

### 10.3 OpenAI Structured Outputs Constraints

When using `response_format=YourPydanticModel`, OpenAI enforces **strict JSON Schema rules**:

| ❌ Forbidden | Why | ✅ Fix |
|-------------|-----|--------|
| `dict[str, str]` | Generates `additionalProperties` | Use `list[SomeModel]` |
| Fields with defaults | Field excluded from `required` | Remove defaults |
| `Optional[str]` without `null` type | Schema mismatch | Use `str \| None` |

The `_make_strict_schema()` helper in both `generate_taxonomy.py` and `service.py` addresses this by recursively adding `additionalProperties: false` to every object node in the schema.

**Testing tip:** Always inspect your schema first:
```python
print(YourModel.model_json_schema())
```

---

### 10.4 vLLM & OpenAI-Compatible Servers

The single most important thing to understand about LLMind's "local model" support is that **it is not vLLM-specific.** The whole mechanism is one function:

```python
def build_vllm_client(base_url: str) -> OpenAI:
    return OpenAI(api_key="vllm", base_url=base_url)
```

It builds the *same* `OpenAI` client object the cloud path uses — it just points it at a different URL. Because the OpenAI Python SDK speaks a standard HTTP protocol (`POST /v1/chat/completions`, `POST /v1/embeddings`), **any server that implements that protocol works**: vLLM, Ollama, LM Studio, llama.cpp's server, text-generation-webui, LocalAI, and others. The `"vllm"` name is historical — read it as "any OpenAI-compatible endpoint."

The choice propagates through the system via the `BackendMode` enum:
- `BackendMode.vllm` → `build_vllm_client(base_url)` → embeddings go in the `embedding_local` column → `VECTOR(384)`
- `BackendMode.openai` → `build_openai_client()` → embeddings go in the `embedding_cloud` column → `VECTOR(1536)`

**When to use a local model:** to avoid API costs, work offline / keep data on-prem, or run a custom/fine-tuned model.

> 📖 **Full setup walkthrough — including a local model on Windows and a remote Linux vLLM server over SSH — is in [Section 11](#11-connecting-a-local-llm-replacing-the-openai-api).**

---

### 10.5 UMAP + KMeans Dimensionality Reduction

**The problem:** Embeddings have 1536 dimensions. Humans can visualize at most 2–3 dimensions.

**UMAP** (Uniform Manifold Approximation and Projection) in `pipeline/ml.py`:
- Non-linear dimensionality reduction.
- Preserves both local and global structure.
- Key parameters: `n_neighbors=15`, `min_dist=0.1`, `metric="cosine"`.
- Warnings about `n_jobs` being overridden by `random_state` are suppressed.

**Optional PCA pre-step:** Reduces from 1536 → 64 dimensions via `sklearn.decomposition.PCA` before UMAP. This speeds up UMAP significantly without much quality loss.

**KMeans** (`kmeans_cluster()`):
- Uses `sklearn.cluster.KMeans` with `n_init="auto"` and `random_state=42`.
- Falls back to `[0] * len(X)` (all same cluster) if sklearn is unavailable.

**Normalisation** (`normalize_to_unit_interval()`):
- Maps values to `[0, 1]` range: `(v - lo) / (hi - lo)`.
- Returns `0.5` for all values if `hi ≈ lo` (using `math.isclose`).

---

### 10.6 Immutable Tree Updates in React

**File:** `llmind-web/src/app/mindmap/page.tsx`

React requires **immutable** state updates — you can't modify an object in place. For tree data, this means creating new objects all the way up the path:

```typescript
function insertChildrenAtNode(
  nodes: ReadonlyArray<MindmapNode>,
  parentId: string,
  childrenToInsert: ReadonlyArray<MindmapNode>
): TreeUpdateResult {
  let inserted = false;
  const nextNodes = nodes.map((node) => {
    if (node.id === parentId) {
      inserted = true;
      const existingChildren = node.children ?? [];
      const existingIds = new Set(existingChildren.map((child) => child.id));
      const uniqueNewChildren = childrenToInsert.filter((child) => !existingIds.has(child.id));
      return {
        ...node,
        children: [...existingChildren, ...uniqueNewChildren],
      };
    }
    if (!node.children?.length) return node;    // leaf — no change needed
    const childResult = insertChildrenAtNode(node.children, parentId, childrenToInsert);
    if (!childResult.inserted) return node;      // nothing changed below
    inserted = true;
    return { ...node, children: childResult.nodes };  // new node with updated subtree
  });
  return { nodes: nextNodes, inserted };
}
```

**Key design details:**
- **De-duplication** — uses a `Set` of existing child IDs to prevent inserting nodes that already exist.
- **Short-circuit** — if no children, returns the original node (saves unnecessary cloning). If recursion finds no change, returns the original node.
- **Return value** — `{ nodes, inserted }` tells the caller whether the insertion succeeded.

**Designer's mental model:** Imagine the tree is made of paper. You can't erase and rewrite — you must photocopy the entire path from root to the changed node, modifying it on the copy. React then compares old and new trees to find what changed in the UI.

---

## 11. Connecting a Local LLM (Replacing the OpenAI API)

By default LLMind sends every generation and embedding request to OpenAI's cloud API. This section shows how to point it at a model you run yourself instead — first on your **local Windows machine**, then on a **remote Linux server running vLLM** that you reach over SSH.

You do **not** need to touch any code. Everything is driven by a handful of environment variables and a `mode` flag.

---

### 11.1 How LLMind Talks to Models

Every LLM call in the backend goes through one of two client factories in [`utils/clients.py`](../llmind-python/utils/clients.py):

```python
def build_openai_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)            # cloud

def build_vllm_client(base_url: str) -> OpenAI:
    return OpenAI(api_key="vllm", base_url=base_url)          # local / remote
```

Both return the **same `OpenAI` SDK object** — the only difference is `base_url`. Any server that speaks the OpenAI HTTP protocol (`/v1/chat/completions`, `/v1/embeddings`) can stand in:

| Server | Runs on Windows natively? | Notes |
|---|---|---|
| **Ollama** | ✅ Yes | Easiest on Windows. Endpoint: `http://localhost:11434/v1` |
| **LM Studio** | ✅ Yes | GUI + one-click server at `http://localhost:1234/v1` |
| **llama.cpp** (`llama-server`) | ✅ Yes | Lightweight, GGUF models |
| **vLLM** | ❌ No (Linux + NVIDIA GPU) | Fastest for serving; use a remote Linux box or WSL2 |

> ⚠️ **vLLM does not run on native Windows.** It requires Linux and a CUDA GPU. On Windows, use Ollama / LM Studio / llama.cpp, or run vLLM inside WSL2 or on a remote server (§11.4).

Three configuration knobs control where requests go — all in `llmind-python/.env`:

```bash
VLLM_BASE_URL=http://localhost:11434/v1   # the OpenAI-compatible endpoint
VLLM_MODEL=qwen2.5:7b-instruct            # chat/generation model name
VLLM_EMBED_MODEL=BAAI/bge-small-en-v1.5   # embedding model name (pipeline only)
```

To *select* the local backend at request time you pass `mode: "vllm"` — either from the frontend dialogs (the **Generate Taxonomy** and **Generate Nodes** dialogs both have an "openai / vllm" dropdown) or directly in the API payload. When `mode = "vllm"`, the backend reads `VLLM_BASE_URL` and `VLLM_MODEL` from your `.env` automatically (see [`taxonomy/service.py`](../llmind-python/backend/taxonomy/service.py)).

---

### 11.2 What `mode = vllm` Actually Switches

This is the part most people get wrong, so be precise about it. `mode = "vllm"` only redirects the **chat/generation** calls. Here is the full map:

| Operation | Honors `mode=vllm`? | Where it goes |
|---|---|---|
| Taxonomy generation (`POST /api/taxonomy/generate`) | ✅ Yes | `VLLM_BASE_URL`, model `VLLM_MODEL` |
| Node generation (`POST /api/related-projects/generate-nodes`) | ✅ Yes | `VLLM_BASE_URL` (or per-request `base_url`) |
| Pipeline embeddings (`ingest` / `cluster` / `farthest` with `--embed-mode vllm`) | ✅ Yes | `--vllm-base-url`, stored in `embedding_local` `VECTOR(384)` |
| **Related-projects search embedding** (`POST /api/related-projects/search`) | ❌ **No** | **Always OpenAI cloud** |

> 🔑 **The search gotcha.** `search_related_projects()` in [`related_projects/service.py:216`](../llmind-python/backend/related_projects/service.py:216) is hardcoded to `build_openai_client()`. It embeds the search query with OpenAI **regardless of `mode`**. So even in "local" mode, the *related projects* panel still needs a valid `OPENAI_API_KEY` — unless you skip Supabase entirely by sending `should_query_supabase: false`.
>
> To go **fully local** for search as well, you'd have to edit that function to branch on `BackendMode` (the same way node generation does) and re-embed your corpus locally so the stored vectors match the query model's dimensions. That's a code change, not a config change.

**Bottom line for a no-code-change setup:** local generation works out of the box; keep `OPENAI_API_KEY` set for the search panel, or bypass search with `should_query_supabase: false`.

---

### 11.3 Case A — A Local LLM on Windows

Goal: run the chat model on your own machine, no cloud generation calls. We'll use **Ollama** (simplest), with notes for LM Studio.

**Step 1 — Install and pull a model**

Install Ollama for Windows from [ollama.com](https://ollama.com/download), then:

```powershell
ollama pull qwen2.5:7b-instruct      # chat model (good JSON adherence)
ollama serve                          # starts the server (usually already running)
```

Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1`.

**Step 2 — Point LLMind at it**

Edit `llmind-python/.env`:

```bash
VLLM_BASE_URL=http://localhost:11434/v1
VLLM_MODEL=qwen2.5:7b-instruct
# Keep OPENAI_API_KEY set — the related-projects search still uses it (see §11.2)
OPENAI_API_KEY=sk-...
```

**Step 3 — Restart the backend and select the local backend**

```bash
cd llmind-python
uv run fastapi dev backend/main.py
```

Then either:
- **From the UI:** open the **Generate Taxonomy** or **Generate Nodes** dialog and choose **vllm** in the backend dropdown, or
- **From the API:** set `"mode": "vllm"` in the request body.

**Step 4 — Smoke test (no Supabase needed)**

```powershell
curl -X POST 'http://localhost:8000/api/taxonomy/generate' `
  -H 'Content-Type: application/json' `
  -d '{\"project_overview\":\"A modular interactive light installation for a public square.\",\"mode\":\"vllm\"}'
```

**LM Studio alternative:** load a model in LM Studio, click **Start Server** (Local Server tab). It serves at `http://localhost:1234/v1`. Set `VLLM_BASE_URL=http://localhost:1234/v1` and `VLLM_MODEL` to the model identifier LM Studio shows.

> 🧩 **Structured-output support matters.** LLMind's vLLM path sends a strict JSON-schema `response_format` (see [`generate_taxonomy.py`](../llmind-python/generate_taxonomy.py) `_make_strict_schema`). Pick a server/model that honors `response_format: json_schema`: recent Ollama (≥ 0.5), LM Studio, and vLLM all do. If responses fail validation, the model isn't returning schema-conformant JSON — switch to a stronger instruct model or a server with constrained decoding.

---

### 11.4 Case B — A Remote Linux vLLM Server over SSH

Goal: run vLLM on a GPU Linux box (lab server, cloud VM) and drive it from your Windows dev machine. This is what the shipped default already assumes — `VLLM_BASE_URL=http://100.73.44.12:8001/v1` is a Tailscale address pointing at such a server.

**Step 1 — On the Linux server: install and serve**

```bash
# Requires Linux + NVIDIA GPU + CUDA
uv pip install vllm        # or: pip install vllm

# Chat/generation server. --served-model-name sets the name clients must request.
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8001 \
  --served-model-name qwen \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192

# (Optional) a separate embedding server on another port, if you also embed locally
vllm serve BAAI/bge-small-en-v1.5 --task embed --host 0.0.0.0 --port 8002
```

Run these under **tmux** or **systemd** so they survive your SSH session closing:

```bash
tmux new -s vllm
# start vllm here, then detach with Ctrl-b d
```

**Step 2 — Reach the server from Windows**

You have two safe options. **Do not expose vLLM directly on the public internet** — it has no authentication.

**Option 1 — SSH tunnel (recommended).** Forward the remote ports to your local machine:

```powershell
# Maps localhost:8001 -> server:8001 (and 8002 for embeddings). -N = no shell, just forward.
ssh -N -L 8001:localhost:8001 -L 8002:localhost:8002 user@your-server.example.com
```

Leave that window open. Then in `.env`, point at the **local** end of the tunnel:

```bash
VLLM_BASE_URL=http://localhost:8001/v1
VLLM_MODEL=qwen          # must match --served-model-name
OPENAI_API_KEY=sk-...    # still needed for related-projects search (§11.2)
```

**Option 2 — Tailscale / private VPN.** If the server is on your tailnet, skip the tunnel and point straight at its tailnet IP (this is the existing default style):

```bash
VLLM_BASE_URL=http://100.73.44.12:8001/v1
VLLM_MODEL=qwen
```

**Step 3 — Restart the backend and use `mode: "vllm"`** — identical to Case A, Steps 3–4.

> 💡 **Keeping the tunnel alive.** For an always-on link, add to your `~/.ssh/config`:
> ```
> Host vllm-server
>     HostName your-server.example.com
>     User you
>     LocalForward 8001 localhost:8001
>     LocalForward 8002 localhost:8002
>     ServerAliveInterval 60
> ```
> Then just `ssh -N vllm-server`.

---

### 11.5 Embedding Dimensions — The 384 vs 1536 Trap

This only matters if you also want **local embeddings** (the data pipeline, `--embed-mode vllm`), not just local chat.

The database has two embedding columns, sized for specific models:

| Backend | Column | Dimensions | Default model |
|---|---|---|---|
| `openai` | `embedding_cloud` | **1536** | `text-embedding-3-small` |
| `vllm` | `embedding_local` | **384** | `BAAI/bge-small-en-v1.5` |

The `embedding_local` column is `VECTOR(384)` — it fits `bge-small-en-v1.5` exactly. If you serve a **different** embedding model (e.g. `bge-m3` is 1024-dim, `nomic-embed-text` is 768-dim), the dimensions won't match and inserts will fail.

To switch local embedding models you must, in order:
1. Set `VLLM_EMBED_MODEL` (and `--vllm-model` on the CLI) to the new model.
2. Change `VECTOR(384)` to the new dimension in [`migrations/media_doc_tables.sql`](../llmind-python/migrations/media_doc_tables.sql) (and the migration file), and re-run the migration.
3. Re-run `uv run python database_pipeline.py ingest --embed-mode vllm` to repopulate `embedding_local`.

If you only need **local generation** (taxonomy + nodes) and are happy to keep search on OpenAI, you can ignore this entire subsection.

---

### 11.6 Verifying & Troubleshooting

**Confirm the server is reachable and the model name is right:**

```powershell
curl http://localhost:8001/v1/models     # or :11434 for Ollama, :1234 for LM Studio
```

The `id` in the response is exactly what `VLLM_MODEL` must be set to.

**One-line backend sanity check (bypasses Supabase):**

```bash
cd llmind-python
uv run python -c "from utils.clients import build_vllm_client; from config import settings; c=build_vllm_client(settings.vllm_base_url); print(c.chat.completions.create(model=settings.vllm_model, messages=[{'role':'user','content':'reply with OK'}]).choices[0].message.content)"
```

**Common failures:**

| Symptom | Likely cause | Fix |
|---|---|---|
| `502` with `provider: vllm` in detail | Server unreachable / wrong `VLLM_BASE_URL` / tunnel not open | `curl …/v1/models`; check the SSH tunnel window is still open |
| `502`, model-not-found | `VLLM_MODEL` ≠ the served model name | Match `VLLM_MODEL` to `--served-model-name` (or the `/v1/models` id) |
| Validation / JSON parse error | Server didn't honor strict `json_schema` | Use a server/model with structured-output support (§11.3 note) |
| Insert fails with dimension mismatch | Local embedding model ≠ 384 dims | Resize the `VECTOR` column (§11.5) and re-ingest |
| Related projects still hit OpenAI / fail without key | Search embedding is always OpenAI | Set `OPENAI_API_KEY`, or send `should_query_supabase: false` (§11.2) |
| `Missing required environment variable: OPENAI_API_KEY` | Search path needs the key even in local mode | Set `OPENAI_API_KEY`, or bypass search as above |

> **Reminder:** a `502` masks the real error. Check the uvicorn logs or `e.__cause__` for the original exception before assuming the server is down — see [§8.6](#86-error-handling--the-502-pattern).

---

## 12. Hands-On Exercises

### Beginner

1. **Read the config:** Open `llmind-python/config.py` and list all the environment variables that need to be set. Create a `.env` file with placeholder values.

2. **Trace a type:** Starting from `ProjectRecord` in `utils/models.py`, follow how a scraped project flows through `scrape_projects.py` → `database_pipeline.py` → `utils/supabase.py`. What happens to each field at each step?

3. **Explore the mind map data:** Open `llmind-web/public/schema_selected.json`. Identify the aspects and options. How does `schema-mindmap-data.ts` transform this JSON into `MindmapNode[]`?

### Intermediate

4. **Follow the API flow:** Starting from the "Generate Nodes" button click in `llmind-web/src/app/mindmap/page.tsx`, trace the complete path:
   - What hook is called?
   - What data gets sent to the backend?
   - What does `flattenMindmapNodes()` do?
   - How does the response get inserted into the tree?

5. **Understand the prompt:** Read `utils/prompts.py`. How does `USER_PROMPT_TEMPLATE` differ from `IDEA_FIRST_PROMPT`? Modify the `USER_PROMPT_TEMPLATE` to ask for exactly 3 options per aspect instead of letting the LLM decide.

6. **Explore the Zustand store:** Compare the `mindmap-store.ts` `partialize` function with the full state interface. Why are some fields excluded from persistence?

7. **Go local:** Following [§11.3](#113-case-a--a-local-llm-on-windows), install Ollama, set `VLLM_BASE_URL`/`VLLM_MODEL`, and generate a taxonomy with `"mode": "vllm"`. Then trace *why* the related-projects panel still calls OpenAI ([§11.2](#112-what-mode--vllm-actually-switches)) — find the hardcoded `build_openai_client()` and sketch the change that would make search local too.

### Advanced

8. **Database schema:** Read the SQL files in `migrations/`. Draw a diagram of the tables and their relationships. How does the `ON DELETE CASCADE` constraint work between `media_doc` and the embedding tables?

9. **Error debugging:** The `ServiceError` pattern wraps all external call errors. Add a logging statement in `service.py` that prints the original error before wrapping it. Why is the `from exc` chain important for debugging?

10. **Custom embedding model:** What would you need to change to switch from OpenAI's `text-embedding-3-small` (1536 dims) to a local model with 384 dimensions? (Hint: config, `EMB_COLUMN_MAP`, and the migration SQL — and see [§11.5](#115-embedding-dimensions--the-384-vs-1536-trap).)

11. **Add a new feature:** Design (in pseudocode) how you would add a "save taxonomy to Supabase" feature that persists the user's mind map modifications. Which files would you create or modify?

---

## 13. Further Reading

### AI & Embeddings
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) — How text embeddings work
- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — Forcing JSON responses
- [UMAP Explained](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html) — Visual guide to dimensionality reduction

### Local & Self-Hosted Models
- [vLLM — OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) — Serving models with `vllm serve`
- [Ollama OpenAI Compatibility](https://github.com/ollama/ollama/blob/main/docs/openai.md) — The `/v1` endpoint LLMind talks to
- [LM Studio Local Server](https://lmstudio.ai/docs/app/api) — One-click OpenAI-compatible server on Windows
- [SSH Port Forwarding](https://www.ssh.com/academy/ssh/tunneling/example) — How the `-L` tunnel in §11.4 works

### Python
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/) — Building APIs in Python
- [Pydantic V2 Docs](https://docs.pydantic.dev/latest/) — Data validation and modeling
- [Typer Docs](https://typer.tiangolo.com/) — Building CLI tools (used by the pipeline)
- [Tenacity Docs](https://tenacity.readthedocs.io/) — Retry library used in the scraper

### Frontend / React
- [Next.js App Router Docs](https://nextjs.org/docs/app) — File-based routing and layouts
- [React Query (TanStack Query)](https://tanstack.com/query/latest/docs/react/overview) — Server state management
- [Zustand](https://zustand-demo.pmnd.rs/) — Minimalist state management
- [Mind Elixir Docs](https://mind-elixir.com/) — Mind map library
- [shadcn/ui](https://ui.shadcn.com/) — UI component library used for buttons, cards, badges, etc.

### Database
- [Supabase Docs](https://supabase.com/docs) — Database, auth, and vector search
- [pgvector](https://github.com/pgvector/pgvector) — PostgreSQL vector similarity extension

### Design Space Theory
- Buruk, O. T. (2020). "Design space" — a conceptual tool for game design exploration
- MacLean, A. et al. (1991). "Design space analysis" — understanding a design through its alternatives

---

> **Remember:** You don't need to understand everything at once. Start with Layer 1 (foundations), connect the dots visually with the architecture diagram, and then dive deeper into whichever layer interests you most. Each file in the codebase has a clear, focused responsibility.
