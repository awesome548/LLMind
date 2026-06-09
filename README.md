# LLMind
LLMind is an LLM-assisted system for exploring the design space.

## Technical snapshot
Monorepo with two independent subsystems (see `CLAUDE.md` for the hub):

- **Frontend (`llmind-web/`):** Next.js + React + TypeScript, Zustand for state, TanStack Query for async data, `mind-elixir` for the mind-map view. A **Design Space** view renders the taxonomy on a 2D lattice of the project corpus — two views of one selection. See `llmind-web/FRONTEND.md`.
- **Backend (`llmind-python/`):** FastAPI, managed with `uv`. Generates structured taxonomies via OpenAI-compatible structured output and serves related-project retrieval. Runs fully local (LM Studio / vLLM for chat + embeddings) or against OpenAI + Supabase. See `llmind-python/BACKEND.md`.
- **AI workflow:** the active mind-map branch + lineage is formatted into prompt templates and sent to the chat model to propose new nodes; structured JSON is parsed back into the node tree.
- **Project retrieval:** query embeddings fetch related project metadata via Supabase pgvector or an offline `.npz` index, surfaced beside the map.
- **Design-space pipeline:** `database_pipeline.py` scrapes, embeds, clusters, and fits a frozen UMAP projection of the corpus; `generate_taxonomy.py` serializes taxonomies. See `DESIGN-SPACE-VIZ.md` and `DESIGN-SPACE-TESTING.md`.

## Run locally

One command starts both servers (each in its own window), builds the design-space
projection if missing, and wires the frontend proxy to the backend:

```powershell
.\dev.ps1                 # start backend + frontend
.\dev.ps1 -Install        # first run: `uv sync` + `bun install`, then start
.\dev.ps1 -NoFrontend     # backend only (or -NoBackend)
```

If PowerShell blocks the script, run it once as:
`powershell -ExecutionPolicy Bypass -File .\dev.ps1`

Then open http://localhost:3000/mindmap. To run things by hand instead, see the
Quick Start in `CLAUDE.md`.

Check each directory's README / doc for more detailed info.