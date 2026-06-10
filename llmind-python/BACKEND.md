# Backend — llmind-python

FastAPI backend. Python 3.13+, managed with **uv**.

---

## Commands

```bash
uv sync                                    # install dependencies
uv run fastapi dev backend/main.py         # dev server → http://localhost:8000
uv run python -c "from config import settings; print(settings)"  # verify env
```

---

## API Endpoints

### `POST /api/taxonomy/generate`
Generate a full design taxonomy from a project overview.

**Request**
| Field | Type | Default | Notes |
|---|---|---|---|
| `project_overview` | `string` | required | 1–10 000 chars |
| `num_reflections` | `int` | `1` | ≥ 1 |
| `content_mode` | `"description" \| "details" \| "hybrid"` | `"details"` | Supabase column to use |
| `ids_file` | `string \| null` | `null` | Simple filename only — no `/` or `..` |
| `reasoning_effort` | `"low" \| "medium" \| "high"` | `"medium"` | |
| `mode` | `"openai" \| "vllm"` | `"openai"` | |

**Response** `200`
```json
{
  "aspects": [
    { "name": "string", "desc": "string", "options": [{ "name": "string", "desc": "string" }] }
  ]
}
```
**Error** `502` — `TaxonomyServiceError`; detail contains `request_id` and `stage`.

**Example (vLLM)**
```bash
curl -X POST 'http://0.0.0.0:8000/api/taxonomy/generate' \
  -H 'Content-Type: application/json' \
  -d '{
    "project_overview": "Design a modular community learning center for teens and seniors.",
    "ids_file": "50_selected_updated.json",
    "reasoning_effort": "medium",
    "mode": "vllm"
  }'
```

---

### `POST /api/related-projects/search`
Semantic search for related projects in Supabase.

**Request**
| Field | Type | Default |
|---|---|---|
| `topic` | `string` | required |
| `lineage` | `string[]` | `[]` |
| `description` | `string \| null` | `null` |
| `should_query_supabase` | `bool` | `true` |
| `limit` | `int` | `5` (1–20) |
| `similarity_threshold` | `float` | `0.0` (0–1) |

**Response** `200` — `{ "projects": RelatedProject[] }`

**Supabase bypass test**
```bash
curl -s -X POST http://localhost:8000/api/related-projects/search \
  -H "Content-Type: application/json" \
  -d '{"topic":"AI","lineage":[],"should_query_supabase":false,"limit":5,"similarity_threshold":0.0}' \
  | python3 -m json.tool
```

---

### `POST /api/related-projects/generate-nodes`
Generate child nodes for a selected mindmap node using an LLM. **Async** (see below).

**Request** — full schema in `backend/related_projects/router.py:GenerateNodesRequest`

Key fields: `taxonomy_nodes`, `focus_node`, `lineage`, `should_query_supabase`, `related_projects`, `mode`, `reasoning_effort`.

**Response** `202` — `{ "job_id": "...", "status": "pending" }`. Poll `GET /api/jobs/{job_id}`; the job `result` is:
```json
{
  "parent_id": "string",
  "options": { "key": "description" },
  "nodes": [{ "node_id": "string", "topic": "string", "parent_node": "string" }],
  "related_projects": [...]
}
```

**Supabase bypass test**
```bash
curl -s -X POST http://localhost:8000/api/related-projects/generate-nodes \
  -H "Content-Type: application/json" \
  -d '{"taxonomy_nodes":[{"id":"1","topic":"AI","isroot":true}],"focus_node":{"id":"1","topic":"AI"},"lineage":[],"should_query_supabase":false}' \
  | python3 -m json.tool
```

---

### Projection (design space)

Frozen 2D projection of the project corpus for the design-space visualization. Fit
once with the CLI (`uv run python database_pipeline.py project`), then served/queried
at runtime. See [`../DESIGN-SPACE-VIZ.md`](../DESIGN-SPACE-VIZ.md) and
[`../DESIGN-SPACE-ITERATION-PLAN.md`](../DESIGN-SPACE-ITERATION-PLAN.md).

| Endpoint | Purpose | Needs embed/LLM server |
|---|---|---|
| `GET /api/projection/surface` | Precomputed corpus background: grid spec, points, density; `meta.trustworthiness` reports layout fidelity | No |
| `POST /api/projection/locate` | Embed node text → coords in the frozen space (`{items:[{node_id,text}]}`); each point carries `confidence` (true-vs-2D neighbourhood Jaccard, None = unscorable) | Yes (embed) |
| `POST /api/projection/generate-at` | **Async.** Location-conditioned generation: clicked `(x,y)` + optional `coords` (located nodes) → seeds that **bracket** the gap, parent aspect **derived from the click**, options with `desc`, per-node `drift` + `mean_drift` | Yes (embed + LLM) |

`generate-at` behaviour: seeds come from `seed_corpus` (`SEED_STRATEGY=bracket` default,
`anchor` = legacy single-neighbourhood; switchable for A/B). Every call appends a JSONL
row to `data/projection/generate_log.jsonl` (`prompt_version`, `seed_strategy`, target,
seeds, per-node drift) — the evaluation dataset for prompt/seeding variants.

`surface`/`locate` return `502` with `ServiceError` detail on failure. Artifacts live in
`data/projection/` (`model.joblib`, `surface.json`); path is `settings.projection_dir`.

### Corpus

| Endpoint | Purpose | Needs embed server |
|---|---|---|
| `GET /api/corpus/projects/{id}` | One corpus project's metadata (the inspectable design-space dots) | No |
| `POST /api/corpus/similar` | `{text, k}` → closest corpus precedents by TRUE cosine similarity (used for candidate designs) | Yes |

### Taxonomy extras

`POST /api/taxonomy/generate` additionally returns `corpus_similarity` — cosine of the
project overview to the corpus centroid (best-effort; `null` when the embedding server
is down). The frontend shows a domain-mismatch notice below ~0.3.

### Calibration

```bash
uv run python database_pipeline.py project-calibrate   # needs the embedding server
```
Re-locates every corpus project by a SHORT text (its name) and reports displacement
from its true coordinate — quantifies how trustworthy short-text node placement is.

### Async generation (jobs)

`generate-at` and `generate-nodes` are long (local LLM ~50-80s). They return
`202 {job_id}` immediately and run on a background thread pool (`backend/jobs.py`);
poll `GET /api/jobs/{job_id}` → `{status: pending|done|error, result, detail}`. This
keeps every HTTP request short (the Next.js dev proxy can't deliver long responses)
and lets the UI show a spinner on the target dot/node. Jobs are in-memory and
process-local (single-process server only), pruned after 30 min.

---

## Architecture

| Layer | Location | Responsibility |
|---|---|---|
| Entry point | `backend/main.py` | Mounts routers; CORS (browser calls backend directly) |
| Router/Service — projection | `backend/projection/{router,service}.py` | Surface, node location (+confidence), generate-at (bracket seeds, derived parent, drift, JSONL log) |
| Router/Service — corpus | `backend/corpus/{router,service}.py` | Corpus metadata reader (shared), project-by-id, true-metric similarity search |
| Projection core | `pipeline/projection.py` | Frozen PCA→UMAP reducer, persistence, grid, density, nearest, trustworthiness |
| Async jobs | `backend/jobs.py` + `backend/jobs_router.py` | Background thread pool + `GET /api/jobs/{id}` polling for long generation |
| Config | `config.py` | Single `Settings` instance via pydantic-settings |
| Router — taxonomy | `backend/taxonomy/router.py` | Request validation; catches `TaxonomyServiceError` → 502 |
| Service — taxonomy | `backend/taxonomy/service.py` | Path resolution, calls `generate_taxonomy` |
| Router — related-projects | `backend/related_projects/router.py` | Request validation; catches `ServiceError` → 502 |
| Service — related-projects | `backend/related_projects/service.py` | Supabase search + LLM node generation |
| Taxonomy CLI | `generate_taxonomy.py` | `OpenAIChat` dataclass, `run_generate`, Typer CLI |
| Clients | `utils/clients.py` | `build_openai_client()`, `build_vllm_client()` |
| Supabase | `utils/supabase.py` | `get_supabase_client()`, `build_artefacts()` |
| Prompts | `utils/prompts.py` | `SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE`, `IDEA_FIRST_PROMPT` |
| Modes | `utils/modes.py` | `BackendMode` (openai/vllm), `ContentMode` (description/details/hybrid) |
| Models | `utils/models.py` | `Taxonomy`, `Aspect`, `Option`, `ProjectRecord`, `EmbedRecord` |

---

## Environment Variables

All loaded from `.env` in `llmind-python/`. Override any by setting the env var.

| Variable | Default | Required | Notes |
|---|---|---|---|
| `OPENAI_API_KEY` | `""` | Yes (openai mode) | Hard fails if missing when client is built |
| `SUPABASE_URL` | `""` | Yes (supabase enabled) | Hard fails at call site |
| `SUPABASE_KEY` | `""` | Yes (supabase enabled) | Hard fails at call site |
| `OPENAI_NODE_MODEL` | `gpt-5-mini-2025-08-07` | No | Verify model exists in your account |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | No | |
| `SUPABASE_MATCH_FUNCTION` | `match_media_docs` | No | Must exist as a Supabase RPC function |
| `SUPABASE_MATCH_COUNT` | `5` | No | |
| `VLLM_BASE_URL` | `http://100.73.44.12:8001/v1` | No | Change to `http://localhost:8001/v1` for local dev |
| `VLLM_MODEL` | `qwen` | No | |
| `VLLM_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | No | |
| `SEED_STRATEGY` | `bracket` | No | generate-at seeding: `bracket` (surround the gap) or `anchor` (legacy) |
| `DATA_DIR` | `data` | No | Root for data files |
| `TAXONOMY_DIR` | `taxonomy` | No | Output path for generated taxonomies |

---

## Error Handling

**Pattern:** every external call in `service.py` is wrapped in `try/except → raise ServiceError/TaxonomyServiceError`. The router converts those to HTTP 502. Read the `detail` field before debugging further.

**Diagnosing 502s — expose `__cause__`:**
```bash
uv run python -c "
import traceback
from backend.related_projects.service import generate_nodes_from_related_projects
try:
    generate_nodes_from_related_projects(
        focus_node_id='1', focus_node_topic='AI',
        taxonomy_nodes=[{'id':'1','topic':'AI','isroot':True}],
        should_query_supabase=False,
    )
except Exception as e:
    traceback.print_exc(); print('cause:', e.__cause__)
"
```

**Key service behaviors:**
- `search_related_projects()`: when Supabase returns empty or throws, returns `CURATED_FALLBACK_PROJECTS` — never propagates the error.
- `generate_nodes_from_related_projects()`: pass `should_query_supabase=false` to skip Supabase entirely in tests.
- `generate_taxonomy()` (taxonomy service): `ids_file` is validated against `data_dir` to prevent path traversal.
- `OpenAIChat.send_message()`: message history is built immutably — state only persists after both API call and validation succeed.
