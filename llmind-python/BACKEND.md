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
at runtime. See [`DESIGN-SPACE-VIZ.md`](../documentations/DESIGN-SPACE-VIZ.md) and
[`DESIGN-SPACE-ITERATION-PLAN.md`](../documentations/DESIGN-SPACE-ITERATION-PLAN.md).

| Endpoint | Purpose | Needs embed/LLM server |
|---|---|---|
| `GET /api/projection/surface` | Precomputed corpus background: grid spec, points, density; `meta.trustworthiness` reports layout fidelity | No |
| `POST /api/projection/locate` | Embed node text → coords in the frozen space (`{items:[{node_id,text}]}`). **Placement is evidence-anchored (Part 11):** the similarity-weighted centroid of the top-k corpus neighbours' frozen coordinates — the same anchors that drive `support` — so a point sits amid its precedents (frozen `UMAP.transform()` is the fallback when corpus/surface artifacts are missing; `clipped` is only meaningful there). Each point carries `confidence` (true-vs-2D neighbourhood Jaccard) and `support` — the corpus-support percentile read against the **short-register baseline** fitted by `project-align` ("as much evidence as a real project described at node length"; falls back to the full-register self-support yardstick when no map exists). When `register_map.npz` exists (and `REGISTER_ALIGNMENT` isn't false), embeddings are register-corrected first | Yes (embed) |
| `POST /api/corpus/annotate` | **Schema annotation (Part 12 A2).** Async job: annotates every corpus project against the taxonomy's options (per option: register-corrected embedding shortlist of 30 → chunked local-LLM membership calls judging from description + Details, `JUDGE_BATCH=5` with window-aware token budgets — see `backend/corpus/llm.py` for the thinking-model rules) → per-option `{count, project_ids, projects:[{id,name}]}` + Halskov granularity diagnostics (`too_broad` ≥80%, `unprecedented` ≤1). Cached per option content hash under `data/projection/annotations/` (`ANNOTATION_VERSION` salts the hash — bump to invalidate); taxonomy edits only re-judge changed options | Yes (embed + LLM) |
| `POST /api/corpus/generate-cell` | **Cross-tab cell generation (Part 12 B2).** Async job: an empty option×option cell (an exact, nameable gap) + half-matching exemplar ids → ONE project concept committing to BOTH poles (`{name, desc}`); logged to `generate_log.jsonl` as `prompt_version="cell-v1"` | Yes (LLM) |
| `POST /api/corpus/rationale` | **The rationale layer (Part 13 L-A).** Async job (keyed — concurrent clients share it): `{aspects:[{id,name,desc,options:[{name,count}]}], n_projects}` → `{rationales:{<aspect_id>: str}}` — one line per aspect answering "why this dimension?", grounded in the annotation counts. Cached per aspect content+counts under `data/projection/rationales/` (`RATIONALE_VERSION` salts); per-aspect LLM failures degrade to `""` (explanation, never a gate) | Yes (LLM) |
| `POST /api/corpus/missing-aspect` | **The coverage probe (Part 13 L-A).** Async job (keyed): `{aspect_names, project_ids}` (the frontend computes the poorly-covered projects from the annotation — pure set arithmetic) → `{proposals:[{name,desc,reason}]}` — what dimension those projects exemplify that the taxonomy misses, deduped against existing names, ≤2. Rides the C1 proposals channel client-side | Yes (LLM) |
| `POST /api/projection/peek` | **Gap preview (E1):** the deterministic seed set a generate-at would use + nearby explored ideas + the derived parent aspect — shown BEFORE any LLM time is spent | No |
| `POST /api/projection/generate-at` | **Async.** Location-conditioned generation: clicked `(x,y)` + optional `coords` (located nodes) → seeds that **bracket** the gap, parent aspect **derived from the click**, options with `desc`, per-node `drift` + `mean_drift` | Yes (embed + LLM) |
| `POST /api/projection/axes` | Semantic-axes perspective: `{x:{pole_a,pole_b}, y:{...}, items}` → exact bipolar cosine coords for corpus + items (clip-flagged), with diagnostics (`x/y_pole_sim`, `axis_corr`) | Yes (embed) |
| `POST /api/projection/metrics` | **Examine strips (Part 10):** a LIST of bipolar metrics → per metric the FULL corpus score distribution, clip-flagged item scores, `pole_sim`; plus the pairwise metric correlation matrix (rubric redundancy) | Yes (embed) |

`generate-at` behaviour: seeds come from `seed_corpus` (`SEED_STRATEGY=bracket` default,
`anchor` = legacy single-neighbourhood; switchable for A/B). Every call appends a JSONL
row to `data/projection/generate_log.jsonl` (`prompt_version`, `seed_strategy`,
`register_aligned`, `placement`, target, seeds, per-node `desc`/drift/`clipped`/`support`)
— the evaluation dataset for prompt/seeding/alignment variants. Drift means different
things under different placement regimes, so stats never aggregate across them.
Analyse with `uv run python database_pipeline.py project-log-stats` (drift mean/median +
clipped rate per `prompt_version` × `seed_strategy` × `register_aligned` × `placement`;
pure logic in `pipeline/log_stats.py`).

### Placement validity (Iterations H + J)

The register gap — short node texts vs the full-description corpus index — is
measured and corrected (ITERATION-PLAN Part 9):

```bash
uv run python database_pipeline.py project-align     # fit short→long register map (needs embed server)
uv run python database_pipeline.py project-diagnose  # reproducible validity report (add --offline to skip embedding)
```

`project-align` learns an affine short→long correction from the corpus's own
(name+first-sentences, full-text) pairs, reports HELD-OUT cosine/displacement/clip
metrics — including the **UMAP-transform vs evidence-anchored kNN** placement
comparison that motivated Part 11 (kNN k=5: median displacement 0.149 vs 0.179,
clip 0% vs 35% on corpus short-register round-trips) — and saves
`data/projection/register_map.npz`; `/locate` applies it (`REGISTER_ALIGNMENT=false`
disables). The same fit also persists the **short-register support baseline**
(sorted mean-top-k cosines of the out-of-fold corrected short texts,
self-excluded) inside the artifact — the yardstick that makes node-length support
percentiles meaningful (the full-register self-support distribution flattens
every short text to the 0th percentile). Every located point carries a continuous
corpus `support` percentile (`backend/corpus/service.py`). The `SOFT_MARGIN`
soft-clip band (`pipeline/projection.py`) now only concerns the fallback
transform path — the primary kNN placement cannot leave the corpus footprint.

`surface`/`locate` return `502` with `ServiceError` detail on failure. Artifacts live in
`data/projection/` (`model.joblib`, `surface.json`); path is `settings.projection_dir`.

### Candidates (dual-layer designs — Part 10)

| Endpoint | Purpose | Needs embed/LLM server |
|---|---|---|
| `POST /api/candidates/draft-brief` | **Async job.** Draft the candidate's BRIEF (identity layer) from its committed choices — project-register prose the designer edits | Yes (LLM) |
| `POST /api/candidates/alignment` | How the brief and the choices agree: `agreement` = cos(brief, composition) + per aspect whether the brief leans toward the chosen option or its strongest (data-picked) competitor | Yes (embed) |
| `POST /api/candidates/steer` | **Steering (Part 12 B3). Async job.** ONE deliberate move on the brief — `mode: metric` (toward a strip pole, `target_score` on the corpus-normalised −1..1 scale), `toward`/`away` (a precedent reference) — made IN LANGUAGE by the LLM, with `preserve` choices. Returns `{revised_text, named_qualities, measurement}` where measurement = requested vs achieved + along/orthogonal displacement (raw cosine space; `null` when the embedding service failed after the revision — the revision survives). Never auto-commits — the client shows a veto diff. Logged to `data/projection/steer_log.jsonl` | Yes (embed + LLM) |
| `POST /api/reflections/draft` | **Reflection drafting (Part 12 C2). Async job.** `{context}` (an exploration-event label) → `{draft}` — the one-line first-person rationale the designer might write; empty draft is NOT an error (the chip opens blank). Burden-inverted PRT: the system drafts, the designer accepts/edits/skips | Yes (LLM) |

`generate-at` accepts an optional `brief` (the active candidate's concept) as
prompt context — the squiggle hypothesis: convergence material feeding a
divergence step. Each log row records `brief_context` so `project-log-stats`
(now `prompt × seeding × aligned × brief`) can measure the effect.

### Corpus

| Endpoint | Purpose | Needs embed server |
|---|---|---|
| `GET /api/corpus/projects/{id}` | One corpus project's metadata (the inspectable design-space dots) | No |
| `POST /api/corpus/similar` | `{text, k}` → closest corpus precedents by TRUE cosine similarity (used for candidate designs) | Yes |
| `POST /api/corpus/relevance` | `{text}` → cosine score for EVERY corpus project + min/max (the relevance lens; client normalises → "relative relevance") | Yes |

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
| Projection core | `pipeline/projection.py` | Frozen PCA→UMAP reducer, persistence, grid, density, nearest, trustworthiness, soft-clip margin, evidence-anchored `place_by_neighbors` (Part 11) |
| Register alignment | `pipeline/register_alignment.py` | Short→long embedding correction: fit (CV: translation vs ridge), apply, persistence |
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
| `REGISTER_ALIGNMENT` | `true` | No | Apply the fitted short→long register correction in `/locate` (needs `register_map.npz`) |
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
