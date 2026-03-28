# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Structure

Monorepo with two independent subsystems:

```
llmind-python/   FastAPI backend  (Python 3.13+, uv)
llmind-web/      Next.js frontend (React 19, Bun)
```

---

## Backend (`llmind-python/`)

```bash
uv sync                                      # install deps
uv run fastapi dev backend/main.py           # dev server → http://localhost:8000
uv run python -c "from config import settings; print(settings)"  # verify env
```

**Test endpoints directly:**
```bash
# Search (Supabase bypass)
curl -s -X POST http://localhost:8000/api/related-projects/search \
  -H "Content-Type: application/json" \
  -d '{"topic":"AI","lineage":[],"should_query_supabase":false,"limit":5,"similarity_threshold":0.0}' | python3 -m json.tool

# Generate nodes (Supabase bypass — isolates LLM call)
curl -s -X POST http://localhost:8000/api/related-projects/generate-nodes \
  -H "Content-Type: application/json" \
  -d '{"taxonomy_nodes":[{"id":"1","topic":"AI","isroot":true}],"focus_node":{"id":"1","topic":"AI"},"lineage":[],"should_query_supabase":false}' | python3 -m json.tool
```

**Diagnose 502s** — always expose `__cause__` before the ServiceError wraps it:
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

### Backend architecture

| Layer | Location | Responsibility |
|---|---|---|
| Config | `config.py` | Single `Settings` instance via pydantic-settings; all env vars map 1-to-1 |
| Router | `backend/related_projects/router.py` | Request/response validation; catches `ServiceError` → 502 |
| Service | `backend/related_projects/service.py` | All business logic; raises `ServiceError` on external failures |
| Clients | `utils/clients.py` | `build_openai_client()`, `build_vllm_client()` |
| Supabase | `utils/supabase.py` | `get_supabase_client()` |
| Prompts | `utils/prompts.py` | `SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE` |

**Error propagation pattern:** every external call in `service.py` is wrapped in `try/except Exception → raise ServiceError(...)`. The router converts any `ServiceError` to HTTP 502. The 502 `detail` field contains the exact ServiceError message — read it before debugging further.

### Key service behaviors

- `search_related_projects()`: when Supabase returns empty or throws, returns `CURATED_FALLBACK_PROJECTS` (never propagates the error).
- `generate_nodes_from_related_projects()`: if `related_projects` is passed in the request it skips Supabase lookup entirely; pass `should_query_supabase=false` to skip in testing.
- `_generate_node_payload()`: OpenAI path uses `client.beta.chat.completions.parse` (Structured Outputs); vLLM path uses `client.chat.completions.create` with a manually constructed JSON schema.

### Required env vars (`.env` in `llmind-python/`)

| Key | Used by |
|---|---|
| `OPENAI_API_KEY` | `build_openai_client()` — hard fails if missing |
| `SUPABASE_URL` | `get_supabase_client()` — hard fails if missing |
| `SUPABASE_KEY` | `get_supabase_client()` — hard fails if missing |
| `OPENAI_NODE_MODEL` | defaults to `gpt-5-mini-2025-08-07` in `config.py:37` — verify this model exists in your account |
| `SUPABASE_MATCH_FUNCTION` | defaults to `match_media_docs` — must exist as a Supabase RPC function |

---

## Frontend (`llmind-web/`)

```bash
bun install        # install deps
bun dev            # dev server → http://localhost:3000
bun build          # production build
bun lint           # eslint
```

**All `/api/*` requests are proxied to the backend** via `next.config.ts` rewrites:
```
/api/related-projects/search         → $BACKEND_URL/api/related-projects/search
/api/related-projects/generate-nodes → $BACKEND_URL/api/related-projects/generate-nodes
```
`BACKEND_URL` defaults to `http://0.0.0.0:8000`. Override with `BACKEND_URL` or `NEXT_PUBLIC_BACKEND_URL` env var.

### Frontend architecture

| Layer | Location | Responsibility |
|---|---|---|
| Types | `src/types/openapi.ts` | Auto-generated from backend OpenAPI spec — **do not edit manually** |
| API client | `src/lib/api-client.ts` | Axios instance with `baseURL: '/'` (proxied by Next.js) |
| Hooks | `src/features/mindmap/hooks/` | `useRelatedProjectsQuery` (React Query), `useGenerateNodesMutation` |
| Store | `src/store/mindmap-store.ts` | Zustand; persists `contextText`, `selectedTopic`, `projects` to localStorage |
| Components | `src/components/mindmap/` | `SimpleMindMap` (mind-elixir wrapper), `SimpleProjectPanel` |
| Data | `src/features/mindmap/data/schema-mindmap-data.ts` | Static initial taxonomy nodes + descriptions by topic |

**Node click → generate flow:**
1. `SimpleMindMap.onSelect(topic, lineage)` → `page.tsx` local state
2. `useRelatedProjectsQuery` auto-fires on selection change (React Query)
3. "Generate Nodes" button → `useGenerateNodesMutation` → `flattenMindmapNodes` (full tree context) → POST `/api/related-projects/generate-nodes`
4. Response nodes inserted immutably via `insertChildrenAtNode` using `response.parent_id`

**Placeholder filter:** the backend returns `{ Name: "Relevant projects will appear here" }` when Supabase has no matches. The page filters this out before passing `relatedProjects` to the generate call (`page.tsx:166-169`).

### Regenerate OpenAPI types

Whenever backend request/response models change, regenerate the frontend types:
```bash
cd llmind-web
bunx openapi-typescript http://localhost:8000/openapi.json -o src/types/openapi.ts
```

---

## Critical: OpenAI Structured Outputs schema constraints

**Root cause of the `dict[str, str]` 502 bug (March 2026):**

`client.beta.chat.completions.parse(response_format=SomePydanticModel)` uses OpenAI [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs). This API enforces strict JSON Schema rules that Pydantic does not enforce by default.

**Forbidden patterns in Pydantic models used as `response_format`:**

| Pattern | Why it fails | Fix |
|---|---|---|
| `dict[str, Any]` / `dict[str, str]` | Generates `additionalProperties: {"type": ...}` — forbidden | Replace with `list[SomeModel]` |
| Fields with `default` or `default_factory` | Field omitted from `required` array — all fields must be required | Remove defaults, or add `required` to all fields |
| Optional fields without explicit `None` in enum | Schema mismatch on nullability | Use `str \| None` and include `null` in the schema |

**Rule:** Every Pydantic model passed to `response_format=` must have all fields in `required`, no `additionalProperties`, and no dynamic key maps. When in doubt, generate and inspect the schema: `Model.model_json_schema()`.

**The 502 masks the real error.** The `ServiceError` catch-all in `service.py` discards the original exception type. Always check `e.__cause__` or uvicorn logs for the real `openai.BadRequestError` / `openai.APIStatusError` before assuming infrastructure failure.
