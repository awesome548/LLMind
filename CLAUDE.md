# CLAUDE.md — LLMind Project Hub

Central reference for Claude Code. All detailed documentation is in subsystem docs below.

---

## Structure

Monorepo with two independent subsystems:

```
llmind-python/   FastAPI backend  (Python 3.13+, uv)
llmind-web/      Next.js frontend (React 19, Bun)
```

---

## Detailed Documentation

| Doc | Contents |
|-----|----------|
| [`PROJECT-REPORT.md`](PROJECT-REPORT.md) | The master report: what the system is, the research argument, iterations, critical reflection, §5.7 code-verification audit + known defects |
| [`PROCESS.md`](PROCESS.md) | Session handoff: execution state, hard-won local-LLM stack rules, next steps |
| [`ITERATION-M-PLAN.md`](ITERATION-M-PLAN.md) | The next-iteration plan: engineering fixes (§5.7 defects, with gates) + design-research questions + pilot sequencing |
| [`llmind-python/BACKEND.md`](llmind-python/BACKEND.md) | API endpoints, env vars, architecture, CLI commands, error patterns |
| [`llmind-web/FRONTEND.md`](llmind-web/FRONTEND.md) | Frontend architecture, scripts, feature flows, component map |
| [`llmind-web/ZUSTAND.md`](llmind-web/ZUSTAND.md) | Zustand store shape, actions, persistence |
| [`llmind-web/REACT-QUERY.md`](llmind-web/REACT-QUERY.md) | React Query hooks — queries and mutations used in this project |

---

## Quick Start

### Backend
```bash
cd llmind-python
uv sync
uv run fastapi dev backend/main.py        # → http://localhost:8000
uv run python -c "from config import settings; print(settings)"  # verify env
```

### Frontend
```bash
cd llmind-web
bun install
bun dev                                    # → http://localhost:3000
```

---

## API connection (direct, not proxied)

The frontend calls the backend **directly**, not through the Next.js rewrite proxy.
`src/lib/api-client.ts` sets `baseURL = NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000'`,
and the backend enables CORS (`backend/main.py`).

**Why:** the Next.js dev `rewrites()` proxy does not deliver responses for
long-running upstream requests (local LLM generation can take 50s+) — the backend
returns 200 but the browser never receives it, leaving the UI stuck. A direct
connection handles long requests reliably. The `next.config.ts` rewrite remains as
a fallback (used only if `NEXT_PUBLIC_API_BASE_URL` is set to an empty string).
`BACKEND_URL` still controls the (now-fallback) proxy target.

---

## Critical: OpenAI Structured Outputs Schema Rules

`client.beta.chat.completions.parse(response_format=Model)` enforces strict JSON Schema rules Pydantic does not apply by default.

| Forbidden pattern | Why it fails | Fix |
|---|---|---|
| `dict[str, Any]` / `dict[str, str]` | Generates `additionalProperties` — forbidden | Replace with `list[SomeModel]` |
| Fields with `default` or `default_factory` | Omitted from `required` array | Remove defaults |
| `Optional` without `null` in enum | Schema nullability mismatch | Use `str \| None` |

**Rule:** Every Pydantic model used as `response_format=` must have all fields in `required` and no `additionalProperties`. Inspect with `Model.model_json_schema()`.

**The 502 masks the real error.** Always check `e.__cause__` or uvicorn logs for the original `openai.BadRequestError` before assuming infrastructure failure.

---

## Regenerate Frontend Types

Run after any backend request/response model change:
```bash
cd llmind-web
bunx openapi-typescript http://localhost:8000/openapi.json -o src/types/openapi.ts
```
