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

**One owner per topic** (2026-07-03 consolidation). Docs are either **LIVE**
(kept current — edit the owner, never a copy) or **ARCHIVED** (historical record —
add dated banners, never rewrite the body).

### Live — single source of truth per topic

| Doc | Owns |
|-----|------|
| [`PROJECT-REPORT.md`](PROJECT-REPORT.md) | The master report: what the system is + why (§1.3 justifications), the research argument, iterations synthesis, §5.7 code-verification audit + known defects |
| [`FEATURE-ATLAS.md`](FEATURE-ATLAS.md) | Writing companion & decision aid: every implemented feature — purpose (design-process stage), mechanism + interconnection diagrams, check recipes, intended-vs-implemented deltas, critique, and the §10 question bank |
| [`llmind-python/BACKEND.md`](llmind-python/BACKEND.md) | **Everything backend**: API endpoints, env vars (deployed values), architecture, projection/annotation subsystems, error patterns, and the full data-pipeline & CLI reference |
| [`llmind-web/FRONTEND.md`](llmind-web/FRONTEND.md) | Frontend architecture, component map, feature flows, the locked design language, study mode |
| [`llmind-web/ZUSTAND.md`](llmind-web/ZUSTAND.md) | Store shape, actions, persistence + session-load trust boundary |
| [`llmind-web/REACT-QUERY.md`](llmind-web/REACT-QUERY.md) | Every query/mutation hook |
| [`PROCESS.md`](PROCESS.md) | Session handoff: execution state + the hard-won local-LLM stack rules (§2) |
| [`ITERATION-M-PLAN.md`](ITERATION-M-PLAN.md) | Next-iteration plan: engineering (Wave 1 shipped) + design-research rationale + pilot sequencing |
| [`documentations/USER-TESTING-PLAN.md`](documentations/USER-TESTING-PLAN.md) + [`USER-FLOWS.md`](documentations/USER-FLOWS.md) | **The study protocol SSOT** (incl. the §9 probes) + task templates |
| [`documentations/LEARN.md`](documentations/LEARN.md) | Onboarding: mental models per layer (inventories live with the owners above) |
| Thin hubs/launchers | root [`README.md`](README.md), [`llmind-python/CLAUDE.md`](llmind-python/CLAUDE.md), the two subsystem `README.md`s — pointers only |

### Archived — provenance, banner-only

`documentations/`: `DESIGN-SPACE-VIZ.md`, `DESIGN-SPACE-ITERATION-PLAN.md` (the
full Parts 1–13 record), `DESIGN-SPACE-PERSPECTIVES-PLAN.md`,
`DESIGN-SPACE-TESTING.md`, `PROJECT_DEV.md` (superseded). Bodies stay unmodified;
corrections go in dated banners at the top.

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
