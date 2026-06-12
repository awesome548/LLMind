# React Query — Hooks Reference

TanStack Query v5. Mindmap hooks live in `src/features/mindmap/hooks/`,
design-space hooks in `src/features/design-space/hooks/`.

Long-running LLM endpoints (generate-nodes, generate-at) return `202 {job_id}`
and are polled via `src/lib/run-job.ts` (1.5 s interval, 5 min timeout,
`AbortSignal` support) — the mutation resolves with the job result.

## Queries

| Hook | Endpoint | Notes |
|---|---|---|
| `useRelatedProjectsQuery` | `POST /api/related-projects/search` | Auto-fires on selection change; key = full request |
| `useSurfaceQuery(enabled)` | `GET /api/projection/surface` | Gated on the space view; cached forever (`staleTime: Infinity`) |
| `useCorpusProjectQuery(id)` | `GET /api/corpus/projects/{id}` | Corpus glyph / provenance-chip detail; cached forever |
| `useCandidatePrecedentsQuery(text)` / `useManyCandidatePrecedents(texts)` | `POST /api/corpus/similar` | True-cosine precedents for composed candidates; 5 min stale |
| `useRelevanceQuery(text)` | `POST /api/corpus/relevance` | Relevance-lens scores for ALL corpus projects, normalized client-side; 5 min stale |
| `useAxesQuery(params)` | `POST /api/projection/axes` | Perspectives "Cross two metrics" tab; key = poles + items; 10 min stale |
| `useAlignmentQuery(params)` | `POST /api/candidates/alignment` | Brief↔choices agreement + per-aspect leans (Examine headline/consistency strips); 5 min stale |
| `useMetricsQuery(params)` | `POST /api/projection/metrics` | Corpus + brief scores along the strip metrics; response order = request order; 10 min stale |
| `useAnnotationQuery(options, enabled)` | `POST /api/corpus/annotate` (job) | Corpus annotation (Part 12 A2): per-option counts + receipts; enabled in the Schema and Cross-tab views; key = option texts; `staleTime: Infinity` (server caches per option) |

## Mutations

| Hook | Endpoint | Notes |
|---|---|---|
| `useGenerateTaxonomyMutation` | `POST /api/taxonomy/generate` | Returns `corpus_similarity` for the domain notice |
| `useGenerateNodesMutation` | `POST /api/related-projects/generate-nodes` (job) | Whole-tree context via `flattenMindmapNodes`; options carry `desc` |
| `useGenerateAtMutation` | `POST /api/projection/generate-at` (job) | Sends node coords (backend derives the parent aspect); result carries coords, `drift`, `clipped`, `support`, seed provenance |
| `useLocateNodesMutation` | `POST /api/projection/locate` | Places nodes in the frozen space; points carry `confidence` + `clipped` + `support` (corpus-support percentile) |
| `usePeekMutation` | `POST /api/projection/peek` | Gap preview (no LLM): seeds, nearby ideas, parent aspect |
| `useDraftBriefMutation` | `POST /api/candidates/draft-brief` (job) | LLM-drafts the candidate's brief from its choices — the designer edits the result |
| `useGenerateCellMutation` | `POST /api/corpus/generate-cell` (job) | B2: one concept into an empty cross-tab cell (pole-conditioned, half-matching exemplar seeds); kept results become candidate skeletons |
| `useSteerMutation` | `POST /api/candidates/steer` (job) | B3: one steering move on the brief (metric / toward / away); returns revision + requested-vs-achieved measurement; ALWAYS shown as a veto card |
| `useDraftReflectionMutation` | `POST /api/reflections/draft` (job) | C2: drafts the one-line rationale for an exploration event; the chip opens immediately and the draft fills only if the designer hasn't typed |

Regenerate the OpenAPI types after backend model changes (see FRONTEND.md);
app code imports types from `src/types/api-aliases.ts`, never `openapi.ts`.
