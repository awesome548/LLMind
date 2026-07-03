# PROJECT_DEV.md — Development Log & Rationale

> ⚠️ **SUPERSEDED (2026-07-03 doc consolidation; body below archived unmodified).**
> This early development log is retained for provenance only. Its content is
> superseded by [`../PROJECT-REPORT.md`](../PROJECT-REPORT.md) (the synthesis) and
> [`DESIGN-SPACE-ITERATION-PLAN.md`](DESIGN-SPACE-ITERATION-PLAN.md) (the full
> critique→iteration record). Do not update this file.

Context and justification for the work done building the **Design Space** view and
hardening the surrounding system. Written for future reference: each entry says
*what* changed, *why*, and *how it was verified*. Companion docs:
[`DESIGN-SPACE-VIZ.md`](DESIGN-SPACE-VIZ.md) (plan/spec) and
[`DESIGN-SPACE-TESTING.md`](DESIGN-SPACE-TESTING.md) (test protocol).

---

## 1. What the Design Space is

A second view of the taxonomy, beside the mind map. The taxonomy is projected
onto a 2-D lattice of dots that *is the manifold of the real project corpus*
(fit once with UMAP). Dots: faint **corpus density** (real projects), **empty**
cells (click to generate), and **taxonomy nodes** colored by branch. The mind map
and design space are two views of one shared selection/tree — "two sides of one
coin."

**Core invariants** (load-bearing — most design follows from these):
1. One shared, **frozen** projection: fit once on the corpus, only `.transform()`
   afterward, so coordinates are **stable** across sessions and edits.
2. One identity per node (`node_id`) shared by both views.
3. The lattice is a presentation layer; underlying coords are continuous floats.

---

## 2. Architecture added

**Backend (`llmind-python/`)**
- `pipeline/projection.py` — frozen reducer: `fit_projection`, `ProjectionModel.transform/invert`, grid/density/nearest helpers, surface payload. Pure numpy/sklearn/umap.
- `database_pipeline.py project` — CLI: fit on the local index (or Supabase) → `data/projection/{model.joblib, surface.json}`.
- `backend/projection/{router,service}.py` — `GET /surface`, `POST /locate`, `POST /generate-at`.
- `backend/jobs.py` + `backend/jobs_router.py` — async job store + `GET /api/jobs/{id}` polling.
- `config.py` — `projection_dir`, prompt-budget settings.

**Frontend (`llmind-web/`)**
- `src/components/design-space/design-space-surface.tsx` — SVG lattice renderer.
- `src/features/design-space/` — types + hooks (`use-surface-query`, `use-locate-nodes`, `use-generate-at-mutation`).
- `src/lib/run-job.ts` — generic job poller.
- `src/lib/node-colors.ts` — shared branch colors (both views).
- `src/app/mindmap/page.tsx` — orchestration (view toggle, shared selection, generation handlers).

---

## 3. Change log with justifications

### 3.1 Frozen projection + surface (foundation)
**What:** Fit PCA→UMAP once on the corpus; persist the reducer; `.transform()` new nodes into the same frame. **Why:** UMAP is non-parametric/stochastic — re-fitting per request would relayout every dot, breaking the "two sides of one coin" mapping. **Verified:** reload + re-transform reproduces persisted coords exactly; unseen vectors clip into `[0,1]`.

### 3.2 API connection: direct, not proxied (CORS)
**What:** Frontend calls the backend directly (`NEXT_PUBLIC_API_BASE_URL`, default `http://localhost:8000`) with CORS enabled, instead of the Next.js `rewrites()` proxy. **Why (root cause):** the Next dev proxy **does not deliver responses for long (~50s+) requests** — the backend returned 200 but the browser never received it, leaving the UI stuck forever. Direct curl returned in 50s; through the proxy it never returned (curl hit its 200s limit). This affected *every* long LLM call. **Verified:** browser direct fetch resolved in 33–50s; CORS preflight 200.

### 3.3 Mind-map filled height
**What:** `SimpleMindMap` root got `h-full` (was `min-h-[520px]` only). **Why:** without `h-full` it collapsed to 520px and sat at the top, leaving a large dead area. **Verified:** container measured 520→878px (full viewport).

### 3.4 Zustand persistence migration
**What:** Added `version: 1` + `migrate` to the `mindmap-store` persist config. **Why:** a stale persisted `taxonomy` (a minimal placeholder from an older build) was pinning the mind map to a wrong schema *and* suppressing the generate-taxonomy prompt (the dialog only auto-opens when `taxonomy` is null). It was **localStorage**, not the DB — which also explained why the embedded preview and the user's browser differed. **Verified:** injected stale v0 state → on reload it's dropped, rich default returns, dialog opens.

### 3.5 Duplicate React key (`n-portable`)
**What:** `ensureUniqueChildIds` remaps generated node ids that collide with existing tree ids; design-space render also dedupes by id. **Why:** node ids are `slugify(name)`, so a generated option matching an existing name produced duplicate ids → broken React keys and a broken id→coord mapping (nodes dropped/duplicated). **Verified:** collision `[portable, portable, fresh]` → `[portable-2, portable-3, fresh]`; real generation produced no warning.

### 3.6 Generate-at 502 + error surfacing
**What:** `/generate-at` derives the generation backend from `settings.vector_store` (local→vLLM) instead of defaulting to OpenAI; 502 detail now includes `__cause__`. **Why:** the frontend hardcoded `mode: 'openai'`, but the stack is full-local with no OpenAI key → 502. The 502 was also masking the real error. **Verified:** with no `mode`, generation returns real nodes.

### 3.7 Context overflow + statelessness
**What:** Bounded the node-generation prompt — focused taxonomy view + per-project description cap (`prompt_max_*` settings). **Why:** the prompt embeds the related projects (full descriptions, ~1,600 chars each) + the whole taxonomy, which grows with every generation, eventually exceeding a 4096 context. **Key clarification:** node generation is **already stateless** — each call sends a fresh system+user message, no accumulation; the overflow was prompt *size*, not conversation history. **Verified:** even with 5 huge descriptions + a 200-node tree, the prompt is ~1,000 tokens.

### 3.8 Structure: generate under the nearest aspect
**What:** Design-space generation attaches new options under the **aspect nearest the clicked dot** (computed from coords), not the currently-selected node. The backend also forces `parent_id = focus_node_id` (a local model's echoed parent is unreliable). **Why:** generating with the *root* selected made option-like content into new top-level aspects (three different colors, broken hierarchy). Spatial attach keeps the 2-level structure and branch color. **Verified:** clicking near "Display Technology" generated options *under* it (Curved/Flexible LED, Micro-LED, …), all parented correctly.

### 3.9 Shared branch colors
**What:** `node-colors.ts` (`nodeColor(branchIndex, depth)`) used by both views. Distinct hue per branch; lighter tint by depth within a branch. **Why:** the same node should be the same color in both views; branches should be clearly distinct, members similar. **Verified:** Display Technology = `hsl(210 …)` in both; options a lighter blue; other branches green/orange/purple/etc.

### 3.10 Async job mode
**What:** `generate-at` and `generate-nodes` return `202 {job_id}` immediately and run on a background thread pool; the client polls `GET /api/jobs/{id}`. **Why:** a 50–80s synchronous request is fragile (proxy/connection timeouts, lost on reload) and gives no progress feedback; short poll requests are robust and drive the spinner. **Verified:** POST returns a job_id in ~0.03s; poll → pending → done.

### 3.11 Target spinners
**What:** A rotating SVG ring around the clicked dot during design-space generation; a spinner over the focused node during mind-map node generation. **Why:** show *where* work is happening, not just *that* it is. **Verified:** ring present on click; node spinner positioned on the focus node.

### 3.12 Zoom / pan / reset
**What:** Wheel-zoom (toward cursor) + drag-pan via window listeners, a Reset-view button. Implemented as a CSS transform on the SVG so data coords + tooltips stay correct. **Why:** the lattice is dense; navigation is needed. **Verified:** scale 1→1.12 on wheel; pan updates transform; reset → identity.

### 3.13 Click interactions fixed (pointer capture)
**What:** Removed `setPointerCapture` from the pan handler; pan now uses window listeners. **Why:** capturing the pointer on the container **redirected `pointerup`/`click` away from the dots**, so dot clicks never fired — selection, related-projects fetch, and cross-view linking all silently failed from the design space (while mind-map→design-space still worked, the tell-tale asymmetry). **Verified:** dot click selects, loads related projects, and highlights the node in the mind map; pan still works and a drag-click is suppressed.

### 3.14 Multi-match glow + range highlight
**What:** Selection draws a radial "range" glow (full color → fading) — and now **one glow per matching dot**, emphasizing the exact one (by lineage) and fainting the rest. **Why (bug):** the old code highlighted the *first* node matching the topic, so two same-named nodes under different branches would glow the *wrong* one. The range glow itself reflects that a node's design-space position is approximate. **Verified:** two same-topic nodes both glow; the clicked one (by lineage) is emphasized; old code emphasized the wrong one.

### 3.15 Option 1 (connector) + Option 3 (faithful seeding) — see §4

### 3.16 Discovered dots + trace + background deselect
**What:** A clicked-and-generated cell becomes a **hollow ring** ("discovered"); the **trace** line(s) from it to the generated nodes appear on completion and again whenever the hollow dot is clicked; clicking empty (non-dot) space **deselects** (and clears the trace). **Why:** ties the often-distant generated placements back to where you clicked, prevents accidental re-generation at the same spot (the original overflow trigger), and makes the spatial relationship inspectable on demand. **Verified:** generate → 1 hollow dot + 6 trace lines; background click → 0 lines + glow cleared; re-click hollow dot → 6 lines again.

---

## 4. The Option-3 investigation (inverse-transform seeding)

This is the most important finding to preserve.

**Problem being solved.** The clicked dot and the generated node often land far apart, because the click only influences *seeding* (which projects/aspect inform the LLM), while *placement* is the generated text's own embedding through a lossy, non-metric 2-D projection. Two improvements were proposed: **(1)** a connector animation tying click→result, and **(3)** *inverse-transform seeding* — invert the clicked 2-D point back to a high-dim vector and seed from the corpus projects nearest it **in the original metric** (more faithful than 2-D-nearest, which UMAP distorts). Theoretically option 3 is the more "conceptual-spaces-faithful" approach.

**What was built for option 3.**
- Re-fit the projection on **L2-normalized vectors with the Euclidean metric** (≡ cosine on unit vectors, but — unlike cosine — it makes `inverse_transform` well-defined). `ProjectionModel.invert(point)` does: `[0,1]` → raw UMAP coords (via stored bounds) → `UMAP.inverse_transform` → `PCA.inverse_transform` → a unit vector.

**The discovery (why pure inverse was abandoned).** Empirically, on this **209-point** corpus, `UMAP.inverse_transform` is **too lossy** to recover the right neighborhood:
- Inverting a point's *own* surface location returned that point at a **median rank of 40/209** (best 12) — it should be ~rank 0.
- Inverse-top-5 vs 2-D-top-5 overlap ≈ **0.1 / 5** (essentially disjoint).
- Dropping PCA made it **worse** (median 131/209), so PCA wasn't the cause — UMAP's inverse interpolation is simply coarse on a small, high-dim dataset.

Pure inverse-seeding would therefore feed the LLM **unrelated** projects — worse than the 2-D seeding it was meant to improve.

**The new solution (anchor-based original-space seeding — `service.seed_corpus`).** Keep option 3's *intent* (faithful, original-metric seeding from the clicked location) without its weak link:
1. **Anchor** at the corpus project nearest the click *in 2-D*. UMAP preserves **local** structure well (its distortion is global), so the nearest 2-D dot is a genuinely nearby real point — a trustworthy anchor.
2. **Expand** the seed set to that anchor's nearest neighbors **in the original 768-d metric** (cosine). These are guaranteed semantically coherent (most-similar to a real anchor), respecting "proximity = similarity" — the conceptual-spaces axiom option 3 was after.

**Why this is correct.** It replaces "reconstruct the embedding at this location" (lossy) with "use a real embedding that *is* at ~this location" (the anchor) — no reconstruction error — then ranks in the faithful metric. **Verified:** clicking near "Depot Boijmans" seeds it + Denmark Pavilion, LED Frieze, Kinetic Façade (coherent facade/media-architecture set), anchor first.

**Status of `invert`.** Retained but unused for retrieval. It becomes viable as the corpus grows (more points → better inverse interpolation); revisit then.

**Option 1 (connector)** shipped as the persistent **trace** (§3.16): the burst from the clicked cell to the landed nodes, shown on completion and re-summonable from the discovered dot.

---

## 5. Known limitations / future work

- **Position vs intuitiveness.** A node's dot is at its *embedding* position, which can be far from the clicked cell. The trace mitigates the confusion honestly rather than faking placement (snap-to-cursor was rejected to keep "position = meaning").
- **Embedding-text faithfulness ("option 5", deferred).** Embedding each node as *title + a summary of its subtree* would make an aspect sit at the center of its options (more faithful), but it makes positions **content-derived and therefore unstable** (a node moves as children are added), conflicting with the stability invariant. Deferred for that reason.
- **Empty ≠ always meaningful.** A blank lattice region can be a real design gap *or* a UMAP layout artifact; generation assumes the former.
- **Whole space is a corpus projection** with the taxonomy overlaid; a purist design space would be spanned by the taxonomy's own aspects as dimensions (large redesign).
- **Jobs are in-memory/process-local** — correct for the single-process dev server; a multi-worker deployment needs a shared store.
- **`openapi.ts`** is stale for the now-async generation endpoints (they return `{job_id}`). The frontend types the *job result* via `runJob<T>`, so it isn't broken; regenerate when convenient.
- **Multimodal** embeddings are out of scope (text-only).

---

## 6. Operational notes

- Run with `.\dev.ps1` (starts backend via uvicorn — **not** `fastapi run`, whose banner emoji crashes the Windows cp1252 console — and the frontend with the right env).
- Re-run `uv run python database_pipeline.py project` after rebuilding the index or changing projection params. The projection is now Euclidean-on-normalized (≈ cosine; layout essentially unchanged from the original cosine fit).
- `/locate` and generation require the embedding/LLM server (LM Studio / vLLM) running; the surface/background does not.
- Generation takes ~50–80s with a local 35B model; the async job + spinner + trace are designed around that latency.
