# Design Space — Critique & Iteration Plan

**Status:** ✅ **Implemented (Iterations A–D)** — verified against the live local stack
(35/35 offline tests; live generate-at end-to-end PASS; browser-verified).
**Scope:** Critique of how the prior implementation reflected the *design space* concept, followed by the iteration plan that was then executed.
**Companion docs:** [DESIGN-SPACE-VIZ.md](DESIGN-SPACE-VIZ.md) (original plan & status), [DESIGN-SPACE-TESTING.md](DESIGN-SPACE-TESTING.md).

## Implementation status (what was built)

| Iteration | Status | Notes / deviations from the plan |
|---|---|---|
| **A — honest, legible, durable substrate** | ✅ | Corpus glyphs inspectable (diamonds → Related Projects detail); trustworthiness at fit time (**0.760**, k=15) in the legend; per-node placement confidence (true-vs-2D k-NN Jaccard, dashed < 0.1 — calibrated against corpus self-confidence ≈ 0.25); store v2 persists the whole exploration; provenance recorded + shown as clickable seed chips. `project-calibrate` CLI added (run pending — needs embed server at the time). |
| **B — genuinely spatial generation** | ✅ | `GENERATE_AT_PROMPT` (v2) with gap instruction + nearby-options dedup; bracket seeding (max-min anchors + true-metric deepening, `SEED_STRATEGY` flag for A/B); `desc` required on generated options end-to-end; parent aspect derived backend-side from the click (Q-B4: coords sent in request, as recommended); per-node drift + `generate_log.jsonl`. Live test: derived parent correct, all descs present, one option landed 0.085 from the click. |
| **C — designs as first-class points** | ✅ | Candidates (per-taxonomy, Q-C1 as recommended) composed via Context-panel "Choose" (instead of an alt-click mode — more discoverable); star glyphs at the composition's embedding; `POST /api/corpus/similar` precedents (live: ~81% matches); compare dialog (choices, per-candidate precedents, 2D distances with fidelity caveat); reject-with-reason + reopen, dimmed in both views; markdown export of the full exploration record. |
| **D — identity & robustness** | ✅ (D1 scoped) | Collision badges + chooser popover; hover decoupled from the 2,304-dot lattice rebuild; surface query gated on view; rename → coord invalidation + re-locate; selection carries `nodeId` (exact identity; full string-identity removal deferred); typed `/surface` response + `api-aliases.ts` (regen-safe types); client-side generation cancel; `corpus_similarity` domain notice on taxonomy generation (Q-A1: heat fades in on zoom-out, as recommended). |

**Calibration result** (`project-calibrate`, 209 projects, name-only text):
median displacement **0.336 (16.1 cells at R=48)**, p90 0.72. Short-text placement
is neighbourhood-level, not position-level — empirically confirming weakness T5 and
justifying the confidence dashes, drift reporting, and required option descriptions.

Remaining (deliberately deferred): full id-based selection refactor (D1 complete form),
backend job cancellation (`DELETE /api/jobs/{id}`), and multi-corpus support.

---

## Part 1 — What the implementation gets right

The system currently holds **two parallel formalizations** of "design space":

1. **Morphological taxonomy** (the mind map): Aspects = dimensions, Options = alternatives per dimension. This matches the classical design-space tradition (Zwicky's morphological charts; MacLean et al.'s Design Space Analysis). The system's own prompt defines it this way: *"a conceptual space, which encompasses the creativity constraints that govern what the outcome of the design process might (and might not) be"* (`utils/prompts.py`).
2. **Empirical similarity landscape** (the Design Space view): a frozen 2D UMAP projection of 209 real-project embeddings, with taxonomy nodes projected into the same space.

Several decisions are genuinely sound and should be preserved:

| Decision | Why it's right |
|---|---|
| **Frozen projection** (fit once, persist, `.transform()` forever — `pipeline/projection.py`) | Coordinate stability is the precondition for any spatial mental model. Persisted bounds + dim guard + mtime caching are well executed. |
| **One identity, two views** (shared node ids, shared selection between map and space) | The "two sides of one coin" framing is the strongest conceptual idea in the project. |
| **Lattice as presentation only** (invariant 4: quantization never mutates true coords) | Keeps the data model honest while making the surface clickable. |
| **Anchor-then-expand seeding** (`seed_corpus`: nearest 2D anchor → expand in original 768-d metric) | A thoughtful workaround for UMAP's local-good/global-bad distortion profile. |
| **Async job pattern** (`backend/jobs.py` + polling) | Correct response to 50–80s local-LLM latency; keeps requests short. |
| **Embedding-dim guard** in `locate_nodes` | Fails loudly on the catastrophic case (runtime model ≠ fit model). |

---

## Part 2 — Weaknesses

### 2a. Conceptual / design weaknesses

#### C1. A point in the space is not a design (the deepest issue)
In the morphological formalization, a *point in the design space* is a **combination** — one option chosen per aspect. The current surface instead plots **individual options** and **whole projects** as the same kind of dot in one text-embedding manifold. The central object of design-space exploration — a candidate design, i.e. a configuration — has **no representation anywhere** in the system. A designer cannot:
- compose a design by selecting options across aspects,
- see where that *composition* sits relative to real precedents,
- compare two candidate designs.

Consequence: the "space" is a similarity map of mixed-granularity text snippets (2-word option labels next to paragraph-length project descriptions), not the combinatorial space the taxonomy defines.

#### C2. "Empty = unexplored territory" is mostly a projection artifact
The core affordance (click an empty cell → generate "in the gap") rests on the claim that emptiness means unexplored design territory ([DESIGN-SPACE-VIZ.md](DESIGN-SPACE-VIZ.md) §1). In practice:
- 209 corpus points over 48×48 = 2,304 cells → **only ~8% of cells are occupied** (measured: 194/2304). 92% of the surface is "empty" from sampling sparsity alone.
- UMAP does **not** preserve density or global distances; the size of gaps between clusters is largely a layout artifact (documented by the UMAP authors themselves).
- The original plan's mitigation — a **trustworthiness score** surfaced in the UI (VIZ §8) — was never implemented. The UI presents the layout with full visual confidence.

#### C3. Generation is not actually conditioned on location
`generate_at` (`backend/projection/service.py`) reuses the generic aspect-exploration prompt (`USER_PROMPT_TEMPLATE`) unchanged. The clicked coordinate never reaches the LLM; the *only* spatial influence is which 5 related projects are injected. The planned instruction — *"propose a node that sits between these neighbors yet is distinct from them — fill the gap"* (VIZ §3.3, step 3) — **was never implemented**. Worse, the seeds are the anchor project's nearest neighbors **in the original metric**, i.e. a tight cluster *around one existing project*, not a set bracketing the gap. Clicking an empty cell therefore effectively means *"generate more options like the project nearest my click."*

The coordinate drift that the trail/discovered-cell UX papers over (generated nodes landing far from the click) is the direct symptom: nodes land wherever their topic text embeds, and nothing in the pipeline pulls that toward the click.

#### C4. Two inconsistent notions of "what is near here"
For one click, two different proximity computations decide two different things:
- **Parent aspect** (frontend, `page.tsx handleGenerateAt`): aspect of the nearest *located taxonomy dot* in 2D — over however few nodes happen to have coords.
- **Seed projects** (backend, `seed_corpus`): nearest *corpus* anchor, expanded in 768-d.

These can disagree arbitrarily: options get attached to an aspect that has nothing to do with the seed projects that inspired them.

#### C5. The corpus — the evidence layer — is invisible
`surface.json` ships every corpus point **including names**, but the renderer never draws them: `surface.points` is unused in `design-space-surface.tsx`; only the anonymous density heat is shown. Hover says "3 nearby projects" with no names; clicking *generates* instead of *inspecting*. For a tool whose framing is "ideas positioned within the space of real work" (VIZ §1), the real work cannot be browsed, identified, or opened from the space view at all. (Q2 in the original plan deferred project dots — that deferral costs the tool its grounding.)

#### C6. Exploration = expansion only; no narrowing, no commitment, no rationale
Design-space exploration in both the research literature and design practice is generate **and** prune: choosing options, excluding alternatives, recording why. The system's own definition emphasizes constraints ("what the outcome might *and might not* be"), yet the UI can only ever grow the tree. There is no way to mark an option chosen or rejected, no constraint propagation, no captured rationale (QOC-style), and no exportable record of the exploration. The "discovered cells" set is the only exploration history, and it is session-bound (see T1).

#### C7. Provenance is discarded
`generate_at` returns `seed_neighbours` and `target`, and the frontend types them (`GenerateAtResponse`), but `page.tsx` drops them. Once nodes appear, nothing records which precedents seeded which idea, with which model and prompt. For a research instrument, this is the cheapest high-value data being thrown away.

#### C8. Domain pinning is silent
A taxonomy can be generated for *any* project overview, but the corpus/space is always media architecture (and `SYSTEM_PROMPT["project"]` hard-codes the Aarhus 2017 brief). A user exploring an unrelated domain gets a meaningless background with no warning.

### 2b. Technical weaknesses

#### T1. Exploration state is volatile (contradicts the original plan)
`nodes`, `coords`, `discoveredMap`, and trails live in **page-local React state** (`page.tsx`). The Zustand store persists only `taxonomy` + selection context (`mindmap-store.ts` `partialize`). Therefore on refresh:
- **all generated nodes are lost** (tree rebuilt from `taxonomy`),
- all coordinates are lost and every node is **re-embedded** next session,
- all discovered cells / trails are lost.

VIZ §3.4 specified `coords` + selection in the shared store as the single source of truth; FRONTEND.md still describes the store as authoritative. The implementation diverged.

#### T2. String-based identity
Selection is `{topic, lineage}` and tree lookups match **topic strings** (`findNodeByLineage`, the surface's match-by-topic with "lineage drift" fallback). Duplicate topics under different branches are ambiguous by construction. Mind-elixir allows manual node renaming (`onDataChange`), which silently breaks: (a) the node's coordinate (embedding of the old text, never re-located because `attemptedRef` already contains the id), and (b) descriptions, which are keyed **by topic** (`descriptionByTopic`).

#### T3. Generated options carry no description
`NodeGenerationPayload` is `{parent_id, options: [{id, topic}]}` — no `desc`, even though the `Taxonomy` schema has `desc` on every option *specifically for embedding-based retrieval* (its own docstring). Generated nodes are therefore located by their topic alone (often 1–3 words) — the weakest possible text for placement (see T5) — and degrade later retrieval.

#### T4. Cell collisions make nodes invisible
`occupied` in `design-space-surface.tsx` is first-wins per cell; any second node snapped to the same cell is **unclickable and invisible**. Semantically similar options embed similarly, so collisions are the *expected* case as a branch grows. VIZ §3.2 specified a count badge + disambiguation list; not implemented.

#### T5. Out-of-distribution inputs to the frozen transform
The reducer was fit on paragraph-length project descriptions. Locate inputs are short labels (topic, or topic + one-line desc). UMAP `.transform()` on out-of-distribution text embeddings tends to collapse points toward dense regions, making node placement systematically less trustworthy than corpus placement — and nothing measures or displays this (no calibration was ever run; relates to C2).

#### T6. Smaller frontend issues
- `baseDots` (2,304 circles, each with handlers) is rebuilt on **every hover change** (`hover` is in the `useMemo` deps).
- `useSurfaceQuery(true)` always fetches, even when the space view is never opened.
- Multiple generations in the same cell overwrite the previous trail (`discoveredMap.set(cellKey, line)`).
- `runJob` supports `AbortSignal`, but the page never passes one — generation can't be cancelled.

#### T7. Jobs are process-local, single-process only
Fine for the prototype (documented), but no cancellation, no progress states, and lost on backend restart while the client keeps polling until timeout.

#### T8. Hand-written API types will drift
`features/design-space/types.ts` duplicates payload shapes by hand. The projection router returns untyped `dict[str, Any]` for `/surface` and `/generate-at`, so even regenerating `openapi.ts` would not fold them in — the planned consolidation can't happen until the router declares response models.

---

## Part 3 — Iteration plan

Four iterations, each independently shippable, ordered so that each one makes the next more valuable. Within each: goal → justification → technical changes → acceptance criteria.

---

### Iteration A — An honest, legible, durable substrate

**Goal:** the space view tells the truth (fidelity shown), shows its evidence (corpus browsable), and nothing the designer does is lost (persistence + provenance).
**Justification:** fixes C2, C5, C7, T1 — all prerequisites for trusting anything generated *from* the space. No ML or prompt changes; lowest risk, highest immediate value for designers.

#### A1. Render and open corpus projects (fixes C5)
- **Renderer** (`design-space-surface.tsx`): draw `surface.points` as small distinct glyphs (e.g. 0.18-cell radius, muted amber, square or diamond to distinguish from lattice/node circles) *on top of* the density shading. Hover → tooltip with project `name`. Click → `onSelectProject(id)` callback.
- **Page**: clicking a corpus point opens that project in the existing Related Projects panel as a detail view. Data: extend `SimpleProjectPanel` to accept a `focusProject`, fetched from a new endpoint.
- **Backend**: `GET /api/corpus/projects/{id}` → serves the record from the existing meta sidecar (`_load_index_meta` in `backend/projection/service.py` already loads `local_index.npz.meta.json`). Response model: `{id, Name, Descriptions, Details, Image}`. Move `_load_index_meta` to a small shared module (e.g. `backend/corpus.py`) so both routers use it.
- **Interaction rule** (to resolve the inspect-vs-generate conflict): corpus glyphs *inspect*; lattice dots *generate*. Update hover copy and the legend accordingly.

#### A2. Fidelity reporting (fixes C2/T5 honesty)
- **At fit time** (`database_pipeline.py project`): compute `sklearn.manifold.trustworthiness(X_highdim, coords, n_neighbors=15)` and store it in `surface.meta.trustworthiness`. Cheap (one-time, corpus-sized).
- **Per-located node** (`locate_nodes`): compute a confidence score = Jaccard overlap between the node's k-NN among corpus vectors in the **original** 768-d metric (cosine; corpus vectors already cached by `_load_corpus_vectors`) and its k-NN among corpus points in **2D** (k=10). Add `confidence: float` to `LocatedPoint`.
- **UI**: legend line "layout fidelity: 0.87 (trustworthiness, k=15)"; node dots with `confidence < 0.3` get a dashed stroke and a tooltip note "placement approximate".
- **Calibration script** (one-off, informs B): for each corpus item, locate it by its *name-only* text and report median displacement from its true coordinate. This quantifies the short-text register problem and tells us how much weight to give node placements.

#### A3. Persist the exploration (fixes T1)
- Move into the persisted Zustand store (version bump to 2 with migration): `nodes` (the working tree), `coords`, `discovered` (serialize the Map as `Array<[cellKey, GenerationTrail]>`), and `provenance` (A4). Keep `attemptedRef` session-local (it's a retry guard, not state).
- `page.tsx` becomes a subscriber: the `useEffect` that rebuilds from `taxonomy` runs only when taxonomy *changes* (compare a stored taxonomy hash), not on every mount — otherwise reload would still wipe generated nodes.
- This also realizes VIZ §3.4 as originally specified and makes FRONTEND.md/ZUSTAND.md true again (update both).

#### A4. Provenance (fixes C7)
- Store per generated node: `{seedProjectIds, target: {x,y}, mode, model, reasoningEffort, createdAt, source: 'generate-at' | 'generate-nodes'}` in `provenance: Record<nodeId, Provenance>`.
- The data already exists in `GenerateAtResponse.seed_neighbours` / `target`; for `generate-nodes`, the backend already returns `related_projects` — include their ids.
- **UI**: selecting a generated node shows "Seeded by: ⟨project names⟩" in the Context panel; each name clickable → A1's project detail.

**Acceptance:** reload mid-exploration → identical tree, dots, discovered cells. Corpus project openable from the space. Legend shows trustworthiness. Every generated node can answer "where did you come from?"

---

### Iteration B — Genuinely spatial generation

**Goal:** "generate here" means *here*: the prompt knows about the location, the seeds bracket the gap, and we measure whether it works.
**Justification:** fixes C3, C4, T3. This is the payoff interaction of the whole design-space concept; currently it is aspect-exploration wearing a spatial costume.

#### B1. A dedicated `GENERATE_AT_PROMPT` (fixes C3)
New template in `utils/prompts.py`, used only by `generate_at`:
- States the spatial intent explicitly: *"The designer pointed at an unoccupied region of the design space. The following real projects surround that region: …"*
- The gap instruction from VIZ §3.3: *"Propose options that conceptually sit between these neighbors yet are distinct from every one of them — fill the gap rather than imitating the nearest project."*
- Includes the **nearby existing taxonomy options** (any located node within radius r=0.12 of the click) under "ALREADY EXPLORED NEARBY — do not duplicate".
- Requires output `{parent_id, options: [{id, topic, desc}]}` — desc mandatory (B3).

#### B2. Bracketing seeds instead of anchor-cluster (fixes C3)
Replace `seed_corpus`'s expand-around-one-anchor with **diverse bracketing**, `seed_corpus_v2`:
1. Take the 2k nearest corpus points to the click in 2D.
2. Greedy max-min selection (farthest-point, already implemented in `pipeline/ml.py: select_farthest`) of m=3 anchors among them — anchors on *different sides* of the gap.
3. For each anchor, add its single nearest original-metric neighbor (from `_load_corpus_vectors`) for semantic depth → up to k seeds total, deduplicated.
4. Keep the current behavior as a fallback when the corpus is tiny.
- *Optional experiment behind a flag:* blend in `ProjectionModel.invert(x, y)` — rank corpus by cosine to the inverted vector and merge with the 2D ranking. `invert()` exists and is currently dead code; the calibration script (A2) will tell us if it's usable.

#### B3. Descriptions on generated options (fixes T3)
- `NodeOption` gains `desc: str` (vLLM strict schema: all fields required — compliant since there's no default needed; for the OpenAI structured-output path this also satisfies the no-defaults rule, see CLAUDE.md).
- `generate_at` locates new nodes by `f"{topic}. {desc}"` (same composition as `nodesToLocateItems`).
- Frontend: store descriptions in a new id-keyed map (`descriptionById`) merged from generation responses; the Context panel and related-projects query read from it. (Full topic→id key migration is D2; here we only *add* the id-keyed path for generated nodes.)
- Regenerate `openapi.ts` after the model change.

#### B4. Consistent parenting (fixes C4)
Derive the parent aspect on the **backend** from the same spatial context as the seeds: among located taxonomy nodes passed in the request, pick the aspect whose *options' mean coordinate* is nearest the click; fall back to the frontend-provided focus node. Return `parent_id` (it already does); delete the frontend nearest-dot heuristic in `handleGenerateAt`. One notion of "near here", owned by one side. Requires the request to carry node coords: add optional `coords: [{node_id, x, y}]` to `GenerateAtRequest`.

#### B5. Measure drift (research instrumentation)
- Response gains `drift: [{node_id, distance}]` (Euclidean click→landed, in [0,1] space).
- UI: show mean drift subtly in the trail tooltip ("landed 0.21 away").
- Backend appends one JSONL line per generation to `data/projection/generate_log.jsonl`: `{ts, target, seeds, prompt_version, model, nodes: [{topic, x, y, drift}]}`. This is the evaluation dataset for comparing the old vs new prompt/seeding — a concrete before/after for the thesis write-up.

**Acceptance:** the prompt visibly references the location and neighbors; seeds come from ≥2 distinct directions around the click; mean drift (logged) decreases vs the pre-B baseline on the same clicks; every generated option has a desc.

---

### Iteration C — Designs as first-class points (configurations)

**Goal:** represent the actual unit of design-space exploration — a candidate design = one option per aspect — and let designers compose, place, compare, and prune.
**Justification:** fixes C1 and C6. This is what turns the tool from a taxonomy browser into a design-space *explorer*. Scoped to stay tractable.

#### C1. Candidate model + composition UI
- Store slice: `candidates: Record<candidateId, {id, name, choices: Record<aspectNodeId, optionNodeId>, note, createdAt}>` (persisted).
- UI: a "Compose" mode — in either view, clicking an option toggles it as the chosen option for its aspect (radio semantics per aspect); a small floating Candidate panel shows current choices, allows naming/saving. Multiple saved candidates, one active.

#### C2. Candidates placed in the space
- Compose the candidate's text: `"{root topic}. " + join(f"{aspect topic}: {option topic} — {option desc}")` and send through the existing `/locate` (no backend change needed — it's just another `{node_id, text}` item with a `cand:` id prefix).
- Render as a distinct glyph (star, larger, branch-neutral color). The active candidate also gets its **nearest real precedents**: reuse the original-metric search (`local_store.search`) via a new `POST /api/corpus/similar {text, k}` → list shown in the Related Projects panel as "Closest precedents to this design". This is the single most useful grounding feature for a designer: *"my current design as a whole is most like these 5 real projects."*

#### C3. Compare candidates
A simple side-by-side table (dialog): rows = aspects, columns = candidates, cells = chosen option; below, each candidate's top-3 precedents and the pairwise 768-d cosine distance between candidate embeddings ("how different are my two directions really?"). All data already client-side or available via `/corpus/similar`.

#### C4. Pruning and rationale (QOC-lite)
- Option states: `open | chosen-in-⟨candidate⟩ | rejected`, with an optional one-line reason on rejection. Store: `optionState: Record<nodeId, {state, reason?}>`.
- Rejected options render dimmed/struck in both views (the mind map wrapper already supports style hooks via mind-elixir node data).
- **Export**: "Export exploration" button → markdown file: taxonomy with states and reasons, candidates with choices and precedents, generation provenance (A4). This is the design-rationale artifact the session produces — the thing a designer can put in a journal or appendix.

**Acceptance:** a designer can compose two named candidates, see both as stars among real projects, read each one's closest precedents, reject options with reasons, and export a readable record of all of it.

---

### Iteration D — Identity & robustness cleanups

**Goal:** retire the string-identity debt and the known UI traps so iterations A–C stand on solid ground. (Can be interleaved with B/C; listed last because A–C deliver designer-visible value first, but D1/D2 should land before C if duplicate-topic bugs start biting.)

| # | Change | Fixes | Detail |
|---|---|---|---|
| D1 | **Id-based selection** | T2 | `MindmapSelection` becomes `{nodeId}` with topic/lineage derived via a memoized id-index. `findNodeByLineage`, surface topic-matching, and the mind-elixir wrapper all switch to id lookups. Delete the "lineage drift" fuzzy fallback. |
| D2 | **Id-keyed descriptions + rename handling** | T2, T5 | `descriptionByTopic` → `descriptionById` (built at taxonomy load; generated nodes contribute via B3). On mind-elixir rename (`onDataChange` diff), drop the node's coord + attempted flag so it re-locates with the new text. |
| D3 | **Collision badge** | T4 | Cells with >1 snapped node render the first dot plus a count badge; click opens a small popover listing the co-located nodes (select on click). Implements VIZ §3.2 as specified. |
| D4 | **Typed projection API** | T8 | Declare `response_model` for `/surface` and the job result schema (Pydantic models in `backend/projection/router.py`); regenerate `openapi.ts`; delete `features/design-space/types.ts`. |
| D5 | **Cancellation** | T6/T7 | Wire `AbortSignal` from a cancel button on the spinner through `runJob`; backend `DELETE /api/jobs/{id}` marks the job abandoned (thread still completes — document this — but the UI frees immediately). |
| D6 | **Renderer perf** | T6 | Split the static lattice from hover effects: render base dots once (no `hover` dep), do hover with a single overlay circle positioned from a `pointermove` hit-test (cell math, no per-dot handlers). Gate `useSurfaceQuery` on `view === 'space'` (first open). |
| D7 | **Domain notice** | C8 | When a taxonomy is generated, embed its overview and compare to the corpus centroid (cosine, original metric); if similarity is below a threshold, show a banner: "The background corpus is media architecture — spatial context may not apply to this brief." Cheap honesty; full multi-corpus support stays out of scope. |

---

## Part 4 — Risks & open questions (please decide before implementation)

**Risks**
- *B2/B4 change generation behavior* — keep the JSONL log (B5) from day one so old vs new is comparable; keep old seeding behind a flag for A/B.
- *Store persistence (A3) of a large tree + coords* — localStorage is fine at this scale (hundreds of nodes); if exports grow, move to IndexedDB later, not now.
- *Strict-schema compliance for `desc`* (B3) — vLLM strict mode and OpenAI structured outputs both require all-fields-required / no `additionalProperties`; the new field must have **no default** (see CLAUDE.md rules). Verify with `NodeGenerationPayload.model_json_schema()`.
- *C introduces real product-design questions* (compose-mode interaction) — prototype the interaction with the existing static taxonomy before wiring generation into it.

**Open questions**
1. **Q-A1:** With corpus glyphs added, should the density heat stay (two encodings of the same data), or be dropped in favor of dots only? My recommendation: drop the heat at default zoom, fade it in only when zoomed out.
2. **Q-B4:** Backend-derived parenting needs node coords in the request — acceptable payload growth (~30 bytes/node), or should the backend re-locate instead (extra embed call)? Recommendation: send coords.
3. **Q-C1:** Are candidates per-taxonomy (cleared when a new taxonomy is generated, like coords) or global? Recommendation: per-taxonomy, archived into the export.
4. **Q-priority:** If only two iterations fit the timeline, my recommendation is **A then B** — honesty + real spatial generation are worth more than configurations for validating the core concept; C is the bigger research contribution but builds on both.

---

## Part 5 — Summary table (critique → fix)

| Weakness | Severity | Fixed by |
|---|---|---|
| C1 — no representation of designs/configurations | High (conceptual core) | C1–C3 |
| C2 — emptiness ≈ projection artifact, no fidelity shown | High | A2, D6 legend |
| C3 — generation not location-conditioned | High | B1, B2, B5 |
| C4 — inconsistent "near here" (parent vs seeds) | Medium | B4 |
| C5 — corpus invisible/uninspectable | High | A1 |
| C6 — no pruning/rationale/commitment | High (for "exploration") | C4 |
| C7 — provenance discarded | Medium (cheap fix) | A4 |
| C8 — silent domain pinning | Low | D7 |
| T1 — exploration state lost on refresh | High | A3 |
| T2 — string-based identity | Medium | D1, D2 |
| T3 — generated options lack desc | Medium | B3 |
| T4 — cell collisions hide nodes | Medium | D3 |
| T5 — OOD placement unmeasured | Medium | A2 (calibration) |
| T6 — renderer perf / no cancel / trail overwrite | Low | D5, D6 |
| T7 — in-memory jobs | Low (accepted for prototype) | documented |
| T8 — hand-written API types | Low | D4 |

---

## Part 6 — Post-implementation review (found issues & next iteration candidates)

Reviewed after A–D landed (2026-06-10), with the live stack running.

### Found issues

| # | Issue | Severity | Status |
|---|---|---|---|
| F1 | **Collision chooser never appeared**: the badged dot's click opened the popover, but the event bubbled to the container's click handler which dismisses choosers — React batched both updates to a net no-op, so badged dots felt dead. | High (UX-blocking) | ✅ Fixed: `stopPropagation()` when opening; popover swallows its own clicks; pan dismisses the (viewport-fixed) popover. Verified in-browser. |
| F2 | An option can be **chosen and rejected simultaneously** — `setChoice` ignores `optionState`, `rejectOption` ignores candidates. Display happens to favour "rejected", but the data is contradictory. | Medium | Open (E3) |
| F3 | **Orphaned state on node deletion**: deleting a node in mind-elixir leaves stale entries in `coords`, `provenance`, `optionState`, `descriptionById`, and candidate `choices` (renders as "—" but persists). | Low–Medium | Open (E4) |
| F4 | **Boundary-clipped placements look like positions**: out-of-hull embeddings clip to x/y = 0 or 1 (visible as a pinned column at the surface edge; the live test produced x=0.000 twice). They read as "this idea sits at the edge" when the honest reading is "outside the corpus's range". Clipping also inflates drift stats. | Medium (honesty) | Open (E2) |
| F5 | **Stale docs**: `REACT-QUERY.md` documents none of the five design-space hooks; `LEARN.md` predates the direct-connection change and everything since. | Low | Open (E8) |
| F6 | **Page bloat**: `page.tsx` is ~914 lines (locate sync, candidate placement, provenance, option actions all inline); `design-space-surface.tsx` ~769. | Low (maintainability) | Open (E7) |
| F7 | The placeholder project name is duplicated as a string literal in `handleGenerateNodes` despite the `PLACEHOLDER_PROJECT_NAME` constant. | Trivial | Open (E7) |
| F8 | `generate_log.jsonl` accumulates but has **no analysis tooling** — the promised prompt/seeding A/B comparison still requires manual JSON wrangling. | Medium (research) | Open (E6) |
| F9 | Discovered cells are permanent — no way to clear a stale trail or dismiss an accidental generation's marker. | Low | Open (E5) |
| F10 | No frontend tests at all: store migrations, candidate actions, and `ensureUniqueChildIds` are pure and unit-testable but unguarded. | Medium | Open (E7) |

### Iteration E candidates (ordered by value)

- **E1 — Gap preview before generating (recommended first).** Clicking an empty cell currently *commits* ~60s of LLM time with no preview and marks the cell forever. Replace with progressive disclosure: click → a lightweight popover shows the bracket seed projects and nearby existing ideas (new cheap `POST /api/projection/peek {x,y,k}` — runs `seed_corpus` + `_format_nearby_options`, no LLM) with a "Generate here" button. Designers see *why* this gap is interesting before spending a generation; accidental clicks cost nothing; the spatial RAG becomes legible. This also gives F9 a natural home (preview ≠ discovered).
- **E2 — Honest edges.** Render clipped placements (coordinate exactly 0/1) as open half-markers pinned to the edge with an "outside corpus range" tooltip; report clipped nodes separately in drift logging so they don't pollute the A/B metric.
- **E3 — Choice/rejection semantics.** Rejecting an option clears it from every candidate's choices (confirm if it was chosen); rejected options are un-choosable until reopened. One invariant: an option is never both.
- **E4 — State GC.** On tree changes, prune `coords`/`provenance`/`optionState`/`descriptionById`/candidate choices for ids no longer in the tree (extend `handleNodesChange`'s diff).
- **E5 — Exploration stats.** A small HUD line (cells explored, options per aspect, aspects chosen in the active candidate) + per-aspect coverage in the Candidate panel; "clear discovery" affordance on discovered cells.
- **E6 — Log analysis CLI.** `database_pipeline.py project-log-stats`: mean/median drift and clipped-rate grouped by `prompt_version` × `seed_strategy` — closes the evaluation loop B5 opened.
- **E7 — Refactor + tests.** Extract `useExplorationSync` (locate/attempted) and `useCandidatePlacement` hooks from `page.tsx`; unit tests for store actions/migration and tree utilities; dedupe the placeholder literal.
- **E8 — Docs refresh.** REACT-QUERY.md (five new hooks) and LEARN.md's design-space sections.
- **Later/bigger:** session snapshots (save/compare whole explorations); letting accepted generated ideas join the corpus background as a distinct stratum; multi-corpus support.

---

## Part 7 — Beyond one static 2D view: perspectives & dimensionality

**The question:** the space is one frozen 2D projection. Would 3D, or multiple
"perspectives", enhance discovery?

**Framing.** A projection is a *lens*, and we have measured how much this one
lens hides: trustworthiness 0.76, corpus self-confidence ≈ 0.25, short-text
calibration ≈ 16 cells. No single layout of 768-d data can be faithful. The
discovery win therefore comes less from *adding a dimension* than from
**re-viewing the same items under different organizing principles** — the way a
designer turns a physical object. Stability stays sacred *within* each
perspective (the frozen-projection invariant applies per view); discovery comes
from switching between frozen views with animated transitions (object constancy
keeps the mental map).

### Idea inventory

| # | Idea | What it gives | Cost | Verdict |
|---|---|---|---|---|
| **P1 — Relevance lens** | Select any node/candidate → recolor or fade ALL corpus dots by TRUE cosine similarity to it (one embed + 209 dot products). Generalizes the related-projects highlight from 5 binary marks to a continuous field; the static map becomes query-responsive. | Small (reuse `/api/corpus/similar` with k=209, or add scores-for-all endpoint) | **Do first** |
| **P6 — Encoding perspectives** | Same geometry, different paint: color-by-cluster (labels already sit unused in `surface.json`!), color-by-similarity (P1), later color-by-metadata. A simple "color by …" toggle. | Trivial | Do with P1 |
| **P3 — Semantic-axis perspectives** (taxonomy-as-axes) | Let the designer pick the axes from their OWN taxonomy: X = an option-pole pair within one aspect (score = sim(project, pole A) − sim(project, pole B)), Y = another. The scatterplot becomes *interpretable* ("x: passive → co-creative; y: permanent → temporary"). This finally aligns the view with the morphological design space (original critique C1): an **empty region now means "no real project is both X and Y" — a genuine morphological gap**, far stronger than UMAP whitespace. Axis views are exact and deterministic (no UMAP, no distortion — a major honesty win), recomputable for any axis pair, and `generate-at` conditioning becomes verbal and precise ("propose options that are highly interactive but temporary"). | Medium: `POST /api/projection/axes` (embed poles, score corpus + nodes, cache per pair); X/Y axis pickers; per-view coords in the store | **The headline next step** |
| **P2 — Ensemble UMAP views** | 2–3 frozen UMAPs with different `n_neighbors` (local k=5 / default 15 / global k=50) or seeds, with a switcher + animated transitions. Items that stay together across views are robust structure; items that scatter are artifacts → a per-point **stability score** that strengthens the existing confidence metric. | Medium: `data/projection/views/{name}/` artifacts, `?view=` param, per-view coords | Phase 3 |
| **P4 — Local re-projection lens** ("semantic zoom") | Select a dense region → re-fit UMAP on just those points → a transient, high-fidelity local view (clearly badged as a lens, not the canonical space). | Medium | Phase 3, if dense clusters demand |
| **P5 — True 3D** | The pipeline already supports `--dims 3` end-to-end except the renderer. But: clicking *empty space* — the core affordance — is ill-posed in 3D (a ray contains infinitely many empty points); the lattice becomes 48³ ≈ 110k voxels for 209 points, so "emptiness" loses all signal; occlusion + weak depth perception on flat screens; new dependency (three.js). | High | **Not as the canvas.** If pursued: fit 3D, render as three stable orthogonal 2D slices (xy/xz/yz switcher — clicks stay 2D), or encode z as halo/size with a z-slider. Decision gate: first measure `--dims 3` trustworthiness (one CLI run); only proceed if the fidelity gain is large. |

### Proposed way forward

1. **Phase 1 (small):** P1 relevance lens + P6 color-by-cluster toggle.
   *(3D decision datapoint already measured: an in-memory 3D fit scores
   trustworthiness **0.790 vs 0.760** for 2D — a +0.03 gain that does not justify
   losing the click-empty-space interaction. 3D is shelved unless the corpus
   grows substantially.)*
2. **Phase 2 (the win):** P3 semantic axes MVP — axis pickers (two option poles
   per axis), exact bipolar scores, lattice + generate-at reused with verbal
   axis conditioning in the prompt. Store: `coords` keyed per view id
   (persist migration v3); UMAP view remains the default "similarity" perspective.
3. **Phase 3:** P2 ensemble views + stability score; P4 local lens as needed.
4. **3D gate:** revisit only if the measured 3D fidelity gain is decisive and a
   concrete task (e.g. collision disambiguation) demands it.

Cross-cutting: every perspective declares its own fidelity in the legend (UMAP
views: trustworthiness; axis views: "exact by construction"); animated dot
transitions between views; `generate_log.jsonl` gains a `view` field so drift is
comparable per perspective.
