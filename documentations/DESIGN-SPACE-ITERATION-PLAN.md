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

> **Update (2026-06-11, Iteration G):** shipped per the review's recommended order —
> **E1** gap preview (`POST /api/projection/peek`, popover with seeds/parent-aspect/nearby
> before any LLM time; generation commits from the popover), **E2** clipped honesty
> end-to-end (`transform_with_flags` → `clipped` on locate/generate-at/log → hollow edge
> markers; clipped excluded from drift aggregates), **E3** choose/reject invariant (store-
> enforced both directions + UI warning + pick-flow guard), **E4** state GC on tree edits,
> **E5** exploration stats (UI strip + export + study instrument), **E6** `project-log-stats`
> CLI (first real reading: 5 gens, drift mean 0.314, **47% clipped rate**), session
> save/load JSON + usage counters, tree-utils extraction with a 29-test bun suite
> (backend harness now 45 tests), and the E8 docs refresh. Still open: E7's larger
> page-hook extraction, a "clear discovered cell" affordance (F9), F2.1, and the pilot
> study itself.

## Part 8 — Critical review after Iteration F (2026-06-11)

State at review: A–D + F shipped (lens is now an on/off overlay with a selection/candidate
anchor switcher; candidates fill via "— pick" click-flow in any view; Perspectives view
read-only). E1–E8 and F2.1 remain open. This review asks: where is the project actually
weak now, and what should happen next?

### W1 — Feature accretion is outpacing the workflow (the new top weakness)
The tool now offers 3 views, a lens with 2 anchor sources, 3 ways to fill a candidate
slot, 2 generation paths, and 4 evidence panels. Each is individually justified; together
they have no *spine* — nothing tells a designer what to do first or why. The
similarity/relevance overlap complaint that triggered this iteration was a symptom:
features are accreting faster than the conceptual model is being communicated.
**Way forward:** (a) progressive disclosure — hide Candidate/lens/Perspectives until a
taxonomy exists, hide Compare until 2 candidates; (b) a first-run guided sequence
(generate → explore → reject/choose → compose → compare → export); (c) instrument
feature usage (simple counters in the store) so the next decision about what to cut or
promote is based on what designers actually touch.

### W2 — The evaluation gap is now the project's deepest weakness
The instrumentation exists (drift JSONL with `prompt_version` × `seed_strategy`; honest
fidelity metrics on screen) but nothing analyses it (E6 unbuilt) and no user has been
observed. For a research project, both halves of the contribution are currently
unmeasured: the *technical* claim (bracket seeding + spatial prompt reduce drift) sits
unverified in `generate_log.jsonl`, and the *design* claim (perspectives broaden
exploration) has no behavioural metric. **Way forward:** E6 (log-stats CLI) is an
afternoon; E5 (exploration stats) doubles as the study instrument — cells explored,
aspects covered, candidate diversity (pairwise distance) are all computable from the
store. Then a small within-subjects pilot (map-only vs +space vs full) closes the loop.
E5/E6 should be re-ranked from "polish" to **research-critical**.

### W3 — The corpus ceiling
Everything (UMAP, lens, axes, precedents, seeds) sits on 209 scraped projects in one
domain with one 768-d local model. Sparsity overdetermines "gaps"; scraped text quality
varies; and single-corpus means the approach's generality is untested. **Way forward:** a
second corpus in a different design domain as a generality probe (the pipeline —
`build_local_index` + `project` fit — already supports it); treat per-corpus artifacts as
a first-class concept then (directory per corpus, corpus picker).

### W4 — F introduced its own debts (self-critique)
- **Lens normalization is per-query**: the same red means different absolute cosine for
  different anchors. The legend says "relative", but cross-anchor comparison is the
  natural use. Consider an absolute-scale option (fixed cos 0.3–0.8 domain) or a tiny
  histogram beside the ramp.
- **Axes default poles are arbitrary** (first vs last option). The endpoint could return
  the *most-distant pair* as a suggested default (it has the embeddings in hand).
- **Honest edges (E2) is now inconsistent**: clip-dashing exists in the axes view but
  boundary-pinned dots in the UMAP space still read as positions.
- The pick flow can select a **rejected** option into a candidate (E3's conflict, third
  path now).

### W5 — Sustainability before a study
`page.tsx` ~1,100 lines; 12+ persisted store slices with zero tests (E7); LEARN.md badly
stale for its designer audience (E8); in-memory single-process jobs block any multi-user
deployment; no exploration import (markdown export only) — a study needs full state
save/load (JSON) for capture and crash recovery.

### Recommended order
1. **E6 + E5** — measurement first; run the bracket-vs-anchor / prompt-v1-vs-v2 analysis
   that the log already contains.
2. **E1 gap preview** — still the right interaction fix; design it once and let F2.1
   (generate-in-axes) inherit the same preview pattern later.
3. **Small integrity batch: E2 (surface edges), E3 (choose/reject invariant incl. the
   pick flow), E4 (state GC).**
4. **Session save/load JSON + usage counters** — study enablement.
5. **E7 refactor + store tests, E8 docs** — before the codebase calcifies.
6. **Pilot study** (3–5 designers, within-subjects, E5 metrics as DVs).

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

---

## Part 9 — Iteration H: Placement validity (register gap, corpus support, soft margin)

*Added 2026-06-11, after the first real `project-log-stats` reading showed a 47%
clipped rate among generated ideas.*

### Diagnostics (what was measured before deciding anything)

Three offline/one online checks against the frozen model (`pre_pca=64 → UMAP`,
209 corpus points, 768-d local embeddings):

| # | Check | Result | Implication |
|---|---|---|---|
| 1 | Corpus round-trip through the frozen `.transform()` | **0 / 209 clipped, displacement 0.000** | The transform is faithful for in-distribution inputs; clipping is NOT transform noise |
| 2 | Bounds tightness (fit coords, 5–95 pct span) | x: [0.13, 0.89], y: [0.06, 0.89] | Bounds are not stretched by outliers — padding them wouldn't help |
| 3 | Re-embed all 17 logged generated ideas → top-1 cosine to corpus | **0.47–0.65** vs corpus-internal baseline **mean 0.834, p5 0.770** | *Every* generated idea is far out-of-distribution in 768-d |
| 4 | Same texts: clipped vs in-bounds top-1 cosine | clipped **0.564**, in-bounds **0.587** | The binary `clipped` flag draws a near-arbitrary line — all generated points are equally extrapolated; overshoot beyond the boundary is ≤ ~5% of the span |

**Root cause (register gap):** the corpus was indexed from ~2,700-char project
descriptions (median 2,645); generated ideas are embedded as ~60–150-char
"Topic. One-line desc" strings. Short fragments and long documents occupy
systematically different regions of embedding space — the earlier
`project-calibrate` run (re-locating corpus projects by Name alone → ~16-cell
median displacement) measured the same effect from the other side.

**Reframe:** the space cannot be validly "expanded" without data — validity =
corpus neighbourhood evidence. What we can do: (a) move generated ideas back
into supported territory (H1, H2), and (b) represent the residual extrapolation
honestly (H3, H4). Rejected after diagnosis: re-tuning/swapping UMAP (the
mapping was cleared of blame; a refit breaks the frozen-coordinate invariant),
fit-time register augmentation (risks register islands), k-NN interpolation
placement (fabricates interior locations for genuinely novel ideas).

### H1 — Close the register gap at the source (prompt v3)

Generated descs become 2–4 sentences written like a real project description
(mechanism/material, interaction, audience experience) instead of one line.
`GENERATE_AT_PROMPT_VERSION` 2 → 3 so `project-log-stats` separates the variants;
`USER_PROMPT_TEMPLATE` (generate-nodes) gets the same desc register so mind-map
generated nodes locate identically. Locate text composition stays `topic. desc`.
Risk: LLM-authored blurbs keep some stylistic signature — H1 narrows, H2 closes.

### H2 — Learned register correction (short → long embedding alignment)

The corpus gives 209 paired examples of the same project in both registers.
New `pipeline/register_alignment.py`:

- `build_short_text(name, description, sentences, max_chars)` — short-register
  exemplar mimicking node text ("Name. first sentences of description").
- `fit_register_map(short, long, alphas, folds)` — candidates: translation
  (`W=I, b=mean(long−short)`) and closed-form multi-output ridge over an alpha
  grid; winner picked by k-fold CV mean cosine(mapped, long); refit on all pairs.
  Held-out 2D metrics per fold (displacement vs true coords, clip rate) are the
  decision evidence — CV cosine is the generalization number, fold-wise transform
  the target metric.
- `RegisterMap.apply(X)` — affine map + re-unit-normalization.
- Persisted to `data/projection/register_map.npz` (+ meta: scores, alpha, n_pairs).

Runtime: `locate_nodes` applies the map (artifact present AND
`REGISTER_ALIGNMENT=true`, a new Settings flag defaulting on) right after the
dim guard; transform, placement confidence, and corpus support all consume the
corrected vector. Applies to everything `/locate` places (taxonomy nodes,
generated ideas, candidate stars) — documented boundary: `/axes`, `/relevance`,
`/similar` stay raw (they compare designer text to the corpus directly).
`generate_log.jsonl` rows gain `register_aligned` (+ node `desc`, so future
diagnostics re-embed the true locate text); `aggregate_generate_log` groups by
(prompt_version, seed_strategy, register_aligned).

CLI `project-align`: embeds the short texts (needs embed server), fits,
prints held-out before/after (cosine, displacement, clip rate), saves the map.
Risk: 209 pairs for a 768→768 map → heavy regularization required (the CV grid
handles it; translation is the safe floor). Risk: over-correcting genuinely
novel ideas toward the corpus → understating novelty — which is why H3 ships
alongside.

### H3 — Continuous corpus support replaces the binary honesty signal

Diagnostic #4 shows `clipped` is a near-coin-flip distinction among uniformly
extrapolated points. The signal that actually varies is 768-d corpus support:

- `backend/corpus/service.py`: cached self-support baseline (per corpus vector,
  mean top-5 cosine to the rest) + `corpus_support(vecs)` → **percentile of the
  query's support within the corpus baseline** (0 = below every corpus point,
  1 = above the median pack). Pure percentile helper unit-tested.
- `locate_nodes` / `generate-at` nodes / log rows gain `support: float | None`
  (best-effort, like confidence).
- UI: node-dot **fill strength** encodes support (washed-out = little corpus
  evidence); tooltip states it; legend explains. `clipped` remains as geometry
  (margin band, drift exclusion) but stops pretending to be the honesty signal.

### H4 — Soft margin instead of hard edge pinning

`_normalize` keeps the clipped mask but replaces `np.clip` with an identity-
inside / tanh-compressed-outside squash (`SOFT_MARGIN = 0.06` ≈ 3 cells at
R=48): coordinates land in [−0.06, 1.06], preserving direction *and ordering*
among out-of-hull points instead of stacking them at exactly 0/1 (which also
manufactured false collision badges). Interior coordinates are bit-identical —
no refit, saved sessions unaffected; old pinned coords simply sit on the
boundary line. The margin is served via surface `meta.soft_margin` (injected at
serve time — no artifact rebuild) and rendered as an explicit **"beyond corpus
range" band** around the lattice; clipped dots draw at their continuous margin
position rather than snapping to an edge cell. Distances inside the band are
not metrically meaningful — the legend says so.

### Validation

- `project-diagnose` CLI (new): reproduces the table above on demand — offline
  round-trip + bounds + support baseline; with the embed server, re-embeds the
  generate-log texts and reports clip rate / top-1 cosine / support, raw vs
  register-corrected. The numbers in this Part were one-off scripts; this makes
  them regression-checkable.
- Unit tests: soft-clip (interior identity, margin bounds, ordering, mask),
  register map (synthetic short/long pairs → mapped closer than raw;
  persistence round-trip; short-text builder), support percentile, log-stats
  grouping with `register_aligned`.
- Manual UI guide: new Iteration H section in DESIGN-SPACE-TESTING.md.
- Acceptance: held-out clip rate and displacement drop measurably under H2;
  H1 measured via log-stats v3 rows; no interior coordinate changes; all
  existing tests stay green.

### Iteration H — measured results (2026-06-11, same day)

All four fixes shipped (H1 prompt v3, H2 `project-align` + runtime correction,
H3 support score end-to-end, H4 soft margin + band UI). Gates: backend
harness 63/63, frontend 29/29, tsc/lint clean, zero browser console errors.

**H2 (register alignment), honest reading.** Fit on the corpus's 209
(name+2-sentences → full-text) pairs, translation won CV (ridge overfits at
n=209, d=768): held-out cosine 0.905 → 0.928. On held-out corpus short texts:
displacement 12.3 → 11.9 cells, clip 34% → 35% — i.e. **the correction barely
moves the 2D outcome for prefix-style texts**, because a description prefix
already embeds very close to its full text. On the *generated-idea* texts from
the log it does more: top-1 cosine 0.578 → 0.633, but clip rate unchanged
(41%) and support percentile 0.00 → 0.00 for the old one-liner texts. The
register map is kept (it is strictly directionally correct and free at
runtime), but it is not the main lever.

**H1 (project-style descs) is the lever.** First live v3 generation
("Content Theme" gap): descs 428–485 chars, support percentiles **0.12 / 0.18
/ 0.45** — the first generated ideas with non-zero corpus support (old
one-liner texts: 0.00 across the board) — and clip 1/3. Drift was higher
(0.508 vs 0.314 historical mean) — one generation, no conclusion;
`project-log-stats` now separates v3/aligned rows as their own variant
(`prompt × seeding × aligned`), so the A/B accumulates from here.

**H3/H4 validated in the browser.** A clipped placement rendered at
cx = −10.06 SVG units — *inside* the band at its continuous position, dashed,
fill-opacity 0.427 = 0.3 + 0.7·support — instead of pinned at the edge cell
(which previously also produced false collision badges). Diagnostic #4's
near-arbitrary binary flag is no longer the honesty signal; support is.

**Standing finding.** Even with both fixes, generated ideas live at the low
end of corpus support (0.12–0.45 pctile). That is the corpus-ceiling
measurement (Part 8 W3) made per-node and visible to the designer — the
quantitative case for corpus expansion when that work is taken up.

Reproduce: `uv run python database_pipeline.py project-diagnose` (add
`--offline` without the embed server); manual UI checks in
DESIGN-SPACE-TESTING.md §6.

---

## Part 10 — Iteration I: Briefs make candidates designs; Perspectives becomes the alignment instrument

*Added 2026-06-11, from the purpose clarification of the Perspectives mode (a
post-candidate examination tool against a focused metric list) and the brief
proposal that followed. Full argument in PROJECT-REPORT.md §4–5 and the
conversation record; this Part is the implementation contract.*

### I0 — The dual-layer candidate (the conceptual change)

A candidate stops being a combination and becomes a design with two layers:

| Layer | What it is | Where it comes from |
|---|---|---|
| **Brief** (identity) | The designer's own project-style prose: what this design *is* | Written by the designer; LLM-drafted from the choices as a starting point |
| **Choices** (commitments) | One option per aspect: what this design *commits to* in the constraint structure | The existing composition flow |

Rules that keep the idea honest:
1. **The brief is the candidate's primary embedding.** It is register-native text
   (the corpus IS project descriptions — Part 9 measured why this matters), so the
   star, precedents, and relevance lens all read the brief when present. The
   composition text survives only as the comparison reference; the two are never
   concatenated.
2. **The layers' divergence is a feature, not an error.** Where the brief embeds
   vs what the choices claim is the central measurement of the revamped
   Perspectives mode (I3). Nothing silently reconciles them.
3. **Neither layer may replace the other.** Brief without choices loses the
   morphological structure (enumerable, comparable, gap-spottable); choices
   without brief are not a design. The UI treats a candidate as complete when it
   has both.
4. **Drafting kills the blank page.** "Draft from choices" asks the LLM to write
   a project-style description committing to the chosen options; the designer
   edits. The edit-diff is itself a study signal (what designers change about the
   machine's synthesis).
5. **Evidence rules carry over.** Brief placements get support/confidence/margin
   treatment like every located text. Brief edits move the star; the previous
   positions persist as a capped trail — the design's trajectory through
   precedent space (the "evolving, not snapshot" requirement, finally literal).

**The squiggle hypothesis (recorded, deliberately untested):** mixing
convergence (brief, examination) and divergence (generation, gap-filling) on one
surface is hypothesized to HELP a modern, rapid-iterating, messy process (design
squiggle), not harm it. We therefore do NOT build a mode boundary between
brief-conditioned and brief-free work. Concretely: when a candidate with a brief
is active, generate-at receives the brief as a context block — and logs
`brief_context: true` — so the fixation-by-context risk is *measurable* in the
study rather than designed away in advance.

### I1 — Dual-layer candidates (store + panel + map)

- Store: `brief?: string` per candidate; `appendCandidateTrail` (previous star
  positions, capped at 10); brief included in the locate signature so editing
  re-places the star. Session save/load and export carry both layers.
- Candidate panel: brief textarea + "Draft from choices" (async job, spinner);
  placement caveats shown for the brief like any node.
- Map: star at the brief's position; faint trail for the active candidate. NO
  ghost marker / tether to the composition position on the map — 2-D distance
  between the layers would imply a measurement the projection cannot honestly
  make; the divergence lives in Perspectives as true cosine.

### I2 — Backend

| Endpoint | Shape | Notes |
|---|---|---|
| `POST /api/candidates/draft-brief` | `{aspects: [{aspect, option, desc}], project_overview?}` → 202 job → `{brief}` | New `DRAFT_BRIEF_PROMPT`, register-matched to Part 9 v3 guidance (concrete nouns, mechanism, experience; 3–5 sentences) |
| `POST /api/candidates/alignment` | `{brief, composition, aspects: [{aspect_id, chosen: {id,text}, alternatives: [{id,text}]}]}` → `{agreement, per_aspect: [{aspect_id, chosen_score, top_alternative: {id, score}, leans_away}]}` | Sync, one batched embed. `agreement` = cos(brief, composition). `top_alternative` = argmax cos(brief, other options) — the strongest competitor is defined by data, not picked |
| `POST /api/projection/metrics` | `{metrics: [{pole_a, pole_b}], items}` → per metric: corpus scores (full array), item scores (clip-flagged), `pole_sim`; plus pairwise metric correlations | Generalizes `/axes` (k=2 special case) via a shared bipolar-scoring helper; corpus arrays let the client draw distributions and speak percentiles |

`generate-at` gains optional `brief`; prompt v4 adds a DESIGNER'S CURRENT
CONCEPT block ("context, not a template — do not restate it"); the log row
records `brief_context` so log-stats can compare drift/novelty with and without
concept conditioning.

### I3 — Perspectives revamp (the alignment instrument)

- **Entry through candidates:** an "Examine" action on the candidate panel opens
  Perspectives pre-loaded with the active candidate. The navigator tab remains
  but teaches when empty ("Compose a candidate and write its brief — this is
  where you examine it").
- **Default representation: metric profile strips** — one horizontal bipolar
  strip per metric; corpus distribution as a rug/density behind; the candidate's
  brief as a star on the strip; pole labels; a percentile sentence under each
  ("more ⟨pole⟩ than NN% of real projects — scaled to this corpus").
- **Two strip families, one mechanism:**
  - *Consistency metrics (automatic):* per aspect with a choice, the axis
    chosen-option ↔ strongest-rejected-alternative; a divergence badge when the
    brief leans toward the alternative. This is only meaningful because the brief
    is written independently of the choices (the old composition-based check was
    partly self-confirming — the composition contains the chosen descs).
  - *Rubric metrics (persisted):* designer-saved aspect-pole pairs (store slice,
    GC'd with the tree, carried in sessions) so every candidate is examined
    against the same project-specific yardstick. Custom free-text criteria are
    the planned second pass (C2), not in this iteration.
- **Headline:** brief↔composition agreement + the largest per-aspect divergence.
- **The 2-D scatter (existing axes view) is demoted to a drill-down tab** for
  crossing two metrics (quadrant/trade-off reading); its diagnostics (pole
  similarity, axis correlation) become rubric-quality warnings.
- **Deferred, recorded:** C2 custom text criteria; E "revise toward pole"
  (await study evidence that designers iterate candidates); map ghost marker;
  multimodal briefs (when needed: caption-bridge images into the brief — no
  geometry change).

### Validation
- Unit: alignment scoring + top-alternative selection (synthetic vectors),
  metrics scoring vs axes parity, prompt assembly; store brief/trail/rubric +
  session round-trip; percentile + consistency-metric builders (pure).
- Manual guide: DESIGN-SPACE-TESTING.md §7 (brief → draft → examine walkthrough).
- Study hooks this iteration creates: brief_context A/B in the generate log;
  draft-vs-edited-brief diffs; trail as a reflection artifact.

### Iteration I — results note (2026-06-11, same day)

Shipped end-to-end: dual-layer candidates (brief textarea + LLM draft-from-
choices job + brief-first star/precedents/lens + capped trail), the
`/api/candidates/{draft-brief,alignment}` and `/api/projection/metrics`
endpoints, prompt v4 with the DESIGNER_BRIEF context block (`brief_context`
logged; log-stats now groups `prompt × seeding × aligned × brief`), and the
Perspectives revamp (Examine strips default, scatter demoted to "Cross two
metrics", persisted rubric with GC + redundancy warnings, entry via the
candidate panel's Examine button, teaching empty-states).

Gates: backend 76/76, frontend 39/39, tsc/lint clean, zero console errors.

Live walkthrough findings (now in TESTING §7): the drafted brief placed with
**confidence 0.54** — roughly double the corpus self-confidence mean (0.25),
live confirmation of the Part 9 premise that register-native prose is what this
space places well. And the alignment instrument produced a real finding on its
first run: the LLM'S OWN DRAFT leaned toward "sensor-driven reactive" (86%)
against the chosen "passive viewing" — the kind of concept↔commitment drift the
mode exists to catch, caught before any designer saw it.

Open (deferred by plan): C2 custom free-text criteria, E "revise toward pole",
the map ghost marker, multimodal briefs, and the squiggle-hypothesis A/B —
which now accumulates data automatically in the generate log.

### Support recalibration addendum (2026-06-12)

A user observation ("many nodes show ~0% corpus support — is this normal?")
exposed a calibration flaw in H3: the support percentile was read against the
corpus's FULL-register self-support (mean 0.811, floor 0.721 mean-top-5 cosine)
— a bar node-length text structurally cannot reach. Measured: even REAL corpus
projects, described in two sentences, scored mean 11% raw / 31% corrected, and
"Public plaza/square" (abundant precedent) scored 0%. The number was signalling
text length, not evidence.

**Fix:** `project-align` now also fits and persists a **short-register support
baseline** inside `register_map.npz` — the sorted mean-top-k cosines of the
out-of-fold corrected short corpus texts (each excluding its own full text,
since runtime queries have no self). `/locate` reads node support as a
percentile of that distribution; the full-register yardstick remains the
fallback when no map exists. Support now answers: *"compared to a real project
described at this length, how much corpus evidence does this idea have?"*

Validated live (one-line option texts, before → after): LED wall panels
0.33 → **0.84**; Passive viewing 0.10 → **0.45**; Haptic 0.02 → **0.16**;
Olfactory 0.01 → **0.10**; bare topic-only labels stay 0. One honest nuance:
pure *siting* statements ("Public plaza/square. The installation is sited in an
open civic square…") still read low (0.06) — corpus prose centres the artifact,
so location-context options carry less direct textual evidence than technology
options. Backend harness: 80 tests green (support_scores self-exclusion,
baseline persistence + legacy-artifact compatibility, explicit-baseline
percentiles).

**Follow-up (same day):** the recalibrated values were invisible in the UI at
first — coords (with their support) persist in localStorage and the locate
effect only placed nodes *without* coords, so anything located under the old
calibration never refreshed. The effect now re-locates every node once per
session (persisted coords still render instantly until the response merges), so
calibration refits propagate on the next space-view visit. Verified: 33 stale
zeroed supports refreshed to the live spread (29/33 nonzero, default schema
mean 18%, LED wall panels 66%, projection mapping 57%) on entering Design Space.

---

## Part 11 — Iteration J: evidence-anchored out-of-sample placement

### J0. Trigger and diagnosis

A user observation exposed a placement contradiction: **"LED wall panels"
rendered in the "beyond corpus range" band with corpus support 66%** — the map
said "outside everything we know" while the evidence metric said "one of the
most precedented ideas on the board". Tracing the point through the pipeline
showed two distinct flaws, both in how *queries* are placed (the frozen corpus
layout itself is not implicated):

1. **The geometric band overstates trivial overshoots.** The point's raw UMAP
   coordinate exceeded the corpus bounding box by **1.7% of the axis range** —
   a hair past the single most extreme corpus project — yet the binary clip
   flag gave it the same "beyond corpus range" label a genuine outlier would
   get.
2. **`UMAP.transform()` placed it far from its own evidence.** Its top-5
   corpus neighbours in the original 768-d metric are all real LED-facade
   projects (Taman Anggrek, Shanghai World Financial Center, Chanel Ginza
   Tower…) clustering near (0.50, 0.16); the transform placed the query at
   (0.27, −0.02) — **0.29 normalized units from the centroid of its own
   precedents**, about a third of the map.

Census over the 26 default-schema options: 7 clipped, of which 1 was a
high-support contradiction. The decisive measurement used the corpus's own
short-register round-trips (each project's "Name. first sentences" text,
register-corrected, placed back into the frozen map, self-excluded — the only
available ground truth for out-of-sample placement):

| placement method | median disp. | mean | p90 | clip rate |
|---|---|---|---|---|
| `UMAP.transform()` (H-era) | 0.182 | 0.247 | 0.514 | **30%** |
| evidence-weighted kNN (k=5) | **0.147** | **0.165** | **0.285** | 0% |
| evidence-weighted kNN (k=10) | 0.165 | 0.183 | 0.317 | 0% |

The 30% clip rate on *real corpus projects* round-tripping into their own map
is the indictment: the geometric "beyond corpus range" channel was mostly
transform noise, not a novelty signal.

### J1. The faithfulness question (design-decision record)

Earlier iterations declined neighbour-interpolation placement on faithfulness
grounds. The standard objections, re-examined against the current system:

- **"kNN interpolation cannot extrapolate — it hides genuine novelty."** True
  by construction: a convex combination of corpus positions can never leave the
  corpus footprint. But the round-trip numbers show the geometric channel never
  carried that signal faithfully (30% false positives on in-distribution data),
  and since Iteration H/Part 10 the system has a *faithful* outsideness channel:
  corpus support, measured in the original 768-d metric against the
  short-register baseline. Novelty detection belongs to support (washed-out
  fill, "thin precedent"), not to 2D geometry. Nothing of value is lost.
- **"UMAP transform is the principled manifold extension; kNN is a hack."**
  UMAP has no principled out-of-sample extension. `.transform()` itself
  initialises new points from their nearest *training-set* neighbours and then
  runs a few stochastic optimisation epochs — it is a noisy cousin of kNN
  interpolation, not a faithful manifold map. UMAP's own documentation
  acknowledges transform placements "concentrated on top of existing classes or
  spread between them" and points to parametric UMAP or interpolation methods
  as remedies.
- **"Disagreeing neighbours produce void placements."** Real and inherent: if
  the top-5 anchors straddle distinct clusters, the centroid lands between
  them, in a region belonging to nothing. This is surfaced — not solved — by
  the existing placement-confidence metric (Jaccard overlap of 768-d vs 2D
  neighbourhoods), which goes to ~0 exactly in this case and renders the dot
  dashed ("placement approximate"). UMAP transform has the same pathology with
  worse tails (p90 0.514 vs 0.285).

Alternatives considered and rejected:

- **Parametric UMAP** (neural encoder): the literature's heavyweight remedy;
  needs a deep-learning dependency, has ~209 training samples to learn a
  768→2 map (overfit regime), and refitting relayouts the map (breaks frozen
  coords + every persisted session). Disproportionate.
- **KRR / RBF interpolation** (the literature's lightweight remedy): a smooth,
  global variant of the same idea — measured against kNN during validation
  (see J4); even where competitive it loses on explainability: 5 nameable
  anchor projects vs opaque weights over all 209.
- **Raising the clip threshold / hull test**: cosmetic — LED would still be
  drawn a third of the map from its precedents.
- **Refit UMAP with other hyperparameters**: relayouts the frozen space,
  breaks every persisted coordinate, and no parameter setting fixes
  `.transform()`'s out-of-sample behaviour.

**Decision: place queries at the similarity-weighted centroid of their top-5
corpus neighbours' frozen coordinates.** The same five anchors then drive
position, corpus support, and the precedents panel — one consistent evidence
story ("placed amid these five projects" becomes a true, clickable statement),
which is the receipts-not-scores direction of the Part 10 reflection, and
matches the map's stated epistemics: *relative neighbourhood structure over
precedent evidence, not absolute coordinates*.

### J2. Design

- `pipeline/projection.py` gains pure `place_by_neighbors(vecs_unit,
  corpus_unit, corpus_coords, k, exclude_rows=None)`: cosine top-k per query,
  weights ∝ positive similarity (uniform fallback if degenerate), returns the
  weighted centroid of the anchors' frozen [0,1] coordinates. `exclude_rows`
  exists for fit-time diagnostics only (self-exclusion), mirroring
  `support_scores`.
- `locate_nodes` places via `place_by_neighbors` with `k = SUPPORT_NEIGHBORS`
  (deliberately the same k as support) whenever corpus vectors + surface
  coordinates are available and dimensions match; **falls back to the frozen
  `UMAP.transform()`** (unchanged semantics, including soft-clip + clipped
  flag) when they are not. On the kNN path `clipped` is always false.
- `generate_at` inherits the placement through `locate_nodes`; its log rows
  gain a `placement` field ("knn" / "umap") and `log-stats` groups by it, so
  drift statistics are never compared across placement regimes (drift's
  *meaning* changes: distance from the clicked gap to where the generated
  idea's evidence lives — bracketing seeds that actually blend should land
  near the gap; cliché drift snaps to the cliché's cluster, honestly).
- Placement confidence (Jaccard) is **unchanged** — it is method-agnostic and
  expected to rise on average, since placement now targets the true
  neighbourhood directly.
- The **"beyond corpus range" band retires from the UI** (legend row, band
  rendering, tooltip phrase, `meta.soft_margin`); outsideness is support's
  job. The backend keeps `clipped` in the API for the fallback path; the
  margin/soft-clip machinery stays in `pipeline/projection.py` (fallback +
  diagnostics baseline).
- `project-align` reports UMAP-transform vs kNN displacement side by side on
  the held-out short-register round-trips (the only ground truth), and
  `project-diagnose` states the active placement method — the decision stays
  reproducible and revisitable.

### J3. Risks owned

- A query whose neighbours disagree gets a void placement → dashed dot via
  confidence (existing rendering); accepted over transform noise.
- Genuinely out-of-domain text is drawn *inside* the map → washed-out fill at
  ~0 support is the honest signal; the band's geometric claim was less honest.
- Persisted coords from the transform era remain until re-located → covered by
  the once-per-session refresh shipped in the Part 10 follow-up.

### J4. Validation (measured, 2026-06-12)

- **Held-out displacement** (`project-align`, OOF-corrected shorts,
  self-excluded): raw transform mean 0.257 / median 0.182 / clipped 34%;
  corrected transform mean 0.249 / median 0.179 / clipped 35%; **kNN k=5 mean
  0.168 (8.0 cells) / median 0.149 / clipped 0%**.
- **KRR/RBF check** (the literature's lightweight alternative, 5-fold CV over
  an (alpha, gamma) grid, best picked by displacement): median 0.178 / mean
  0.190 / p90 0.319 — loses to kNN on every statistic even tuned, and its
  weights over all 209 projects are unexplainable next to 5 nameable anchors.
  kNN k=5 settled.
- **The trigger case, live**: "LED wall panels" now placed at (0.501, 0.158) —
  the centroid of its five LED-facade precedents — `clipped` false, support
  66% (unchanged, same anchors). Its confidence stays low (0.11), honestly:
  the anchors are somewhat spread, so the dot draws dashed. Public plaza
  (0.544, 0.579) support 12%; olfactory support 1% — washed-out, inside the
  map, as designed.
- **Browser**: band, legend entry, and tooltip phrase gone; viewBox back to
  the unit square; stale transform-era coords (including their `clipped` keys)
  replaced by the once-per-session relocate on entering Design Space.
- **Gates**: backend 87/87 (new: 6 `place_by_neighbors` properties, placement
  log grouping); frontend tsc clean, 39/39 bun tests, eslint 0 errors.

---

## Part 12 — Iteration K: the living schema (PLANNED — awaiting review)

*Status: plan only. Nothing below is implemented. Written 2026-06-12 after
reviewing six sources against the "one field, three projections, one
inspector" proposal; the evidence kept the lens architecture but overturned
its center of gravity.*

### K0. The evidence trail (what each source forces)

| Source | What it contributes | What it corrects in the prior proposal |
|---|---|---|
| Halskov & Lundqvist (2021), *Filtering and Informing the Design Space*, TOCHI 28(1) | Informing = establishing/transforming the space; **filtering = extracting a slice for investigation (NOT pruning)**; the two loop at activity scale (Schön's seeing–moving); divergence occurs late (a content tool added 5 new aspects months in); the design-space schema with per-activity dynamics (italics = informed, dashed = filtered) is the analysis instrument | The proposal's instruments are one-way (measure only). The literature's defining dynamic is **filter→inform**: investigation generates new aspects/options. Every instrument needs an informing-back channel |
| Halskov (2021), *A Media Architecture Design Space: The MAB 2012–2018 Nominees*, MAB '20 | 54 MAB nominees hand-annotated against a faceted schema; per-option instance **counts**; ± faceted search; **option×option cross-tabs whose empty cells are exact, nameable gaps**; granularity principles (avoid options matching ~all or ~1 instance) | The canonical criteria view is the **discrete cross-tab with real projects in cells**, not continuous bipolar axes (those stay as drill-down). The corpus⇄taxonomy bridge is **annotation**, which LLMind can automate |
| Suh et al. (2024), *Luminate*, CHI '24 | Dimension-driven re-layout validated by users (9/10 divergence support); generate-into-filtered-subspace; user-added dimension retroactively re-annotates all items; semantic zoom; **fade-don't-remove filtering** | Confirms the lens direction; warns that ungrounded LLM dimensions are "syntactically valid but semantically weak" — corpus grounding is the differentiator, never to be traded away |
| Onarheim & Biskjaer (2013), *An Introduction to 'Creativity Constraints'* | Constraints both restrain and enable; Elster's intrinsic / imposed / **self-imposed**; inverted-U between constrainedness and creativity | Choices, rejections, and briefs ARE self-imposed constraints — name them so in the model. Constrainedness is worth mirroring to the designer, never enforcing |
| Dalsgaard & Halskov (2012), *Reflective Design Documentation*, DIS '12 | Process capture converts design into knowledge; it dies of documentation burden; reflections attach to events; benefits often deferred to write-up time | LLMind auto-captures what PRT asked humans to type — except the **why**. Add burden-inverted reflection: AI drafts, designer accepts/edits/skips |
| Dissertation (Uchikoga) | Mind map as bridge between QOC / point-cloud / schema; participant asked for a **table overview** and **rationale**; Luminate critique: scope-grounding + manipulability of dimensions; named future work: temporal layers/replay | The schema table pays the oldest P1 debt; manipulability everywhere; temporal snapshots are in-scope, not exotic |

### K1. The re-centering

**The system's deep model becomes a living design-space schema; every view is
a lens on it.** Entities (most already exist in the store/logs — this is
largely a re-description, not a rebuild):

- **Aspects** and **options** — descs, provenance
  (`taxonomy | generate-at | steer | manual`), state (open / chosen-by /
  rejected+reason). Choices, rejections, and briefs are *self-imposed
  constraints* (Elster, via Onarheim & Biskjaer).
- **Corpus projects** — plus NEW **annotations**: which options each project
  exemplifies (A2). The discrete bridge between evidence and taxonomy.
- **Candidates** — constraint bundles (choices) + identity (brief) + trail.
- **Events** (NEW, lightweight) and **reflections** (NEW) — the process record.

Lenses over this model: **Schema table** (canonical overview — new),
**Map** (similarity/evidence lens — existing, demoted from protagonist),
**Cross-tabs** (morphological lens — new, generalizes the axes view),
**Inspector** (filtering instruments — Examine relocated), **Mind map**
(structure editing — unchanged until schema-table editing reaches parity).

Standing principles: manipulability everywhere (everything addable /
renamable / deletable); filtering fades rather than removes (Luminate);
every instrument can inform back (TOCHI); reflection never blocks (PRT).

### K2. Phase A — the schema spine

**A1. Schema table view.** A new view: aspects as columns, options as cells
(Halskov's schema form). Styling encodes the model: chosen = ring, rejected =
struck + dimmed, generated-origin = italic (Halskov's "informed" mark),
per-option **count badge** from A2. Click = shared selection (receipts appear
in Related Projects); in-table actions reuse existing store actions (choose /
reject / reopen / rename / add option = manual informing). Pays the
dissertation participant's table request verbatim.

**A2. Corpus annotation (the bridge).**
- `POST /api/corpus/annotate` (202 job): body = the taxonomy; result =
  `{taxonomy_hash, options: {<option_id>: {count, project_ids}}, diagnostics}`.
- Pipeline: per option, register-corrected option vector → top-30 corpus
  shortlist by true cosine; then per project, ONE local-LLM call listing its
  shortlisted options → membership ids (structured output: `list[str]`, no
  dict fields — CLAUDE.md schema rules). ≤209 LLM calls, cached as
  `data/projection/annotations/<taxonomy_hash>.json`; incremental
  re-annotation by option id on taxonomy edits.
- Diagnostics per Halskov's granularity principles: `too_broad`
  (count ≥ ~80% of corpus), `unprecedented` (count ≤ 1 — possibly novel,
  possibly vague). Badges in the schema table.
- **This delivers the receipts goal** (PROJECT-REPORT §6 item 5) in
  categorical form: support becomes a count with a clickable project list;
  the percentile retreats to diagnostics.
- Validation gate: ~20 hand-checked (project, option) pairs before trusting
  the pipeline; the job reports embedding-shortlist vs LLM-verdict agreement.

**A3. Faceted filtering.** Store gains transient
`facets: {include: Set<optionId>, exclude: Set<optionId>}` (NOT persisted).
Schema chips and the map respond: non-matching corpus glyphs fade to low
opacity (never removed — spatial context preserved). Combinable ± exactly as
in Halskov's tool.

### K3. Phase B — lenses and instruments

**B1. Inspector dock.** The Examine strips render in a right dock inside the
map view whenever a candidate is selected (same Collapsible / icon-collapse
grammar as the existing panels). Kills the steering mode ping-pong by
construction. The Perspectives tab remains as an alias until K5.

**B2. Cross-tab lens.** Pick two aspects → option×option grid; each cell
shows its annotated projects (count + names, click-through) and any candidate
whose choices include both options. **Empty cell = exact, nameable gap** →
"generate into this cell": pole-conditioned generation seeded with the two
option texts + exemplars from the adjacent half-matching cells; the prompt
states "no precedent combines A and B". A kept result becomes a **candidate
skeleton** (choices = {aspectA: optA, aspectB: optB}, brief = generated desc)
— the morphological-combination→candidate flow. Reuses the generate job
machinery; logged like generate-at with `cell: [optA, optB]`. The continuous
axes view remains as the cross-tab's drill-down ("show as continuous
scatter").

**B3. Steering v1.** `POST /api/candidates/steer` (202 job):
`{text, mode: 'metric'|'toward'|'away', metric?: {pole_a_text, pole_b_text,
target_score}, reference?: {text, weight}, preserve: string[]}` →
`{revised_text, named_qualities: string[], measurement}`.
- *Strip rails*: drag the star along an Examine strip to a target percentile
  → revision → ghost shows requested vs achieved (the language-feasibility
  gap — the steering analog of drift).
- *Pull-toward-precedent*: from the inspector or a precedent's context menu.
  Measurement = displacement decomposed into along-direction + orthogonal
  components (raw cosine space — briefs are long-register).
- Brief **diff shown for veto before commit** (the peek transparency
  pattern). Every steer appends a labeled trail segment.
- `steer_log.jsonl`: `{ts, mode, requested, achieved, along, orthogonal,
  named_qualities, brief_chars_before, brief_chars_after}` — study fodder.
- Deltas are **rulers and briefs, never constructors** (Part 11's evidence
  rule): the LLM makes the move in language; embeddings only measure it.

### K4. Phase C — the loops

**C1. Informing-back (the TOCHI loop).** Uniform channel: any instrument may
emit *proposals* `{kind: 'option'|'aspect', text, desc, source, evidence}`
rendered as accept/dismiss chips. v1 emitters: steer's `named_qualities`
("add 'durational rhythm' under Temporal Strategy?"); alignment's
uncovered-quality detection ("the brief emphasizes X — no aspect covers it");
cell generation (kept ideas inform both aspects). Accepted proposals enter
the taxonomy with provenance `source: 'steer'|'alignment'|'cell'`.

**C2. Reflection capture (PRT, burden-inverted).** On choose, reject (extends
the existing reason field), steer-commit, candidate-create, and
generation-keep: the local LLM drafts a ONE-LINE rationale from context,
prefilled in a small inline input — Enter accepts, typing edits, Esc skips.
Never modal, never required, drafts generated async. Stored as
`reflections: Record<eventId, {text, edited, ts}>` (persisted; in session
files + markdown export). Pays the "why these seven?" debt in both
directions: the system explains its proposals; the designer's choices carry
their why.

**C3. Temporal snapshots.** Append-only `events: Array<{ts, kind, refs}>` in
the store (persisted, capped 500), unifying what is already timestamped
(provenance.createdAt, trails, usage, logs). The schema table gains a replay
slider: the space at time t, with Halskov's dynamics styling (what each event
informed/filtered). Simultaneously the dissertation's named future work,
PRT's timeline, and the study's richest instrument.

### K5. End state (after A–C land and survive use)

The mode bar becomes a lens bar: **Schema | Map | Cross-tabs** + Mind map
(retained for tree editing until the schema table reaches editing parity).
Perspectives dissolves — strips into the Inspector, scatter into the
cross-tab drill-down. The map is the evidence lens, no longer the thesis.

Optional (decide after real use): a **constrainedness mirror** — a quiet
header chip ("4/6 aspects locked · 12 rejected") per the inverted-U;
informative, never enforcing.

### K6. Phasing, gates, and review points

Each phase is independently shippable and reviewed before the next:

| Phase | Contents | Gate |
|---|---|---|
| A | A1 schema table, A2 annotation job + receipts, A3 facets | Annotation spot-check passed; schema table drives shared selection; suites green; manual walkthrough added to TESTING §9 |
| B | B1 inspector dock, B2 cross-tabs + cell generation, B3 steer | Steer measurement pure tests; requested-vs-achieved visible; cell counts = annotation counts; a full examine→steer cycle needs no mode switch |
| C | C1 proposals, C2 reflections, C3 events + replay | Session round-trip with new slices (defaults-first restore); export contains reflections; replay reproduces a recorded session's schema states |

Implementation-order note: B1 (inspector dock) is the cheapest item in the
plan and independent of A — it may land first as a quick win; the grouping
above is thematic, not a strict sequence.

### K7. Risks owned

- **Annotation quality.** LLM membership judgments will be imperfect.
  Mitigations: receipts always visible (errors inspectable), the spot-check
  gate precedes trust, counts presented as evidence rather than verdicts.
  Per-pair manual correction deferred (v2).
- **Annotation cost.** ≤209 local-LLM calls per fresh taxonomy (minutes,
  async job, cached by hash, incremental on edits). Acceptable for a research
  prototype; stated plainly in the UI ("annotating corpus…").
- **UI density.** Schema + facets + inspector + chips risks the honesty-stack
  failure mode at interaction level. Mitigations: progressive disclosure
  along the spine (inspector appears with the first candidate; cross-tabs
  invite once ≥2 aspects have annotations); per-lens affordance pruning;
  fade, don't add chrome.
- **Steering overshoot.** LLMs rewrite more than asked; the
  requested-vs-achieved gap makes that visible rather than hidden; step-size
  capping is prompt engineering, iterated on steer_log data.
- **Scope.** Three phases ≈ three iterations of effort. The study remains the
  bottleneck for every claim (REPORT §6.2) and MUST NOT wait for C: it can
  run after B, with A+B as the tested artifact.

### K8. Explicitly deferred

Mind-map dissolution (schema editing parity first); per-annotation manual
correction; multimodal briefs/images; corpus expansion (after the study, per
REPORT §6.7); retrieval-path unification (REPORT §6.5 — A2's categorical
receipts reduce its urgency, but the embedding-side unification still
stands); semantic zoom on the map (Luminate pattern — nice-to-have, not
load-bearing).
