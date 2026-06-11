# Design-Space Visualization — Implementation Plan & Status

**Status:** ✅ Prototype implemented and verified (M0–M3). See *Implementation status* below.
**Scope:** Text-only embeddings. Multimodal deferred (separate future axis).
**Dimensions:** 2D (the pipeline supports `--dims 3`; renderer is 2D).

---

## Implementation status

Built and tested end-to-end against the live local stack (`PASSED 31 FAILED 0` —
see [DESIGN-SPACE-TESTING.md](DESIGN-SPACE-TESTING.md)).

**Backend**
- `llmind-python/pipeline/projection.py` — frozen reducer: `fit_projection`, `ProjectionModel.transform`, persistence (joblib), grid quantization, density, nearest-neighbour, surface payload.
- `llmind-python/database_pipeline.py` — new `project` command fits on the local index (or Supabase) → `data/projection/{model.joblib,surface.json}`.
- `llmind-python/backend/projection/{service,router}.py` — `GET /api/projection/surface`, `POST /api/projection/locate`, `POST /api/projection/generate-at`; mounted in `backend/main.py`.
- `config.py` — added `projection_dir` (anchored to `data/`, independent of the misconfigured `DATA_DIR` in `.env`).

**Frontend**
- `llmind-web/src/components/design-space/design-space-surface.tsx` — SVG lattice renderer.
- `llmind-web/src/features/design-space/` — `types.ts` + hooks (`use-surface-query`, `use-locate-nodes`, `use-generate-at-mutation`).
- `llmind-web/src/app/mindmap/page.tsx` — Mind Map / Design Space toggle, shared `coords` + selection, locate-on-change, generate-at handler.

**Decisions taken during build** (differ from / resolve the original draft):
- **Local-first.** The design space is fit on `data/local_index.npz` (the active full-local path), not Supabase. Fitting needs no server; only `/locate` + `/generate-at` need the embedding/LLM server.
- **Embedding dim is 768** (the served model emits 768d), not 384 — irrelevant to the math (dim-agnostic), but a **dim guard** now fails `/locate` loudly if the runtime model ≠ the fit model.
- **Renderer is hand-rolled SVG**, not Plotly — zero new deps, exact "array of basic dots" control, clean click handling.
- **Resolved open questions:** Q1 → nodes appear at their **true transformed coordinate** (snapped to the lattice cell); Q2 → corpus shown as **density background** (individual project dots deferred); Q3 → **local model** defines the space; Q4 → generated node is a **child of the focused branch**; Q5 → **toggle** (not split), preserving selection across both views.

---

## Original scope (as proposed)

**Scope:** Text-only embeddings. Multimodal deferred (separate future axis).
**Dimensions:** 2D for now (3D as a later toggle — see §10).

---

## 1. The concept

Add a second view of the taxonomy that is **the same information as the mind map, projected onto a 2D "design space"** instead of a tree.

The surface is a **regular lattice of selectable dots** (a grid). Every dot is a position in the design space. A dot is in one of these states:

| Dot state | Meaning | Interaction |
|---|---|---|
| **Empty** | Unexplored design territory | Click → ask the LLM to generate a node *here* |
| **Node** (colored) | Occupied by a taxonomy node | Click → selects the node (highlights it in the mind map) |
| **Project** (colored) | Occupied by a real corpus project | Click → opens the related-project detail |
| **Branch-highlighted** | Belongs to the currently selected mind-map branch | Visual emphasis only |

The mind map and the design-space view are **two sides of one coin**: every taxonomy node has exactly one identity and one position, shared by both views. Selecting in one highlights in the other; generating in one appears in both.

### Why this framing is coherent

The design space is defined by **the real project corpus**, not by the taxonomy. We fit the dimensionality reduction once on the corpus embeddings, freeze it, and treat that 2D manifold as "the space of media-architecture design." Taxonomy ideas are then *positioned within the space of real work*. Consequences:

- **Colored node dots** = ideas the taxonomy has articulated.
- **Faint project dots** = real evidence (where actual projects cluster).
- **Empty dots** = design territory neither the taxonomy nor the corpus has reached → the most interesting place to generate.

So "generate in the surrounding space" has a real meaning: *fill a gap in the design space, seeded by whatever real projects and existing ideas are nearest to that gap.*

---

## 2. Invariants (the things that must stay true)

These are the load-bearing constraints. Most of the design follows from them.

1. **One shared coordinate space.** All dots — projects and nodes — are projected through the *same* frozen reducer fit on the *same* embedding model. Cloud (1536d) and local (384d) vectors are NOT comparable; the visualization commits to **one model** (local `bge-small-en-v1.5`, 384d — see §9 Q3).
2. **Coordinates are stable across sessions and incremental growth.** Expanding a branch must not relayout existing dots. This forbids re-running `fit_transform` per request. The reducer is fit once, persisted, and only `.transform()` is used afterward.
3. **One identity per entity, shared by both views.** A taxonomy node's `node_id` is the join key between the mind-map tree and the design-space dot. The frontend store is the single source of truth for both.
4. **The lattice is a presentation layer.** Underlying coordinates are continuous floats; the grid is a quantization applied at render time. Snapping a point to a cell never mutates its true coordinate.

---

## 3. The three hard problems & decisions

### 3.1 Coordinate stability → a persisted "projection model"

**Problem.** UMAP is non-parametric and stochastic. `umap_reduce` ([`pipeline/ml.py`](../llmind-python/pipeline/ml.py)) currently calls `reducer.fit_transform(X)` on whatever batch it gets, and `normalize_to_unit_interval` normalizes by that batch's min/max. Both are batch-dependent → no stability.

**Decision.** Introduce a **projection artifact** fit once on the corpus and persisted to disk:

```
data/projection/
  reducer.joblib        # fitted PCA + fitted UMAP (cosine, n_components=2)
  bounds.json           # {x_min, x_max, y_min, y_max} from the reference fit
  grid.json             # {resolution, ...} grid spec
  corpus_points.json    # [{id, x, y, kind:"project", density}] precomputed background
```

- **Fit** (`pipeline/ml.py: fit_projection`): on corpus embeddings, fit PCA(64) → UMAP(2, cosine), record the resulting min/max bounds. Save via `joblib` (already a dependency).
- **Transform** (`pipeline/ml.py: transform_points`): for any new text (taxonomy node), embed → PCA.transform → UMAP.transform → normalize using the **persisted bounds** (not the new batch's). Returns stable `(x, y)` in `[0,1]`.

This is the single keystone change. Everything else depends on it.

### 3.2 Grid quantization & collisions

- Quantize `[0,1]²` into an `R×R` lattice (start `R = 48`, configurable). Cell of a point = `(floor(x*R), floor(y*R))`.
- With ~50–500 occupied positions over ~2300 cells, most cells are empty → plenty of generation affordances. Good.
- **Collisions** (two entities in one cell): for v1, render a small count badge and let selection disambiguate via a list. Do **not** jitter true coordinates (violates invariant 4).
- **Optional heat shading on empty cells:** shade each empty cell by local corpus density (precomputed in `corpus_points.json`). This folds the earlier "heatmap" idea into the lattice — the surface itself becomes a discretized density/whitespace map, and the densest *empty* regions read as "adjacent to real work but unexplored."

### 3.3 "Generate in the surrounding space" → spatial-neighbor RAG

**Problem.** A clicked empty cell is a 2D coordinate. The LLM needs semantic context, not `(x, y)`. There is no clean text-decoder for an arbitrary coordinate.

**Decision (v1): spatial-neighbor retrieval.** When the user clicks empty cell `(gx, gy)`:

1. Map the cell center back to continuous `(x, y)`.
2. Find the *k* nearest **real** entities by 2D distance — both corpus projects and existing taxonomy nodes.
3. Feed those neighbors as the seed context into the existing generation flow ([`generate_nodes_from_related_projects`](../llmind-python/backend/related_projects/service.py)), with an added instruction: *"propose a node that sits between these neighbors yet is distinct from them — fill the gap."*
4. The new node is embedded → `transform_points` → lands at its true coordinate (which may differ from the clicked cell; see Q1).
5. The node is inserted into the mind map (child of the current focus branch) **and** lit up on the surface — same `node_id`, both views.

This reuses the existing retrieval-augmented node generation almost verbatim; the only new input is "retrieve by spatial proximity to a clicked location" instead of "retrieve by topic text." It also keeps the tree semantics intact: the generated node is a child of the **currently focused branch**, positioned in space by its own embedding.

**Enhancement (later): inverse transform.** UMAP supports `inverse_transform([[x,y]])` → synthetic high-dim embedding → nearest neighbors in the *original* metric (more faithful than 2D distance). Defer; the 2D-neighbor seed is good enough for v1 and far simpler.

### 3.4 Bidirectional sync (the "two sides of one coin")

The existing Zustand store ([`src/store/mindmap-store.ts`](../llmind-web/src/store/mindmap-store.ts)) already persists selection, nodes, and taxonomy. Make it the shared source of truth:

- Add `coords: Record<node_id, {x,y}>` and `selectedEntityId` to the store.
- Mind-map `onSelect` and surface `onDotSelect` both write `selectedEntityId`.
- Both components subscribe; selection highlight is derived state. Branch highlight = all `node_id`s whose lineage includes the selected node (we already compute lineage for retrieval).
- Node generation (from either view) appends to `nodes` + `coords` immutably, exactly as `insertChildrenAtNode` does today.

No new sync protocol — it's one store, two subscribers.

---

## 4. Data model

A single **projection document** the frontend holds (and the backend returns incrementally):

```ts
type DesignSpacePoint = {
  entity_id: string;          // node_id for taxonomy nodes; project id for corpus
  kind: "node" | "project";
  label: string;
  branch_path: string[];      // lineage; [] for corpus projects
  x: number; y: number;       // continuous, in [0,1], stable
  // grid cell derived at render time
};
```

Backend persists the reference/background (§3.1). Taxonomy node points are computed on demand and merged in.

---

## 5. Backend changes

| File | Change |
|---|---|
| [`pipeline/ml.py`](../llmind-python/pipeline/ml.py) | Add `fit_projection(X, dims=2)` and `transform_points(X)`; refactor `umap_reduce` to share the UMAP/PCA construction. Persist/load via `joblib`. Normalize with persisted bounds. |
| [`database_pipeline.py`](../llmind-python/database_pipeline.py) | New CLI command `project fit` → fits on corpus embeddings, writes `data/projection/*`. New `project corpus` → precomputes `corpus_points.json` (+ density). |
| `backend/projection/` (new) | Router + service. Endpoints below. |
| [`backend/related_projects/service.py`](../llmind-python/backend/related_projects/service.py) | Add `nearest_entities(x, y, k)` (spatial) and a thin wrapper that injects spatial neighbors + the "fill the gap" instruction into the existing node-generation path. |
| [`backend/main.py`](../llmind-python/backend/main.py) | Mount the new projection router. |

### New endpoints

- `GET  /api/projection/surface` → grid spec, bounds, and precomputed corpus background points (+ density). Served from disk; cheap.
- `POST /api/projection/locate` → body: list of taxonomy node texts (`name + desc` + `node_id`). Returns `[{node_id, x, y}]`. Called after taxonomy generation and after every node generation so nodes get coordinates.
- `POST /api/projection/generate-at` → body: `{x, y, focus_node_id, lineage, k}`. Does §3.3: spatial-neighbor retrieval → node generation → locate new nodes → returns the same shape as `generate-nodes` **plus** coordinates.

`generate-at` is intentionally a superset of the existing `generate-nodes` response so the frontend insertion logic is shared.

---

## 6. Frontend changes

| Area | Change |
|---|---|
| Store ([`mindmap-store.ts`](../llmind-web/src/store/mindmap-store.ts)) | Add `coords`, `selectedEntityId`, `surface` (background); actions to merge coords and set selection. |
| New component `DesignSpaceSurface` | Renders the lattice + dots; handles dot selection and empty-cell generation. |
| New hook `use-surface-query` | Fetches `/api/projection/surface` once. |
| New hook `use-generate-at-mutation` | Wraps `/api/projection/generate-at`; on success merges nodes + coords (reuse insertion logic). |
| Existing taxonomy/node mutations | After success, also call `/api/projection/locate` for the new node ids and merge coords. |
| Page ([`app/mindmap/page.tsx`](../llmind-web/src/app/mindmap/page.tsx)) | Lay out mind map and surface side-by-side (or tabs); wire shared selection. |

**Renderer choice:** start with **Plotly** (`react-plotly.js`) — `scattergl` for dots, fast to stand up, trivial 2D→3D later, built-in hover/select. Graduate to **deck.gl** (`ScatterplotLayer` + GPU picking) only if dot counts or interactivity demand it. (react-three-fiber only if 3D orbit becomes the centerpiece.)

---

## 7. Milestones (each independently shippable)

- **M0 — Frozen projection (backend).** `fit_projection`/`transform_points`, persist artifact, `project fit` + `project corpus` CLI. Validate stability: transform the same node twice → identical coords; add a node → existing coords unchanged. *Keystone; do first.*
- **M1 — Surface read-only (frontend).** `GET /surface` + `DesignSpaceSurface` rendering corpus background as a lattice with density shading. No taxonomy yet.
- **M2 — Taxonomy overlay + sync.** `/locate` + colored node dots + branch highlighting + bidirectional selection with the mind map.
- **M3 — Generate-at.** Empty-cell click → `/generate-at` → new node in both views. The payoff interaction.
- **M4 (optional).** Inverse-transform seeding (§3.3 enhancement); trustworthiness reporting; 3D toggle.

---

## 8. Risks

- **UMAP distance distortion.** 2D neighbors ≠ semantic neighbors exactly. Acceptable for a generation *seed*; surface a trustworthiness score (sklearn `trustworthiness`) so the layout's fidelity is honest. (USYD research context — worth reporting.)
- **Cold-start corpus.** The space only exists once the corpus is embedded and `project fit` has run. Document this as a prerequisite step.
- **Coordinate drift on generation** (Q1). A generated node may land away from the clicked cell. This is *informative*, not a bug — but needs a UX decision.
- **UMAP `.transform()` cost.** Per-call transform of a few nodes is fast; batch where possible. Not expected to be a bottleneck at this corpus size.

---

## 9. Open questions (please confirm before M0)

- **Q1 — Where does a generated node appear?** Options: (a) at its *true* transformed coordinate, with a connector to the clicked cell if it drifts (honest, my recommendation); (b) snap-display to the clicked cell (matches intent, hides reality). Which?
- **Q2 — What occupies dots: nodes only, or nodes + projects?** I assumed both (projects as faint background, nodes as colored). Confirm projects should be visible/selectable, or background-only.
- **Q3 — Which embedding model defines the space?** I assumed **local `bge-small` (384d)** for cost/offline consistency with `VECTOR_STORE=local`. Switch to OpenAI 1536d if you want cloud-quality geometry (must be consistent end-to-end).
- **Q4 — Generation parent.** I assumed a generated node is a **child of the currently-focused branch**, positioned by its embedding. Alternative: parent = nearest node to the clicked cell (more spatial, less tree-coherent). Which?
- **Q5 — Layout.** Side-by-side panes, or a tab/toggle between mind map and surface?

---

## 10. Explicitly out of scope (for now)

- **Multimodal embeddings** — deferred until a suitable model is chosen. When added, it's a *new embedding axis / new column*, not a change to this plan's geometry; the surface logic is model-agnostic given §2.1.
- **3D** — the manifold and grid generalize to 3D by `n_components=3` + a `z`; deferred behind a toggle (M4). All §2 invariants already hold in 3D.
- **Inverse-transform seeding** — M4 enhancement; v1 uses 2D spatial neighbors.
