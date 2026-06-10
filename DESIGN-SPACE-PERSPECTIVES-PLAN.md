# Iteration F — Perspectives: Relevance Lens + Semantic Axes

**Status:** Plan for review — NOT implemented. Awaiting approval/decisions on the open questions.
**Prerequisite (done in this pass):** unified view interactions — both the mind map and the
design space now share the same grammar (plain wheel = zoom toward cursor, factor 1.12,
range 0.5–8; left-drag = pan; click = select; a "Reset view" button in both). Constants live
in `src/lib/view-interactions.ts`; the mind map uses mind-elixir's `handleWheel` +
`mouseSelectionButton: 2` (multi-select box moved to right-drag). This matters for F2: a
third canvas view only works if all canvases feel identical.

---

## F1 — Relevance lens (a second mode *inside* the Design Space view)

### Concept
The current view paints corpus dots one color — the map answers "what is similar to what"
only through (distorted) 2D proximity. The lens makes it **query-responsive**: pick an
anchor (the selected node, or the active candidate), and every corpus dot is recolored by
its **true cosine similarity** to the anchor — computed in the original 768-d metric, not
2D distance, so the painting is faithful even where the layout is not. It generalizes the
existing related-projects highlight from 5 binary marks to a continuous field over all 209.

### UI
- A small segmented control inside the space view (top-center, under the navigator):
  **Similarity map** (current) | **Relevance lens**.
- Lens mode:
  - Corpus diamonds: color ramp slate→amber by normalized score; subtle size ramp.
  - The anchor's own dot/star gets a marker; other taxonomy nodes fade to ~25% (declutter).
  - Legend: the ramp + the anchor label + "relative relevance (normalized per query)".
  - Lattice, generation, discovered cells, trails: **unchanged** (lens changes paint, not
    behavior — see Q1).
- Anchor resolution: selected node (topic + desc) → else active candidate (composed text)
  → else lens disabled with a hint ("select a node or candidate to use the lens").

### Backend
New endpoint (scores-only; `/api/corpus/similar` returns full metadata and is wasteful at k=209):

```
POST /api/corpus/relevance  {text: str}
  →  {scores: [{id: str, score: float}], min: float, max: float}
```
Implementation: embed `text` with the local model, dot-product against the cached
unit-normalized corpus matrix (the projection service already maintains this cache — move
`_load_corpus_vectors` into `backend/corpus/service.py` where it now belongs, fixing the
private-import debt from D7). Cost ≈ one embed call; response <5 KB.

### Frontend
- `useRelevanceQuery(text | null)` — react-query, `staleTime` 5 min, keyed by text.
- Surface props: `lensMode: boolean`, `relevance?: Record<string, number>` (client
  normalizes via min/max). The existing `relatedProjects` highlight remains the
  similarity-mode affordance (Q4).
- Color ramp hand-rolled HSL interpolation — no new dependencies.

### Critique / risks
- **Score compression:** corpus cosines typically span a narrow band (~0.5–0.8), so raw
  scores would all look mid-ramp. Min-max normalization per query is required — and the
  legend must therefore say "relative", or the lens overclaims. (Honesty pattern continues.)
- **Weak anchors:** short topic-only anchors place poorly (measured ~16-cell displacement
  in calibration); the lens uses topic+desc text and displays the anchor text so the user
  sees what was actually asked.
- **Server dependency:** needs the embedding server; on failure the lens falls back to
  similarity mode with an error toast.
- **Mode proliferation:** this adds the project's first *sub*-mode. Mitigation: the toggle
  lives visually inside the space view and defaults to Similarity; the lens auto-disables
  without an anchor.

---

## F2 — Semantic-axis Perspectives (a third top-level view)

### Concept
The UMAP map shows *statistical* similarity with measured distortion (trustworthiness
0.76). This view lets the designer pick the axes **from their own taxonomy**: each axis is
a bipolar semantic dimension between two option "poles" of one aspect. Every corpus
project, taxonomy option, and candidate is scored **exactly**:

```
score_axis(item) = cos(emb(item), emb(pole_A)) − cos(emb(item), emb(pole_B))
```
normalized per axis over the corpus to [−1, 1]. No UMAP, no stochasticity, no distortion —
**exact by construction** — and an empty region finally means something morphological:
*"no real project is both strongly ⟨pole_Ax⟩ and strongly ⟨pole_Ay⟩"*. This is the view
where the design space (in the literature's sense) and the visualization coincide.

### Visualization choice (delegated decision — rationale)
| Option | Verdict |
|---|---|
| Reuse the 48×48 lattice | ✗ Wrong tool: axes are continuous and *interpretable*; quantizing hides marginal structure and visually implies UMAP semantics. The lattice's job (generation affordance over a similarity manifold) doesn't transfer. |
| Parallel coordinates | ✗ Handles >2 aspects but is poor for gap-spotting and impossible to point at ("generate here" has no 2D location). |
| SPLOM (all aspect pairs) | ✗ ~15 tiny panels for 6 aspects; defeats focus; expensive to score all pairs. |
| **Bipolar scatterplot** | ✓ **Chosen.** Continuous scatter; pole labels at the four axis ends; light quadrant shading by corpus density; rug ticks on both axes for marginal distributions; quadrant corner labels ("⟨pole_Ax⟩ + ⟨pole_Ay⟩"). Gaps are visible *and readable*. Same interaction grammar as the other canvases (the UX unification above is the prerequisite). |

Elements: corpus diamonds (clickable → detail, same as space view), option dots **of the
two chosen aspects** emphasized in branch colors (others hidden or faded — Q2), candidate
stars (which double as a consistency check: a candidate that chose pole A's option should
sit toward pole A — if it doesn't, that exposes desc quality honestly), axis-quality
warnings (below).

### Axis definition UX
- Per axis: pick an **aspect** (dropdown; only aspects with ≥2 options) → poles default to
  the aspect's two **most mutually distant options** (max pairwise cosine distance among
  its options' embeddings) → either pole overridable by dropdown.
- Axis quality diagnostics, displayed inline:
  - pole similarity: warn when `cos(pole_A, pole_B) > 0.85` ("poles too similar — the axis
    collapses; pick more contrasting options");
  - axis correlation: Pearson r of corpus x-scores vs y-scores ("axes overlap r=0.72 —
    points will hug the diagonal").

### Backend
```
POST /api/projection/axes
  {x: {pole_a: {text}, pole_b: {text}}, y: {...}, items: [{node_id, text}]}
  → {corpus: [{id, x, y}], items: [{node_id, x, y, clipped}],
     meta: {x_pole_sim, y_pole_sim, axis_corr, x_range, y_range}}
```
- One batched embed call (4 poles + N item texts); corpus matrix from the shared cache.
- Min-max normalize per axis over the corpus; items outside the corpus range are clipped
  and flagged (`clipped: true` → rendered as edge markers — the E2 "honest edges" idea
  applies here from day one).
- Stateless; the client caches per axis-pair signature. Axis *picks* persist in the store
  (`axisConfig`), axis *coordinates* do not (cheap to recompute; avoids staleness).

### Generation in the axes view (Q3)
The eventual payoff: click an empty region → generate options conditioned **verbally and
precisely** — "options that are strongly ⟨pole_Ax⟩ and moderately ⟨pole_By⟩" (derived from
the click's normalized position), seeds = corpus nearest in *axis space* (an exact metric,
better than anything the UMAP view can offer). Reuses the existing job/prompt machinery
with one new template field. **Recommendation: ship F2 read-only first** (scatter +
pickers + diagnostics), add generation as F2.1 — it roughly doubles F2's scope and the
read-only view already delivers the discovery value.

### Critique / risks
- **Pole quality = desc quality.** Axes built from thin option descriptions will be mushy.
  Mitigation: poles embed `topic + desc` (descs now exist for generated options too); the
  pole-similarity warning catches degenerate cases.
- **Corpus-relative normalization:** positions shift if the corpus changes — the legend
  must say "scaled to this corpus".
- **1-D semantics per axis are a simplification:** an aspect with 5 diverse options isn't
  truly bipolar. Defaulting to the most-distant pair is a reasonable v1; a future version
  could offer "similarity to each option" small multiples. State the simplification in
  the UI ("axis spans ⟨A⟩ ↔ ⟨B⟩; other options of this aspect are not on this line").
- **Scope:** a new ~450-line view component + endpoint + pickers — the largest single
  feature since Iteration C. The read-only cut keeps it tractable.

---

## Build order
1. **F1** — relevance endpoint + lens mode in the space view (small; also pays down the
   `_load_corpus_vectors` placement debt).
2. **F2** — axes endpoint + read-only bipolar scatter + axis pickers + diagnostics
   (medium-large).
3. **F2.1** — generation-in-axes with verbal conditioning (optional follow-up).

## Open questions before implementation
- **Q1 (lens behavior):** in lens mode, keep empty-cell generation active (lens = paint
  only — my recommendation), or make the lens read-only?
- **Q2 (axes clutter):** show only the two chosen aspects' options as dots (my
  recommendation), or all taxonomy nodes faded?
- **Q3 (axes generation):** read-only F2 first (my recommendation), or include
  generate-in-axes in v1?
- **Q4 (highlight vs lens):** keep the binary related-projects highlight in Similarity
  mode and let the lens subsume it in Lens mode (my recommendation), or remove the binary
  highlight entirely once the lens exists?
