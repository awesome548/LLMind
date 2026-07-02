# Frontend — llmind-web

Next.js 16 frontend. React 19, Bun, TanStack Query, Zustand.

---

## Scripts

| Command | Description |
|---|---|
| `bun dev` | Dev server → http://localhost:3000 (Turbopack) |
| `bun build` | Production build |
| `bun start` | Production server |
| `bun lint` | ESLint |
| `bun test src` | Unit tests (bun:test) — tree-utils, candidate-utils, stats, session-io, export, store invariants |

---

## Architecture

| Layer | Location | Responsibility |
|---|---|---|
| Types — generated | `src/types/openapi.ts` | Auto-generated from backend OpenAPI spec — **do not edit manually** |
| Types — aliases | `src/types/api-aliases.ts` | Regen-safe aliases over `openapi.ts` + hand-written async **job result** shapes (invisible to OpenAPI). App code imports from here, never from `openapi.ts`. |
| API client | `src/lib/api-client.ts` | Axios instance; calls the backend directly (see CLAUDE.md) |
| Hooks — queries | `src/features/mindmap/hooks/use-related-projects-query.ts` | React Query: fetch related projects on topic select |
| Hooks — mutations | `src/features/mindmap/hooks/use-generate-nodes-mutation.ts` | React Query: generate child nodes via LLM (async job) |
| Hooks — mutations | `src/features/mindmap/hooks/use-generate-taxonomy-mutation.ts` | React Query: generate full taxonomy (returns `corpus_similarity` for the domain notice) |
| Store | `src/store/mindmap-store.ts` | Zustand ^5 (persist schema version 2, with a v1 migration); persists the WHOLE exploration — tree, coords, discovered, provenance, candidates, pruning (see ZUSTAND.md) |
| Components | `src/components/mindmap/` | `SimpleMindMap` (mind-elixir wrapper; `nodeStates` styles rejected/chosen), `SimpleProjectPanel` (accepts `focusProject`) |
| Dialog | `src/features/mindmap/components/generate-taxonomy-dialog.tsx` | Taxonomy generation form (project overview, reasoning, mode) |
| Data | `src/features/mindmap/data/schema-mindmap-data.ts` | Static initial taxonomy + `taxonomyToMindmapNodes()` converter |
| Page | `src/app/mindmap/page.tsx` | Main orchestrator — wires store, hooks, components; Mind Map / Design Space / Perspectives view toggle + Similarity/Relevance-lens mode toggle |
| Design space — surface | `src/components/design-space/design-space-surface.tsx` | SVG lattice: corpus glyphs (inspectable), node dots (confidence-dashed, **fill strength = corpus support**, placed amid their top-5 precedents — Part 11), candidate stars, collision badges + chooser, zoom-faded density heat, trustworthiness legend, cancel button; **relevance-lens painting** (cool-to-warm ramp, anchor-faded nodes, "relative" legend). Emphasis follows the luminance hierarchy (see Design language): corpus = pale `CORPUS_MUTED` field, related-to-selection = vivid `CORPUS_COLOR` + glow, selected node = own-hue halo + saturation lift — never dark outlines |
| Structure — schema view | `src/components/design-space/schema-table.tsx` | **The design-space schema (Part 12 A1/A3)**, second view of the Structure mode (Tree ↔ Schema ↔ Cross-tab top-center toggle) — aspects × options as a **pan-zoom canvas** (shared `usePanZoom` grammar: wheel zoom, left-drag pan, Reset view) whose cards pack into balanced CSS columns (~2 aspects per column, max 4 wide) so the sheet uses the vertical space and nothing sits clipped under the floating panels. Cell styling: chosen = ring, rejected = struck, generated = italic, count badges = corpus annotation with click-through receipts, granularity badges, ± facet chips that fade non-matching map glyphs, in-table choose/reject/reopen + add-option (manual informing). Pure view models in `features/design-space/schema-utils.ts` |
| Structure — cross-tab view | `src/components/design-space/cross-tab-view.tsx` | **The morphological lens (Part 12 B2)**, third Structure view — two aspects → option×option grid from the annotation (`buildCrossTabCells`), rendered as a **pan-zoom canvas** (shared grammar; controls float in a top pill, Reset view bottom-left): cells list exemplifying projects (click-through receipts; popovers flip toward the viewport center at edges) + candidates committing to both options; **empty cell = exact, nameable gap** → "Generate into this gap" (seeded with `halfMatchingExemplars`) → veto preview → "Keep as candidate" creates a candidate skeleton (choices = the two options, brief = the concept) and opens the Candidate panel. "Show as continuous scatter" deep-links to Perspectives' scatter tab |
| Design space — examine | `src/components/design-space/examine-view.tsx` | "Perspectives", revamped (Part 10): the **alignment instrument** — candidate switcher, rubric editor, and the scatter drill-down around the shared strips component; entered via the candidate panel's **Examine** |
| Design space — strips | `src/components/design-space/candidate-strips.tsx` | **The examine instrument's core, shared (Part 12 B1)** — concept↔commitments agreement headline, consistency strips (chosen option ↔ data-picked strongest alternative, "leans" badges), persisted rubric strips, percentile sentences, redundancy warnings. Rendered by both Perspectives and the **Inspector dock** (right column of Design Space while a candidate is active — examine ⇄ map without a mode switch; Related Projects shifts to `compact` column caps below it). **Every strip track is a steering rail (Part 12 B3):** click OR drag along the rail (pointer-captured; the ghost target star follows) → Steer/Cancel render INSIDE the strip card under the rail (target and commitment as one object) → LLM revision + requested-vs-achieved measurement → veto card (`steer-result-card.tsx`). Clipped scores (outside the corpus range) render the star solid at reduced opacity with a tooltip, clamped to the rail edge — a dashed outline at 20px reads as a rendering artifact, not a meaning |
| Design space — steering | `src/features/design-space/hooks/use-steer-mutation.ts`, `src/components/design-space/steer-result-card.tsx` | **B3:** one deliberate move on the brief (metric rail / pull-toward / push-away precedent). The LLM moves in language with `preserve` choices; embeddings only measure (along/orthogonal in raw cosine space). Results are ALWAYS veto cards — Apply commits via `setCandidateBrief` (the star's trail records the hop). Precedent pulls live on the Candidate panel's precedent rows (hover: ⇢ pull / ⇠ push) |
| Design space — axes view | `src/components/design-space/axes-view.tsx` | The "Cross two metrics" drill-down tab inside Perspectives: bipolar scatter on designer-chosen aspect/option poles — exact cosine scores, quadrant density shading, axis-quality warnings, clip-dashed items |
| Design space — candidates | `src/components/design-space/candidate-panel.tsx`, `compare-candidates-dialog.tsx` | **Dual-layer candidates (Part 10):** choices (commitments) + BRIEF (identity; textarea + LLM "Draft from choices"). The brief drives the star, precedents, and lens; brief revisions leave a trail on the map. Compare; export |
| Examine utils | `src/features/design-space/examine-utils.ts` | Pure strip helpers: consistency/rubric metric defs, corpus percentile — unit-tested |
| Loops — reflections (C2) | `src/components/design-space/reflection-chip.tsx` + `hooks/use-draft-reflection-mutation.ts` | **Burden-inverted reflection capture**: reflectable events (choose/reject/steer apply/candidate create/cell keep/generate) pop a bottom-right chip; the local LLM drafts the one-line "why" async (fills only if the designer hasn't typed); Enter accepts, typing edits (tracked as `edited`), Esc skips. Stored as `reflections[eventId]`; in session files + markdown export |
| Loops — proposals (C1) | `src/components/design-space/proposal-chips.tsx` | **Informing-back chips**: applied steers propose their `named_qualities` as options (aspect known for strip steers, picker otherwise); kept cell ideas propose themselves under both parent aspects. Accept inserts into the taxonomy with provenance `source: 'steer'\|'cell'` + an `option_added` event; dismiss drops. Transient page state — only accepted proposals persist |
| Loops — timeline (C3) | `src/features/design-space/replay-utils.ts` + `src/components/design-space/replay-timeline.tsx` | **The exploration timeline** (Fusion-360 style): one icon-coded marker per event (✓ chose, ✕ rejected, ✦ generated/kept, + added, → steered, 💡 dismissed idea…), playhead with a few-words bubble, applied/future marker dimming, horizontal scroll for long logs. Clicking a marker scrubs the schema to the state AFTER that step (read-only; `buildReplayOverlay` is pure + tested) AND opens its detail card — full label, time, the kept reflection (📓 adornment on the marker), and **Reconsider** for dismissed suggestions (the proposal travels in `event.detail`, re-enqueued as a chip). While scrubbing, options that didn't exist yet render **ghosted** (`notYet` set → faded/grayscale + tooltip), the clicked step's subject cells get an **amber outline**, and the schema's status strip shows an amber **"Replay — step N of M · read-only"** badge with its own Back-to-now (Nielsen H1: the mode is announced where clicks get ignored), so every scrub visibly answers. Floored at the last taxonomy change; honest simplification labelled: latest commitment per aspect across all candidates. Deliberately NOT built: restart-from-moment (worth building later, scoped to commitments/filters as an append-only `rolled_back` event) and git-like branching (rejected — candidates + session save/load already are the branching mechanism); rationale in ITERATION-PLAN K9 |
| Design space — hooks | `src/features/design-space/hooks/` | `use-surface-query` (gated on view — space OR schema, the probe needs the project universe), `use-locate-nodes`, `use-generate-at-mutation` (sends coords + AbortSignal), `use-corpus-project`, `use-candidate-precedents`, `use-relevance-query` (lens), `use-axes-query`, `use-pan-zoom` (shared canvas grammar) |
| Rationale layer (Part 13 L-A) | `hooks/use-rationale-query.ts`, `hooks/use-missing-aspect-mutation.ts` | **The structure explains itself** (the study's "why these seven?"): per-aspect one-line rationale grounded in annotation counts (annotation-gated query, server-cached per aspect+evidence) → "why:" lines under schema column headers + a violet callout in the Context panel when an aspect is selected. **Coverage probe:** `poorlyCoveredProjects` (pure, in schema-utils) finds corpus projects the taxonomy barely describes → strip chip "N projects fit poorly — probe for a missing dimension" → LLM names the missing dimension(s) → amber chips (`kind: 'aspect'`, source `coverage`); accepting inserts a new root-child aspect column (provenance `coverage`, `option_added` event) |
| First-run choice (Part 13 L-B) | `page.tsx` (Dialog) | Once-only on first structure-mode entry: **Start from your brief** (opens the taxonomy dialog — its overview field IS the brief; the study's inform→filter layered model) vs **Discover first** (explore the prebuilt space). Tracked `first_run_brief`/`first_run_discover`. The brief persists as `projectBrief` (store + sessions); once a taxonomy exists the navigator's "Generate Taxonomy" button becomes **"Edit Brief & Taxonomy"** and reopens the dialog prefilled (derived-value pattern — typing takes over, closing resets) |
| Design space — utils/types | `src/features/design-space/candidate-utils.ts`, `types.ts` | Candidate text composition + hand-written projection payload types (incl. axes) |
| Shared interactions | `src/lib/view-interactions.ts`, `src/lib/svg-glyphs.ts` | One zoom factor/range for ALL canvases (the mind map mirrors it via mind-elixir `handleWheel` + `mouseSelectionButton: 2`); shared star glyph |
| Tree utils | `src/features/mindmap/tree-utils.ts` | Pure tree ops (find/insert/collect/unique-ids) — extracted from the page, unit-tested |
| Gap preview (E1) | `use-peek-mutation.ts` + popover in the surface | Click empty cell → seeds/parent-aspect/nearby preview FIRST; generation commits from the popover only |
| Exploration stats (E5) | `src/features/design-space/exploration-stats.ts` | Pure stats (options, generated, rejected, chosen aspects, cells, candidate diversity) — UI strip + export + study instrument |
| Session I/O | `src/lib/session-io.ts` + navigator Save/Load | Full exploration state as versioned JSON (capture, crash recovery, sharing); `restoreSession` store action |
| Usage counters | store `usage` + `trackUsage` | Feature-usage instrumentation; included in session files |
| Export | `src/lib/export-exploration.ts` | Markdown exploration record (taxonomy + states, candidates, provenance) |

---

## Design language (applies to EVERY component, new or edited)

The project's core UI philosophy is a **clear information hierarchy**, grounded
in standard interaction-design theory (Norman's principles, Nielsen's
heuristics, Gestalt). Concretely:

- **Hierarchy by luminance, never ink.** Context layers are pale and
  desaturated; the current focus gets saturation, brightness, size, or a soft
  same-hue halo. Dark borders/outlines are NOT an emphasis channel on dense
  visualizations (they read as clutter). Canonical example: the surface's
  corpus field (`CORPUS_MUTED hsl(28 52% 79%)`) vs related-to-selection
  (`CORPUS_COLOR hsl(28 92% 50%)` + radial glow) vs the selected node
  (own-hue halo + `saturate(1.4)`); all dot strokes stay white.
- **One color, one meaning, everywhere:** amber = real corpus evidence,
  violet = the designer's candidates/commitments, sky = already discovered,
  emerald = LLM-generated, red = rejected, slate = neutral chrome. A new
  feature reuses these before inventing a hue.
- **One emphasis channel per meaning.** If size already encodes depth, pick a
  free channel (glow/saturation/opacity) for selection — don't stack a second
  meaning onto the same channel.
- **Norman basics on every control:** visible signifier (hover state, cursor,
  tooltip on icon-only targets), immediate feedback (<100ms acknowledgment
  even when the real work is a 50s LLM job), controls spatially near what
  they affect, constraints over error states.
- **Nielsen basics on every view:** status always visible (jobs, active
  candidate/lens/view), every state exitable (Esc, Reset view, Back to now,
  breadcrumbs), recognition over recall (legends and labels in situ),
  fade-don't-remove for filtering.

---

## Key Flows

### Node click → generate child nodes
1. `SimpleMindMap.onSelect(topic, lineage)` → page local state
2. `useRelatedProjectsQuery` auto-fires (React Query, on selection change)
3. "Generate Nodes" button → `useGenerateNodesMutation` → `flattenMindmapNodes` (full tree context) → `POST /api/related-projects/generate-nodes`
4. Response nodes inserted immutably via `insertChildrenAtNode` using `response.parent_id`

### Taxonomy generation → mindmap rebuild
1. "Generate Taxonomy" button opens `GenerateTaxonomyDialog`
2. User inputs project overview, reasoning effort, backend mode
3. Submit → `useGenerateTaxonomyMutation` → `POST /api/taxonomy/generate`
4. `onSuccess` → `useMindmapStore.setTaxonomy(result)`
5. `page.tsx` `useEffect` on `taxonomy` → `taxonomyToMindmapNodes(taxonomy)` → replaces `nodes` state and resets selection to root

### Design Space ⇄ Mind Map (two views, one selection)
1. Top-center toggle switches `view` between `'map'` and `'space'` — both read the same `nodes` + `selection` (selection carries `nodeId` for exact identity).
2. `useSurfaceQuery` loads the corpus background on first visit to the space view (`GET /api/projection/surface`; cached forever). The legend shows the layout's **trustworthiness**.
3. On `nodes` change, nodes are embedded + placed via `POST /api/projection/locate` (best-effort; embeddings are register-corrected backend-side when the map is fitted). Placement is **evidence-anchored** (Part 11): a node sits at the weighted centroid of its top-5 corpus precedents — the same anchors behind its **corpus support** percentile (fill strength + tooltip) — so position and evidence tell one story and nothing lands outside the map. A dashed dot marks low **placement confidence** (the 2D neighbourhood disagrees with the true one — e.g. spread anchors). Coords persist in the store and render instantly, but **every node re-locates once per session** so values cached under an older calibration (register-map refit, placement change) refresh on the first space-view visit; renames drop the stale coord immediately.
4. Clicking an **empty** lattice cell → `useGenerateAtMutation` (`POST /api/projection/generate-at`, async job, cancellable). The backend brackets the gap with seed projects, derives the **parent aspect from the click**, and returns options **with descriptions, coordinates, and drift**. Seeds/target are recorded as per-node **provenance** (chips in the Context panel).
5. Clicking a **corpus diamond** opens that real project pinned at the top of the Related Projects panel; the pin releases on the next node selection (otherwise it would shadow every later node's results as a stuck first entry). Clicking a **node** dot updates `selection`; co-located nodes get a count badge + chooser popover.
6. **Candidates**: choose one option per aspect (Context panel button) → the composition is embedded and drawn as a **star**, with its closest real precedents in the Candidate panel; compare and export from there. While a candidate is active, the **Inspector dock** (Part 12 B1) renders its Examine strips in the right column of the map view, so alignment is visible while moving on the map. See [`DESIGN-SPACE-VIZ.md`](../documentations/DESIGN-SPACE-VIZ.md), [`DESIGN-SPACE-ITERATION-PLAN.md`](../documentations/DESIGN-SPACE-ITERATION-PLAN.md), and [`DESIGN-SPACE-TESTING.md`](../documentations/DESIGN-SPACE-TESTING.md).

### Placeholder filter
The backend returns `{ Name: "Relevant projects will appear here" }` when Supabase has no matches. The page filters this out before passing `relatedProjects` to the generate-nodes call.

---

## Component Map

*(Rebuilt 2026-07-03 from the actual tree — the previous map predated the entire
design-space subsystem.)*

```
src/
├── app/
│   ├── layout.tsx / providers.tsx    # Root layout, QueryClient provider
│   ├── page.tsx                      # Landing page (stale demo copy — see PROJECT-REPORT §5.7)
│   └── mindmap/
│       └── page.tsx                  # Main orchestrator — all view wiring (~2.3k lines)
├── components/
│   ├── design-space/
│   │   ├── design-space-surface.tsx  # The map: SVG lattice, glyphs, honesty layer, lens
│   │   ├── schema-table.tsx          # The living schema (aspects × options, receipts, facets)
│   │   ├── cross-tab-view.tsx        # Option×option morphological lens + generate-into-gap
│   │   ├── axes-view.tsx             # Bipolar scatter (Perspectives drill-down)
│   │   ├── examine-view.tsx          # Perspectives: alignment instrument
│   │   ├── candidate-strips.tsx      # Shared strips + steering rails (dock + Perspectives)
│   │   ├── candidate-panel.tsx       # Dual-layer candidates (choices + brief)
│   │   ├── compare-candidates-dialog.tsx
│   │   ├── steer-result-card.tsx     # The veto card
│   │   ├── proposal-chips.tsx        # C1 informing-back chips
│   │   ├── reflection-chip.tsx       # C2 burden-inverted reflections
│   │   └── replay-timeline.tsx       # C3 Fusion-style timeline
│   ├── mindmap/
│   │   ├── simple-mindmap.tsx        # mind-elixir wrapper
│   │   └── simple-project-panel.tsx  # related projects list
│   └── ui/                           # shadcn/ui atoms (badge, button, card, collapsible,
│                                     #   dialog, input, scroll-area, separator, sheet, sonner)
├── features/
│   ├── design-space/
│   │   ├── hooks/                    # 18 hooks — see REACT-QUERY.md
│   │   ├── schema-utils.ts           # Pure schema/annotation/coverage view models (+tests)
│   │   ├── replay-utils.ts           # Pure replay overlay (+tests)
│   │   ├── examine-utils.ts          # Pure strip metrics (+tests)
│   │   ├── candidate-utils.ts        # Candidate text composition (+tests)
│   │   ├── exploration-stats.ts      # Pure study stats (+tests)
│   │   └── types.ts                  # Projection payload types
│   └── mindmap/
│       ├── components/               # generate-taxonomy-dialog, generate-nodes-dialog
│       ├── data/schema-mindmap-data.ts  # Default taxonomy (public/schema_selected.json, 6 aspects)
│       ├── hooks/                    # 3 hooks — see REACT-QUERY.md
│       ├── tree-utils.ts             # Pure tree ops (+tests)
│       └── types.ts                  # MindmapNode, MindmapSelection, etc.
├── store/
│   └── mindmap-store.ts              # THE store — see ZUSTAND.md (+ invariant tests)
├── types/
│   ├── openapi.ts                    # Auto-generated — do not edit
│   └── api-aliases.ts                # Regen-safe aliases + job-result shapes
└── lib/
    ├── api-client.ts                 # Direct backend baseURL (see CLAUDE.md)
    ├── run-job.ts                    # 202-job polling (1.5s interval, 5min default timeout)
    ├── session-io.ts                 # Versioned session save/load (+tests)
    ├── export-exploration.ts         # Markdown exploration record (+tests)
    ├── view-interactions.ts          # Shared zoom/pan grammar
    ├── svg-glyphs.ts / node-colors.ts
    └── utils.ts
```

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `next` | ^16.2.7 | Framework |
| `react` | 19.2.3 | UI |
| `@tanstack/react-query` | ^5.90 | Server state / async data |
| `zustand` | ^5.0 | Client state (persist schema **version 2** — see ZUSTAND.md) |
| `axios` | ^1.13 | HTTP client |
| `mind-elixir` | ^5.9 | Mindmap renderer |
| `radix-ui` | ^1 | Headless UI primitives |
| `tailwindcss` | ^4 | Styling |
| `lucide-react` | ^0.577 | Icons |

---

## Regenerate OpenAPI Types

Run whenever backend request/response models change (backend must be running):
```bash
cd llmind-web
npx -y openapi-typescript http://localhost:8000/openapi.json -o src/types/openapi.ts
```

`openapi.ts` is rewritten wholesale — that is safe because app code imports from
`src/types/api-aliases.ts`, which aliases the generated component schemas and
hand-declares **async job result** shapes (generate-nodes / generate-at results
travel through `GET /api/jobs/{id}`, which OpenAPI types as `unknown`). If a
backend Pydantic model used by a job result changes, update `api-aliases.ts` by
hand to match.
