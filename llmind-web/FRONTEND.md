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
| Store | `src/store/mindmap-store.ts` | Zustand v2; persists the WHOLE exploration — tree, coords, discovered, provenance, candidates, pruning (see ZUSTAND.md) |
| Components | `src/components/mindmap/` | `SimpleMindMap` (mind-elixir wrapper; `nodeStates` styles rejected/chosen), `SimpleProjectPanel` (accepts `focusProject`) |
| Dialog | `src/features/mindmap/components/generate-taxonomy-dialog.tsx` | Taxonomy generation form (project overview, reasoning, mode) |
| Data | `src/features/mindmap/data/schema-mindmap-data.ts` | Static initial taxonomy + `taxonomyToMindmapNodes()` converter |
| Page | `src/app/mindmap/page.tsx` | Main orchestrator — wires store, hooks, components; Mind Map / Design Space / Perspectives view toggle + Similarity/Relevance-lens mode toggle |
| Design space — surface | `src/components/design-space/design-space-surface.tsx` | SVG lattice: corpus glyphs (inspectable), node dots (confidence-dashed), candidate stars, collision badges + chooser, zoom-faded density heat, trustworthiness legend, cancel button; **relevance-lens painting** (single-hue amber ramp, anchor-faded nodes, "relative" legend) |
| Design space — axes view | `src/components/design-space/axes-view.tsx` | "Perspectives": bipolar scatter on designer-chosen aspect/option poles — exact cosine scores, quadrant density shading, rug ticks, pole labels, axis-quality warnings (pole similarity, axis correlation), clip-dashed items. Read-only v1 |
| Design space — candidates | `src/components/design-space/candidate-panel.tsx`, `compare-candidates-dialog.tsx` | Compose one option per aspect; precedents for the composition; compare; export |
| Design space — hooks | `src/features/design-space/hooks/` | `use-surface-query` (gated on view), `use-locate-nodes`, `use-generate-at-mutation` (sends coords + AbortSignal), `use-corpus-project`, `use-candidate-precedents`, `use-relevance-query` (lens), `use-axes-query`, `use-pan-zoom` (shared canvas grammar) |
| Design space — utils/types | `src/features/design-space/candidate-utils.ts`, `types.ts` | Candidate text composition + hand-written projection payload types (incl. axes) |
| Shared interactions | `src/lib/view-interactions.ts`, `src/lib/svg-glyphs.ts` | One zoom factor/range for ALL canvases (the mind map mirrors it via mind-elixir `handleWheel` + `mouseSelectionButton: 2`); shared star glyph |
| Export | `src/lib/export-exploration.ts` | Markdown exploration record (taxonomy + states, candidates, provenance) |

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
3. On `nodes` change, missing nodes are embedded + placed via `POST /api/projection/locate` (best-effort). Each located point carries a **placement confidence** (dashed dot when low). Coords persist in the store; renames drop the stale coord so the node re-locates.
4. Clicking an **empty** lattice cell → `useGenerateAtMutation` (`POST /api/projection/generate-at`, async job, cancellable). The backend brackets the gap with seed projects, derives the **parent aspect from the click**, and returns options **with descriptions, coordinates, and drift**. Seeds/target are recorded as per-node **provenance** (chips in the Context panel).
5. Clicking a **corpus diamond** opens that real project in the Related Projects panel. Clicking a **node** dot updates `selection`; co-located nodes get a count badge + chooser popover.
6. **Candidates**: choose one option per aspect (Context panel button) → the composition is embedded and drawn as a **star**, with its closest real precedents in the Candidate panel; compare and export from there. See [`../DESIGN-SPACE-VIZ.md`](../DESIGN-SPACE-VIZ.md), [`../DESIGN-SPACE-ITERATION-PLAN.md`](../DESIGN-SPACE-ITERATION-PLAN.md), and [`../DESIGN-SPACE-TESTING.md`](../DESIGN-SPACE-TESTING.md).

### Placeholder filter
The backend returns `{ Name: "Relevant projects will appear here" }` when Supabase has no matches. The page filters this out before passing `relatedProjects` to the generate-nodes call.

---

## Component Map

```
src/
├── app/
│   └── mindmap/
│       └── page.tsx                  # Main page — all wiring here
├── components/
│   ├── mindmap/
│   │   ├── simple-mindmap.tsx        # mind-elixir wrapper
│   │   └── simple-project-panel.tsx  # related projects list
│   └── ui/                           # shadcn/ui atoms
│       ├── button.tsx
│       ├── dialog.tsx
│       ├── badge.tsx
│       ├── collapsible.tsx
│       ├── input.tsx
│       ├── scroll-area.tsx
│       └── separator.tsx
├── features/
│   └── mindmap/
│       ├── components/
│       │   └── generate-taxonomy-dialog.tsx
│       ├── data/
│       │   └── schema-mindmap-data.ts
│       ├── hooks/
│       │   ├── use-related-projects-query.ts
│       │   ├── use-generate-nodes-mutation.ts
│       │   └── use-generate-taxonomy-mutation.ts
│       └── types.ts                  # MindmapNode, MindmapSelection, etc.
├── store/
│   └── mindmap-store.ts
├── types/
│   └── openapi.ts                    # Auto-generated — do not edit
└── lib/
    ├── api-client.ts
    └── utils.ts
```

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `next` | 16.1.6 | Framework |
| `react` | 19.2.3 | UI |
| `@tanstack/react-query` | ^5 | Server state / async data |
| `zustand` | ^5 | Client state |
| `axios` | ^1 | HTTP client |
| `mind-elixir` | ^5 | Mindmap renderer |
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
