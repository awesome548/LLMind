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
| Types | `src/types/openapi.ts` | Auto-generated from backend OpenAPI spec — **do not edit manually** |
| API client | `src/lib/api-client.ts` | Axios instance; `baseURL: '/'` (proxied by Next.js) |
| Hooks — queries | `src/features/mindmap/hooks/use-related-projects-query.ts` | React Query: fetch related projects on topic select |
| Hooks — mutations | `src/features/mindmap/hooks/use-generate-nodes-mutation.ts` | React Query: generate child nodes via LLM |
| Hooks — mutations | `src/features/mindmap/hooks/use-generate-taxonomy-mutation.ts` | React Query: generate full taxonomy from project overview |
| Store | `src/store/mindmap-store.ts` | Zustand; persists selection, projects, and generated taxonomy |
| Components | `src/components/mindmap/` | `SimpleMindMap` (mind-elixir wrapper), `SimpleProjectPanel` |
| Dialog | `src/features/mindmap/components/generate-taxonomy-dialog.tsx` | Taxonomy generation form (project overview, reasoning, mode) |
| Data | `src/features/mindmap/data/schema-mindmap-data.ts` | Static initial taxonomy + `taxonomyToMindmapNodes()` converter |
| Page | `src/app/mindmap/page.tsx` | Main orchestrator — wires store, hooks, components |

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

Run whenever backend request/response models change:
```bash
cd llmind-web
bunx openapi-typescript http://localhost:8000/openapi.json -o src/types/openapi.ts
```
