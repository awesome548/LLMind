# Zustand — mindmap-store

Single store for the mindmap page. Uses `devtools` + `persist` middleware.
Persisted to `localStorage` under the key `mindmap-store`.

---

## State Shape

```ts
interface MindmapStoreState {
  // ── UI context ─────────────────────────────────────────────────
  contextText: string;           // Breadcrumb label built from lineage
  contextDescription: string;   // Description of the selected topic
  selectedTopic: string;         // Currently active topic name

  // ── Projects ───────────────────────────────────────────────────
  projects: MindmapProjectSchema[];
  projectsLoading: boolean;

  // ── Generated taxonomy ─────────────────────────────────────────
  taxonomy: TaxonomyInput | null;  // Last result from POST /api/taxonomy/generate

  // ── Internal ───────────────────────────────────────────────────
  jmRef: unknown | null;           // mind-elixir instance ref

  // ── Actions ────────────────────────────────────────────────────
  selectTopic(input: MindmapSelectionInput): void;
  setTaxonomy(taxonomy: TaxonomyInput): void;
  setProjects(projects: MindmapProjectSchema[]): void;
  setProjectsLoading(isLoading: boolean): void;
  setContext(context: { contextText: string; contextDescription: string }): void;
  setMindmapData(payload: Partial<{ contextText; contextDescription; projects; projectsLoading }>): void;
  setJmRef(ref: unknown | null): void;
  resetMindmapStore(): void;
}
```

---

## Persistence

The following fields are persisted to `localStorage` via `partialize`:

| Field | Reason |
|---|---|
| `contextText` | Restore last breadcrumb on revisit |
| `contextDescription` | Restore topic description |
| `selectedTopic` | Restore last selected node |
| `projects` | Avoid re-fetch on revisit |
| `taxonomy` | Preserve generated taxonomy across sessions |

`jmRef` and loading flags are **not** persisted.

---

## Usage in Components

```tsx
// Read a single value (subscribe only to that slice)
const taxonomy = useMindmapStore((state) => state.taxonomy);

// Read an action
const setTaxonomy = useMindmapStore((state) => state.setTaxonomy);

// Select a topic (updates contextText + contextDescription atomically)
const selectTopic = useMindmapStore((state) => state.selectTopic);
selectTopic({ topic: 'Interaction', lineage: ['Design Aspects', 'Interaction'], contextDescription: '...' });
```

---

## TaxonomyInput Type

Defined in `src/features/mindmap/data/schema-mindmap-data.ts`:

```ts
interface TaxonomyInput {
  aspects: ReadonlyArray<{
    name: string;
    desc: string;
    options: ReadonlyArray<{ name: string; desc: string }>;
  }>;
}
```

This matches the shape returned by `POST /api/taxonomy/generate`.
When `setTaxonomy` is called, `page.tsx` converts it to `MindmapNode[]` via `taxonomyToMindmapNodes()` and replaces the mindmap tree.
