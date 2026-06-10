# Zustand — mindmap-store

Single store for the mindmap page. Uses `devtools` + `persist` middleware.
Persisted to `localStorage` under the key `mindmap-store` (**version 2**, with a
migration that upgrades v1 state by rebuilding the tree from its taxonomy).

The store is the **single source of truth for the exploration**: the working
tree, design-space coordinates, discovered cells, provenance, candidates, and
pruning state all live (and persist) here — a reload loses nothing.

---

## State Shape

```ts
interface MindmapStoreState {
  // ── UI context ─────────────────────────────────────────────────
  contextText: string;           // Breadcrumb label built from lineage
  contextDescription: string;    // Description of the selected topic
  selectedTopic: string;         // Currently active topic name

  // ── Projects ───────────────────────────────────────────────────
  projects: MindmapProjectSchema[];
  projectsLoading: boolean;

  // ── Generated taxonomy + working tree ──────────────────────────
  taxonomy: TaxonomyInput | null;       // Last result from POST /api/taxonomy/generate
  nodes: ReadonlyArray<MindmapNode>;    // The working tree (incl. generated nodes)

  // ── Design-space exploration state ─────────────────────────────
  coords: CoordMap;                     // node.id → {x, y, z?, confidence?}
  discovered: Record<string, GenerationTrail>;  // "gx,gy" → trail (+ meanDrift)
  provenance: Record<string, NodeProvenance>;   // node.id → seeds/click/source
  descriptionById: Record<string, string>;      // generated nodes' one-line descs

  // ── Candidates + pruning (Iteration C) ─────────────────────────
  candidates: Record<string, DesignCandidate>;  // id → {name, choices, note}
  activeCandidateId: string | null;
  optionState: Record<string, OptionStateEntry>; // node.id → {state:'rejected', reason?}

  // ── Internal ───────────────────────────────────────────────────
  jmRef: unknown | null;           // mind-elixir instance ref

  // ── Actions ────────────────────────────────────────────────────
  selectTopic(input: MindmapSelectionInput): void;
  setTaxonomy(taxonomy: TaxonomyInput): void;  // ALSO rebuilds nodes + wipes all
                                               // exploration state (new taxonomy
                                               // = new design-space overlay)
  setNodes(nodes): void;
  mergeCoords(coords): void;
  removeCoords(ids): void;          // e.g. after a rename → re-locate
  recordDiscovery(cellKey, trail): void;
  recordProvenance(entries): void;
  mergeDescriptions(entries): void;
  createCandidate(name?): string;   // creates + activates, returns id
  deleteCandidate(id): void;
  setActiveCandidate(id | null): void;
  renameCandidate(id, name): void;
  setChoice(aspectId, optionId | null): void;  // active candidate, radio per aspect
  rejectOption(nodeId, reason?): void;
  reopenOption(nodeId): void;
  setProjects / setProjectsLoading / setContext / setMindmapData / setJmRef / resetMindmapStore
}
```

---

## Persistence

Persisted via `partialize`: `contextText`, `contextDescription`, `selectedTopic`,
`projects`, `taxonomy`, **`nodes`**, **`coords`**, **`discovered`**,
**`provenance`**, **`descriptionById`**, **`candidates`**,
**`activeCandidateId`**, **`optionState`**.

`jmRef` and loading flags are **not** persisted. The locate "attempted once"
guard is session-local React state (a retry guard, not data).

**Versioning:** bump `version` and extend `migrate` whenever the persisted shape
changes. v1 → v2 rebuilt `nodes` from the persisted taxonomy and started the
exploration maps empty.

---

## Usage in Components

```tsx
// Read a single value (subscribe only to that slice)
const nodes = useMindmapStore((state) => state.nodes);

// Read an action
const mergeCoords = useMindmapStore((state) => state.mergeCoords);

// Compose a candidate: create (auto-activates), then choose options
const createCandidate = useMindmapStore((s) => s.createCandidate);
const setChoice = useMindmapStore((s) => s.setChoice);
createCandidate('Variant A');
setChoice('display-technology', 'led-wall-panels');
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
`setTaxonomy` itself converts it via `taxonomyToMindmapNodes()` and replaces the
working tree (the page no longer does this in an effect — that pattern wiped
generated nodes on reload).
