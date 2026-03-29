# React Query — Hooks Reference

TanStack Query v5. All hooks live in `src/features/mindmap/hooks/`.

---

## Queries

### `useRelatedProjectsQuery`
**File:** `hooks/use-related-projects-query.ts`

Auto-fires on topic selection change. Fetches related projects from Supabase via the backend.

```ts
const { data, isFetching } = useRelatedProjectsQuery({ request });
// data: { projects: MindmapProjectSchema[] } | undefined
```

- `queryKey`: derived from the full request object — refetches when topic or lineage changes
- `staleTime`: `0` — always re-fetches on selection change
- Backend endpoint: `POST /api/related-projects/search`

---

## Mutations

### `useGenerateNodesMutation`
**File:** `hooks/use-generate-nodes-mutation.ts`

Generates child nodes for the selected mindmap node using an LLM.

```ts
const { mutateAsync, isPending } = useGenerateNodesMutation();

const response = await mutateAsync({
  allNodes,           // full mindmap tree (for deduplication context)
  focusNode,          // { id, topic } of the selected node
  description,        // optional topic description
  shouldQuerySupabase,
  relatedProjects,    // pass fetched projects to skip Supabase lookup
  reasoningEffort,    // "low" | "medium" | "high"
  mode,               // "openai" | "vllm"
});
// response: { parent_id, nodes: GeneratedNode[], related_projects, options }
```

Internally: flattens the full tree with `flattenMindmapNodes()`, derives lineage from `focusNode.id`, then posts to `POST /api/related-projects/generate-nodes`.

**Exported utilities:**
- `flattenMindmapNodes(nodes)` → `TaxonomyNodeInputSchema[]` — converts tree to flat list
- `generatedNodesToMindmapNodes(response)` → `MindmapNode[]` — converts response to tree nodes

---

### `useGenerateTaxonomyMutation`
**File:** `hooks/use-generate-taxonomy-mutation.ts`

Generates a full taxonomy from a project overview. Called from `GenerateTaxonomyDialog`.

```ts
const { mutate, isPending, error, data, reset } = useGenerateTaxonomyMutation();

mutate(
  { project_overview, reasoning_effort, mode },
  { onSuccess: (result) => setTaxonomy(result) }
);
// result: { aspects: TaxonomyAspect[] }
```

Backend endpoint: `POST /api/taxonomy/generate`

**Types** (defined in the hook file):
```ts
type ReasoningEffort = 'low' | 'medium' | 'high';
type BackendMode = 'openai' | 'vllm';

interface GenerateTaxonomyResponse {
  aspects: Array<{
    name: string;
    desc: string;
    options: Array<{ name: string; desc: string }>;
  }>;
}
```

---

## Error Handling Pattern

All API calls follow the same pattern:

```ts
try {
  const { data } = await api.post('/api/...', payload);
  return data;
} catch (error) {
  if (isAxiosError(error)) {
    const detail = error.response?.data?.detail ?? error.message;
    throw new Error(`Failed to ... (HTTP ${error.response?.status}): ${detail}`);
  }
  throw new Error('Failed to ...', { cause: error });
}
```

The `detail` field in 502 responses contains the backend `ServiceError` message including `request_id` and `stage` for tracing.

---

## Provider Setup

Configured in `src/app/layout.tsx` with `QueryClientProvider`. Default stale/cache times apply unless overridden per-query.
