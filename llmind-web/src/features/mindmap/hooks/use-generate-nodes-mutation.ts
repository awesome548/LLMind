import { useMutation } from '@tanstack/react-query';
import { runJob } from '@/src/lib/run-job';
import type {
  GenerateNodesRequestSchema,
  GenerateNodesResponseSchema,
  TaxonomyNodeInputSchema,
} from '@/src/types/api-aliases';
import type { MindmapNode } from '../types';

// ── Utilities ────────────────────────────────────────────────────────────────

/**
 * Flattens the whole MindmapNode tree so the backend sees every existing topic.
 * This is the "whole-map context" — prevents the LLM from generating duplicates.
 */
export function flattenMindmapNodes(
  nodes: ReadonlyArray<MindmapNode>
): TaxonomyNodeInputSchema[] {
  const result: TaxonomyNodeInputSchema[] = [];

  function traverse(node: MindmapNode, parentId: string | null, isRoot: boolean): void {
    result.push({ id: node.id, topic: node.topic, parentid: parentId ?? undefined, isroot: isRoot });
    for (const child of node.children ?? []) {
      traverse(child, node.id, false);
    }
  }

  for (const node of nodes) {
    traverse(node, null, true);
  }
  return result;
}

/**
 * Derives the lineage (root → focusNode) from the flattened node list.
 * Walks parentid links upward to build the ancestor chain.
 */
function deriveLineage(
  flat: TaxonomyNodeInputSchema[],
  focusNodeId: string
): string[] {
  const byId = new Map(flat.map((n) => [n.id, n]));

  const lineage: string[] = [];
  let current = byId.get(focusNodeId);
  while (current) {
    lineage.unshift(current.topic);
    current = current.parentid ? byId.get(current.parentid) : undefined;
  }
  return lineage;
}

/**
 * Converts the backend's flat GeneratedNode list to MindmapNode[].
 *
 * All returned nodes are direct children of the focus node — the backend
 * never returns multi-level trees here, so a simple map is correct.
 * The caller attaches them under `response.parent_id`.
 */
export function generatedNodesToMindmapNodes(
  response: GenerateNodesResponseSchema
): MindmapNode[] {
  return response.nodes.map((n) => ({ id: n.node_id, topic: n.topic }));
}

// ── API call ─────────────────────────────────────────────────────────────────

const generateNodes = (
  request: GenerateNodesRequestSchema
): Promise<GenerateNodesResponseSchema> =>
  // Long generation runs as a backend job; poll to completion (short requests,
  // no dropped long-held connection).
  runJob<GenerateNodesResponseSchema>('/api/related-projects/generate-nodes', request);

// ── Hook ─────────────────────────────────────────────────────────────────────

export interface GenerateNodesParams {
  /**
   * The full mindmap tree. Sent as context so the backend avoids generating
   * topics that already exist anywhere in the map.
   */
  allNodes: ReadonlyArray<MindmapNode>;
  /**
   * The node the user selected — this is where new children will be added.
   */
  focusNode: Pick<MindmapNode, 'id' | 'topic'>;
  description?: string;
  shouldQuerySupabase?: boolean;
  relatedProjects?: GenerateNodesRequestSchema['related_projects'];
  model?: string;
  mode?: GenerateNodesRequestSchema['mode'];
  baseUrl?: string;
  reasoningEffort?: string;
}

/**
 * Mutation hook for node generation.
 *
 * Usage:
 *   const { mutate, isPending, data } = useGenerateNodesMutation();
 *   mutate({ allNodes: nodes, focusNode: clickedNode });
 *
 * On success:
 *   data.parent_id   — id of the focus node (attach children here)
 *   generatedNodesToMindmapNodes(data) — new MindmapNode[] to insert
 */
export const useGenerateNodesMutation = () =>
  useMutation({
    mutationFn: ({
      allNodes,
      focusNode,
      description,
      shouldQuerySupabase = true,
      relatedProjects,
      model,
      mode = 'openai',
      baseUrl,
      reasoningEffort = 'medium',
    }: GenerateNodesParams) => {
      const taxonomyNodes = flattenMindmapNodes(allNodes);
      const lineage = deriveLineage(taxonomyNodes, focusNode.id);

      const request: GenerateNodesRequestSchema = {
        taxonomy_nodes: taxonomyNodes,
        focus_node: { id: focusNode.id, topic: focusNode.topic },
        lineage,
        description: description ?? null,
        should_query_supabase: shouldQuerySupabase,
        related_projects: relatedProjects ?? null,
        model: model ?? null,
        mode,
        base_url: baseUrl ?? null,
        reasoning_effort: reasoningEffort,
      };
      return generateNodes(request);
    },
  });
