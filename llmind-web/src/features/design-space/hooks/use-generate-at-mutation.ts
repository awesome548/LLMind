import { useMutation } from '@tanstack/react-query';
import { runJob } from '@/src/lib/run-job';
import { flattenMindmapNodes } from '@/src/features/mindmap/hooks/use-generate-nodes-mutation';
import type { MindmapNode, MindmapSelection } from '@/src/features/mindmap/types';
import type { CoordMap, GenerateAtResponse } from '../types';

export interface GenerateAtParams {
  x: number;
  y: number;
  allNodes: ReadonlyArray<MindmapNode>;
  focusNode: Pick<MindmapNode, 'id' | 'topic'>;
  lineage: MindmapSelection['lineage'];
  /** Current node coordinates — lets the backend derive the parent aspect and
   * the nearby-ideas list from the same click that picks the seeds. */
  coords?: CoordMap;
  k?: number;
  mode?: 'openai' | 'vllm';
  reasoningEffort?: string;
  /** Aborts the client-side wait (the backend job itself runs to completion). */
  signal?: AbortSignal;
}

const generateAt = async (params: GenerateAtParams): Promise<GenerateAtResponse> => {
  const request = {
    x: params.x,
    y: params.y,
    focus_node_id: params.focusNode.id,
    focus_node_topic: params.focusNode.topic,
    taxonomy_nodes: flattenMindmapNodes(params.allNodes),
    lineage: [...params.lineage],
    k: params.k ?? 5,
    coords: params.coords
      ? Object.entries(params.coords).map(([node_id, c]) => ({
          node_id,
          x: c.x,
          y: c.y,
        }))
      : undefined,
    // Omitted unless explicitly set: the backend derives the generation backend
    // from its config (local stack when VECTOR_STORE=local), matching how the
    // design space embeds and retrieves.
    mode: params.mode,
    reasoning_effort: params.reasoningEffort ?? 'medium',
  };

  // Long generation runs as a backend job; poll to completion (keeps each
  // request short and avoids dropped long-held connections).
  return runJob<GenerateAtResponse>('/api/projection/generate-at', request, {
    signal: params.signal,
  });
};

/**
 * Location-conditioned generation: turn a clicked empty location into new
 * options that fill that gap, seeded by corpus projects bracketing the spot.
 * The response carries the new nodes, their coordinates, their drift from the
 * click, and the seed projects (provenance).
 */
export const useGenerateAtMutation = () =>
  useMutation({ mutationFn: generateAt });
