import { useMutation } from '@tanstack/react-query';
import { runJob } from '@/src/lib/run-job';
import { flattenMindmapNodes } from '@/src/features/mindmap/hooks/use-generate-nodes-mutation';
import type { MindmapNode, MindmapSelection } from '@/src/features/mindmap/types';
import type { GenerateAtResponse } from '../types';

export interface GenerateAtParams {
  x: number;
  y: number;
  allNodes: ReadonlyArray<MindmapNode>;
  focusNode: Pick<MindmapNode, 'id' | 'topic'>;
  lineage: MindmapSelection['lineage'];
  k?: number;
  mode?: 'openai' | 'vllm';
  reasoningEffort?: string;
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
    // Omitted unless explicitly set: the backend derives the generation backend
    // from its config (local stack when VECTOR_STORE=local), matching how the
    // design space embeds and retrieves.
    mode: params.mode,
    reasoning_effort: params.reasoningEffort ?? 'medium',
  };

  // Long generation runs as a backend job; poll to completion (keeps each
  // request short and avoids dropped long-held connections).
  return runJob<GenerateAtResponse>('/api/projection/generate-at', request);
};

/**
 * Spatial-neighbour RAG: turn a clicked empty location into new child nodes of
 * the focus branch, seeded by the corpus projects nearest that spot. The
 * response carries both the new nodes and their coordinates in the frozen space.
 */
export const useGenerateAtMutation = () =>
  useMutation({ mutationFn: generateAt });
