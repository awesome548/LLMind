import { useMutation } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import api from '@/src/lib/api-client';
import { flattenMindmapNodes } from '@/src/features/mindmap/hooks/use-generate-nodes-mutation';
import type { MindmapNode } from '@/src/features/mindmap/types';
import type { CoordMap } from '../types';

export interface PeekSeed {
  id: string | null;
  Name: string;
  Descriptions: string;
  x: number;
  y: number;
}

export interface PeekResponse {
  seeds: PeekSeed[];
  nearby_options: string[];
  parent_aspect: string | null;
}

interface PeekParams {
  x: number;
  y: number;
  allNodes: ReadonlyArray<MindmapNode>;
  coords: Readonly<CoordMap>;
  k?: number;
}

const peekAt = async (params: PeekParams): Promise<PeekResponse> => {
  try {
    const { data } = await api.post<PeekResponse>('/api/projection/peek', {
      x: params.x,
      y: params.y,
      k: params.k ?? 5,
      taxonomy_nodes: flattenMindmapNodes(params.allNodes),
      coords: Object.entries(params.coords)
        .filter(([id]) => !id.startsWith('cand:'))
        .map(([node_id, c]) => ({ node_id, x: c.x, y: c.y })),
    });
    return data;
  } catch (error) {
    if (isAxiosError(error)) {
      const detail =
        error.response?.data && typeof error.response.data === 'object' && 'detail' in error.response.data
          ? String((error.response.data as { detail: unknown }).detail)
          : error.message;
      throw new Error(`Gap preview failed: ${detail}`);
    }
    throw new Error('Gap preview failed.');
  }
};

/**
 * Gap preview (E1): what a generation at a clicked location WOULD be briefed
 * with — its deterministic seed set, nearby explored ideas, and the parent
 * aspect — before any LLM time is committed. Fast: no LLM, no embed server.
 */
export const usePeekMutation = () => useMutation({ mutationFn: peekAt });
