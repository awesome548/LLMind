import { useMutation } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import api from '@/src/lib/api-client';
import type { MindmapNode } from '@/src/features/mindmap/types';
import type { CoordMap, LocateResponse } from '../types';

export interface LocateItem {
  node_id: string;
  text: string;
}

/**
 * Flatten a mind-map tree into locate items. Each node is embedded as
 * `topic` plus its description (when available) — the same text the taxonomy
 * carries for embedding-based retrieval. Id-keyed descriptions (generated
 * nodes) take precedence over topic-keyed ones (taxonomy nodes).
 */
export function nodesToLocateItems(
  nodes: ReadonlyArray<MindmapNode>,
  descriptionByTopic: Readonly<Record<string, string>> = {},
  descriptionById: Readonly<Record<string, string>> = {}
): LocateItem[] {
  const items: LocateItem[] = [];
  const walk = (node: MindmapNode) => {
    const desc = descriptionById[node.id] ?? descriptionByTopic[node.topic] ?? '';
    const text = desc ? `${node.topic}. ${desc}` : node.topic;
    items.push({ node_id: node.id, text });
    for (const child of node.children ?? []) walk(child);
  };
  for (const node of nodes) walk(node);
  return items;
}

const locateNodes = async (items: LocateItem[]): Promise<CoordMap> => {
  if (items.length === 0) return {};
  try {
    const { data } = await api.post<LocateResponse>('/api/projection/locate', { items });
    const coords: CoordMap = {};
    for (const p of data.points) {
      coords[p.node_id] = {
        x: p.x,
        y: p.y,
        ...(p.z != null ? { z: p.z } : {}),
        ...(p.confidence != null ? { confidence: p.confidence } : {}),
        ...(p.support != null ? { support: p.support } : {}),
      };
    }
    return coords;
  } catch (error) {
    if (isAxiosError(error)) {
      const detail =
        error.response?.data && typeof error.response.data === 'object' && 'detail' in error.response.data
          ? String((error.response.data as { detail: unknown }).detail)
          : error.message;
      throw new Error(`Failed to locate nodes: ${detail}`);
    }
    throw new Error('Failed to locate nodes.');
  }
};

/** Embeds taxonomy nodes and returns their coordinates in the frozen space. */
export const useLocateNodesMutation = () =>
  useMutation({ mutationFn: locateNodes });
