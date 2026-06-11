import { useQuery } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import api from '@/src/lib/api-client';

interface RelevanceResponse {
  scores: Array<{ id: string; score: number }>;
  min: number;
  max: number;
}

export interface RelevanceMap {
  /** corpus id → normalized 0..1 relevance (min-max over this query). */
  byId: Record<string, number>;
  /** Raw cosine range — shown in the legend so "relative" is explicit. */
  min: number;
  max: number;
}

const fetchRelevance = async (text: string): Promise<RelevanceMap> => {
  try {
    const { data } = await api.post<RelevanceResponse>('/api/corpus/relevance', { text });
    const span = data.max - data.min || 1;
    const byId: Record<string, number> = {};
    for (const { id, score } of data.scores) {
      byId[id] = (score - data.min) / span;
    }
    return { byId, min: data.min, max: data.max };
  } catch (error) {
    if (isAxiosError(error)) {
      const detail =
        error.response?.data && typeof error.response.data === 'object' && 'detail' in error.response.data
          ? String((error.response.data as { detail: unknown }).detail)
          : error.message;
      throw new Error(`Relevance lens unavailable: ${detail}`);
    }
    throw new Error('Relevance lens unavailable.');
  }
};

/**
 * True cosine relevance of every corpus project to an anchor text — the
 * design-space relevance lens. Faithful (original 768-d metric) even where the
 * 2D layout is distorted; normalized per query, so the painting is RELATIVE.
 */
export const useRelevanceQuery = (text: string | null) =>
  useQuery({
    queryKey: ['corpus-relevance', text],
    queryFn: () => fetchRelevance(text as string),
    enabled: Boolean(text && text.trim()),
    staleTime: 5 * 60 * 1000,
    retry: 1,
  });
