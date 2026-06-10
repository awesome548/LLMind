import { useQueries, useQuery } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import api from '@/src/lib/api-client';

export interface PrecedentProject {
  id: string;
  Name: string;
  Descriptions: string;
  Details: string;
  Image?: string | null;
  /** True (original-metric) cosine similarity to the candidate text. */
  score: number;
}

interface SimilarProjectsResponse {
  projects: PrecedentProject[];
}

const fetchPrecedents = async (text: string, k: number): Promise<PrecedentProject[]> => {
  try {
    const { data } = await api.post<SimilarProjectsResponse>('/api/corpus/similar', {
      text,
      k,
    });
    return data.projects;
  } catch (error) {
    if (isAxiosError(error)) {
      const detail =
        error.response?.data && typeof error.response.data === 'object' && 'detail' in error.response.data
          ? String((error.response.data as { detail: unknown }).detail)
          : error.message;
      throw new Error(`Precedent search failed: ${detail}`);
    }
    throw new Error('Precedent search failed.');
  }
};

const precedentsQueryOptions = (text: string | null, k: number) => ({
  queryKey: ['corpus-similar', text, k] as const,
  queryFn: () => fetchPrecedents(text as string, k),
  enabled: Boolean(text && text.trim()),
  staleTime: 5 * 60 * 1000,
  retry: 1,
});

/** Closest real precedents to a composed candidate text (original metric). */
export const useCandidatePrecedentsQuery = (text: string | null, k = 5) =>
  useQuery(precedentsQueryOptions(text, k));

/** Precedents for several candidates at once (compare dialog). */
export const useManyCandidatePrecedents = (texts: Array<string | null>, k = 3) =>
  useQueries({ queries: texts.map((text) => precedentsQueryOptions(text, k)) });
