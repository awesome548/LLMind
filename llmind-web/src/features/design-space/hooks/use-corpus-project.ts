import { useQuery } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import api from '@/src/lib/api-client';

export interface CorpusProject {
  id: string;
  Name: string;
  Descriptions: string;
  Details: string;
  Image?: string | null;
}

const fetchCorpusProject = async (projectId: string): Promise<CorpusProject> => {
  try {
    const { data } = await api.get<CorpusProject>(`/api/corpus/projects/${projectId}`);
    return data;
  } catch (error) {
    if (isAxiosError(error)) {
      const detail =
        error.response?.data && typeof error.response.data === 'object' && 'detail' in error.response.data
          ? String((error.response.data as { detail: unknown }).detail)
          : error.message;
      throw new Error(`Failed to load project: ${detail}`);
    }
    throw new Error('Failed to load project.');
  }
};

/**
 * One corpus project's metadata — used when a real-project glyph on the design
 * space (or a provenance chip) is opened. Corpus records never change at
 * runtime, so cache indefinitely.
 */
export const useCorpusProjectQuery = (projectId: string | null) =>
  useQuery({
    queryKey: ['corpus-project', projectId],
    queryFn: () => fetchCorpusProject(projectId as string),
    enabled: Boolean(projectId),
    staleTime: Infinity,
    gcTime: 10 * 60 * 1000,
    retry: 1,
  });
