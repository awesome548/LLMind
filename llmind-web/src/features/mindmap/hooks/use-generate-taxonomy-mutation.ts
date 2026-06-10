import { useMutation } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import api from '@/src/lib/api-client';

export type ReasoningEffort = 'low' | 'medium' | 'high';
export type BackendMode = 'openai' | 'vllm';

export interface GenerateTaxonomyParams {
  project_overview: string;
  reasoning_effort?: ReasoningEffort;
  mode?: BackendMode;
}

export interface TaxonomyOption {
  name: string;
  desc: string;
}

export interface TaxonomyAspect {
  name: string;
  desc: string;
  options: TaxonomyOption[];
}

export interface GenerateTaxonomyResponse {
  aspects: TaxonomyAspect[];
  /** Cosine similarity of the brief to the corpus centroid (original metric).
   * Low values mean the design-space background may not apply to this brief.
   * Null/undefined = unscored (embedding server unavailable). */
  corpus_similarity?: number | null;
}

const generateTaxonomy = async (
  params: GenerateTaxonomyParams
): Promise<GenerateTaxonomyResponse> => {
  try {
    const { data } = await api.post<GenerateTaxonomyResponse>(
      '/api/taxonomy/generate',
      params
    );
    return data;
  } catch (error) {
    if (isAxiosError(error)) {
      const status = error.response?.status;
      const detail =
        typeof error.response?.data === 'string'
          ? error.response.data
          : error.response?.data &&
              typeof error.response.data === 'object' &&
              'detail' in error.response.data
            ? String(error.response.data.detail)
            : null;

      const statusLabel = status ? `HTTP ${status}` : 'Network error';
      const reason = detail ?? error.message;
      throw new Error(`Failed to generate taxonomy (${statusLabel}): ${reason}`);
    }
    throw new Error('Failed to generate taxonomy.', { cause: error });
  }
};

export const useGenerateTaxonomyMutation = () =>
  useMutation({
    mutationFn: (params: GenerateTaxonomyParams) =>
      generateTaxonomy({
        reasoning_effort: 'medium',
        mode: 'openai',
        ...params,
      }),
  });
