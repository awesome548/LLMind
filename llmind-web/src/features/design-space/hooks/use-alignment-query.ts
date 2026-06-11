import { useQuery } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import api from '@/src/lib/api-client';
import type { AlignmentAspectInput } from '../candidate-utils';
import type { AlignmentResponse } from '../types';

export interface AlignmentQueryParams {
  brief: string;
  composition: string;
  aspects: AlignmentAspectInput[];
}

const fetchAlignment = async (
  params: AlignmentQueryParams
): Promise<AlignmentResponse> => {
  try {
    const { data } = await api.post<AlignmentResponse>(
      '/api/candidates/alignment',
      params
    );
    return data;
  } catch (error) {
    if (isAxiosError(error)) {
      const detail =
        error.response?.data && typeof error.response.data === 'object' && 'detail' in error.response.data
          ? String((error.response.data as { detail: unknown }).detail)
          : error.message;
      throw new Error(`Alignment unavailable: ${detail}`);
    }
    throw new Error('Alignment unavailable.');
  }
};

/**
 * How the candidate's two layers agree: cos(brief, composition) overall, and
 * per aspect whether the brief leans toward the chosen option or its strongest
 * competitor. Enabled only when both layers exist.
 */
export const useAlignmentQuery = (params: AlignmentQueryParams | null) =>
  useQuery({
    queryKey: ['candidate-alignment', params],
    queryFn: () => fetchAlignment(params as AlignmentQueryParams),
    enabled: Boolean(params),
    staleTime: 5 * 60 * 1000,
    retry: 1,
  });
