import { useQuery } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import api from '@/src/lib/api-client';
import type { AxesResponse } from '../types';

export interface AxesQueryParams {
  xPoleA: string;
  xPoleB: string;
  yPoleA: string;
  yPoleB: string;
  items: Array<{ node_id: string; text: string }>;
}

const fetchAxes = async (params: AxesQueryParams): Promise<AxesResponse> => {
  try {
    const { data } = await api.post<AxesResponse>('/api/projection/axes', {
      x: { pole_a: { text: params.xPoleA }, pole_b: { text: params.xPoleB } },
      y: { pole_a: { text: params.yPoleA }, pole_b: { text: params.yPoleB } },
      items: params.items,
    });
    return data;
  } catch (error) {
    if (isAxiosError(error)) {
      const detail =
        error.response?.data && typeof error.response.data === 'object' && 'detail' in error.response.data
          ? String((error.response.data as { detail: unknown }).detail)
          : error.message;
      throw new Error(`Axes unavailable: ${detail}`);
    }
    throw new Error('Axes unavailable.');
  }
};

/**
 * Exact bipolar coordinates for the Perspectives view. Deterministic per pole
 * pair, so results cache aggressively; recomputed when poles or items change.
 */
export const useAxesQuery = (params: AxesQueryParams | null) =>
  useQuery({
    queryKey: ['projection-axes', params],
    queryFn: () => fetchAxes(params as AxesQueryParams),
    enabled: Boolean(params),
    staleTime: 10 * 60 * 1000,
    retry: 1,
  });
