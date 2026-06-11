import { useQuery } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import api from '@/src/lib/api-client';
import type { MetricsResponse } from '../types';

export interface MetricsQueryParams {
  metrics: Array<{ poleA: string; poleB: string }>;
  items: Array<{ node_id: string; text: string }>;
}

const fetchMetrics = async (params: MetricsQueryParams): Promise<MetricsResponse> => {
  try {
    const { data } = await api.post<MetricsResponse>('/api/projection/metrics', {
      metrics: params.metrics.map((m) => ({
        pole_a: { text: m.poleA },
        pole_b: { text: m.poleB },
      })),
      items: params.items,
    });
    return data;
  } catch (error) {
    if (isAxiosError(error)) {
      const detail =
        error.response?.data && typeof error.response.data === 'object' && 'detail' in error.response.data
          ? String((error.response.data as { detail: unknown }).detail)
          : error.message;
      throw new Error(`Metrics unavailable: ${detail}`);
    }
    throw new Error('Metrics unavailable.');
  }
};

/**
 * Corpus + candidate scores along a LIST of bipolar metrics — the data behind
 * the Examine strips (exact cosine; response order matches the request order).
 */
export const useMetricsQuery = (params: MetricsQueryParams | null) =>
  useQuery({
    queryKey: ['projection-metrics', params],
    queryFn: () => fetchMetrics(params as MetricsQueryParams),
    enabled: Boolean(params && params.metrics.length > 0),
    staleTime: 10 * 60 * 1000,
    retry: 1,
  });
