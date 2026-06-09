import { useQuery } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import api from '@/src/lib/api-client';
import type { Surface } from '../types';

const fetchSurface = async (): Promise<Surface> => {
  try {
    const { data } = await api.get<Surface>('/api/projection/surface');
    return data;
  } catch (error) {
    if (isAxiosError(error)) {
      const status = error.response?.status;
      const detail =
        error.response?.data && typeof error.response.data === 'object' && 'detail' in error.response.data
          ? String((error.response.data as { detail: unknown }).detail)
          : error.message;
      throw new Error(`Failed to load design-space surface (${status ?? 'network'}): ${detail}`);
    }
    throw new Error('Failed to load design-space surface.');
  }
};

/**
 * Loads the precomputed corpus background once. The surface rarely changes
 * (only when `database_pipeline.py project` is re-run), so it is cached
 * aggressively.
 */
export const useSurfaceQuery = (enabled = true) =>
  useQuery({
    queryKey: ['design-space', 'surface'],
    queryFn: fetchSurface,
    enabled,
    staleTime: Infinity,
    gcTime: Infinity,
    retry: 1,
  });
