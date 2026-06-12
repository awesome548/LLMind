import { useMutation } from '@tanstack/react-query';
import { runJob } from '@/src/lib/run-job';

export interface MissingAspectInput {
  aspect_names: string[];
  /** The poorly-covered corpus projects (computed client-side from the
   * annotation — pure set arithmetic, no extra backend round-trip). */
  project_ids: string[];
}

/** Async-job result shape (travels through GET /api/jobs/{id}). */
export interface MissingAspectResult {
  proposals: Array<{ name: string; desc: string; reason: string }>;
}

/**
 * The coverage probe (Part 13 L-A): ask what dimension the poorly-covered
 * projects exemplify that the taxonomy misses. Designer-triggered, one
 * local-LLM call; results ride the C1 proposals channel as chips.
 */
export const useMissingAspectMutation = () =>
  useMutation({
    mutationFn: (input: MissingAspectInput) =>
      runJob<MissingAspectResult>('/api/corpus/missing-aspect', input, {
        timeoutMs: 5 * 60 * 1000,
      }),
  });
