import { useMutation } from '@tanstack/react-query';
import { runJob } from '@/src/lib/run-job';

export interface GenerateCellInput {
  aspect_a: string;
  option_a: { name: string; desc: string };
  aspect_b: string;
  option_b: { name: string; desc: string };
  /** Half-matching receipts from the annotation (the prompt's seeds). */
  exemplar_ids: string[];
  /** Cancels the client-side wait (e.g. on view unmount); the job itself
   * completes server-side and its result is discarded. */
  signal?: AbortSignal;
}

/** Async-job result shape (travels through GET /api/jobs/{id} — invisible to
 * OpenAPI, hence hand-written here like the annotation response). */
export interface GenerateCellResult {
  name: string;
  desc: string;
  cell: [string, string];
  exemplars_used: number;
}

/**
 * Generate ONE concept into an empty cross-tab cell (Part 12 B2): the
 * morphological-combination → candidate-skeleton flow. One local-LLM call;
 * thinking models take a minute or two — generous job budget.
 */
export const useGenerateCellMutation = () =>
  useMutation({
    mutationFn: ({ signal, ...input }: GenerateCellInput) =>
      runJob<GenerateCellResult>('/api/corpus/generate-cell', input, {
        timeoutMs: 5 * 60 * 1000,
        signal,
      }),
  });
