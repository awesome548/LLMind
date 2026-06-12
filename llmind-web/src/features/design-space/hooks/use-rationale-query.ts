import { useQuery } from '@tanstack/react-query';
import { runJob } from '@/src/lib/run-job';

export interface RationaleAspectInput {
  id: string;
  name: string;
  desc: string;
  options: Array<{ name: string; count: number }>;
}

export interface RationaleResponse {
  rationales: Record<string, string>;
  meta: { model: string; version: number };
}

/**
 * The rationale layer (Part 13 L-A): one line per aspect answering the
 * study's "why these seven?" — grounded in the annotation counts, which is
 * why the hook is gated on the annotation being ready. Server-side cached
 * per aspect content + counts, so only new evidence re-drafts.
 */
export const useRationaleQuery = (
  aspects: ReadonlyArray<RationaleAspectInput>,
  nProjects: number,
  enabled: boolean
) =>
  useQuery({
    queryKey: [
      'aspect-rationale',
      aspects
        .map((a) => `${a.name}|${a.options.map((o) => `${o.name}:${o.count}`).join(',')}`)
        .sort(),
    ],
    queryFn: () =>
      // Cold run = one small LLM call per aspect (~6); cached re-runs resolve
      // in seconds.
      runJob<RationaleResponse>(
        '/api/corpus/rationale',
        { aspects, n_projects: nProjects },
        { timeoutMs: 10 * 60 * 1000 }
      ),
    enabled: enabled && aspects.length > 0,
    staleTime: Infinity,
    retry: false,
  });
