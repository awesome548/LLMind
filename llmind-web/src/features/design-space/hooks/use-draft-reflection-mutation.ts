import { useMutation } from '@tanstack/react-query';
import { runJob } from '@/src/lib/run-job';

/**
 * Draft a one-line rationale for an exploration event (Part 12 C2). Slow on
 * a thinking model — the chip opens immediately with an empty input and the
 * draft fills in IF it arrives before the designer types; never blocking.
 */
export const useDraftReflectionMutation = () =>
  useMutation({
    mutationFn: ({ context }: { context: string }) =>
      // Generous budget: a thinking model under concurrent load (annotation,
      // steering) can hold a one-liner in queue for minutes.
      runJob<{ draft: string }>(
        '/api/reflections/draft',
        { context },
        { timeoutMs: 5 * 60 * 1000 }
      ),
  });
