import { useMutation } from '@tanstack/react-query';
import { runJob } from '@/src/lib/run-job';

export interface SteerInput {
  text: string;
  mode: 'metric' | 'toward' | 'away';
  metric?: { pole_a_text: string; pole_b_text: string; target_score: number };
  reference?: { text: string; weight: number };
  preserve: string[];
}

export interface SteerMeasurement {
  mode: string;
  /** What the designer asked for (target score / signed precedent weight). */
  requested: number;
  /** What the revision actually moved, on the same scale. */
  achieved: number;
  /** Displacement along the requested direction (raw cosine space). */
  along: number;
  /** Displacement orthogonal to it — the move's side effects. */
  orthogonal: number;
  score_before: number | null;
  score_after: number | null;
}

/** Async-job result shape (via GET /api/jobs/{id} — hand-written like the
 * annotation response). `measurement` is null when the embedding service
 * failed AFTER the revision was generated — the revision survives unmeasured. */
export interface SteerResult {
  revised_text: string;
  named_qualities: string[];
  measurement: SteerMeasurement | null;
}

/**
 * One steering move on the active candidate's brief (Part 12 B3). The move is
 * made in language by the local LLM; the measurement reports requested vs
 * achieved. The result is ALWAYS shown for veto — never auto-committed.
 */
export const useSteerMutation = () =>
  useMutation({
    mutationFn: (input: SteerInput) =>
      runJob<SteerResult>('/api/candidates/steer', input, { timeoutMs: 5 * 60 * 1000 }),
  });
