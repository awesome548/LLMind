import { useMutation } from '@tanstack/react-query';
import { runJob } from '@/src/lib/run-job';

export interface DraftBriefParams {
  /** One row per committed choice: the dimensions the brief must embody. */
  aspects: Array<{ aspect: string; option: string; desc?: string }>;
  projectOverview?: string;
  signal?: AbortSignal;
}

const draftBrief = async (params: DraftBriefParams): Promise<{ brief: string }> =>
  runJob<{ brief: string }>(
    '/api/candidates/draft-brief',
    {
      aspects: params.aspects.map((a) => ({
        aspect: a.aspect,
        option: a.option,
        desc: a.desc ?? '',
      })),
      project_overview: params.projectOverview ?? '',
    },
    { signal: params.signal }
  );

/**
 * LLM-drafts the candidate's brief from its committed choices (async job) —
 * the starting point the designer edits, never the final word (Part 10 I1).
 */
export const useDraftBriefMutation = () => useMutation({ mutationFn: draftBrief });
