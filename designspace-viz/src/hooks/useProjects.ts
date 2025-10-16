// useProjects.ts
import { useCallback, useMemo, useState } from 'react';
import { supabase } from './supabaseClient';           // <-- use the singleton
import OpenAI from 'openai';
import type { ProjectDetails } from '../utils/type';

const FALLBACK_LOADING_MESSAGE = 'Searching Supabase for related projects...';
const FALLBACK_EMPTY_MESSAGE = 'No related projects found for this selection yet.';

const curatedFallbackProjects: ProjectDetails[] = [
  { Name: 'Relevant projects will appear here', Descriptions: '', Details: '' }
];

export function useProjects() {
  const [loadingText, setLoadingText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [projects, setProjects] = useState<ProjectDetails[]>(curatedFallbackProjects);

  const setLoading = useCallback((on: boolean) => {
    setIsLoading(on);
    setLoadingText(on ? FALLBACK_LOADING_MESSAGE : '');
    if (on) {
      setProjects([]);
    }
  }, []);

  const search = useCallback(async (topic: string, lineage: string[], shouldQuerySupabase = false) => {
    if (!shouldQuerySupabase) {
      setProjects(curatedFallbackProjects);
      setLoading(false);
      return;
    }

    setLoading(true);

    const hasEnv =
      Boolean(import.meta.env.VITE_SUPABASE_URL) &&
      Boolean(import.meta.env.VITE_SUPABASE_KEY) &&
      Boolean(import.meta.env.VITE_OPENAI_API_KEY);

    if (hasEnv) {
      try {
        // Create OpenAI client once per module (could also move to its own singleton if you like)
        const openai = new OpenAI({
          apiKey: import.meta.env.VITE_OPENAI_API_KEY as string,
          dangerouslyAllowBrowser: true,
        });

        const OPENAI_MODEL =
          (import.meta.env.VITE_OPENAI_EMBED_MODEL as string) || 'text-embedding-3-small';
        const MATCH_FN =
          (import.meta.env.VITE_SUPABASE_MATCH_FN as string) || 'match_media_docs';

        const queryText =
          (lineage?.slice(1)?.join(' > ') || topic || '').trim() || topic || 'test';

        const embResp = await openai.embeddings.create({
          model: OPENAI_MODEL,
          input: [queryText],
        });
        const queryEmbedding = embResp.data?.[0]?.embedding;
        if (!Array.isArray(queryEmbedding)) throw new Error('Failed to create embedding');

        const { data, error } = await supabase.rpc(MATCH_FN, {
          query_embedding: queryEmbedding,
          match_count: 3,
          similarity_threshold: 0.0,
        });
        if (error) throw error;

        const rows = Array.isArray(data) ? data : [];
        if (rows.length) {
          const mapped: ProjectDetails[] = rows.map((r: any) => {
            const md = r?.metadata ?? {};
            const rawImage = md.Image ?? null;
            const derivedId = md.id ?? r?.id ?? md.Id ?? r?.Id ?? undefined;
            return {
              id: typeof derivedId === 'string' ? derivedId : undefined,
              Id: md.Id ?? undefined,
              Name: md.Name ?? '(untitled)',
              Descriptions: md.Descriptions ?? '',
              Details: md.Details ?? r?.content ?? '',
              Image: typeof rawImage === 'string' && rawImage.trim() ? rawImage.trim() : undefined,
            } as ProjectDetails;
          });
          setProjects(mapped);
          setLoading(false);
          return;
        }
      } catch (e) {
        console.warn('Supabase vector lookup failed, falling back:', e);
      }
    }

    setProjects(curatedFallbackProjects);
    setLoading(false);
  }, [setLoading]);

  const statusText = useMemo(
    () => loadingText || (projects.length ? '' : FALLBACK_EMPTY_MESSAGE),
    [loadingText, projects.length]
  );

  return { projects, search, statusText, isLoading };
}
