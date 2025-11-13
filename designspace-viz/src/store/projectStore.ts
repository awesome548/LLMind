import { create } from 'zustand';
import { supabase } from '../hooks/supabaseClient';
import OpenAI from 'openai';
import type { ProjectDetails } from '../types/taxonomy';

const curatedFallbackProjects: ProjectDetails[] = [
  { Name: 'Relevant projects will appear here', Descriptions: '', Details: '' },
];

interface ProjectState {
  projects: ProjectDetails[];
  projectsLoading: boolean;
  searchProjects: (
    topic: string,
    lineage: string[],
    shouldQuerySupabase?: boolean,
    description?: string
  ) => Promise<void>;
}

export const useProjectStore = create<ProjectState>((set) => ({
  projects: curatedFallbackProjects,
  projectsLoading: false,

  searchProjects: async (topic, lineage, shouldQuerySupabase = false, description) => {
    if (!shouldQuerySupabase) {
      set({
        projects: curatedFallbackProjects,
        projectsLoading: false,
      });
      return;
    }

    set({
      projectsLoading: true,
    });

    const hasEnv =
      Boolean(import.meta.env.VITE_SUPABASE_URL) &&
      Boolean(import.meta.env.VITE_SUPABASE_KEY) &&
      Boolean(import.meta.env.VITE_OPENAI_API_KEY);

    if (hasEnv) {
      try {
        const openai = new OpenAI({
          apiKey: import.meta.env.VITE_OPENAI_API_KEY as string,
          dangerouslyAllowBrowser: true,
        });

        const OPENAI_MODEL =
          (import.meta.env.VITE_OPENAI_EMBED_MODEL as string) || 'text-embedding-3-small';
        const MATCH_FN =
          (import.meta.env.VITE_SUPABASE_MATCH_FN as string) || 'match_media_docs';

        const lineageText = (lineage?.slice(1)?.join(' > ') || '').trim();
        const descriptionText = description?.trim() ?? '';
        const querySource = [lineageText, descriptionText, topic]
          .filter(Boolean)
          .join(' | ');
        const queryText = (querySource || topic || '').trim() || 'test';

        const embResp = await openai.embeddings.create({
          model: OPENAI_MODEL,
          input: [queryText],
        });
        const queryEmbedding = embResp.data?.[0]?.embedding;
        if (!Array.isArray(queryEmbedding)) throw new Error('Failed to create embedding');

        const { data, error } = await supabase.rpc(MATCH_FN, {
          query_embedding: queryEmbedding,
          match_count: 5,
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
          set({
            projects: mapped,
            projectsLoading: false,
          });
          return;
        }
      } catch (error) {
        console.warn('Supabase vector lookup failed, falling back:', error);
      }
    }

    set({
      projects: curatedFallbackProjects,
      projectsLoading: false,
    });
  },
}));
