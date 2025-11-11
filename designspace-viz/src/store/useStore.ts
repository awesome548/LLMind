import { create } from 'zustand';
import { supabase } from '../hooks/supabaseClient';
import OpenAI from 'openai';
import type { SchemaDoc, ProjectDetails } from '../types/taxonomy';
import { DEFAULT_TOPIC } from '../types/taxonomy';

interface AppState {
  // Schema state
  schema: SchemaDoc | null;
  schemaStatus: 'idle' | 'loading' | 'error' | 'loaded';
  
  // Projects state
  projects: ProjectDetails[];
  projectsLoading: boolean;
  projectsStatusText: string;
  
  // UI state
  activeTab: 'mindmap' | 'table';
  contextText: string;
  contextDescription: string;
  
  // jsMind ref
  jmRef: any | null;
  
  // Actions
  loadSchema: () => Promise<void>;
  searchProjects: (topic: string, lineage: string[], shouldQuerySupabase?: boolean) => Promise<void>;
  setActiveTab: (tab: 'mindmap' | 'table') => void;
  selectTopic: (topic: string, lineage: string[], userInitiated: boolean) => void;
  setJmRef: (ref: any) => void;
}

const FALLBACK_LOADING_MESSAGE = 'Searching Supabase for related projects...';
const FALLBACK_EMPTY_MESSAGE = 'No related projects found for this selection yet.';

const curatedFallbackProjects: ProjectDetails[] = [
  { Name: 'Relevant projects will appear here', Descriptions: '', Details: '' }
];

export const useStore = create<AppState>((set, get) => ({
  // Initial state
  schema: null,
  schemaStatus: 'idle',
  projects: curatedFallbackProjects,
  projectsLoading: false,
  projectsStatusText: '',
  activeTab: 'mindmap',
  contextText: DEFAULT_TOPIC,
  contextDescription: '',
  jmRef: null,

  // Set jmRef
  setJmRef: (ref) => {
    set({ jmRef: ref });
  },

  // Load schema
  loadSchema: async () => {
    set({ schemaStatus: 'loading' });
    try {
      const response = await fetch('/taxonomy/schema.json', { cache: 'no-store' });
      if (!response.ok) throw new Error(String(response.status));
      const json: SchemaDoc = await response.json();
      set({ schema: json, schemaStatus: 'loaded' });
    } catch (error) {
      console.error('Failed to load schema:', error);
      set({ schemaStatus: 'error' });
    }
  },

  // Search projects
  searchProjects: async (topic: string, lineage: string[], shouldQuerySupabase = false) => {
    if (!shouldQuerySupabase) {
      set({ 
        projects: curatedFallbackProjects, 
        projectsLoading: false,
        projectsStatusText: FALLBACK_EMPTY_MESSAGE 
      });
      return;
    }

    set({ 
      projectsLoading: true, 
      projectsStatusText: FALLBACK_LOADING_MESSAGE,
      projects: [] 
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
            projectsStatusText: '' 
          });
          return;
        }
      } catch (e) {
        console.warn('Supabase vector lookup failed, falling back:', e);
      }
    }

    set({ 
      projects: curatedFallbackProjects, 
      projectsLoading: false,
      projectsStatusText: FALLBACK_EMPTY_MESSAGE 
    });
  },

  // Set active tab
  setActiveTab: (tab) => {
    set({ activeTab: tab });
  },

  // Select topic (handles both context and project search)
  selectTopic: (topic: string, lineage: string[], userInitiated: boolean) => {
    const state = get();
    const schema = state.schema;
    
    const hierarchySegments = (Array.isArray(lineage) ? lineage : []).filter(Boolean);
    const label = hierarchySegments.length ? hierarchySegments.slice(1).join(' > ') : (topic || DEFAULT_TOPIC);
    
    // If schema hasn't loaded yet, clear and bail early
    if (!schema?.Taxonomy?.length) {
      set({ 
        contextText: label, 
        contextDescription: '' 
      });
      state.searchProjects(topic, lineage, userInitiated);
      return;
    }

    const normalize = (s: string) => s?.normalize('NFKC').replace(/\s+/g, ' ').trim().toLowerCase();
    const aspects = schema.Taxonomy;
    const aspectNames = new Set(aspects.map(a => normalize(a.Aspect)));

    // Try to find any lineage segment that is a known Aspect
    const lineageAspect =
      hierarchySegments.find(seg => aspectNames.has(normalize(seg))) ||
      (aspectNames.has(normalize(topic)) ? topic : undefined);

    let matchingAspect = undefined;
    if (lineageAspect) {
      const target = normalize(lineageAspect);
      matchingAspect = aspects.find(a => normalize(a.Aspect) === target);
    }

    set({ 
      contextText: label,
      contextDescription: matchingAspect?.Description ?? '' 
    });

    state.searchProjects(topic, lineage, userInitiated);
  },
}));