import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { TaxonomyInput } from '../features/mindmap/data/schema-mindmap-data';
import type { MindmapProjectSchema } from '../types/openapi';

const DEFAULT_CONTEXT_TEXT = 'Mindmap';

const DEFAULT_PROJECTS: ReadonlyArray<MindmapProjectSchema> = [
  {
    Name: 'Relevant projects will appear here',
    Descriptions: '',
    Details: '',
  },
];

const getDefaultProjects = (): MindmapProjectSchema[] =>
  DEFAULT_PROJECTS.map((project) => ({ ...project }));

const buildContextText = (topic: string, lineage: string[]): string => {
  const hierarchySegments = lineage.filter(Boolean);
  const lineageLabel = hierarchySegments.length
    ? hierarchySegments.slice(1).join(' > ')
    : '';
  return lineageLabel || topic.trim() || DEFAULT_CONTEXT_TEXT;
};

export interface MindmapSelectionInput {
  topic: string;
  lineage?: string[];
  contextDescription?: string;
}

interface MindmapStoreState {
  jmRef: unknown | null;
  contextText: string;
  contextDescription: string;
  selectedTopic: string;
  projects: MindmapProjectSchema[];
  projectsLoading: boolean;
  taxonomy: TaxonomyInput | null;
  setJmRef: (ref: unknown | null) => void;
  selectTopic: (input: MindmapSelectionInput) => void;
  setContext: (context: { contextText: string; contextDescription: string }) => void;
  setProjects: (projects: MindmapProjectSchema[]) => void;
  setProjectsLoading: (isLoading: boolean) => void;
  setTaxonomy: (taxonomy: TaxonomyInput) => void;
  setMindmapData: (payload: {
    contextText?: string;
    contextDescription?: string;
    projects?: MindmapProjectSchema[];
    projectsLoading?: boolean;
  }) => void;
  resetMindmapStore: () => void;
}

const createInitialState = () => ({
  jmRef: null as unknown | null,
  contextText: DEFAULT_CONTEXT_TEXT,
  contextDescription: '',
  selectedTopic: '',
  projects: getDefaultProjects(),
  projectsLoading: false,
  taxonomy: null as TaxonomyInput | null,
});

export const useMindmapStore = create<MindmapStoreState>()(
  devtools(
    persist(
      (set) => ({
        ...createInitialState(),
        setJmRef: (ref) => set(() => ({ jmRef: ref })),
        selectTopic: ({ topic, lineage = [], contextDescription = '' }) =>
          set(() => ({
            contextText: buildContextText(topic, lineage),
            contextDescription,
            selectedTopic: topic,
          })),
        setContext: ({ contextText, contextDescription }) =>
          set(() => ({
            contextText,
            contextDescription,
          })),
        setProjects: (projects) =>
          set(() => ({
            projects: projects.map((project) => ({ ...project })),
          })),
        setProjectsLoading: (isLoading) =>
          set(() => ({
            projectsLoading: isLoading,
          })),
        setTaxonomy: (taxonomy) =>
          set(() => ({ taxonomy })),
        setMindmapData: (payload) =>
          set((state) => ({
            contextText: payload.contextText ?? state.contextText,
            contextDescription:
              payload.contextDescription ?? state.contextDescription,
            projects: payload.projects
              ? payload.projects.map((project) => ({ ...project }))
              : state.projects,
            projectsLoading: payload.projectsLoading ?? state.projectsLoading,
          })),
        resetMindmapStore: () => set(() => createInitialState()),
      }),
      {
        name: 'mindmap-store',
        partialize: (state) => ({
          contextText: state.contextText,
          contextDescription: state.contextDescription,
          selectedTopic: state.selectedTopic,
          projects: state.projects,
          taxonomy: state.taxonomy,
        }),
      }
    ),
    { name: 'mindmap-store' }
  )
);
