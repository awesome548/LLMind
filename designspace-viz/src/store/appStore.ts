import { create } from 'zustand';
import { DEFAULT_TOPIC } from '../types/taxonomy';
import { useSchemaStore } from './schemaStore';
import { useProjectStore } from './projectStore';

type Tab = 'mindmap' | 'table';

interface AppState {
  activeTab: Tab;
  contextText: string;
  contextDescription: string;
  setActiveTab: (tab: Tab) => void;
  selectTopic: (topic: string, lineage: string[], userInitiated: boolean) => void;
}

export const useAppStore = create<AppState>((set) => ({
  activeTab: 'mindmap',
  contextText: DEFAULT_TOPIC,
  contextDescription: '',

  setActiveTab: (tab) => set({ activeTab: tab }),

  selectTopic: (topic, lineage, userInitiated) => {
    const schema = useSchemaStore.getState().schema;
    const searchProjects = useProjectStore.getState().searchProjects;

    const hierarchySegments = (Array.isArray(lineage) ? lineage : []).filter(Boolean);
    const label = hierarchySegments.length
      ? hierarchySegments.slice(1).join(' > ')
      : (topic || DEFAULT_TOPIC);

    if (!schema?.Taxonomy?.length) {
      set({
        contextText: label,
        contextDescription: '',
      });
      searchProjects(topic, lineage, userInitiated);
      return;
    }

    const normalize = (s: string) =>
      (s || '')
        .normalize('NFKC')
        .replace(/\s+/g, ' ')
        .trim()
        .toLowerCase();

    const aspects = schema.Taxonomy;
    const aspectNames = new Set(aspects.map((aspect) => normalize(aspect.Aspect)));

    const lineageAspect =
      hierarchySegments.find((segment) => aspectNames.has(normalize(segment))) ||
      (aspectNames.has(normalize(topic)) ? topic : undefined);

    const matchingAspect = lineageAspect
      ? aspects.find((aspect) => normalize(aspect.Aspect) === normalize(lineageAspect))
      : undefined;

    set({
      contextText: label,
      contextDescription: matchingAspect?.Description ?? '',
    });

    searchProjects(topic, lineage, userInitiated, matchingAspect?.Description ?? '');
  },
}));
