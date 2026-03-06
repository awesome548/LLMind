import { create } from 'zustand';
import { DEFAULT_TOPIC } from '../types/taxonomy';
import { getSchemaAspects } from '../types/taxonomySchema';
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

    const aspects = getSchemaAspects(schema);

    if (!aspects.length) {
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

    const aspectNames = new Set(aspects.map((aspect) => normalize(aspect.name)));

    const lineageAspect =
      hierarchySegments.find((segment) => aspectNames.has(normalize(segment))) ||
      (aspectNames.has(normalize(topic)) ? topic : undefined);

    const matchingAspect = lineageAspect
      ? aspects.find((aspect) => normalize(aspect.name) === normalize(lineageAspect))
      : undefined;

    const optionNames = new Set(
      (matchingAspect?.options ?? []).map((option) => normalize(option.name))
    );
    const lineageOption =
      hierarchySegments.find((segment) => optionNames.has(normalize(segment))) ||
      (optionNames.has(normalize(topic)) ? topic : undefined);
    const matchingOption = lineageOption
      ? matchingAspect?.options?.find((option) => normalize(option.name) === normalize(lineageOption))
      : undefined;
    const contextDescription = matchingOption?.desc ?? matchingAspect?.desc ?? '';

    set({
      contextText: label,
      contextDescription,
    });

    searchProjects(topic, lineage, userInitiated, contextDescription);
  },
}));
