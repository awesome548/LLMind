import { create } from 'zustand';
import type { SchemaDoc, SchemaAspect } from '../types/taxonomy';
import type { TaxonomyNode } from '../types/chatCompletion';
import { useMindMapStore } from './mindMapStore';

type SchemaStatus = 'idle' | 'loading' | 'error' | 'loaded';

interface SchemaState {
  schema: SchemaDoc | null;
  schemaStatus: SchemaStatus;
  loadSchema: () => Promise<void>;
}

type JsMindNode = TaxonomyNode & {
  data?: {
    description?: string;
    [key: string]: unknown;
  };
};

const extractNodeArray = (mindData: unknown): JsMindNode[] => {
  if (!mindData) return [];
  if (Array.isArray(mindData)) {
    return mindData as JsMindNode[];
  }
  if (typeof mindData === 'object' && mindData !== null) {
    const maybeArray = (mindData as { data?: JsMindNode[] }).data;
    if (Array.isArray(maybeArray)) {
      return maybeArray as JsMindNode[];
    }
  }
  return [];
};

const readMindMapNodes = (): JsMindNode[] => {
  const jmRef = useMindMapStore.getState().jmRef;
  if (!jmRef || typeof jmRef.get_data !== 'function') {
    return [];
  }

  try {
    const mindData = jmRef.get_data('node_array');
    return extractNodeArray(mindData);
  } catch (error) {
    console.error('Failed to read jsMind data:', error);
    return [];
  }
};

const buildSchemaFromMindMap = (): SchemaDoc | null => {
  const nodes = readMindMapNodes();
  if (!nodes.length) {
    return null;
  }

  const rootNode = nodes.find((node) => node.isroot) || nodes.find((node) => !node.parentid);
  const rootId = rootNode?.id;

  const aspectCandidates = rootId
    ? nodes.filter((node) => node.parentid === rootId)
    : nodes.filter((node) => !node.parentid);

  const aspects: SchemaAspect[] = aspectCandidates
    .map((aspectNode, index) => {
      const aspectName = typeof aspectNode.topic === 'string' && aspectNode.topic.trim()
        ? aspectNode.topic.trim()
        : `Aspect ${index + 1}`;

      const description = typeof aspectNode.description === 'string' && aspectNode.description.trim()
        ? aspectNode.description.trim()
        : (typeof aspectNode.data?.description === 'string' ? aspectNode.data.description.trim() : '');

      const optionNodes = nodes.filter((node) => node.parentid === aspectNode.id);
      const options = optionNodes
        .map((option) => (typeof option.topic === 'string' ? option.topic.trim() : ''))
        .filter((topic) => Boolean(topic));

      const formatted: SchemaAspect = {
        Aspect: aspectName,
      };

      if (description) {
        formatted.Description = description;
      }

      if (options.length) {
        formatted.Options = options;
      }

      return formatted;
    })
    .filter((aspect) => Boolean(aspect.Aspect));

  if (!aspects.length) {
    return null;
  }

  return { Taxonomy: aspects };
};

export const useSchemaStore = create<SchemaState>((set) => ({
  schema: null,
  schemaStatus: 'idle',

  loadSchema: async () => {
    set({ schemaStatus: 'loading' });
    try {
      const liveSchema = buildSchemaFromMindMap();
      if (liveSchema) {
        set({ schema: liveSchema, schemaStatus: 'loaded' });
        return;
      }

      const response = await fetch('/taxonomy/schema.json', { cache: 'no-store' });
      if (!response.ok) throw new Error(String(response.status));
      const json: SchemaDoc = await response.json();
      set({ schema: json, schemaStatus: 'loaded' });
    } catch (error) {
      console.error('Failed to load schema:', error);
      set({ schemaStatus: 'error' });
    }
  },
}));
