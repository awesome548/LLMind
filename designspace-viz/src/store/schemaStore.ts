import { create } from 'zustand';
import type { SchemaDoc, SchemaOption } from '../types/taxonomy';
import type { TaxonomyNode } from '../types/chatCompletion';
import { safeParseSchemaDoc } from '../types/taxonomySchema';
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

  const aspects = aspectCandidates
    .map((aspectNode, index) => {
      const aspectName = typeof aspectNode.topic === 'string' && aspectNode.topic.trim()
        ? aspectNode.topic.trim()
        : `Aspect ${index + 1}`;

      const description = typeof aspectNode.description === 'string' && aspectNode.description.trim()
        ? aspectNode.description.trim()
        : (typeof aspectNode.data?.description === 'string' ? aspectNode.data.description.trim() : '');

      const optionNodes = nodes.filter((node) => node.parentid === aspectNode.id);
      const options: SchemaOption[] = optionNodes
        .map((option) => {
          const name = typeof option.topic === 'string' ? option.topic.trim() : '';
          if (!name) {
            return null;
          }

          const optionDescription = typeof option.description === 'string' && option.description.trim()
            ? option.description.trim()
            : (typeof option.data?.description === 'string' ? option.data.description.trim() : '');

          return optionDescription
            ? { name, desc: optionDescription }
            : { name };
        })
        .filter((option): option is SchemaOption => Boolean(option));

      return {
        name: aspectName,
        ...(description ? { desc: description } : {}),
        ...(options.length ? { options } : {}),
      };
    })
    .filter((aspect) => Boolean(aspect.name));

  if (!aspects.length) {
    return null;
  }

  const parsedSchema = safeParseSchemaDoc({ aspects });
  if (!parsedSchema.success) {
    console.error('Invalid schema extracted from mind map:', parsedSchema.error);
    return null;
  }

  return parsedSchema.data;
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

      const response = await fetch('/taxonomy/schema_selected.json', { cache: 'no-store' });
      if (!response.ok) throw new Error(String(response.status));
      const json: unknown = await response.json();
      const parsedSchema = safeParseSchemaDoc(json);
      if (!parsedSchema.success) {
        throw new Error(`Schema validation failed: ${parsedSchema.error}`);
      }
      set({ schema: parsedSchema.data, schemaStatus: 'loaded' });
    } catch (error) {
      console.error('Failed to load schema:', error);
      set({ schemaStatus: 'error' });
    }
  },
}));
