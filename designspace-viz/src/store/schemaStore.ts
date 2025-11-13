import { create } from 'zustand';
import type { SchemaDoc } from '../types/taxonomy';

type SchemaStatus = 'idle' | 'loading' | 'error' | 'loaded';

interface SchemaState {
  schema: SchemaDoc | null;
  schemaStatus: SchemaStatus;
  loadSchema: () => Promise<void>;
}

export const useSchemaStore = create<SchemaState>((set) => ({
  schema: null,
  schemaStatus: 'idle',

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
}));
