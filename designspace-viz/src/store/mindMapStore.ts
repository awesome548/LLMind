import { create } from 'zustand';

interface MindMapState {
  jmRef: any | null;
  setJmRef: (ref: any) => void;
}

export const useMindMapStore = create<MindMapState>((set) => ({
  jmRef: null,
  setJmRef: (ref) => set({ jmRef: ref }),
}));
