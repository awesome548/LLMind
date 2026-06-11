import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { TaxonomyInput } from '../features/mindmap/data/schema-mindmap-data';
import {
  SCHEMA_MINDMAP_NODES,
  taxonomyToMindmapNodes,
} from '../features/mindmap/data/schema-mindmap-data';
import type {
  AxesConfig,
  DesignCandidate,
  MindmapNode,
  NodeProvenance,
  OptionStateEntry,
} from '../features/mindmap/types';
import type { CoordMap, GenerationTrail } from '../features/design-space/types';
import type { MindmapProjectSchema } from '../types/api-aliases';

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

function cloneNodes(nodes: ReadonlyArray<MindmapNode>): MindmapNode[] {
  return nodes.map((node) => ({
    ...node,
    children: node.children ? cloneNodes(node.children) : undefined,
  }));
}

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
  /** The working tree — shared by the mind map and the design space, persisted
   * so generated nodes survive a reload (they are NOT derivable from taxonomy). */
  nodes: ReadonlyArray<MindmapNode>;
  /** node.id → frozen design-space coordinate (+ placement confidence). */
  coords: CoordMap;
  /** "gx,gy" cell key → the generation trail produced from that cell. */
  discovered: Record<string, GenerationTrail>;
  /** node.id → which precedents/click produced it. */
  provenance: Record<string, NodeProvenance>;
  /** node.id → one-line description for GENERATED nodes (taxonomy nodes keep
   * their topic-keyed descriptions; this id-keyed map is precise for ids whose
   * topics may collide). */
  descriptionById: Record<string, string>;
  /** Candidate designs (one option per aspect) — per-taxonomy, persisted. */
  candidates: Record<string, DesignCandidate>;
  activeCandidateId: string | null;
  /** node.id → pruning state (rejected + reason). */
  optionState: Record<string, OptionStateEntry>;
  /** Perspectives view: the chosen axis poles (null until first configured). */
  axesConfig: AxesConfig | null;
  setJmRef: (ref: unknown | null) => void;
  selectTopic: (input: MindmapSelectionInput) => void;
  setContext: (context: { contextText: string; contextDescription: string }) => void;
  setProjects: (projects: MindmapProjectSchema[]) => void;
  setProjectsLoading: (isLoading: boolean) => void;
  /** Replaces the taxonomy AND rebuilds the working tree from it, invalidating
   * all exploration state (coords/discovered/provenance) — a new taxonomy is a
   * new design space overlay. */
  setTaxonomy: (taxonomy: TaxonomyInput) => void;
  setNodes: (nodes: ReadonlyArray<MindmapNode>) => void;
  mergeCoords: (coords: CoordMap) => void;
  /** Drop coordinates (e.g. after a rename — the old embedding no longer applies). */
  removeCoords: (ids: ReadonlyArray<string>) => void;
  recordDiscovery: (cellKey: string, trail: GenerationTrail) => void;
  recordProvenance: (entries: Record<string, NodeProvenance>) => void;
  mergeDescriptions: (entries: Record<string, string>) => void;
  /** Creates a candidate, makes it active, and returns its id. */
  createCandidate: (name?: string) => string;
  deleteCandidate: (id: string) => void;
  setActiveCandidate: (id: string | null) => void;
  renameCandidate: (id: string, name: string) => void;
  /** Sets/clears the ACTIVE candidate's choice for an aspect (radio semantics). */
  setChoice: (aspectId: string, optionId: string | null) => void;
  rejectOption: (nodeId: string, reason?: string) => void;
  reopenOption: (nodeId: string) => void;
  setAxesConfig: (config: AxesConfig | null) => void;
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
  nodes: cloneNodes(SCHEMA_MINDMAP_NODES) as ReadonlyArray<MindmapNode>,
  coords: {} as CoordMap,
  discovered: {} as Record<string, GenerationTrail>,
  provenance: {} as Record<string, NodeProvenance>,
  descriptionById: {} as Record<string, string>,
  candidates: {} as Record<string, DesignCandidate>,
  activeCandidateId: null as string | null,
  optionState: {} as Record<string, OptionStateEntry>,
  axesConfig: null as AxesConfig | null,
});

type PersistedState = ReturnType<typeof createInitialState>;

export const useMindmapStore = create<MindmapStoreState>()(
  devtools(
    persist(
      (set, get) => ({
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
          set(() => ({
            taxonomy,
            nodes: taxonomyToMindmapNodes(taxonomy).nodes,
            coords: {},
            discovered: {},
            provenance: {},
            descriptionById: {},
            candidates: {},
            activeCandidateId: null,
            optionState: {},
            axesConfig: null,
          })),
        setNodes: (nodes) => set(() => ({ nodes })),
        mergeCoords: (coords) =>
          set((state) => ({ coords: { ...state.coords, ...coords } })),
        removeCoords: (ids) =>
          set((state) => {
            const next = { ...state.coords };
            for (const id of ids) delete next[id];
            return { coords: next };
          }),
        recordDiscovery: (cellKey, trail) =>
          set((state) => ({
            discovered: { ...state.discovered, [cellKey]: trail },
          })),
        recordProvenance: (entries) =>
          set((state) => ({
            provenance: { ...state.provenance, ...entries },
          })),
        mergeDescriptions: (entries) =>
          set((state) => ({
            descriptionById: { ...state.descriptionById, ...entries },
          })),
        createCandidate: (name) => {
          const id = `cand-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
          const count = Object.keys(get().candidates).length;
          set((state) => ({
            candidates: {
              ...state.candidates,
              [id]: {
                id,
                name: name?.trim() || `Candidate ${count + 1}`,
                choices: {},
                createdAt: Date.now(),
              },
            },
            activeCandidateId: id,
          }));
          return id;
        },
        deleteCandidate: (id) =>
          set((state) => {
            const next = { ...state.candidates };
            delete next[id];
            return {
              candidates: next,
              activeCandidateId:
                state.activeCandidateId === id
                  ? (Object.keys(next)[0] ?? null)
                  : state.activeCandidateId,
            };
          }),
        setActiveCandidate: (id) => set(() => ({ activeCandidateId: id })),
        renameCandidate: (id, name) =>
          set((state) => {
            const candidate = state.candidates[id];
            if (!candidate) return {};
            return {
              candidates: { ...state.candidates, [id]: { ...candidate, name } },
            };
          }),
        setChoice: (aspectId, optionId) =>
          set((state) => {
            const id = state.activeCandidateId;
            const candidate = id ? state.candidates[id] : undefined;
            if (!id || !candidate) return {};
            const choices = { ...candidate.choices };
            if (optionId === null) delete choices[aspectId];
            else choices[aspectId] = optionId;
            return {
              candidates: { ...state.candidates, [id]: { ...candidate, choices } },
            };
          }),
        rejectOption: (nodeId, reason) =>
          set((state) => ({
            optionState: {
              ...state.optionState,
              [nodeId]: { state: 'rejected', ...(reason ? { reason } : {}) },
            },
          })),
        reopenOption: (nodeId) =>
          set((state) => {
            const next = { ...state.optionState };
            delete next[nodeId];
            return { optionState: next };
          }),
        setAxesConfig: (config) => set(() => ({ axesConfig: config })),
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
        // v1 dropped pre-v1 state (stale placeholder taxonomy). v2 adds the
        // persisted exploration state (nodes/coords/discovered/provenance) so a
        // reload no longer loses generated work; v1 state is upgraded by
        // rebuilding the tree from its taxonomy.
        version: 2,
        migrate: (persisted, version) => {
          if (version < 1 || !persisted) return createInitialState();
          if (version < 2) {
            const prev = persisted as Partial<PersistedState>;
            return {
              ...createInitialState(),
              ...prev,
              nodes: prev.taxonomy
                ? taxonomyToMindmapNodes(prev.taxonomy).nodes
                : cloneNodes(SCHEMA_MINDMAP_NODES),
              coords: {},
              discovered: {},
              provenance: {},
            };
          }
          return persisted as PersistedState;
        },
        partialize: (state) => ({
          contextText: state.contextText,
          contextDescription: state.contextDescription,
          selectedTopic: state.selectedTopic,
          projects: state.projects,
          taxonomy: state.taxonomy,
          nodes: state.nodes,
          coords: state.coords,
          discovered: state.discovered,
          provenance: state.provenance,
          descriptionById: state.descriptionById,
          candidates: state.candidates,
          activeCandidateId: state.activeCandidateId,
          optionState: state.optionState,
          axesConfig: state.axesConfig,
        }),
      }
    ),
    { name: 'mindmap-store' }
  )
);
