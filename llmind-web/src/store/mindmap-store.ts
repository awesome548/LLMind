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
  RubricMetric,
} from '../features/mindmap/types';
import type { CoordMap, GenerationTrail } from '../features/design-space/types';

const DEFAULT_CONTEXT_TEXT = 'Mindmap';

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
  contextText: string;
  contextDescription: string;
  selectedTopic: string;
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
  /** Perspectives rubric: the project's persistent examination metrics. */
  rubric: RubricMetric[];
  /** Feature-usage counters (research instrumentation; included in session export). */
  usage: Record<string, number>;
  selectTopic: (input: MindmapSelectionInput) => void;
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
  /** Sets the candidate's BRIEF (its identity layer — the primary embedding). */
  setCandidateBrief: (id: string, brief: string) => void;
  /** Records a previous star position before the brief moves it (capped). */
  appendCandidateTrail: (id: string, point: { x: number; y: number }) => void;
  addRubricMetric: (metric: RubricMetric) => void;
  removeRubricMetric: (metricId: string) => void;
  /** Sets/clears the ACTIVE candidate's choice for an aspect (radio semantics).
   * Refuses rejected options — an option is never chosen AND rejected. */
  setChoice: (aspectId: string, optionId: string | null) => void;
  /** Rejects an option and removes it from EVERY candidate's choices. */
  rejectOption: (nodeId: string, reason?: string) => void;
  reopenOption: (nodeId: string) => void;
  setAxesConfig: (config: AxesConfig | null) => void;
  /** Drop state for nodes no longer in the tree (after deletions). Candidate
   * coordinates (`cand:` keys) are exempt — candidates are pruned by choices. */
  pruneMissingNodes: (validIds: ReadonlySet<string>) => void;
  trackUsage: (event: string) => void;
  /** Replace the whole exploration with an imported session snapshot. */
  restoreSession: (snapshot: SessionSnapshot) => void;
  resetMindmapStore: () => void;
}

const createInitialState = () => ({
  contextText: DEFAULT_CONTEXT_TEXT,
  contextDescription: '',
  selectedTopic: '',
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
  rubric: [] as RubricMetric[],
  usage: {} as Record<string, number>,
});

type PersistedState = ReturnType<typeof createInitialState>;

/** The persisted exploration slices — also the session save/load payload. */
export type SessionSnapshot = Pick<
  MindmapStoreState,
  | 'contextText'
  | 'contextDescription'
  | 'selectedTopic'
  | 'taxonomy'
  | 'nodes'
  | 'coords'
  | 'discovered'
  | 'provenance'
  | 'descriptionById'
  | 'candidates'
  | 'activeCandidateId'
  | 'optionState'
  | 'axesConfig'
  | 'rubric'
  | 'usage'
>;

/** Single definition of what persists / what a session file contains. */
export const selectSessionSnapshot = (state: SessionSnapshot): SessionSnapshot => ({
  contextText: state.contextText,
  contextDescription: state.contextDescription,
  selectedTopic: state.selectedTopic,
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
  rubric: state.rubric,
  usage: state.usage,
});

const bumpUsage = (usage: Record<string, number>, event: string) => ({
  ...usage,
  [event]: (usage[event] ?? 0) + 1,
});

export const useMindmapStore = create<MindmapStoreState>()(
  devtools(
    persist(
      (set, get) => ({
        ...createInitialState(),
        selectTopic: ({ topic, lineage = [], contextDescription = '' }) =>
          set(() => ({
            contextText: buildContextText(topic, lineage),
            contextDescription,
            selectedTopic: topic,
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
            rubric: [],
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
            usage: bumpUsage(state.usage, 'candidate_created'),
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
        setCandidateBrief: (id, brief) =>
          set((state) => {
            const candidate = state.candidates[id];
            if (!candidate) return {};
            return {
              candidates: { ...state.candidates, [id]: { ...candidate, brief } },
            };
          }),
        appendCandidateTrail: (id, point) =>
          set((state) => {
            const candidate = state.candidates[id];
            if (!candidate) return {};
            const trail = [...(candidate.trail ?? []), point].slice(-10);
            return {
              candidates: { ...state.candidates, [id]: { ...candidate, trail } },
            };
          }),
        addRubricMetric: (metric) =>
          set((state) => ({ rubric: [...state.rubric, metric] })),
        removeRubricMetric: (metricId) =>
          set((state) => ({
            rubric: state.rubric.filter((m) => m.id !== metricId),
          })),
        setChoice: (aspectId, optionId) =>
          set((state) => {
            const id = state.activeCandidateId;
            const candidate = id ? state.candidates[id] : undefined;
            if (!id || !candidate) return {};
            // Invariant: an option is never chosen AND rejected.
            if (optionId !== null && state.optionState[optionId]) return {};
            const choices = { ...candidate.choices };
            if (optionId === null) delete choices[aspectId];
            else choices[aspectId] = optionId;
            return {
              candidates: { ...state.candidates, [id]: { ...candidate, choices } },
              ...(optionId !== null ? { usage: bumpUsage(state.usage, 'choice_set') } : {}),
            };
          }),
        rejectOption: (nodeId, reason) =>
          set((state) => {
            // Invariant: rejecting clears the option from every candidate.
            const candidates = Object.fromEntries(
              Object.entries(state.candidates).map(([id, candidate]) => [
                id,
                {
                  ...candidate,
                  choices: Object.fromEntries(
                    Object.entries(candidate.choices).filter(
                      ([, optionId]) => optionId !== nodeId
                    )
                  ),
                },
              ])
            );
            return {
              candidates,
              optionState: {
                ...state.optionState,
                [nodeId]: { state: 'rejected', ...(reason ? { reason } : {}) },
              },
              usage: bumpUsage(state.usage, 'option_rejected'),
            };
          }),
        reopenOption: (nodeId) =>
          set((state) => {
            const next = { ...state.optionState };
            delete next[nodeId];
            return { optionState: next };
          }),
        setAxesConfig: (config) => set(() => ({ axesConfig: config })),
        pruneMissingNodes: (validIds) =>
          set((state) => {
            const keepEntries = <V,>(record: Record<string, V>) =>
              Object.fromEntries(
                Object.entries(record).filter(
                  ([id]) => validIds.has(id) || id.startsWith('cand:')
                )
              );
            const candidates = Object.fromEntries(
              Object.entries(state.candidates).map(([id, candidate]) => [
                id,
                {
                  ...candidate,
                  choices: Object.fromEntries(
                    Object.entries(candidate.choices).filter(
                      ([aspectId, optionId]) =>
                        validIds.has(aspectId) && validIds.has(optionId)
                    )
                  ),
                },
              ])
            );
            return {
              coords: keepEntries(state.coords),
              provenance: keepEntries(state.provenance),
              optionState: keepEntries(state.optionState),
              descriptionById: keepEntries(state.descriptionById),
              candidates,
              rubric: state.rubric.filter(
                (m) =>
                  validIds.has(m.aspectId) &&
                  validIds.has(m.poleAId) &&
                  validIds.has(m.poleBId)
              ),
            };
          }),
        trackUsage: (event) =>
          set((state) => ({
            usage: { ...state.usage, [event]: (state.usage[event] ?? 0) + 1 },
          })),
        // Defaults first, so sessions saved before a slice existed (e.g. rubric)
        // reset it instead of leaking the current exploration's state.
        restoreSession: (snapshot) =>
          set(() => ({ ...selectSessionSnapshot(createInitialState()), ...snapshot })),
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
        partialize: (state) => selectSessionSnapshot(state),
      }
    ),
    { name: 'mindmap-store' }
  )
);
