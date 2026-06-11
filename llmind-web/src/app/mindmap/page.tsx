'use client';

import {
  ChevronRight,
  Compass,
  Focus,
  FolderOpen,
  Grid3x3,
  Home,
  Info,
  Loader2,
  Network,
  PanelsRightBottom,
  Save,
  Sparkles,
  Star,
  Zap,
  type LucideIcon,
} from 'lucide-react';
import Link from 'next/link';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SimpleMindMap } from '@/src/components/mindmap/simple-mindmap';
import { SimpleProjectPanel } from '@/src/components/mindmap/simple-project-panel';
import { DesignSpaceSurface } from '@/src/components/design-space/design-space-surface';
import { useSurfaceQuery } from '@/src/features/design-space/hooks/use-surface-query';
import {
  nodesToLocateItems,
  useLocateNodesMutation,
} from '@/src/features/design-space/hooks/use-locate-nodes';
import { useGenerateAtMutation } from '@/src/features/design-space/hooks/use-generate-at-mutation';
import { useCorpusProjectQuery } from '@/src/features/design-space/hooks/use-corpus-project';
import { useRelevanceQuery } from '@/src/features/design-space/hooks/use-relevance-query';
import { ExamineView } from '@/src/components/design-space/examine-view';
import { CandidatePanel } from '@/src/components/design-space/candidate-panel';
import { CompareCandidatesDialog } from '@/src/components/design-space/compare-candidates-dialog';
import {
  candidateCoordKey,
  candidateEmbeddingText,
  indexNodesById,
} from '@/src/features/design-space/candidate-utils';
import type { CandidateMarker } from '@/src/components/design-space/design-space-surface';
import type { CoordMap, GenerationTrail } from '@/src/features/design-space/types';
import { Badge } from '@/src/components/ui/badge';
import { Button } from '@/src/components/ui/button';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/src/components/ui/collapsible';
import { ScrollArea } from '@/src/components/ui/scroll-area';
import {
  SCHEMA_MINDMAP_NODES,
  SCHEMA_DESCRIPTION_BY_TOPIC,
  taxonomyToMindmapNodes,
} from '@/src/features/mindmap/data/schema-mindmap-data';
import {
  generatedNodesToMindmapNodes,
  useGenerateNodesMutation,
} from '@/src/features/mindmap/hooks/use-generate-nodes-mutation';
import { useRelatedProjectsQuery } from '@/src/features/mindmap/hooks/use-related-projects-query';
import type {
  MindmapNode,
  MindmapSelection,
  NodeProvenance,
} from '@/src/features/mindmap/types';
import {
  collectIds,
  ensureUniqueChildIds,
  findNodeByLineage,
  insertChildrenAtNode,
} from '@/src/features/mindmap/tree-utils';
import { usePeekMutation } from '@/src/features/design-space/hooks/use-peek-mutation';
import type { GapPreview } from '@/src/components/design-space/design-space-surface';
import {
  computeExplorationStats,
  formatExplorationStats,
} from '@/src/features/design-space/exploration-stats';
import { buildSessionFile, parseSessionFile } from '@/src/lib/session-io';
import { downloadTextFile } from '@/src/lib/export-exploration';
import { selectSessionSnapshot } from '@/src/store/mindmap-store';
import { GenerateTaxonomyDialog } from '@/src/features/mindmap/components/generate-taxonomy-dialog';
import { GenerateNodesDialog } from '@/src/features/mindmap/components/generate-nodes-dialog';
import type { GenerateNodesParams } from '@/src/features/mindmap/hooks/use-generate-nodes-mutation';
import { useMindmapStore } from '@/src/store/mindmap-store';
import type { FetchRelatedProjectsRequestSchema } from '@/src/types/api-aliases';

const INITIAL_TOPIC = SCHEMA_MINDMAP_NODES[0]?.topic ?? 'Design Aspects';

const INITIAL_SELECTION: MindmapSelection = {
  topic: INITIAL_TOPIC,
  lineage: [INITIAL_TOPIC],
};

const buildRequest = (
  selection: MindmapSelection,
  description: string
): FetchRelatedProjectsRequestSchema => ({
  topic: selection.topic,
  lineage: [...selection.lineage],
  description: description || null,
  should_query_supabase: true,
  limit: 5,
  similarity_threshold: 0.0,
});

const PLACEHOLDER_PROJECT_NAME = 'Relevant projects will appear here';

/** Normalise generation-response project rows into provenance seed entries. */
function toSeedProjects(rows: unknown): NodeProvenance['seedProjects'] {
  if (!Array.isArray(rows)) return [];
  return rows
    .filter(
      (row): row is Record<string, unknown> => typeof row === 'object' && row !== null
    )
    .map((row) => ({
      id: typeof row.id === 'string' && row.id ? row.id : null,
      name: typeof row.Name === 'string' && row.Name ? row.Name : '(untitled)',
    }))
    .filter((seed) => seed.name !== PLACEHOLDER_PROJECT_NAME);
}

/** A closed panel in the Examine view: thin icon button instead of a header bar. */
function PanelIconButton({
  icon: Icon,
  label,
  onClick,
}: {
  icon: LucideIcon;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={`Open ${label}`}
      aria-label={`Open ${label}`}
      className="pointer-events-auto flex h-10 w-10 items-center justify-center rounded-full border bg-background/90 shadow-md backdrop-blur transition-colors hover:bg-background"
    >
      <Icon className="h-4 w-4 text-muted-foreground" />
    </button>
  );
}

export default function MindmapPage() {
  const selectTopic = useMindmapStore((state) => state.selectTopic);
  const taxonomy = useMindmapStore((state) => state.taxonomy);
  const setTaxonomy = useMindmapStore((state) => state.setTaxonomy);
  // The working tree + exploration state are persisted in the store so that
  // generated nodes, coordinates, and discovered cells survive a reload.
  const nodes = useMindmapStore((state) => state.nodes);
  const setNodes = useMindmapStore((state) => state.setNodes);
  const coords = useMindmapStore((state) => state.coords);
  const mergeCoords = useMindmapStore((state) => state.mergeCoords);
  const removeCoords = useMindmapStore((state) => state.removeCoords);
  const discovered = useMindmapStore((state) => state.discovered);
  const recordDiscovery = useMindmapStore((state) => state.recordDiscovery);
  const provenance = useMindmapStore((state) => state.provenance);
  const recordProvenance = useMindmapStore((state) => state.recordProvenance);
  const descriptionById = useMindmapStore((state) => state.descriptionById);
  const mergeDescriptions = useMindmapStore((state) => state.mergeDescriptions);
  const candidates = useMindmapStore((state) => state.candidates);
  const activeCandidateId = useMindmapStore((state) => state.activeCandidateId);
  const setActiveCandidate = useMindmapStore((state) => state.setActiveCandidate);
  const createCandidate = useMindmapStore((state) => state.createCandidate);
  const appendCandidateTrail = useMindmapStore((state) => state.appendCandidateTrail);
  const setChoice = useMindmapStore((state) => state.setChoice);
  const optionState = useMindmapStore((state) => state.optionState);
  const rejectOption = useMindmapStore((state) => state.rejectOption);
  const reopenOption = useMindmapStore((state) => state.reopenOption);
  const pruneMissingNodes = useMindmapStore((state) => state.pruneMissingNodes);
  const trackUsage = useMindmapStore((state) => state.trackUsage);
  const restoreSession = useMindmapStore((state) => state.restoreSession);
  const { mutateAsync: generateNodes, isPending: isGeneratingNodes } = useGenerateNodesMutation();

  const [selection, setSelection] = useState<MindmapSelection>({
    topic: INITIAL_SELECTION.topic,
    lineage: [...INITIAL_SELECTION.lineage],
  });
  const [generateError, setGenerateError] = useState<string | null>(null);
  // First-run helper: auto-open the taxonomy dialog only AFTER the persisted
  // store has rehydrated. At first render `taxonomy` is still null even when
  // one is persisted (the persist middleware hydrates asynchronously), which
  // used to flash this dialog open on every reload.
  const [taxonomyDialogOpen, setTaxonomyDialogOpen] = useState(false);
  useEffect(() => {
    const openIfFirstRun = () => {
      if (!useMindmapStore.getState().taxonomy) setTaxonomyDialogOpen(true);
    };
    if (useMindmapStore.persist.hasHydrated()) {
      openIfFirstRun();
      return;
    }
    return useMindmapStore.persist.onFinishHydration(openIfFirstRun);
  }, []);
  const [generateNodesDialogOpen, setGenerateNodesDialogOpen] = useState(false);

  const activeDescriptionByTopic = useMemo(
    () => (taxonomy ? taxonomyToMindmapNodes(taxonomy).descriptionByTopic : SCHEMA_DESCRIPTION_BY_TOPIC),
    [taxonomy]
  );

  // ── Design-space view ──────────────────────────────────────────────────────
  const [view, setView] = useState<'map' | 'space' | 'axes'>('map');
  // The side panels are canvas overlays; the Examine view is a document. Below
  // xl the two cannot sit side by side, so entering Perspectives collapses the
  // panels — and in that view a closed panel shrinks to a small icon button
  // instead of a full-width header bar (re-expanding overlaps by the user's
  // deliberate choice). Leaving restores the default open state.
  const [contextPanelOpen, setContextPanelOpen] = useState(true);
  const [projectsPanelOpen, setProjectsPanelOpen] = useState(true);
  const [candidatePanelOpen, setCandidatePanelOpen] = useState(false);
  useEffect(() => {
    if (view === 'axes' && !window.matchMedia('(min-width: 1280px)').matches) {
      setContextPanelOpen(false);
      setProjectsPanelOpen(false);
      setCandidatePanelOpen(false);
    } else if (view !== 'axes') {
      setContextPanelOpen(true);
      setProjectsPanelOpen(true);
    }
  }, [view]);
  // Relevance lens: an on/off overlay on the design space (not a separate
  // mode — same view, same interactions, extra paint). Anchor is switchable
  // between the selected node and the active candidate.
  const [lensOn, setLensOn] = useState(false);
  const [lensSource, setLensSource] = useState<'selection' | 'candidate'>('selection');
  // Candidate "pick" flow: clicking an empty aspect slot ("—") in the panel
  // arms this; the next click on an option of that aspect (in ANY view)
  // fills the slot.
  const [pendingChoiceAspectId, setPendingChoiceAspectId] = useState<string | null>(null);
  const [pendingCell, setPendingCell] = useState<[number, number] | null>(null);
  // The currently-traced connector (transient; the discovered set persists).
  const [activeLine, setActiveLine] = useState<GenerationTrail | null>(null);
  // A corpus project opened for inspection (design-space glyph / provenance chip).
  const [focusProjectId, setFocusProjectId] = useState<string | null>(null);
  const { data: focusProject } = useCorpusProjectQuery(focusProjectId);
  // Fetched on first visit to the space view (cached forever afterwards).
  const { data: surface } = useSurfaceQuery(view === 'space');
  const { mutateAsync: locateNodes } = useLocateNodesMutation();
  const { mutateAsync: generateAt, isPending: isGeneratingAt } = useGenerateAtMutation();
  const { mutateAsync: peekAt } = usePeekMutation();
  // Gap preview (E1): a clicked empty cell opens this; generation happens only
  // from its "Generate here" button.
  const [gapPreview, setGapPreview] = useState<GapPreview | null>(null);
  const sessionFileRef = useRef<HTMLInputElement>(null);
  const locatingRef = useRef(false);
  // Cancels the client-side wait on a running generate-at (job completes
  // server-side but its result is discarded).
  const generateAbortRef = useRef<AbortController | null>(null);

  const attemptedRef = useRef<Set<string>>(new Set());

  // Warns when a generated brief sits far from the background corpus, so the
  // spatial context isn't read as meaningful where it isn't.
  const [corpusNotice, setCorpusNotice] = useState<string | null>(null);
  const CORPUS_SIMILARITY_FLOOR = 0.3;

  const handleTaxonomyGenerated = (
    result: Parameters<typeof setTaxonomy>[0] & { corpus_similarity?: number | null }
  ) => {
    // setTaxonomy rebuilds the tree and wipes coords/discovered/provenance.
    setTaxonomy(result);
    setSelection({ topic: INITIAL_SELECTION.topic, lineage: [...INITIAL_SELECTION.lineage] });
    setActiveLine(null);
    setPendingChoiceAspectId(null);
    attemptedRef.current.clear();
    const similarity = result.corpus_similarity;
    setCorpusNotice(
      similarity != null && similarity < CORPUS_SIMILARITY_FLOOR
        ? `This brief sits far from the background corpus (similarity ${similarity.toFixed(2)}). ` +
            'The design-space surface shows media-architecture projects — its spatial context may not transfer to this domain.'
        : null
    );
  };

  // Best-effort: embed + locate every node once per session — including nodes
  // with persisted coords, so values cached before a calibration change
  // (register-map or support-baseline refit) refresh on load while the cached
  // coords still render instantly. Failures (e.g. embedding server down) leave
  // the background surface intact and are retried on the next change.
  useEffect(() => {
    if (!surface) return;
    const items = nodesToLocateItems(nodes, activeDescriptionByTopic, descriptionById).filter(
      (it) => !attemptedRef.current.has(it.node_id)
    );
    if (items.length === 0 || locatingRef.current) return;

    locatingRef.current = true;
    items.forEach((it) => attemptedRef.current.add(it.node_id));
    locateNodes(items)
      .then((located) => mergeCoords(located))
      .catch(() => {
        items.forEach((it) => attemptedRef.current.delete(it.node_id));
      })
      .finally(() => {
        locatingRef.current = false;
      });
  }, [surface, nodes, activeDescriptionByTopic, descriptionById, locateNodes, mergeCoords]);

  // The currently selected node, its description (id-keyed for generated nodes,
  // topic-keyed for taxonomy nodes), and its provenance (when generated).
  // Resolution prefers the exact node id (set by node clicks in either view);
  // the topic/lineage walk is the fallback for legacy selections.
  const selectedNode = useMemo(() => {
    if (selection.nodeId) {
      const byId = indexNodesById(nodes).get(selection.nodeId);
      if (byId) return byId;
    }
    return findNodeByLineage(nodes, selection.lineage);
  }, [nodes, selection]);
  const description =
    (selectedNode ? descriptionById[selectedNode.id] : undefined) ??
    activeDescriptionByTopic[selection.topic] ??
    '';
  const request = buildRequest(selection, description);
  const { data, isFetching } = useRelatedProjectsQuery({ request });

  const selectedProvenance = selectedNode ? provenance[selectedNode.id] : undefined;

  // Corpus ids of the selection's related projects — the panel's examples are
  // also highlighted as places on the design-space map.
  const relatedProjectIds = useMemo(() => {
    const ids = new Set<string>();
    for (const project of data?.projects ?? []) {
      if (project.Name === PLACEHOLDER_PROJECT_NAME) continue;
      if (project.id) ids.add(project.id);
    }
    return ids;
  }, [data]);

  // ── Candidates: locate each design in the frozen space ──────────────────────
  // A candidate's position is the embedding of its BRIEF when present (the
  // identity layer — Part 10), else its composed option text; re-located
  // whenever that text changes (signature-tracked). When the star moves, the
  // old position joins the candidate's trail (its trajectory across revisions).
  const candidateTextSignatures = useRef<Map<string, string>>(new Map());
  useEffect(() => {
    if (!surface) return;
    const items: Array<{ node_id: string; text: string }> = [];
    for (const candidate of Object.values(candidates)) {
      const text = candidateEmbeddingText(
        candidate,
        nodes,
        activeDescriptionByTopic,
        descriptionById
      );
      const key = candidateCoordKey(candidate.id);
      if (!text) {
        candidateTextSignatures.current.delete(candidate.id);
        continue;
      }
      if (candidateTextSignatures.current.get(candidate.id) === text && coords[key]) continue;
      candidateTextSignatures.current.set(candidate.id, text);
      items.push({ node_id: key, text });
    }
    if (items.length === 0) return;
    locateNodes(items)
      .then((located) => {
        for (const [key, coord] of Object.entries(located)) {
          const previous = useMindmapStore.getState().coords[key];
          if (
            previous &&
            Math.hypot(previous.x - coord.x, previous.y - coord.y) > 0.02
          ) {
            appendCandidateTrail(key.replace(/^cand:/, ''), {
              x: previous.x,
              y: previous.y,
            });
          }
        }
        mergeCoords(located);
      })
      .catch(() => {
        // Allow a retry on the next composition change.
        for (const it of items) {
          candidateTextSignatures.current.delete(it.node_id.replace(/^cand:/, ''));
        }
      });
  }, [surface, candidates, nodes, activeDescriptionByTopic, descriptionById, coords, locateNodes, mergeCoords, appendCandidateTrail]);

  const candidateMarkers = useMemo<CandidateMarker[]>(() => {
    const markers: CandidateMarker[] = [];
    for (const candidate of Object.values(candidates)) {
      const coord = coords[candidateCoordKey(candidate.id)];
      if (!coord) continue;
      markers.push({
        id: candidate.id,
        name: candidate.name,
        x: coord.x,
        y: coord.y,
        active: candidate.id === activeCandidateId,
        trail: candidate.trail,
      });
    }
    return markers;
  }, [candidates, coords, activeCandidateId]);

  const rejectedIds = useMemo(() => new Set(Object.keys(optionState)), [optionState]);

  // Mind-map styling: rejected options muted; the active candidate's choices bold.
  const nodeStates = useMemo(() => {
    const states: Record<string, 'rejected' | 'chosen'> = {};
    const active = activeCandidateId ? candidates[activeCandidateId] : undefined;
    for (const optionId of Object.values(active?.choices ?? {})) states[optionId] = 'chosen';
    for (const nodeId of Object.keys(optionState)) states[nodeId] = 'rejected';
    return states;
  }, [activeCandidateId, candidates, optionState]);

  // ── Option actions (choose / reject) for the selected node ──────────────────
  // An "option" is any non-root, non-aspect node: lineage [root, aspect, ...].
  const selectedAspect = useMemo(
    () =>
      selection.lineage.length >= 3
        ? findNodeByLineage(nodes, selection.lineage.slice(0, 2))
        : null,
    [nodes, selection.lineage]
  );
  const isOptionSelected = Boolean(selectedNode && selectedAspect);
  const activeCandidate = activeCandidateId ? candidates[activeCandidateId] : undefined;
  const isChosen = Boolean(
    selectedNode &&
      selectedAspect &&
      activeCandidate?.choices[selectedAspect.id] === selectedNode.id
  );
  const selectedRejection = selectedNode ? optionState[selectedNode.id] : undefined;
  const [rejectReason, setRejectReason] = useState('');
  const [showRejectInput, setShowRejectInput] = useState(false);
  const [compareOpen, setCompareOpen] = useState(false);

  // ── Relevance lens: an overlay on the design space, anchored to either the
  //    selected node or the active candidate (explicitly switchable) ───────────
  const selectionAnchor = useMemo(() => {
    if (!selectedNode) return null;
    const desc =
      descriptionById[selectedNode.id] ?? activeDescriptionByTopic[selectedNode.topic] ?? '';
    return {
      id: selectedNode.id,
      label: selectedNode.topic,
      text: desc ? `${selectedNode.topic}. ${desc}` : selectedNode.topic,
    };
  }, [selectedNode, descriptionById, activeDescriptionByTopic]);

  const candidateAnchor = useMemo(() => {
    if (!activeCandidate) return null;
    // Brief-first: the lens asks about the actual design, not the choice list.
    const text = candidateEmbeddingText(
      activeCandidate,
      nodes,
      activeDescriptionByTopic,
      descriptionById
    );
    return text
      ? { id: candidateCoordKey(activeCandidate.id), label: activeCandidate.name, text }
      : null;
  }, [activeCandidate, nodes, activeDescriptionByTopic, descriptionById]);

  const lensAnchor =
    lensSource === 'candidate'
      ? candidateAnchor ?? selectionAnchor
      : selectionAnchor ?? candidateAnchor;

  const lensActive = view === 'space' && lensOn && Boolean(lensAnchor);
  const { data: relevance, error: relevanceError } = useRelevanceQuery(
    lensActive ? lensAnchor?.text ?? null : null
  );

  const handleChooseOption = () => {
    if (!selectedNode || !selectedAspect || selectedRejection) return;
    if (!activeCandidateId || !candidates[activeCandidateId]) createCandidate();
    setChoice(selectedAspect.id, isChosen ? null : selectedNode.id);
  };

  // How many candidates currently include the selected option — shown as a
  // warning before rejecting (rejection clears it from all of them).
  const chosenInCandidates = useMemo(
    () =>
      selectedNode
        ? Object.values(candidates).filter((candidate) =>
            Object.values(candidate.choices).includes(selectedNode.id)
          ).length
        : 0,
    [candidates, selectedNode]
  );

  // Exploration stats (E5) — the study instrument, shown live and exported.
  const explorationStats = useMemo(
    () =>
      computeExplorationStats({
        nodes,
        coords,
        discovered,
        provenance,
        optionState,
        candidates,
        activeCandidateId,
      }),
    [nodes, coords, discovered, provenance, optionState, candidates, activeCandidateId]
  );

  useEffect(() => {
    selectTopic({
      topic: selection.topic,
      lineage: [...selection.lineage],
      contextDescription: description,
    });
  }, [selection, description, selectTopic]);

  const handleSelect = (nextSelection: MindmapSelection) => {
    // Armed pick flow: if the click landed on an option of the awaited aspect
    // (works from the mind map AND the design space — both route here), fill
    // the candidate's slot and disarm. Rejected options stay un-choosable.
    if (pendingChoiceAspectId && nextSelection.lineage.length >= 3) {
      const aspect = findNodeByLineage(nodes, nextSelection.lineage.slice(0, 2));
      const optionId =
        nextSelection.nodeId ?? findNodeByLineage(nodes, nextSelection.lineage)?.id;
      if (aspect?.id === pendingChoiceAspectId && optionId && !optionState[optionId]) {
        setChoice(pendingChoiceAspectId, optionId);
        setPendingChoiceAspectId(null);
      }
    }
    setSelection({
      topic: nextSelection.topic,
      lineage: [...nextSelection.lineage],
      ...(nextSelection.nodeId ? { nodeId: nextSelection.nodeId } : {}),
    });
    setActiveLine(null); // a fresh selection clears any traced connector
    // Release any pinned corpus project: the inspection pin would otherwise
    // shadow the new selection's related projects at the top of the panel
    // ("stuck first project") for every node selected after a glyph click.
    setFocusProjectId(null);
  };

  // Structural edits from mind-elixir. A rename invalidates the node's position
  // (it was embedded from the old text) — drop the coord + attempted flag so the
  // node re-locates with its new label.
  const handleNodesChange = useCallback(
    (nextNodes: ReadonlyArray<MindmapNode>) => {
      const prevTopics = new Map<string, string>();
      const collect = (node: MindmapNode) => {
        prevTopics.set(node.id, node.topic);
        for (const child of node.children ?? []) collect(child);
      };
      for (const node of nodes) collect(node);

      const nextIds = new Set<string>();
      collectIds(nextNodes, nextIds);

      // A rename invalidates the node's position (embedded from the old text).
      const renamed: string[] = [];
      const diff = (node: MindmapNode) => {
        const prev = prevTopics.get(node.id);
        if (prev !== undefined && prev !== node.topic) renamed.push(node.id);
        for (const child of node.children ?? []) diff(child);
      };
      for (const node of nextNodes) diff(node);
      if (renamed.length > 0) {
        removeCoords(renamed);
        renamed.forEach((id) => attemptedRef.current.delete(id));
      }

      // A deletion orphans coords/provenance/option-state/choices — prune them.
      const removedAny = [...prevTopics.keys()].some((id) => !nextIds.has(id));
      if (removedAny) pruneMissingNodes(nextIds);

      setNodes(nextNodes);
    },
    [nodes, removeCoords, pruneMissingNodes, setNodes]
  );

  const handleGenerateAt = useCallback(
    async (x: number, y: number) => {
      if (!surface) return;

      // The backend derives the parent aspect from the click + current node
      // coordinates (one spatial notion of "here", shared with the seeds); the
      // current selection is only the fallback focus when nothing is located.
      const fallback =
        findNodeByLineage(nodes, selection.lineage) ?? (nodes[0] ?? null);
      if (!fallback) {
        setGenerateError('No node available to attach generated options to.');
        return;
      }

      const resolution = surface.grid.resolution;
      setPendingCell([Math.floor(x * resolution), Math.floor(y * resolution)]);
      setGenerateError(null);
      // NOTE: we deliberately do NOT move the selection to the focus aspect here.
      // The aspect's dot sits at its own embedding position — often far from both
      // the click and the generated children — so highlighting it looked like a
      // random, unconnected glow. The clicked cell glows instead (via the trail).

      const abort = new AbortController();
      generateAbortRef.current = abort;

      try {
        const response = await generateAt({
          x,
          y,
          allNodes: nodes,
          focusNode: { id: fallback.id, topic: fallback.topic },
          lineage: selection.lineage.length ? selection.lineage : [fallback.topic],
          coords,
          // Squiggle hypothesis (Part 10): the active candidate's brief
          // conditions gap-filling; the backend logs it for the A/B. Read from
          // the store at call time so the callback never carries a stale brief.
          brief: (() => {
            const s = useMindmapStore.getState();
            return s.activeCandidateId
              ? s.candidates[s.activeCandidateId]?.brief ?? null
              : null;
          })(),
          signal: abort.signal,
        });

        const rawChildren: MindmapNode[] = response.nodes.map((n) => ({
          id: n.node_id,
          topic: n.topic,
        }));
        const { children, remap } = ensureUniqueChildIds(nodes, rawChildren);
        const treeUpdate = insertChildrenAtNode(nodes, response.parent_id, children);
        if (!treeUpdate.inserted) {
          setGenerateError('Generated nodes were returned, but no matching parent was found.');
          return;
        }
        setNodes(treeUpdate.nodes);

        // Provenance: every generated node remembers the precedents that seeded
        // it and the location that was clicked.
        const seedProjects = toSeedProjects(response.seed_neighbours);
        const provenanceEntries: Record<string, NodeProvenance> = {};
        for (const child of children) {
          provenanceEntries[child.id] = {
            source: 'generate-at',
            seedProjects,
            target: { x, y },
            createdAt: Date.now(),
          };
        }
        recordProvenance(provenanceEntries);

        // Generated descriptions (id-keyed; remap-aware) — used for the Context
        // panel, retrieval, and any future re-locate of these nodes.
        const descriptionEntries: Record<string, string> = {};
        for (const n of response.nodes) {
          const id = remap[n.node_id] ?? n.node_id;
          if (n.desc) descriptionEntries[id] = n.desc;
        }
        mergeDescriptions(descriptionEntries);

        // Coordinates come back with the generation — no extra /locate call.
        // Keep them aligned with any ids that were remapped to stay unique.
        const merged: CoordMap = {};
        for (const c of response.coords) {
          const id = remap[c.node_id] ?? c.node_id;
          merged[id] = {
            x: c.x,
            y: c.y,
            ...(c.z != null ? { z: c.z } : {}),
            ...(c.confidence != null ? { confidence: c.confidence } : {}),
            ...(c.support != null ? { support: c.support } : {}),
          };
          attemptedRef.current.add(id);
        }
        mergeCoords(merged);
        trackUsage('generate_at');

        // Mark the clicked cell "discovered" (drawn hollow) and store + show the
        // connector to where the nodes landed, so the (often distant) placement is
        // visibly tied to the click. Re-clicking the hollow dot re-traces it.
        const targets = Object.values(merged);
        if (targets.length > 0) {
          const line: GenerationTrail = {
            from: { x, y },
            to: targets.map((c) => ({ x: c.x, y: c.y })),
            meanDrift: response.mean_drift ?? null,
          };
          const cellKey = `${Math.floor(x * resolution)},${Math.floor(y * resolution)}`;
          recordDiscovery(cellKey, line);
          setActiveLine(line);
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Failed to generate at location.';
        // A user-initiated cancel is not an error worth alarming about.
        if (!/cancelled/i.test(message)) setGenerateError(message);
      } finally {
        generateAbortRef.current = null;
        setPendingCell(null);
      }
    },
    [
      surface,
      nodes,
      coords,
      selection,
      generateAt,
      setNodes,
      mergeCoords,
      recordDiscovery,
      recordProvenance,
      mergeDescriptions,
      trackUsage,
    ]
  );

  // Gap preview (E1): read the neighbourhood before committing LLM time.
  const handlePeekAt = useCallback(
    async (x: number, y: number, screen: { x: number; y: number }) => {
      trackUsage('peek');
      setGapPreview({ x, y, screenX: screen.x, screenY: screen.y, data: null });
      try {
        const data = await peekAt({ x, y, allNodes: nodes, coords });
        // Ignore stale responses if the user has already peeked elsewhere.
        setGapPreview((prev) =>
          prev && prev.x === x && prev.y === y ? { ...prev, data } : prev
        );
      } catch (error) {
        setGapPreview(null);
        setGenerateError(error instanceof Error ? error.message : 'Gap preview failed.');
      }
    },
    [nodes, coords, peekAt, trackUsage]
  );

  const handleConfirmGenerate = useCallback(
    (x: number, y: number) => {
      setGapPreview(null);
      void handleGenerateAt(x, y);
    },
    [handleGenerateAt]
  );

  // ── Session save/load: the machine-restorable exploration record ────────────
  const handleSaveSession = () => {
    trackUsage('session_save');
    downloadTextFile(
      `llmind-session-${new Date().toISOString().slice(0, 10)}.json`,
      buildSessionFile(selectSessionSnapshot(useMindmapStore.getState())),
      'application/json;charset=utf-8'
    );
  };

  const handleLoadSession = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = ''; // allow re-selecting the same file
    if (!file) return;
    try {
      const snapshot = parseSessionFile(await file.text());
      if (!window.confirm(`Replace the current exploration with "${file.name}"?`)) return;
      restoreSession(snapshot);
      setSelection({ topic: INITIAL_SELECTION.topic, lineage: [...INITIAL_SELECTION.lineage] });
      setActiveLine(null);
      setGapPreview(null);
      setPendingChoiceAspectId(null);
      setLensOn(false);
      attemptedRef.current.clear();
      trackUsage('session_load');
    } catch (error) {
      setGenerateError(error instanceof Error ? error.message : 'Could not load session.');
    }
  };

  const handleGenerateNodes = async (
    dialogParams?: Pick<GenerateNodesParams, 'description' | 'mode' | 'reasoningEffort'>
  ) => {
    const focusNode = findNodeByLineage(nodes, selection.lineage);
    if (!focusNode) {
      setGenerateError('Unable to locate the selected topic in the current mind map.');
      return;
    }

    setGenerateError(null);
    setGenerateNodesDialogOpen(false);

    try {
      const fetchedProjects = data?.projects ?? [];
      const hasRealProjects = fetchedProjects.some(
        (p) => p.Name !== PLACEHOLDER_PROJECT_NAME
      );
      const relatedProjects = hasRealProjects ? fetchedProjects : null;

      const contextDescription = dialogParams?.description ?? description;

      const response = await generateNodes({
        allNodes: nodes,
        focusNode: { id: focusNode.id, topic: focusNode.topic },
        description: contextDescription,
        shouldQuerySupabase: !hasRealProjects,
        relatedProjects,
        mode: dialogParams?.mode,
        reasoningEffort: dialogParams?.reasoningEffort,
      });

      const { children: generatedChildren, remap } = ensureUniqueChildIds(
        nodes,
        generatedNodesToMindmapNodes(response)
      );
      const treeUpdate = insertChildrenAtNode(nodes, response.parent_id, generatedChildren);
      if (!treeUpdate.inserted) {
        setGenerateError('Generated nodes were returned, but no matching parent was found.');
        return;
      }

      setNodes(treeUpdate.nodes);

      // Provenance: record which related projects seeded these options.
      const seedProjects = toSeedProjects(response.related_projects);
      const provenanceEntries: Record<string, NodeProvenance> = {};
      for (const child of generatedChildren) {
        provenanceEntries[child.id] = {
          source: 'generate-nodes',
          seedProjects,
          createdAt: Date.now(),
        };
      }
      recordProvenance(provenanceEntries);

      // Generated descriptions (id-keyed; remap-aware).
      const descriptionEntries: Record<string, string> = {};
      for (const n of response.nodes) {
        const id = remap[n.node_id] ?? n.node_id;
        if (n.desc) descriptionEntries[id] = n.desc;
      }
      mergeDescriptions(descriptionEntries);
      trackUsage('generate_nodes');
    } catch (error) {
      setGenerateError(
        error instanceof Error ? error.message : 'Failed to generate nodes. Please try again.'
      );
    }
  };

  return (
    <main className="relative flex h-screen w-full flex-col overflow-hidden bg-background">
      {/* Floating Navigator (Bottom) — content-sized so the view toggles and the
          generate buttons never wrap or overlap. */}
      <header className="absolute bottom-8 left-1/2 z-50 flex max-w-[calc(100vw-2rem)] -translate-x-1/2 items-center gap-4 rounded-full border bg-background/80 px-4 py-2 shadow-xl backdrop-blur-md transition-all hover:bg-background/90">
        <div className="flex items-center justify-start gap-1">
          <Link
            href="/"
            className="flex items-center gap-2 whitespace-nowrap rounded-full px-4 py-2 text-xs font-semibold text-muted-foreground transition-all hover:bg-muted hover:text-foreground active:scale-95"
          >
            <Home className="h-4 w-4" />
            Home
          </Link>
          <button
            type="button"
            onClick={handleSaveSession}
            title="Save session (full exploration state as JSON)"
            className="rounded-full p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <Save className="h-4 w-4" />
          </button>
          <button
            type="button"
            onClick={() => sessionFileRef.current?.click()}
            title="Load a saved session (replaces the current exploration)"
            className="rounded-full p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <FolderOpen className="h-4 w-4" />
          </button>
          <input
            ref={sessionFileRef}
            type="file"
            accept="application/json,.json"
            className="hidden"
            onChange={handleLoadSession}
          />
        </div>

        <div className="flex items-center justify-center">
          <div className="flex shrink-0 items-center gap-0.5 rounded-full bg-muted/60 p-0.5">
            <button
              type="button"
              onClick={() => setView('map')}
              aria-pressed={view === 'map'}
              className={`flex items-center gap-1.5 whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-semibold transition-all active:scale-95 ${
                view === 'map'
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Network className="h-3.5 w-3.5" />
              Mind Map
            </button>
            <button
              type="button"
              onClick={() => {
                setView('space');
                trackUsage('view_space');
              }}
              aria-pressed={view === 'space'}
              className={`flex items-center gap-1.5 whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-semibold transition-all active:scale-95 ${
                view === 'space'
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Grid3x3 className="h-3.5 w-3.5" />
              Design Space
            </button>
            <button
              type="button"
              onClick={() => {
                setView('axes');
                trackUsage('view_axes');
              }}
              aria-pressed={view === 'axes'}
              className={`flex items-center gap-1.5 whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-semibold transition-all active:scale-95 ${
                view === 'axes'
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Compass className="h-3.5 w-3.5" />
              Perspectives
            </button>
          </div>
        </div>

        <div className="flex items-center justify-end gap-3">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setTaxonomyDialogOpen(true)}
            className="h-9 gap-2 whitespace-nowrap rounded-full px-5 text-xs font-bold shadow-sm transition-all hover:bg-accent hover:text-accent-foreground active:scale-95"
          >
            <Sparkles className="h-4 w-4 text-primary" />
            Generate Taxonomy
          </Button>
          <Button
            type="button"
            size="sm"
            onClick={() => setGenerateNodesDialogOpen(true)}
            disabled={isGeneratingNodes || isFetching}
            className="h-9 gap-2 whitespace-nowrap rounded-full px-5 text-xs font-bold shadow-sm transition-all active:scale-95"
          >
            {isGeneratingNodes ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Zap className="h-4 w-4 fill-current" />
            )}
            {isGeneratingNodes ? 'Generating...' : 'Generate Nodes'}
          </Button>
        </div>
      </header>

      {/* Floating Lineage / Info Panel. The Examine view is a document, not a
          pannable canvas — its content clears the panels at xl widths; below
          that, entering Perspectives collapses them, and a closed panel there
          renders as a thin icon button (pointer-events pass through the rest). */}
      <div className="pointer-events-none absolute top-4 left-4 z-40 w-full max-w-sm">
        {view === 'axes' && !contextPanelOpen ? (
          <PanelIconButton
            icon={Info}
            label="Context"
            onClick={() => setContextPanelOpen(true)}
          />
        ) : (
        <div className="pointer-events-auto">
        <Collapsible open={contextPanelOpen} onOpenChange={setContextPanelOpen}>
          <section className="overflow-hidden rounded-2xl border bg-background/90 shadow-xl backdrop-blur-md">
            <div className="flex items-center justify-between px-4 py-3">
              <div className="flex items-center gap-2">
                <Info className="h-4 w-4 text-primary" />
                <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                  Context
                </h2>
              </div>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="icon" className="group h-6 w-6 rounded-full">
                  <ChevronRight className="h-4 w-4 transition-transform duration-200 group-data-[state=open]:rotate-90" />
                </Button>
              </CollapsibleTrigger>
            </div>

            <CollapsibleContent>
              <ScrollArea className="max-h-[300px]">
                <div className="space-y-4 px-4 pb-4">
                  <div className="flex flex-wrap items-center gap-1.5">
                    {selection.lineage.map((topic, idx) => (
                      <span key={`${topic}-${idx}`} className="flex items-center gap-1.5">
                        <Badge variant="secondary" className="px-2 py-0 font-medium">
                          {topic}
                        </Badge>
                        {idx < selection.lineage.length - 1 && (
                          <ChevronRight className="h-3 w-3 text-muted-foreground/40" />
                        )}
                      </span>
                    ))}
                  </div>

                  {corpusNotice ? (
                    <p className="rounded-lg bg-amber-500/10 p-2.5 text-xs leading-snug text-amber-700">
                      {corpusNotice}
                    </p>
                  ) : null}

                  {description ? (
                    <div className="space-y-1 text-sm leading-relaxed">
                      <p className="text-muted-foreground">{description}</p>
                    </div>
                  ) : null}

                  {selectedProvenance && selectedProvenance.seedProjects.length > 0 ? (
                    <div className="space-y-1.5">
                      <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                        Seeded by{' '}
                        {selectedProvenance.source === 'generate-at'
                          ? 'projects near the clicked location'
                          : 'related projects'}
                      </p>
                      <div className="flex flex-wrap gap-1">
                        {selectedProvenance.seedProjects.map((seed, idx) => (
                          <button
                            key={`${seed.id ?? seed.name}-${idx}`}
                            type="button"
                            disabled={!seed.id}
                            onClick={() => seed.id && setFocusProjectId(seed.id)}
                            className="rounded-full border bg-muted/60 px-2 py-0.5 text-[10px] font-medium text-foreground transition-colors enabled:hover:bg-muted disabled:cursor-default disabled:opacity-60"
                            title={seed.id ? 'View this project' : undefined}
                          >
                            {seed.name}
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  {isOptionSelected && selectedNode && selectedAspect ? (
                    <div className="space-y-1.5">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <button
                          type="button"
                          onClick={handleChooseOption}
                          disabled={Boolean(selectedRejection)}
                          title={
                            selectedRejection
                              ? 'Rejected options cannot be chosen — reopen it first'
                              : undefined
                          }
                          className={`rounded-full border px-2.5 py-0.5 text-[11px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                            isChosen
                              ? 'border-violet-500 bg-violet-500/10 text-violet-700 hover:bg-violet-500/20'
                              : 'text-foreground hover:bg-muted'
                          }`}
                        >
                          {isChosen
                            ? `✓ Chosen for ${selectedAspect.topic}`
                            : `Choose for ${selectedAspect.topic}`}
                        </button>
                        {selectedRejection ? (
                          <button
                            type="button"
                            onClick={() => reopenOption(selectedNode.id)}
                            className="rounded-full border px-2.5 py-0.5 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-muted"
                          >
                            Reopen
                          </button>
                        ) : (
                          <button
                            type="button"
                            onClick={() => setShowRejectInput((v) => !v)}
                            className="rounded-full border px-2.5 py-0.5 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
                          >
                            Reject…
                          </button>
                        )}
                      </div>
                      {selectedRejection ? (
                        <p className="text-[11px] text-destructive">
                          Rejected{selectedRejection.reason ? ` — ${selectedRejection.reason}` : ''}
                        </p>
                      ) : null}
                      {showRejectInput && !selectedRejection && chosenInCandidates > 0 ? (
                        <p className="text-[10px] font-medium text-amber-700">
                          Also removes it from {chosenInCandidates} candidate
                          {chosenInCandidates > 1 ? 's' : ''}.
                        </p>
                      ) : null}
                      {showRejectInput && !selectedRejection ? (
                        <form
                          className="flex items-center gap-1.5"
                          onSubmit={(e) => {
                            e.preventDefault();
                            rejectOption(selectedNode.id, rejectReason.trim() || undefined);
                            setRejectReason('');
                            setShowRejectInput(false);
                          }}
                        >
                          <input
                            value={rejectReason}
                            onChange={(e) => setRejectReason(e.target.value)}
                            placeholder="Why rule this out? (optional)"
                            className="h-7 flex-1 rounded-md border bg-background px-2 text-[11px]"
                          />
                          <Button type="submit" size="sm" variant="destructive" className="h-7 px-2.5 text-[11px]">
                            Reject
                          </Button>
                        </form>
                      ) : null}
                    </div>
                  ) : null}

                  {generateError ? (
                    <div className="flex items-center justify-between gap-2 rounded-lg bg-destructive/10 p-2.5 text-xs font-medium text-destructive">
                      <span>{generateError}</span>
                      <button
                        type="button"
                        className="underline underline-offset-2"
                        onClick={() => setGenerateNodesDialogOpen(true)}
                        disabled={isGeneratingNodes}
                      >
                        Retry
                      </button>
                    </div>
                  ) : null}

                  {/* Exploration stats (E5) — live progress + study instrument */}
                  <p className="border-t pt-2 text-[10px] leading-relaxed text-muted-foreground">
                    {formatExplorationStats(explorationStats)}
                  </p>
                </div>
              </ScrollArea>
            </CollapsibleContent>
          </section>
        </Collapsible>

        </div>
        )}

        {/* Candidate composition panel */}
        {view === 'axes' && !candidatePanelOpen ? (
          <div className="mt-3">
            <PanelIconButton
              icon={Star}
              label="Candidate"
              onClick={() => setCandidatePanelOpen(true)}
            />
          </div>
        ) : (
        <div className="pointer-events-auto mt-3">
          <CandidatePanel
            open={candidatePanelOpen}
            onOpenChange={setCandidatePanelOpen}
            descriptionByTopic={activeDescriptionByTopic}
            onOpenProject={setFocusProjectId}
            onOpenCompare={() => {
              trackUsage('compare_opened');
              setCompareOpen(true);
            }}
            pendingAspectId={pendingChoiceAspectId}
            onStartPickChoice={setPendingChoiceAspectId}
            onCancelPickChoice={() => setPendingChoiceAspectId(null)}
            onInspectRelevance={() => {
              trackUsage('lens_on');
              setLensSource('candidate');
              setLensOn(true);
              setView('space');
            }}
            onOpenExamine={() => {
              trackUsage('examine_opened');
              setView('axes');
            }}
          />
        </div>
        )}
      </div>

      {/* Relevance lens: an overlay toggle on the design space (not a mode) */}
      {view === 'space' && (
        <div className="absolute top-4 left-1/2 z-40 flex -translate-x-1/2 flex-col items-center gap-1">
          <div className="flex items-center gap-1 rounded-full border bg-background/90 p-0.5 shadow-md backdrop-blur">
            <button
              type="button"
              onClick={() => {
                if (!lensOn) trackUsage('lens_on');
                setLensOn(!lensOn);
              }}
              aria-pressed={lensOn}
              disabled={!lensOn && !lensAnchor}
              title={
                !lensAnchor
                  ? 'Select a node or candidate to anchor the lens'
                  : 'Color real projects by relevance to the anchor'
              }
              className={`flex items-center gap-1.5 whitespace-nowrap rounded-full px-3 py-1 text-[11px] font-semibold transition-all disabled:cursor-not-allowed disabled:opacity-50 ${
                lensOn && lensAnchor
                  ? 'bg-muted text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Focus className="h-3 w-3" />
              Relevance lens {lensOn && lensAnchor ? 'on' : 'off'}
            </button>
            {/* Anchor switcher — only when both anchors exist */}
            {lensOn && selectionAnchor && candidateAnchor && (
              <>
                <span className="h-3.5 w-px bg-border" />
                {(
                  [
                    ['selection', selectionAnchor.label],
                    ['candidate', candidateAnchor.label],
                  ] as const
                ).map(([source, label]) => (
                  <button
                    key={source}
                    type="button"
                    onClick={() => setLensSource(source)}
                    className={`max-w-[9rem] truncate whitespace-nowrap rounded-full px-2 py-1 text-[10px] font-medium transition-colors ${
                      lensSource === source
                        ? 'bg-violet-500/10 text-violet-700'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                    title={`Anchor the lens to ${source === 'selection' ? 'the selected node' : 'the active candidate'}: ${label}`}
                  >
                    {source === 'selection' ? '◉' : '★'} {label}
                  </button>
                ))}
              </>
            )}
          </div>
          {lensActive && relevanceError ? (
            <p className="rounded-md bg-destructive/10 px-2 py-0.5 text-[10px] font-medium text-destructive">
              {relevanceError instanceof Error ? relevanceError.message : 'Lens unavailable.'}
            </p>
          ) : null}
        </div>
      )}

      {/* Main view layer — all views share nodes + selection */}
      <div className="relative h-full w-full">
        {view === 'axes' ? (
          <ExamineView
            nodes={nodes}
            selection={selection}
            onSelectNode={handleSelect}
            onSelectProject={setFocusProjectId}
            descriptionByTopic={activeDescriptionByTopic}
          />
        ) : view === 'space' ? (
          surface ? (
            <DesignSpaceSurface
              surface={surface}
              nodes={nodes}
              coords={coords}
              selection={selection}
              onSelectNode={handleSelect}
              onGenerateAt={handleConfirmGenerate}
              onPeekAt={handlePeekAt}
              preview={gapPreview}
              onDismissPreview={() => setGapPreview(null)}
              onSelectProject={setFocusProjectId}
              isGenerating={isGeneratingAt}
              pendingCell={pendingCell}
              trail={activeLine}
              discovered={discovered}
              onShowDiscovery={(key) => setActiveLine(discovered[key] ?? null)}
              onBackgroundClick={() => {
                setSelection({ topic: '', lineage: [] });
                setActiveLine(null);
              }}
              candidates={candidateMarkers}
              onSelectCandidate={setActiveCandidate}
              rejected={rejectedIds}
              onCancelGenerate={() => generateAbortRef.current?.abort()}
              relatedProjects={!lensActive ? relatedProjectIds : undefined}
              relevance={lensActive ? relevance ?? null : null}
              lensAnchorId={lensActive ? lensAnchor?.id ?? null : null}
              lensAnchorLabel={lensActive ? lensAnchor?.label ?? null : null}
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center p-8 text-center">
              <p className="max-w-md text-sm text-muted-foreground">
                Design-space surface unavailable. Build it with{' '}
                <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">
                  uv run python database_pipeline.py project
                </code>{' '}
                and ensure the backend is running.
              </p>
            </div>
          )
        ) : (
          <SimpleMindMap
            nodes={nodes}
            activeTopic={selection.topic}
            onSelect={handleSelect}
            onDataChange={handleNodesChange}
            generatingNodeId={
              isGeneratingNodes ? findNodeByLineage(nodes, selection.lineage)?.id ?? null : null
            }
            nodeStates={nodeStates}
          />
        )}
      </div>

      <GenerateTaxonomyDialog
        open={taxonomyDialogOpen}
        onOpenChange={setTaxonomyDialogOpen}
        onSuccess={handleTaxonomyGenerated}
      />

      <GenerateNodesDialog
        open={generateNodesDialogOpen}
        onOpenChange={setGenerateNodesDialogOpen}
        onConfirm={handleGenerateNodes}
        isPending={isGeneratingNodes}
        selectedTopic={selection.topic}
      />

      <CompareCandidatesDialog
        open={compareOpen}
        onOpenChange={setCompareOpen}
        descriptionByTopic={activeDescriptionByTopic}
      />

      {/* Floating Related Projects Panel (collapses to an icon in Examine — see
          the lineage panel note) */}
      <div className="pointer-events-none absolute top-4 right-4 z-40 flex w-full max-w-md justify-end">
        {view === 'axes' && !projectsPanelOpen ? (
          <PanelIconButton
            icon={PanelsRightBottom}
            label="Related projects"
            onClick={() => setProjectsPanelOpen(true)}
          />
        ) : (
        <div className="pointer-events-auto w-full">
        <Collapsible open={projectsPanelOpen} onOpenChange={setProjectsPanelOpen}>
          <section className="overflow-hidden rounded-2xl border bg-background/90 shadow-xl backdrop-blur-md">
            <div className="flex items-center justify-between px-4 py-3">
              <div className="flex items-center gap-2">
                <PanelsRightBottom className="h-4 w-4 text-primary" />
                <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                  Related Projects
                </h2>
                {data?.projects?.length ? (
                  <Badge variant="outline" className="h-5 px-1.5 text-[10px]">
                    {data.projects.length}
                  </Badge>
                ) : null}
              </div>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="icon" className="group h-6 w-6 rounded-full">
                  <ChevronRight className="h-4 w-4 transition-transform duration-200 group-data-[state=open]:rotate-90" />
                </Button>
              </CollapsibleTrigger>
            </div>

            <CollapsibleContent>
              {/* No outer ScrollArea: the panel's two columns scroll
                  independently (a wrapping scroller would scroll them as one). */}
              <div className="p-4 pt-0">
                <SimpleProjectPanel
                  projects={data?.projects ?? []}
                  isLoading={isFetching}
                  focusProject={focusProject ?? null}
                />
              </div>
            </CollapsibleContent>
          </section>
        </Collapsible>
        </div>
        )}
      </div>
    </main>
  );
}
