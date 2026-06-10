'use client';

import {
  ChevronRight,
  Grid3x3,
  Home,
  Info,
  Loader2,
  Network,
  PanelsRightBottom,
  Sparkles,
  Zap,
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
import { CandidatePanel } from '@/src/components/design-space/candidate-panel';
import { CompareCandidatesDialog } from '@/src/components/design-space/compare-candidates-dialog';
import {
  candidateCoordKey,
  composeCandidateText,
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

interface TreeUpdateResult {
  nodes: ReadonlyArray<MindmapNode>;
  inserted: boolean;
}

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

function findNodeByLineage(
  nodes: ReadonlyArray<MindmapNode>,
  lineage: ReadonlyArray<string>
): MindmapNode | null {
  let currentNodes = nodes;
  let currentNode: MindmapNode | undefined;

  for (const topic of lineage) {
    currentNode = currentNodes.find((node) => node.topic === topic);
    if (!currentNode) {
      return null;
    }
    currentNodes = currentNode.children ?? [];
  }

  return currentNode ?? null;
}

function insertChildrenAtNode(
  nodes: ReadonlyArray<MindmapNode>,
  parentId: string,
  childrenToInsert: ReadonlyArray<MindmapNode>
): TreeUpdateResult {
  let inserted = false;
  const nextNodes = nodes.map((node) => {
    if (node.id === parentId) {
      inserted = true;
      const existingChildren = node.children ?? [];
      const existingIds = new Set(existingChildren.map((child) => child.id));
      const uniqueNewChildren = childrenToInsert.filter((child) => !existingIds.has(child.id));
      return {
        ...node,
        children: [...existingChildren, ...uniqueNewChildren],
      };
    }

    if (!node.children?.length) {
      return node;
    }

    const childResult = insertChildrenAtNode(node.children, parentId, childrenToInsert);
    if (!childResult.inserted) {
      return node;
    }

    inserted = true;
    return {
      ...node,
      children: childResult.nodes,
    };
  });

  return { nodes: nextNodes, inserted };
}

function collectIds(nodes: ReadonlyArray<MindmapNode>, into: Set<string>): void {
  for (const node of nodes) {
    into.add(node.id);
    if (node.children?.length) collectIds(node.children, into);
  }
}

/**
 * Node ids are slugified names (and LLM-supplied ids for generated nodes), so a
 * newly generated option can collide with an existing node anywhere in the tree
 * — duplicate ids break React keys (e.g. `n-portable`) and the id→coordinate
 * mapping in the design space. Remap any colliding child id to a unique one and
 * report the remap so callers can keep coordinates aligned.
 */
function ensureUniqueChildIds(
  allNodes: ReadonlyArray<MindmapNode>,
  children: ReadonlyArray<MindmapNode>
): { children: MindmapNode[]; remap: Record<string, string> } {
  const used = new Set<string>();
  collectIds(allNodes, used);
  const remap: Record<string, string> = {};

  const result = children.map((child) => {
    let id = child.id;
    if (used.has(id)) {
      let n = 2;
      while (used.has(`${child.id}-${n}`)) n++;
      id = `${child.id}-${n}`;
      remap[child.id] = id;
    }
    used.add(id);
    return { ...child, id };
  });

  return { children: result, remap };
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
  const setChoice = useMindmapStore((state) => state.setChoice);
  const optionState = useMindmapStore((state) => state.optionState);
  const rejectOption = useMindmapStore((state) => state.rejectOption);
  const reopenOption = useMindmapStore((state) => state.reopenOption);
  const { mutateAsync: generateNodes, isPending: isGeneratingNodes } = useGenerateNodesMutation();

  const [selection, setSelection] = useState<MindmapSelection>({
    topic: INITIAL_SELECTION.topic,
    lineage: [...INITIAL_SELECTION.lineage],
  });
  const [generateError, setGenerateError] = useState<string | null>(null);
  const [taxonomyDialogOpen, setTaxonomyDialogOpen] = useState(() => !taxonomy);
  const [generateNodesDialogOpen, setGenerateNodesDialogOpen] = useState(false);

  const activeDescriptionByTopic = useMemo(
    () => (taxonomy ? taxonomyToMindmapNodes(taxonomy).descriptionByTopic : SCHEMA_DESCRIPTION_BY_TOPIC),
    [taxonomy]
  );

  // ── Design-space view ──────────────────────────────────────────────────────
  const [view, setView] = useState<'map' | 'space'>('map');
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
    attemptedRef.current.clear();
    const similarity = result.corpus_similarity;
    setCorpusNotice(
      similarity != null && similarity < CORPUS_SIMILARITY_FLOOR
        ? `This brief sits far from the background corpus (similarity ${similarity.toFixed(2)}). ` +
            'The design-space surface shows media-architecture projects — its spatial context may not transfer to this domain.'
        : null
    );
  };

  // Best-effort: embed + locate any nodes that lack coordinates. Each node is
  // attempted at most once per session; failures (e.g. embedding server down)
  // leave the background surface intact and are retried on the next change.
  useEffect(() => {
    if (!surface) return;
    const items = nodesToLocateItems(nodes, activeDescriptionByTopic, descriptionById).filter(
      (it) => !coords[it.node_id] && !attemptedRef.current.has(it.node_id)
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
  }, [surface, nodes, activeDescriptionByTopic, descriptionById, coords, locateNodes, mergeCoords]);

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

  // ── Candidates: locate each composed design in the frozen space ─────────────
  // A candidate's position is the embedding of its combined option text; it is
  // re-located whenever its composition changes (signature-tracked).
  const candidateTextSignatures = useRef<Map<string, string>>(new Map());
  useEffect(() => {
    if (!surface) return;
    const items: Array<{ node_id: string; text: string }> = [];
    for (const candidate of Object.values(candidates)) {
      const text = composeCandidateText(
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
      .then((located) => mergeCoords(located))
      .catch(() => {
        // Allow a retry on the next composition change.
        for (const it of items) {
          candidateTextSignatures.current.delete(it.node_id.replace(/^cand:/, ''));
        }
      });
  }, [surface, candidates, nodes, activeDescriptionByTopic, descriptionById, coords, locateNodes, mergeCoords]);

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

  const handleChooseOption = () => {
    if (!selectedNode || !selectedAspect) return;
    if (!activeCandidateId || !candidates[activeCandidateId]) createCandidate();
    setChoice(selectedAspect.id, isChosen ? null : selectedNode.id);
  };

  useEffect(() => {
    selectTopic({
      topic: selection.topic,
      lineage: [...selection.lineage],
      contextDescription: description,
    });
  }, [selection, description, selectTopic]);

  const handleSelect = (nextSelection: MindmapSelection) => {
    setSelection({
      topic: nextSelection.topic,
      lineage: [...nextSelection.lineage],
      ...(nextSelection.nodeId ? { nodeId: nextSelection.nodeId } : {}),
    });
    setActiveLine(null); // a fresh selection clears any traced connector
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
      setNodes(nextNodes);
    },
    [nodes, removeCoords, setNodes]
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
          };
          attemptedRef.current.add(id);
        }
        mergeCoords(merged);

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
    ]
  );

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
    } catch (error) {
      setGenerateError(
        error instanceof Error ? error.message : 'Failed to generate nodes. Please try again.'
      );
    }
  };

  return (
    <main className="relative flex h-screen w-full flex-col overflow-hidden bg-background">
      {/* Floating Navigator (Bottom) */}
      <header className="absolute bottom-8 left-1/2 z-50 grid w-[calc(100%-2rem)] max-w-4xl -translate-x-1/2 grid-cols-3 items-center rounded-full border bg-background/80 px-4 py-2 shadow-xl backdrop-blur-md transition-all hover:bg-background/90">
        <div className="flex items-center justify-start">
          <Link
            href="/"
            className="flex items-center gap-2 rounded-full px-4 py-2 text-xs font-semibold text-muted-foreground transition-all hover:bg-muted hover:text-foreground active:scale-95"
          >
            <Home className="h-4 w-4" />
            Home
          </Link>
        </div>

        <div className="flex items-center justify-center">
          <div className="flex items-center gap-0.5 rounded-full bg-muted/60 p-0.5">
            <button
              type="button"
              onClick={() => setView('map')}
              aria-pressed={view === 'map'}
              className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold transition-all active:scale-95 ${
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
              onClick={() => setView('space')}
              aria-pressed={view === 'space'}
              className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold transition-all active:scale-95 ${
                view === 'space'
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Grid3x3 className="h-3.5 w-3.5" />
              Design Space
            </button>
          </div>
        </div>

        <div className="flex items-center justify-end gap-3">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setTaxonomyDialogOpen(true)}
            className="h-9 gap-2 rounded-full px-5 text-xs font-bold shadow-sm transition-all hover:bg-accent hover:text-accent-foreground active:scale-95"
          >
            <Sparkles className="h-4 w-4 text-primary" />
            Generate Taxonomy
          </Button>
          <Button
            type="button"
            size="sm"
            onClick={() => setGenerateNodesDialogOpen(true)}
            disabled={isGeneratingNodes || isFetching}
            className="h-9 gap-2 rounded-full px-5 text-xs font-bold shadow-sm transition-all active:scale-95"
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

      {/* Floating Lineage / Info Panel */}
      <div className="absolute top-4 left-4 z-40 w-full max-w-sm">
        <Collapsible defaultOpen={true}>
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
                          className={`rounded-full border px-2.5 py-0.5 text-[11px] font-medium transition-colors ${
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
                </div>
              </ScrollArea>
            </CollapsibleContent>
          </section>
        </Collapsible>

        {/* Candidate composition panel */}
        <div className="mt-3">
          <CandidatePanel
            descriptionByTopic={activeDescriptionByTopic}
            onOpenProject={setFocusProjectId}
            onOpenCompare={() => setCompareOpen(true)}
          />
        </div>
      </div>

      {/* Main view layer — mind map and design space share nodes + selection */}
      <div className="relative h-full w-full">
        {view === 'space' ? (
          surface ? (
            <DesignSpaceSurface
              surface={surface}
              nodes={nodes}
              coords={coords}
              selection={selection}
              onSelectNode={handleSelect}
              onGenerateAt={handleGenerateAt}
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
              relatedProjects={relatedProjectIds}
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

      {/* Floating Related Projects Panel */}
      <div className="absolute top-4 right-4 z-40 w-full max-w-md">
        <Collapsible defaultOpen={true}>
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
              <ScrollArea className="max-h-[calc(100vh-12rem)]">
                <div className="p-4 pt-0">
                  <SimpleProjectPanel
                    projects={data?.projects ?? []}
                    isLoading={isFetching}
                    focusProject={focusProject ?? null}
                  />
                </div>
              </ScrollArea>
            </CollapsibleContent>
          </section>
        </Collapsible>
      </div>
    </main>
  );
}
