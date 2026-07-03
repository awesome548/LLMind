'use client';

import {
  ChevronRight,
  Compass,
  FlaskConical,
  Focus,
  FolderOpen,
  Grid3x3,
  Home,
  Info,
  LayoutGrid,
  Loader2,
  Microscope,
  Network,
  PanelsRightBottom,
  Save,
  Sparkles,
  Star,
  Table2,
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
import { CandidateStrips } from '@/src/components/design-space/candidate-strips';
import { CrossTabView, type KeepCellIdea } from '@/src/components/design-space/cross-tab-view';
import { ExamineView } from '@/src/components/design-space/examine-view';
import { ProposalChips, type OptionProposal } from '@/src/components/design-space/proposal-chips';
import {
  ReflectionChip,
  type ReflectionPromptState,
} from '@/src/components/design-space/reflection-chip';
import { ReplayTimeline } from '@/src/components/design-space/replay-timeline';
import { buildReplayOverlay } from '@/src/features/design-space/replay-utils';
import { useDraftReflectionMutation } from '@/src/features/design-space/hooks/use-draft-reflection-mutation';
import { SchemaTable } from '@/src/components/design-space/schema-table';
import {
  buildSchemaColumns,
  computeFacetMatches,
  poorlyCoveredProjects,
} from '@/src/features/design-space/schema-utils';
import { useAnnotationQuery } from '@/src/features/design-space/hooks/use-annotation-query';
import { useRationaleQuery } from '@/src/features/design-space/hooks/use-rationale-query';
import { useMissingAspectMutation } from '@/src/features/design-space/hooks/use-missing-aspect-mutation';
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
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/src/components/ui/dialog';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/src/components/ui/collapsible';
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
import { toast } from 'sonner';
import { buildSessionFile, parseSessionFile } from '@/src/lib/session-io';
import { buildStudyBundle, studyBundleFilename } from '@/src/lib/study-bundle';
import { buildExplorationMarkdown, downloadTextFile } from '@/src/lib/export-exploration';
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
  const projectBrief = useMindmapStore((state) => state.projectBrief);
  const setProjectBrief = useMindmapStore((state) => state.setProjectBrief);
  const participantId = useMindmapStore((state) => state.participantId);
  const setParticipantId = useMindmapStore((state) => state.setParticipantId);
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
  const setCandidateBrief = useMindmapStore((state) => state.setCandidateBrief);
  const optionState = useMindmapStore((state) => state.optionState);
  const rejectOption = useMindmapStore((state) => state.rejectOption);
  const reopenOption = useMindmapStore((state) => state.reopenOption);
  const pruneMissingNodes = useMindmapStore((state) => state.pruneMissingNodes);
  const trackUsage = useMindmapStore((state) => state.trackUsage);
  const restoreSession = useMindmapStore((state) => state.restoreSession);
  const events = useMindmapStore((state) => state.events);
  const reflections = useMindmapStore((state) => state.reflections);
  const recordEvent = useMindmapStore((state) => state.recordEvent);
  const addReflection = useMindmapStore((state) => state.addReflection);
  const { mutateAsync: generateNodes, isPending: isGeneratingNodes } = useGenerateNodesMutation();

  const [selection, setSelection] = useState<MindmapSelection>({
    topic: INITIAL_SELECTION.topic,
    lineage: [...INITIAL_SELECTION.lineage],
  });
  const [generateError, setGenerateError] = useState<string | null>(null);
  // First-run choice (Part 13 L-B, per the study's layered inform→filter
  // model): brief-first ("write down what you're imagining" → a space scoped
  // to it) or discover-first (explore the prebuilt space, generate later).
  // Opens only AFTER the persisted store has rehydrated (at first render
  // `taxonomy` is still null even when one is persisted — the persist
  // middleware hydrates asynchronously, which used to flash dialogs open on
  // every reload), and only ONCE (tracked in the persisted usage counters) —
  // working with the default schema is a valid choice, not a nag target.
  const [taxonomyDialogOpen, setTaxonomyDialogOpen] = useState(false);
  const [firstRunChoiceOpen, setFirstRunChoiceOpen] = useState(false);
  useEffect(() => {
    const openIfFirstRun = () => {
      const store = useMindmapStore.getState();
      if (!store.taxonomy && !store.usage['taxonomy_dialog_offered']) {
        setFirstRunChoiceOpen(true);
        store.trackUsage('taxonomy_dialog_offered');
      }
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
  const [view, setView] = useState<'map' | 'space' | 'axes' | 'schema' | 'crosstab'>('map');
  // Which tab Perspectives opens on — 'scatter' when entered via the
  // cross-tab's "show as continuous scatter" drill-down.
  const [examineInitialTab, setExamineInitialTab] = useState<'strips' | 'scatter'>('strips');
  // Document views (vs canvas views) — the floating panels icon-collapse
  // below xl in these, sharing one layout grammar.
  const dockedView = view === 'axes' || view === 'schema' || view === 'crosstab';
  // A3 facets: transient (never persisted) ± option filters driving the map's
  // faceted fading; set from the schema table.
  const [facetInclude, setFacetInclude] = useState<ReadonlySet<string>>(new Set());
  const [facetExclude, setFacetExclude] = useState<ReadonlySet<string>>(new Set());
  // The side panels are canvas overlays; the Examine view is a document. Below
  // xl the two cannot sit side by side, so entering Perspectives collapses the
  // panels — and in that view a closed panel shrinks to a small icon button
  // instead of a full-width header bar (re-expanding overlaps by the user's
  // deliberate choice). Leaving restores the default open state.
  const [contextPanelOpen, setContextPanelOpen] = useState(true);
  const [projectsPanelOpen, setProjectsPanelOpen] = useState(true);
  const [candidatePanelOpen, setCandidatePanelOpen] = useState(false);
  // B1 inspector dock: the Examine strips inside the map view, so examining a
  // candidate needs no trip to Perspectives. Renders only while a candidate
  // is active; open by default.
  const [inspectorOpen, setInspectorOpen] = useState(true);
  useEffect(() => {
    const docked = view === 'axes' || view === 'schema' || view === 'crosstab';
    const xl = window.matchMedia('(min-width: 1280px)');
    const collapseIfCramped = () => {
      // Also fires when the window RESIZES below xl while already in a
      // docked view — entering alone would leave open panels overlapping.
      if (docked && !xl.matches) {
        setContextPanelOpen(false);
        setProjectsPanelOpen(false);
        setCandidatePanelOpen(false);
      }
    };
    collapseIfCramped();
    if (!docked) {
      setContextPanelOpen(true);
      setProjectsPanelOpen(true);
    }
    xl.addEventListener('change', collapseIfCramped);
    return () => xl.removeEventListener('change', collapseIfCramped);
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
  // Armed pick mode is a MODE — it needs a global escape hatch, not only the
  // panel button that armed it (a mis-armed slot would otherwise require
  // navigating back to the Candidate panel to cancel).
  useEffect(() => {
    if (!pendingChoiceAspectId) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setPendingChoiceAspectId(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [pendingChoiceAspectId]);
  const [pendingCell, setPendingCell] = useState<[number, number] | null>(null);
  // The currently-traced connector (transient; the discovered set persists).
  const [activeLine, setActiveLine] = useState<GenerationTrail | null>(null);
  // A corpus project opened for inspection (design-space glyph / provenance chip).
  const [focusProjectId, setFocusProjectId] = useState<string | null>(null);
  const { data: focusProject } = useCorpusProjectQuery(focusProjectId);
  // Fetched on first visit to the space OR schema view (cached forever
  // afterwards) — the schema's coverage probe needs the full project
  // universe to find what the taxonomy never annotates.
  const { data: surface } = useSurfaceQuery(view === 'space' || view === 'schema');
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
    result: Parameters<typeof setTaxonomy>[0] & { corpus_similarity?: number | null },
    overview: string
  ) => {
    // setTaxonomy rebuilds the tree and wipes coords/discovered/provenance.
    setTaxonomy(result);
    // The overview IS the project brief (Part 13 L-B) — persisted so "Edit
    // Brief & Taxonomy" can reopen the dialog prefilled.
    setProjectBrief(overview.trim());
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

  // ── Schema view (Part 12 A1–A3): columns, corpus annotation, facets ────────
  const schemaColumns = useMemo(
    () =>
      buildSchemaColumns(
        nodes,
        activeDescriptionByTopic,
        descriptionById,
        optionState,
        (activeCandidateId ? candidates[activeCandidateId]?.choices : undefined) ?? {},
        provenance
      ),
    [nodes, activeDescriptionByTopic, descriptionById, optionState, activeCandidateId, candidates, provenance]
  );
  const annotationInputs = useMemo(
    () =>
      schemaColumns.flatMap((col) =>
        col.options.map((o) => ({ id: o.id, name: o.name, desc: o.desc }))
      ),
    [schemaColumns]
  );
  const {
    data: annotation,
    isFetching: isAnnotating,
    error: annotationError,
  } = useAnnotationQuery(annotationInputs, view === 'schema' || view === 'crosstab');
  // The rationale layer (Part 13 L-A): the system's one-line why per aspect,
  // grounded in the annotation counts — hence gated on the annotation.
  const rationaleAspects = useMemo(
    () =>
      schemaColumns.map((col) => ({
        id: col.id,
        name: col.name,
        desc: col.desc,
        options: col.options.map((o) => ({
          name: o.name,
          count: annotation?.options[o.id]?.count ?? 0,
        })),
      })),
    [schemaColumns, annotation]
  );
  const { data: rationaleData } = useRationaleQuery(
    rationaleAspects,
    annotation?.meta.n_projects ?? 0,
    view === 'schema' && Boolean(annotation)
  );
  // The selected aspect's why, for the Context panel (lineage root → aspect).
  const selectedAspectRationale = useMemo(() => {
    if (selection.lineage.length !== 2 || !rationaleData) return null;
    const id = selection.nodeId ?? findNodeByLineage(nodes, [...selection.lineage])?.id;
    return id ? rationaleData.rationales[id] || null : null;
  }, [selection, rationaleData, nodes]);
  // The coverage probe's detection half — pure set arithmetic: which real
  // projects does the current taxonomy barely describe?
  const poorlyCovered = useMemo(
    () =>
      annotation && surface
        ? poorlyCoveredProjects(
            annotation.options,
            surface.points.map((p) => ({ id: p.id, name: p.name || '(untitled)' }))
          )
        : [],
    [annotation, surface]
  );
  const facetMatched = useMemo(() => {
    if (!annotation) return null;
    return computeFacetMatches(
      annotation.options,
      [...facetInclude],
      [...facetExclude],
      (surface?.points ?? []).map((p) => p.id)
    );
  }, [annotation, facetInclude, facetExclude, surface]);
  const handleToggleFacet = (optionId: string, kind: 'include' | 'exclude') => {
    const [get, set, other, setOther] =
      kind === 'include'
        ? ([facetInclude, setFacetInclude, facetExclude, setFacetExclude] as const)
        : ([facetExclude, setFacetExclude, facetInclude, setFacetInclude] as const);
    const next = new Set(get);
    if (next.has(optionId)) next.delete(optionId);
    else {
      next.add(optionId);
      if (other.has(optionId)) {
        const o = new Set(other);
        o.delete(optionId);
        setOther(o);
      }
    }
    set(next);
    trackUsage('facet_toggle');
  };
  const handleSchemaSelect = (optionId: string) => {
    for (const col of schemaColumns) {
      const opt = col.options.find((o) => o.id === optionId);
      if (opt) {
        handleSelect({
          topic: opt.name,
          lineage: [nodes[0]?.topic ?? '', col.name, opt.name],
          nodeId: optionId,
        });
        return;
      }
    }
  };
  // Informing the space (Halskov): an added option joins the tree with its
  // provenance (manual typing or an accepted C1 proposal), gets a
  // description, locates automatically, and is logged as an event. Reads the
  // tree FRESH from the store (not the render closure) so back-to-back
  // inserts — two proposals accepted quickly — never clobber each other, and
  // refuses gracefully when the target aspect was deleted meanwhile.
  const insertOptionNode = (
    aspectId: string,
    name: string,
    desc: string,
    source: 'manual' | 'steer' | 'cell'
  ): boolean => {
    const current = useMindmapStore.getState().nodes;
    const aspect = current[0]?.children?.find((a) => a.id === aspectId);
    if (!aspect) return false;
    const base = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
    const existing = new Set<string>();
    collectIds(current, existing);
    let id = base || `option-${existing.size}`;
    for (let n = 2; existing.has(id); n++) id = `${base}-${n}`;
    const insert = (list: ReadonlyArray<MindmapNode>): MindmapNode[] =>
      list.map((node) =>
        node.id === aspectId
          ? { ...node, children: [...(node.children ?? []), { id, topic: name }] }
          : { ...node, children: node.children ? insert(node.children) : undefined }
      );
    setNodes(insert(current));
    if (desc) mergeDescriptions({ [id]: desc });
    recordProvenance({ [id]: { source, seedProjects: [], createdAt: Date.now() } });
    recordEvent('option_added', `Added option "${name}" under ${aspect.topic} (${source})`, [id]);
    return true;
  };
  const handleAddOption = (aspectId: string, name: string, desc: string) => {
    insertOptionNode(aspectId, name, desc, 'manual');
    trackUsage('schema_add_option');
  };
  // Informing at the STRUCTURE level (Part 13 L-A): an accepted
  // missing-dimension proposal becomes a new root-child aspect — an empty
  // schema column the designer fills by hand or by generation. Same
  // fresh-read discipline as insertOptionNode.
  const insertAspectNode = (name: string, desc: string): boolean => {
    const current = useMindmapStore.getState().nodes;
    const root = current[0];
    if (!root) return false;
    if ((root.children ?? []).some((a) => a.topic.toLowerCase() === name.toLowerCase()))
      return false;
    const base = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
    const existing = new Set<string>();
    collectIds(current, existing);
    let id = base || `aspect-${existing.size}`;
    for (let n = 2; existing.has(id); n++) id = `${base}-${n}`;
    setNodes([
      { ...root, children: [...(root.children ?? []), { id, topic: name, children: [] }] },
      ...current.slice(1),
    ]);
    if (desc) mergeDescriptions({ [id]: desc });
    recordProvenance({ [id]: { source: 'coverage', seedProjects: [], createdAt: Date.now() } });
    recordEvent('option_added', `Added dimension "${name}" (coverage probe)`, [id]);
    return true;
  };

  // ── C1: informing-back proposals (transient — only ACCEPTED ones persist;
  // dismissals are EVENTS, so the timeline can resurface them) ───────────────
  const [proposals, setProposals] = useState<OptionProposal[]>([]);
  const enqueueProposals = useCallback(
    (
      items: Array<{ text: string; desc?: string; evidence?: string }>,
      aspectId: string | null,
      evidence: string,
      source: OptionProposal['source'],
      kind: 'option' | 'aspect' = 'option'
    ) => {
      setProposals((prev) => {
        const next = [...prev];
        for (const item of items) {
          const text = item.text.trim();
          if (!text) continue;
          if (
            next.some(
              (p) => p.text.toLowerCase() === text.toLowerCase() && p.aspectId === aspectId
            )
          )
            continue;
          next.push({
            id: `prop-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
            kind,
            aspectId,
            text,
            desc: item.desc ?? '',
            source,
            evidence: item.evidence ?? evidence,
          });
        }
        return next.slice(-4); // keep only the NEWEST few — chips, not a backlog
      });
    },
    []
  );
  const handleProposeQualities = useCallback(
    (qualities: string[], aspectId: string | null, evidence: string) =>
      enqueueProposals(qualities.map((text) => ({ text })), aspectId, evidence, 'steer'),
    [enqueueProposals]
  );
  const handleAcceptProposal = (proposal: OptionProposal, aspectId: string) => {
    // If the aspect vanished while the chip waited, the proposal is moot —
    // the chip clears either way, but nothing phantom is recorded.
    const inserted =
      proposal.kind === 'aspect'
        ? insertAspectNode(proposal.text, proposal.desc)
        : insertOptionNode(
            aspectId,
            proposal.text,
            proposal.desc,
            proposal.source === 'coverage' ? 'manual' : proposal.source
          );
    setProposals((prev) => prev.filter((p) => p.id !== proposal.id));
    trackUsage(inserted ? 'proposal_accepted' : 'proposal_aspect_missing');
  };
  const handleDismissProposal = (proposalId: string) => {
    const proposal = proposals.find((p) => p.id === proposalId);
    setProposals((prev) => prev.filter((p) => p.id !== proposalId));
    if (proposal) {
      // A dismissed idea is history, not garbage — the event carries the
      // full proposal so the timeline's "Reconsider" can re-offer it.
      recordEvent(
        'proposal_dismissed',
        `Dismissed suggestion "${proposal.text}" (from ${proposal.evidence})`,
        proposal.aspectId ? [proposal.aspectId] : [],
        JSON.stringify(proposal)
      );
    }
    trackUsage('proposal_dismissed');
  };
  const handleReconsiderProposal = (proposal: OptionProposal) => {
    enqueueProposals(
      [{ text: proposal.text, desc: proposal.desc }],
      proposal.aspectId,
      proposal.evidence,
      proposal.source
    );
    trackUsage('proposal_reconsidered');
  };
  const aspectList = useMemo(
    () => (nodes[0]?.children ?? []).map((a) => ({ id: a.id, name: a.topic })),
    [nodes]
  );

  // The coverage probe's naming half (Part 13 L-A): ask what dimension the
  // poorly-covered projects exemplify; answers ride the proposals channel.
  const { mutateAsync: probeMissingAspect, isPending: probing } = useMissingAspectMutation();
  const [probeError, setProbeError] = useState<string | null>(null);
  const handleProbeMissingAspect = async () => {
    if (poorlyCovered.length === 0 || aspectList.length === 0) return;
    setProbeError(null);
    trackUsage('coverage_probe');
    try {
      const result = await probeMissingAspect({
        aspect_names: aspectList.map((a) => a.name),
        project_ids: poorlyCovered.map((p) => p.id),
      });
      const names = poorlyCovered.map((p) => p.name);
      const evidence = `the coverage probe (${names.slice(0, 3).join(', ')}${
        names.length > 3 ? '…' : ''
      } fit the taxonomy poorly)`;
      if (result.proposals.length === 0) {
        setProbeError('the probe found no missing dimension to name');
        return;
      }
      enqueueProposals(
        result.proposals.map((p) => ({
          text: p.name,
          desc: p.desc,
          ...(p.reason ? { evidence: `${evidence} — ${p.reason}` } : {}),
        })),
        null,
        evidence,
        'coverage',
        'aspect'
      );
    } catch (error) {
      setProbeError(error instanceof Error ? error.message : 'probe failed');
    }
  };

  // B2: a kept cell idea becomes a candidate skeleton — committed to the two
  // options that named the gap, with the generated concept as its brief (the
  // morphological-combination → candidate flow). The new candidate is active,
  // so setChoice lands on it; the brief drives its star via the locate effect.
  const handleKeepCellIdea = ({ aspectAId, optionAId, aspectBId, optionBId, idea }: KeepCellIdea) => {
    const id = createCandidate(idea.name);
    setChoice(aspectAId, optionAId);
    setChoice(aspectBId, optionBId);
    setCandidateBrief(id, idea.desc);
    recordEvent('cell_kept', `Kept gap concept "${idea.name}" as a candidate`, [id]);
    // C1: the kept concept informs BOTH parent aspects — offer it as an
    // option under each (the designer decides whether it earns vocabulary).
    const item = [{ text: idea.name, desc: idea.desc }];
    enqueueProposals(item, aspectAId, `the kept gap concept "${idea.name}"`, 'cell');
    enqueueProposals(item, aspectBId, `the kept gap concept "${idea.name}"`, 'cell');
    // Visible landing: open the Candidate panel so the new skeleton is seen
    // arriving (a silently-kept idea reads as a no-op).
    setCandidatePanelOpen(true);
    trackUsage('cell_kept');
  };

  // ── C2: reflection capture (burden-inverted, never modal) ──────────────────
  const [reflectionPrompt, setReflectionPrompt] = useState<ReflectionPromptState | null>(null);
  const { mutateAsync: draftReflection } = useDraftReflectionMutation();
  const lastEventId = events.length > 0 ? events[events.length - 1]!.id : null;
  const seenEventRef = useRef<string | null | undefined>(undefined);
  useEffect(() => {
    // Baseline on mount so restored/persisted logs never pop a chip.
    if (seenEventRef.current === undefined) {
      seenEventRef.current = lastEventId;
      return;
    }
    if (!lastEventId || seenEventRef.current === lastEventId) return;
    seenEventRef.current = lastEventId;
    const stored = useMindmapStore.getState();
    const event = stored.events[stored.events.length - 1];
    if (!event || event.id !== lastEventId) return;
    const REFLECTABLE = ['choose', 'reject', 'steer_applied', 'candidate_created', 'cell_kept', 'generated'];
    if (!REFLECTABLE.includes(event.kind)) return;
    // Session restores replay old events wholesale — only fresh acts prompt.
    if (Date.now() - event.ts > 10_000) return;
    setReflectionPrompt({
      eventId: event.id,
      label: event.label,
      drafted: false,
      draftValue: '',
    });
    // Drafting waits a beat: when acts come in bursts (choosing several
    // options), each chip replaces the last — only the SURVIVOR is worth a
    // slow local-LLM call. The cleanup cancels superseded drafts unfired.
    const draftTimer = setTimeout(() => {
      // Labels can carry long user-typed rejection reasons — keep the request
      // inside the backend's 600-char bound rather than 422ing the draft.
      draftReflection({ context: event.label.slice(0, 500) })
        .then(({ draft }) =>
          setReflectionPrompt((prev) =>
            prev && prev.eventId === event.id
              ? { ...prev, drafted: true, draftValue: draft }
              : prev
          )
        )
        .catch(() =>
          setReflectionPrompt((prev) =>
            prev && prev.eventId === event.id ? { ...prev, drafted: true } : prev
          )
        );
    }, 1200);
    return () => clearTimeout(draftTimer);
  }, [lastEventId, draftReflection]);
  const acceptReflection = (value: string) => {
    if (!reflectionPrompt || !value.trim()) return;
    addReflection(
      reflectionPrompt.eventId,
      value,
      value.trim() !== reflectionPrompt.draftValue.trim()
    );
    trackUsage('reflection_accepted');
    setReflectionPrompt(null);
  };
  const skipReflection = () => {
    trackUsage('reflection_skipped');
    setReflectionPrompt(null);
  };

  // ── C3: schema replay (a derived PAST state — the store never mutates) ─────
  const [replayIndex, setReplayIndex] = useState<number | null>(null);
  useEffect(() => {
    if (view !== 'schema') setReplayIndex(null);
  }, [view]);
  // The replay scrubs WITHIN the current space: events before the last
  // taxonomy_set reference a tree that no longer exists, so positions before
  // that boundary would render misleading empties.
  const replayFloor = useMemo(() => {
    for (let i = events.length - 1; i >= 0; i--) {
      if (events[i]!.kind === 'taxonomy_set') return i + 1;
    }
    return 0;
  }, [events]);
  const replaying = view === 'schema' && replayIndex !== null;
  const replayColumns = useMemo(() => {
    if (!replaying || replayIndex === null) return null;
    const overlay = buildReplayOverlay(events, replayIndex);
    // Italic ("informed") is TIMED for nodes whose informing event is in the
    // log; real provenance fills in only for PRE-LOG nodes (no event anywhere
    // — they were informed before recording started, so italic throughout).
    const logged = new Set([...Object.keys(overlay.informed), ...Object.keys(overlay.notYet)]);
    const provenanceLike = {
      ...Object.fromEntries(Object.entries(provenance).filter(([id]) => !logged.has(id))),
      ...Object.fromEntries(
        Object.keys(overlay.informed).map((id) => [
          id,
          { source: 'manual' as const, seedProjects: [], createdAt: 0 },
        ])
      ),
    };
    return buildSchemaColumns(
      nodes,
      activeDescriptionByTopic,
      descriptionById,
      overlay.optionState,
      overlay.activeChoices,
      provenanceLike,
      new Set(Object.keys(overlay.notYet))
    );
  }, [replaying, replayIndex, events, nodes, activeDescriptionByTopic, descriptionById, provenance]);
  // The selected step's subject cells — outlined so every scrub visibly
  // answers, even when the step itself changes no schema state (steers,
  // dismissals, candidate events highlight nothing — the card carries those).
  const replayHighlight = useMemo(() => {
    if (!replaying || replayIndex === null || replayIndex <= replayFloor) return undefined;
    return new Set(events[replayIndex - 1]?.refs ?? []);
  }, [replaying, replayIndex, replayFloor, events]);

  // The real matches, with the backend's "nothing found" placeholder row
  // removed at the source — so the count badge, the panel, the map highlight,
  // and the generate-nodes seed set all agree about how many projects exist
  // (M-E6; the placeholder is never a real, clickable project).
  const realProjects = useMemo(
    () => (data?.projects ?? []).filter((p) => p.Name !== PLACEHOLDER_PROJECT_NAME),
    [data]
  );

  // Corpus ids of the selection's related projects — the panel's examples are
  // also highlighted as places on the design-space map.
  const relatedProjectIds = useMemo(() => {
    const ids = new Set<string>();
    for (const project of realProjects) {
      if (project.id) ids.add(project.id);
    }
    return ids;
  }, [realProjects]);

  // ── Candidates: locate each design in the frozen space ──────────────────────
  // A candidate's position is the embedding of its BRIEF when present (the
  // identity layer — Part 10), else its composed option text; re-located
  // whenever that text changes (signature-tracked). When the star moves, the
  // old position joins the candidate's trail (its trajectory across revisions).
  const candidateTextSignatures = useRef<Map<string, string>>(new Map());
  useEffect(() => {
    if (!surface) return;
    // Debounced: the brief textarea changes this text PER KEYSTROKE — one
    // /locate (an embedding round-trip) after typing pauses, not one per
    // character hammering the local embed server.
    const timer = setTimeout(() => {
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
    }, 900);
    return () => clearTimeout(timer);
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
        recordEvent(
          'generated',
          `Generated ${children.length} option${children.length === 1 ? '' : 's'} at a map gap (${children
            .map((c) => c.topic)
            .slice(0, 3)
            .join(', ')}${children.length > 3 ? '…' : ''})`,
          children.map((c) => c.id)
        );

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
      recordEvent,
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

  // Study mode: a `?p=<id>` URL param tags this session's exports (M-E12). Set
  // only when unset, so a reload doesn't clobber a facilitator-entered id.
  useEffect(() => {
    const p = new URLSearchParams(window.location.search).get('p')?.trim();
    if (p && !useMindmapStore.getState().participantId) setParticipantId(p);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
      const { snapshot, warnings } = parseSessionFile(await file.text());
      if (!window.confirm(`Replace the current exploration with "${file.name}"?`)) return;
      restoreSession(snapshot);
      setSelection({ topic: INITIAL_SELECTION.topic, lineage: [...INITIAL_SELECTION.lineage] });
      setActiveLine(null);
      setGapPreview(null);
      setPendingChoiceAspectId(null);
      setLensOn(false);
      attemptedRef.current.clear();
      trackUsage('session_load');
      if (warnings.length) {
        // Non-fatal: the session loaded, but some slices were malformed and
        // reset. Say so rather than silently dropping data (M-E3).
        toast.warning('Session loaded with repairs', {
          description: warnings.join(' '),
        });
      }
    } catch (error) {
      setGenerateError(error instanceof Error ? error.message : 'Could not load session.');
    }
  };

  // One-click study bundle (M-E12): the machine-restorable session (event log +
  // usage + reflections live inside it), the markdown record, and the computed
  // stats — one file, tagged by participant. Prompts for the id if unset.
  const handleExportStudyBundle = () => {
    const entered =
      participantId ||
      (window.prompt('Participant ID for this study bundle:', '') ?? '').trim();
    if (!entered) return;
    if (entered !== participantId) setParticipantId(entered);
    const markdown = buildExplorationMarkdown({
      nodes,
      descriptionByTopic: activeDescriptionByTopic,
      descriptionById,
      optionState,
      candidates,
      provenance,
      coords,
      discovered,
      activeCandidateId,
      events,
      reflections,
    });
    const session = {
      ...selectSessionSnapshot(useMindmapStore.getState()),
      participantId: entered,
    };
    downloadTextFile(
      studyBundleFilename(entered, new Date().toISOString().slice(0, 10)),
      buildStudyBundle({ participantId: entered, session, markdown, stats: explorationStats }),
      'application/json;charset=utf-8'
    );
    trackUsage('study_bundle_export');
    toast.success('Study bundle exported', { description: `Participant ${entered}` });
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
      const hasRealProjects = realProjects.length > 0;
      const relatedProjects = hasRealProjects ? realProjects : null;

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
      recordEvent(
        'generated',
        `Generated ${generatedChildren.length} option${generatedChildren.length === 1 ? '' : 's'} under "${focusNode.topic}"`,
        generatedChildren.map((c) => c.id)
      );

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
          <button
            type="button"
            onClick={handleExportStudyBundle}
            title={
              participantId
                ? `Export study bundle — participant ${participantId}`
                : 'Export study bundle (session + record + stats, tagged by participant)'
            }
            className="relative rounded-full p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <FlaskConical className="h-4 w-4" />
            {participantId ? (
              <span className="absolute right-1 top-1 h-1.5 w-1.5 rounded-full bg-violet-500" />
            ) : null}
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
              aria-pressed={view === 'map' || view === 'schema' || view === 'crosstab'}
              className={`flex items-center gap-1.5 whitespace-nowrap rounded-full px-3 py-1.5 text-xs font-semibold transition-all active:scale-95 ${
                view === 'map' || view === 'schema' || view === 'crosstab'
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Network className="h-3.5 w-3.5" />
              Structure
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
                setExamineInitialTab('strips');
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
            title={
              taxonomy
                ? 'Edit your project brief and regenerate the design-space taxonomy (replaces the current space; exploration history survives in the timeline)'
                : 'Describe your project and generate a design-space taxonomy from it'
            }
            className="h-9 gap-2 whitespace-nowrap rounded-full px-5 text-xs font-bold shadow-sm transition-all hover:bg-accent hover:text-accent-foreground active:scale-95"
          >
            <Sparkles className="h-4 w-4 text-primary" />
            {/* After the first brief+generation round the action is a REVISION
                of the standing brief, not a fresh start (Part 13 L-B). */}
            {taxonomy ? 'Edit Brief & Taxonomy' : 'Generate Taxonomy'}
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
      {/* Narrower below xl so the two floating columns cannot collide at
          ~900-1100px window widths. */}
      <div className="pointer-events-none absolute top-4 left-4 z-40 w-full max-w-xs xl:max-w-sm">
        {dockedView && !contextPanelOpen ? (
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
              {/* Native scroll, not Radix ScrollArea: a max-h on its Root
                  doesn't bound the Viewport, so long content (generated node
                  descs) was clipped unscrollably at the panel bottom. */}
              <div className="max-h-[min(45dvh,420px)] overflow-y-auto">
                <div className="space-y-4 px-4 pb-4">
                  {/* Breadcrumb: ancestors navigate back to their level. */}
                  <div className="flex flex-wrap items-center gap-1.5">
                    {selection.lineage.map((topic, idx) => {
                      const isCurrent = idx === selection.lineage.length - 1;
                      const ancestorLineage = selection.lineage.slice(0, idx + 1);
                      return (
                        <span key={`${topic}-${idx}`} className="flex items-center gap-1.5">
                          {isCurrent ? (
                            <Badge variant="secondary" className="px-2 py-0 font-medium">
                              {topic}
                            </Badge>
                          ) : (
                            <button
                              type="button"
                              title={`Back to ${topic}`}
                              onClick={() => {
                                const ancestorId = findNodeByLineage(nodes, ancestorLineage)?.id;
                                handleSelect({
                                  topic,
                                  lineage: ancestorLineage,
                                  ...(ancestorId ? { nodeId: ancestorId } : {}),
                                });
                              }}
                            >
                              <Badge
                                variant="secondary"
                                className="cursor-pointer px-2 py-0 font-medium transition-colors hover:bg-primary/15 hover:text-primary"
                              >
                                {topic}
                              </Badge>
                            </button>
                          )}
                          {idx < selection.lineage.length - 1 && (
                            <ChevronRight className="h-3 w-3 text-muted-foreground/40" />
                          )}
                        </span>
                      );
                    })}
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

                  {/* The rationale layer (Part 13 L-A): the system's why for
                      this dimension — labelled as AI explanation from corpus
                      evidence, never a verdict. */}
                  {selectedAspectRationale ? (
                    <p className="rounded-lg bg-violet-500/5 p-2.5 text-xs italic leading-snug text-muted-foreground">
                      <span className="font-semibold not-italic text-violet-700">
                        Why this dimension:
                      </span>{' '}
                      {selectedAspectRationale}
                      <span className="not-italic"> (AI, from corpus evidence)</span>
                    </p>
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
              </div>
            </CollapsibleContent>
          </section>
        </Collapsible>

        </div>
        )}

        {/* Candidate composition panel */}
        {dockedView && !candidatePanelOpen ? (
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
              setExamineInitialTab('strips');
              setView('axes');
            }}
            onProposeQualities={handleProposeQualities}
          />
        </div>
        )}
      </div>

      {/* Structure mode's views of the same constraint structure: the
          editable tree, the design-space schema table (Part 12 A1), and the
          cross-tab lens (Part 12 B2) — all VIEWS of the structure, folded
          into one mode per review. */}
      {(view === 'map' || view === 'schema' || view === 'crosstab') && (
        <div className="pointer-events-none absolute left-1/2 top-4 z-40 -translate-x-1/2">
          <div className="pointer-events-auto flex items-center gap-0.5 rounded-full border bg-background/90 p-0.5 shadow-md backdrop-blur-md">
            <button
              type="button"
              onClick={() => setView('map')}
              aria-pressed={view === 'map'}
              className={`flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-semibold transition-colors ${
                view === 'map' ? 'bg-muted text-foreground' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Network className="h-3 w-3" /> Tree
            </button>
            <button
              type="button"
              onClick={() => {
                setView('schema');
                trackUsage('view_schema');
              }}
              aria-pressed={view === 'schema'}
              className={`flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-semibold transition-colors ${
                view === 'schema' ? 'bg-muted text-foreground' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Table2 className="h-3 w-3" /> Schema
            </button>
            <button
              type="button"
              onClick={() => {
                setView('crosstab');
                trackUsage('view_crosstab');
              }}
              aria-pressed={view === 'crosstab'}
              className={`flex items-center gap-1 rounded-full px-2.5 py-1 text-[11px] font-semibold transition-colors ${
                view === 'crosstab' ? 'bg-muted text-foreground' : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <LayoutGrid className="h-3 w-3" /> Cross-tab
            </button>
          </div>
        </div>
      )}

      {/* Armed pick mode: a global banner (Nielsen H1 / Norman mode error) —
          the pulsing panel button alone is invisible once the eye is on a
          canvas. Sits below whatever each view pins to the top center. */}
      {pendingChoiceAspectId && (
        <div
          className={`pointer-events-none absolute left-1/2 z-40 -translate-x-1/2 ${
            view === 'schema' || view === 'crosstab' ? 'top-28' : 'top-14'
          }`}
        >
          <div className="pointer-events-auto flex items-center gap-2 rounded-full border border-violet-300 bg-violet-500/10 px-3 py-1 text-[11px] font-medium text-violet-700 shadow-sm backdrop-blur">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-violet-400 opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-violet-500" />
            </span>
            Picking an option for{' '}
            <strong className="max-w-[12rem] truncate">
              {nodes[0]?.children?.find((a) => a.id === pendingChoiceAspectId)?.topic ??
                'this aspect'}
            </strong>{' '}
            — click one in any view
            <button
              type="button"
              onClick={() => setPendingChoiceAspectId(null)}
              className="rounded-full border border-violet-300 px-2 py-0.5 font-semibold transition-colors hover:bg-violet-500/15"
              title="Cancel the pick (Esc)"
            >
              Cancel · Esc
            </button>
          </div>
        </div>
      )}

      {/* Relevance lens: an overlay toggle on the design space (not a mode) */}
      {view === 'space' && (
        <div className="absolute top-4 left-1/2 z-40 flex -translate-x-1/2 flex-col items-center gap-1">
          <div className="flex items-center gap-1 rounded-full border bg-background/90 p-0.5 shadow-md backdrop-blur">
            {/* The lens is a SWITCH, not a gated button: it can be armed with
                nothing selected (grayed) and paints the moment an anchor
                exists — no chicken-and-egg with selection. */}
            <button
              type="button"
              role="switch"
              aria-checked={lensOn}
              onClick={() => {
                if (!lensOn) trackUsage('lens_on');
                setLensOn(!lensOn);
              }}
              title="Color real projects by relevance to the selected node or candidate"
              className="flex items-center gap-1.5 whitespace-nowrap rounded-full px-3 py-1 text-[11px] font-semibold"
            >
              <Focus
                className={`h-3 w-3 ${lensActive ? 'text-foreground' : 'text-muted-foreground'}`}
              />
              <span className={lensActive ? 'text-foreground' : 'text-muted-foreground'}>
                Relevance lens
              </span>
              <span
                className={`relative inline-flex h-3.5 w-6 shrink-0 items-center rounded-full transition-colors ${
                  lensOn ? (lensAnchor ? 'bg-violet-500' : 'bg-violet-300') : 'bg-muted-foreground/25'
                }`}
              >
                <span
                  className={`inline-block h-2.5 w-2.5 rounded-full bg-white shadow transition-transform ${
                    lensOn ? 'translate-x-3' : 'translate-x-0.5'
                  }`}
                />
              </span>
            </button>
            {lensOn && !lensAnchor && (
              <span className="whitespace-nowrap pr-2 text-[10px] text-muted-foreground">
                waiting — select a node or candidate
              </span>
            )}
            {/* Only one possible anchor: still NAME it, so "relevance to
                what?" is always answered at the control itself. */}
            {lensOn && lensAnchor && !(selectionAnchor && candidateAnchor) && (
              <span
                className="max-w-[9rem] truncate whitespace-nowrap pr-2 text-[10px] text-violet-700"
                title={`The lens is anchored to ${lensAnchor === candidateAnchor ? 'the active candidate' : 'the selected node'}: ${lensAnchor.label}`}
              >
                → {lensAnchor === candidateAnchor ? '★' : '◉'} {lensAnchor.label}
              </span>
            )}
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
        {view === 'schema' ? (
          <SchemaTable
            columns={replayColumns ?? schemaColumns}
            annotation={annotation ?? null}
            annotating={isAnnotating}
            annotationError={
              annotationError instanceof Error ? annotationError.message : null
            }
            facets={{ include: facetInclude, exclude: facetExclude }}
            onToggleFacet={handleToggleFacet}
            onSelectOption={handleSchemaSelect}
            selectedOptionId={selectedNode?.id ?? null}
            onChoose={
              activeCandidateId && !replaying
                ? (aspectId, optionId) => setChoice(aspectId, optionId)
                : undefined
            }
            onReject={(optionId) => rejectOption(optionId)}
            onReopen={(optionId) => reopenOption(optionId)}
            onAddOption={handleAddOption}
            onOpenProject={setFocusProjectId}
            readOnly={replaying}
            highlightIds={replayHighlight}
            replay={
              replaying && replayIndex !== null
                ? {
                    step: replayIndex - replayFloor,
                    total: events.length - replayFloor,
                    onLive: () => setReplayIndex(null),
                  }
                : null
            }
            rationales={rationaleData?.rationales}
            probe={
              annotation
                ? {
                    count: poorlyCovered.length,
                    running: probing,
                    error: probeError,
                    onRun: handleProbeMissingAspect,
                  }
                : null
            }
          />
        ) : view === 'crosstab' ? (
          <CrossTabView
            columns={schemaColumns}
            annotation={annotation ?? null}
            annotating={isAnnotating}
            annotationError={
              annotationError instanceof Error ? annotationError.message : null
            }
            onOpenProject={setFocusProjectId}
            onKeepIdea={handleKeepCellIdea}
            onShowScatter={() => {
              setExamineInitialTab('scatter');
              setView('axes');
              trackUsage('view_axes');
            }}
          />
        ) : view === 'axes' ? (
          <ExamineView
            nodes={nodes}
            selection={selection}
            onSelectNode={handleSelect}
            onSelectProject={setFocusProjectId}
            descriptionByTopic={activeDescriptionByTopic}
            initialTab={examineInitialTab}
            onProposeQualities={handleProposeQualities}
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
              facetMatched={facetMatched}
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
            activeNodeId={selection.nodeId}
            onSelect={handleSelect}
            onDataChange={handleNodesChange}
            generatingNodeId={
              isGeneratingNodes ? findNodeByLineage(nodes, selection.lineage)?.id ?? null : null
            }
            nodeStates={nodeStates}
          />
        )}
      </div>

      {/* First-run choice (Part 13 L-B): two honest entry points into the
          inform→filter loop. Brief-first opens the taxonomy dialog (whose
          overview field IS the brief); discover-first just closes — the
          prebuilt space is a valid starting point, and Generate Taxonomy
          stays one click away in the navigator. */}
      <Dialog open={firstRunChoiceOpen} onOpenChange={setFirstRunChoiceOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>How do you want to start?</DialogTitle>
            <DialogDescription>
              Both paths lead to the same exploration — pick what fits how you think.
            </DialogDescription>
          </DialogHeader>
          <div className="flex flex-col gap-2">
            <button
              type="button"
              onClick={() => {
                trackUsage('first_run_brief');
                setFirstRunChoiceOpen(false);
                setTaxonomyDialogOpen(true);
              }}
              className="rounded-xl border p-3 text-left transition-colors hover:border-violet-300 hover:bg-violet-500/5"
            >
              <p className="flex items-center gap-1.5 text-sm font-semibold">
                <Sparkles className="h-3.5 w-3.5 text-violet-600" />
                Start from your brief
              </p>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Write down what you&apos;re imagining — the system builds a design space
                scoped to it, with dimensions and options to explore.
              </p>
            </button>
            <button
              type="button"
              onClick={() => {
                trackUsage('first_run_discover');
                setFirstRunChoiceOpen(false);
              }}
              className="rounded-xl border p-3 text-left transition-colors hover:border-violet-300 hover:bg-violet-500/5"
            >
              <p className="flex items-center gap-1.5 text-sm font-semibold">
                <Compass className="h-3.5 w-3.5 text-violet-600" />
                Discover first
              </p>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Explore the prebuilt media-architecture space and its real projects;
                generate your own taxonomy any time from the bottom bar.
              </p>
            </button>
          </div>
        </DialogContent>
      </Dialog>

      <GenerateTaxonomyDialog
        open={taxonomyDialogOpen}
        onOpenChange={setTaxonomyDialogOpen}
        onSuccess={handleTaxonomyGenerated}
        initialOverview={projectBrief}
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

      {/* Floating right column: Related Projects (collapses to an icon in
          Examine — see the lineage panel note) + the B1 inspector dock. */}
      <div className="pointer-events-none absolute top-4 right-4 z-40 flex max-h-[calc(100%-2rem)] w-full max-w-sm flex-col items-end gap-3 xl:max-w-md">
        {dockedView && !projectsPanelOpen ? (
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
                {realProjects.length ? (
                  <Badge variant="outline" className="h-5 px-1.5 text-[10px]">
                    {realProjects.length}
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
                  projects={realProjects}
                  isLoading={isFetching}
                  focusProject={focusProject ?? null}
                  compact={view === 'space' && Boolean(activeCandidate) && inspectorOpen}
                />
              </div>
            </CollapsibleContent>
          </section>
        </Collapsible>
        </div>
        )}

        {/* B1 inspector dock: the Examine strips beside the map while a
            candidate is active — examine ⇄ map without a mode switch (also
            anchors the C1/C2 chips' column, rendered after this block). */}
        {view === 'space' && activeCandidate && (
          <div className="pointer-events-auto w-full">
            <Collapsible open={inspectorOpen} onOpenChange={setInspectorOpen}>
              <section className="overflow-hidden rounded-2xl border bg-background/90 shadow-xl backdrop-blur-md">
                <div className="flex items-center justify-between px-4 py-3">
                  <div className="flex min-w-0 items-center gap-2">
                    <Microscope className="h-4 w-4 text-primary" />
                    <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                      Inspector
                    </h2>
                    <Badge
                      variant="outline"
                      className="h-5 max-w-40 truncate px-1.5 text-[10px]"
                      title={activeCandidate.name}
                    >
                      {activeCandidate.name}
                    </Badge>
                  </div>
                  <CollapsibleTrigger asChild>
                    <Button variant="ghost" size="icon" className="group h-6 w-6 rounded-full">
                      <ChevronRight className="h-4 w-4 transition-transform duration-200 group-data-[state=open]:rotate-90" />
                    </Button>
                  </CollapsibleTrigger>
                </div>
                <CollapsibleContent>
                  <div className="max-h-[38vh] space-y-3 overflow-y-auto p-4 pt-0">
                    <CandidateStrips
                      nodes={nodes}
                      descriptionByTopic={activeDescriptionByTopic}
                      onProposeQualities={handleProposeQualities}
                    />
                  </div>
                </CollapsibleContent>
              </section>
            </Collapsible>
          </div>
        )}
      </div>

      {/* C1 proposals + C2 reflection — transient chips, bottom-right, never
          modal: the exploration keeps moving whether or not they're answered.
          Lifted above the replay bar's slot whenever the schema view shows it. */}
      <div
        className={`pointer-events-none absolute right-4 z-40 flex w-80 flex-col items-stretch gap-2 ${
          // Clear whatever the bottom-center slot holds: the OPEN timeline is
          // ~3× taller than its closed pill, and on ~1200px windows the two
          // would otherwise overlap horizontally.
          view === 'schema' && replayIndex !== null
            ? 'bottom-72'
            : view === 'schema' && events.length > replayFloor
              ? 'bottom-32'
              : 'bottom-24'
        }`}
      >
        <ProposalChips
          proposals={proposals}
          aspects={aspectList}
          onAccept={handleAcceptProposal}
          onDismiss={handleDismissProposal}
        />
        {reflectionPrompt && (
          <ReflectionChip
            key={reflectionPrompt.eventId}
            prompt={reflectionPrompt}
            onAccept={acceptReflection}
            onSkip={skipReflection}
          />
        )}
      </div>

      {/* C3 — the exploration timeline (Fusion-style markers; scrubbing shows
          the schema as it stood after the clicked step, read-only). Floored
          at the last taxonomy_set: earlier events belong to a tree that no
          longer exists and would render misleading empties. */}
      {view === 'schema' && (
        <div className="pointer-events-none absolute inset-x-0 bottom-24 z-30 flex justify-center">
          <ReplayTimeline
            events={events}
            floor={replayFloor}
            index={replayIndex}
            reflections={reflections}
            onOpen={() => {
              setReplayIndex(events.length);
              trackUsage('replay_opened');
            }}
            onScrub={setReplayIndex}
            onLive={() => setReplayIndex(null)}
            onReconsider={handleReconsiderProposal}
          />
        </div>
      )}
    </main>
  );
}
