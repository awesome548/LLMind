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
import {
  DesignSpaceSurface,
  type GenerationTrail,
} from '@/src/components/design-space/design-space-surface';
import { useSurfaceQuery } from '@/src/features/design-space/hooks/use-surface-query';
import {
  nodesToLocateItems,
  useLocateNodesMutation,
} from '@/src/features/design-space/hooks/use-locate-nodes';
import { useGenerateAtMutation } from '@/src/features/design-space/hooks/use-generate-at-mutation';
import type { CoordMap } from '@/src/features/design-space/types';
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
import type { MindmapNode, MindmapSelection } from '@/src/features/mindmap/types';
import { GenerateTaxonomyDialog } from '@/src/features/mindmap/components/generate-taxonomy-dialog';
import { GenerateNodesDialog } from '@/src/features/mindmap/components/generate-nodes-dialog';
import type { GenerateNodesParams } from '@/src/features/mindmap/hooks/use-generate-nodes-mutation';
import { useMindmapStore } from '@/src/store/mindmap-store';
import type { FetchRelatedProjectsRequestSchema } from '@/src/types/openapi';

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

function cloneNodes(nodes: ReadonlyArray<MindmapNode>): ReadonlyArray<MindmapNode> {
  return nodes.map((node) => ({
    ...node,
    children: node.children ? cloneNodes(node.children) : undefined,
  }));
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

interface AspectRef {
  id: string;
  topic: string;
  lineage: string[];
}

/** node id → its depth-1 ancestor (the aspect), or null for the root. */
function buildAspectIndex(
  nodes: ReadonlyArray<MindmapNode>
): Map<string, AspectRef | null> {
  const index = new Map<string, AspectRef | null>();
  const walk = (node: MindmapNode, lineage: string[], depth: number, aspect: AspectRef | null) => {
    const nextLineage = [...lineage, node.topic];
    const myAspect: AspectRef | null =
      depth === 1 ? { id: node.id, topic: node.topic, lineage: nextLineage } : aspect;
    index.set(node.id, myAspect);
    for (const child of node.children ?? []) walk(child, nextLineage, depth + 1, myAspect);
  };
  for (const node of nodes) walk(node, [], 0, null);
  return index;
}

export default function MindmapPage() {
  const selectTopic = useMindmapStore((state) => state.selectTopic);
  const taxonomy = useMindmapStore((state) => state.taxonomy);
  const setTaxonomy = useMindmapStore((state) => state.setTaxonomy);
  const { mutateAsync: generateNodes, isPending: isGeneratingNodes } = useGenerateNodesMutation();

  const [nodes, setNodes] = useState<ReadonlyArray<MindmapNode>>(() =>
    cloneNodes(SCHEMA_MINDMAP_NODES)
  );
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
  const [coords, setCoords] = useState<CoordMap>({});
  const [pendingCell, setPendingCell] = useState<[number, number] | null>(null);
  // The currently-traced connector, and every cell that has generated nodes
  // (drawn hollow as "discovered"; re-clicking re-traces its line).
  const [activeLine, setActiveLine] = useState<GenerationTrail | null>(null);
  const [discoveredMap, setDiscoveredMap] = useState<Map<string, GenerationTrail>>(
    () => new Map()
  );
  const discoveredCells = useMemo(() => new Set(discoveredMap.keys()), [discoveredMap]);
  const { data: surface } = useSurfaceQuery(true);
  const { mutateAsync: locateNodes } = useLocateNodesMutation();
  const { mutateAsync: generateAt, isPending: isGeneratingAt } = useGenerateAtMutation();
  const locatingRef = useRef(false);

  const attemptedRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!taxonomy) return;
    const { nodes: nextNodes } = taxonomyToMindmapNodes(taxonomy);
    setNodes(nextNodes);
    setSelection({ topic: INITIAL_SELECTION.topic, lineage: [...INITIAL_SELECTION.lineage] });
    // New taxonomy → invalidate every design-space coordinate.
    setCoords({});
    attemptedRef.current.clear();
  }, [taxonomy]);

  // Best-effort: embed + locate any nodes that lack coordinates. Each node is
  // attempted at most once per session; failures (e.g. embedding server down)
  // leave the background surface intact and are retried on the next change.
  useEffect(() => {
    if (!surface) return;
    const items = nodesToLocateItems(nodes, activeDescriptionByTopic).filter(
      (it) => !coords[it.node_id] && !attemptedRef.current.has(it.node_id)
    );
    if (items.length === 0 || locatingRef.current) return;

    locatingRef.current = true;
    items.forEach((it) => attemptedRef.current.add(it.node_id));
    locateNodes(items)
      .then((located) => setCoords((prev) => ({ ...prev, ...located })))
      .catch(() => {
        items.forEach((it) => attemptedRef.current.delete(it.node_id));
      })
      .finally(() => {
        locatingRef.current = false;
      });
  }, [surface, nodes, activeDescriptionByTopic, coords, locateNodes]);

  const description = activeDescriptionByTopic[selection.topic] ?? '';
  const request = buildRequest(selection, description);
  const { data, isFetching } = useRelatedProjectsQuery({ request });

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
    });
    setActiveLine(null); // a fresh selection clears any traced connector
  };

  const handleGenerateAt = useCallback(
    async (x: number, y: number) => {
      if (!surface) return;

      // Spatial intent: attach new options under the ASPECT nearest the clicked
      // dot — keeps the 2-level structure (options under their branch, matching
      // color) instead of dumping them under whatever node happens to be selected.
      const aspectIndex = buildAspectIndex(nodes);
      let nearestId: string | null = null;
      let bestDist = Infinity;
      for (const [id, c] of Object.entries(coords)) {
        const d = (c.x - x) ** 2 + (c.y - y) ** 2;
        if (d < bestDist) {
          bestDist = d;
          nearestId = id;
        }
      }
      const nearestAspect = nearestId ? aspectIndex.get(nearestId) ?? null : null;
      const fallback = findNodeByLineage(nodes, selection.lineage);
      const focusId = nearestAspect?.id ?? fallback?.id;
      const focusTopic = nearestAspect?.topic ?? fallback?.topic;
      const focusLineage = nearestAspect?.lineage ?? selection.lineage;
      if (!focusId || !focusTopic) {
        setGenerateError('No aspect found nearby to attach generated options to.');
        return;
      }

      const resolution = surface.grid.resolution;
      setPendingCell([Math.floor(x * resolution), Math.floor(y * resolution)]);
      setGenerateError(null);
      // Reflect the target aspect in the selection so its region highlights and
      // its related projects load.
      setSelection({ topic: focusTopic, lineage: [...focusLineage] });

      try {
        const response = await generateAt({
          x,
          y,
          allNodes: nodes,
          focusNode: { id: focusId, topic: focusTopic },
          lineage: focusLineage,
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

        // Coordinates come back with the generation — no extra /locate call.
        // Keep them aligned with any ids that were remapped to stay unique.
        const merged: CoordMap = {};
        for (const c of response.coords) {
          const id = remap[c.node_id] ?? c.node_id;
          merged[id] = { x: c.x, y: c.y, ...(c.z != null ? { z: c.z } : {}) };
          attemptedRef.current.add(id);
        }
        setCoords((prev) => ({ ...prev, ...merged }));

        // Mark the clicked cell "discovered" (drawn hollow) and store + show the
        // connector to where the nodes landed, so the (often distant) placement is
        // visibly tied to the click. Re-clicking the hollow dot re-traces it.
        const targets = Object.values(merged);
        if (targets.length > 0) {
          const line: GenerationTrail = {
            from: { x, y },
            to: targets.map((c) => ({ x: c.x, y: c.y })),
          };
          const cellKey = `${Math.floor(x * resolution)},${Math.floor(y * resolution)}`;
          setDiscoveredMap((prev) => new Map(prev).set(cellKey, line));
          setActiveLine(line);
        }
      } catch (error) {
        setGenerateError(
          error instanceof Error ? error.message : 'Failed to generate at location.'
        );
      } finally {
        setPendingCell(null);
      }
    },
    [surface, nodes, coords, selection, generateAt]
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
        (p) => p.Name !== 'Relevant projects will appear here'
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

      const { children: generatedChildren } = ensureUniqueChildIds(
        nodes,
        generatedNodesToMindmapNodes(response)
      );
      const treeUpdate = insertChildrenAtNode(nodes, response.parent_id, generatedChildren);
      if (!treeUpdate.inserted) {
        setGenerateError('Generated nodes were returned, but no matching parent was found.');
        return;
      }

      setNodes(treeUpdate.nodes);
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

                  {description ? (
                    <div className="space-y-1 text-sm leading-relaxed">
                      <p className="text-muted-foreground">{description}</p>
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
              isGenerating={isGeneratingAt}
              pendingCell={pendingCell}
              trail={activeLine}
              discovered={discoveredCells}
              onShowDiscovery={(key) => setActiveLine(discoveredMap.get(key) ?? null)}
              onBackgroundClick={() => {
                setSelection({ topic: '', lineage: [] });
                setActiveLine(null);
              }}
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
            onDataChange={setNodes}
            generatingNodeId={
              isGeneratingNodes ? findNodeByLineage(nodes, selection.lineage)?.id ?? null : null
            }
          />
        )}
      </div>

      <GenerateTaxonomyDialog
        open={taxonomyDialogOpen}
        onOpenChange={setTaxonomyDialogOpen}
        onSuccess={setTaxonomy}
      />

      <GenerateNodesDialog
        open={generateNodesDialogOpen}
        onOpenChange={setGenerateNodesDialogOpen}
        onConfirm={handleGenerateNodes}
        isPending={isGeneratingNodes}
        selectedTopic={selection.topic}
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
