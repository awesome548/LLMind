'use client';

import {
  Brain,
  ChevronRight,
  Home,
  Info,
  Loader2,
  PanelsRightBottom,
  Sparkles,
  Zap,
} from 'lucide-react';
import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import { SimpleMindMap } from '@/src/components/mindmap/simple-mindmap';
import { SimpleProjectPanel } from '@/src/components/mindmap/simple-project-panel';
import { Badge } from '@/src/components/ui/badge';
import { Button } from '@/src/components/ui/button';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/src/components/ui/collapsible';
import { ScrollArea } from '@/src/components/ui/scroll-area';
import { Separator } from '@/src/components/ui/separator';
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

  useEffect(() => {
    if (!taxonomy) return;
    const { nodes: nextNodes } = taxonomyToMindmapNodes(taxonomy);
    setNodes(nextNodes);
    setSelection({ topic: INITIAL_SELECTION.topic, lineage: [...INITIAL_SELECTION.lineage] });
  }, [taxonomy]);

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

      const generatedChildren = generatedNodesToMindmapNodes(response);
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
          <h1 className="text-xs font-black tracking-[0.2em] uppercase">LLMind</h1>
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

      {/* Main Mind Map Layer */}
      <div className="relative h-full w-full">
        <SimpleMindMap
          nodes={nodes}
          activeTopic={selection.topic}
          onSelect={handleSelect}
          onDataChange={setNodes}
        />
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
