'use client';

import MindElixir, {
  type MindElixirData,
  type MindElixirInstance,
  type NodeObj,
} from 'mind-elixir';
import 'mind-elixir/style.css';
import { useEffect, useMemo, useRef } from 'react';
import type { MindmapNode, MindmapSelection } from '../../features/mindmap/types';

interface SimpleMindMapProps {
  nodes: ReadonlyArray<MindmapNode>;
  activeTopic: string;
  onSelect: (selection: MindmapSelection) => void;
}

interface MindElixirModel {
  data: MindElixirData;
  lineageById: Readonly<Record<string, string[]>>;
  topicToId: Readonly<Record<string, string>>;
}

function convertNode(
  node: MindmapNode,
  parentLineage: string[],
  lineageById: Record<string, string[]>,
  topicToId: Record<string, string>
): object {
  const lineage = [...parentLineage, node.topic];
  lineageById[node.id] = lineage;
  if (!topicToId[node.topic]) topicToId[node.topic] = node.id;

  const children = (node.children ?? []).map((child) =>
    convertNode(child, lineage, lineageById, topicToId)
  );
  return children.length
    ? { id: node.id, topic: node.topic, children }
    : { id: node.id, topic: node.topic };
}

function buildModel(nodes: ReadonlyArray<MindmapNode>): MindElixirModel {
  const lineageById: Record<string, string[]> = {};
  const topicToId: Record<string, string> = {};

  if (nodes.length === 0) {
    lineageById['root'] = ['Mindmap'];
    topicToId['Mindmap'] = 'root';
    return { data: { nodeData: { id: 'root', topic: 'Mindmap' } }, lineageById, topicToId };
  }

  if (nodes.length === 1) {
    const nodeData = convertNode(nodes[0]!, [], lineageById, topicToId) as MindElixirData['nodeData'];
    return { data: { nodeData }, lineageById, topicToId };
  }

  // Multiple root nodes — wrap in synthetic root
  lineageById['__root__'] = ['Mind Map'];
  topicToId['Mind Map'] = '__root__';
  const children = nodes.map((n) => convertNode(n, ['Mind Map'], lineageById, topicToId));
  const nodeData = { id: '__root__', topic: 'Mind Map', children } as MindElixirData['nodeData'];
  return { data: { nodeData }, lineageById, topicToId };
}

function buildLineageFromParent(node: NodeObj): string[] {
  return node.parent
    ? [...buildLineageFromParent(node.parent), node.topic]
    : [node.topic];
}

export function SimpleMindMap({ nodes, activeTopic, onSelect }: SimpleMindMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mindRef = useRef<MindElixirInstance | null>(null);
  const onSelectRef = useRef(onSelect);
  const isSyncingRef = useRef(false);
  const model = useMemo(() => buildModel(nodes), [nodes]);
  const modelRef = useRef(model);

  useEffect(() => { onSelectRef.current = onSelect; }, [onSelect]);
  useEffect(() => { modelRef.current = model; }, [model]);

  // Init mind-elixir once
  useEffect(() => {
    const container = containerRef.current;
    if (!container || mindRef.current) return;

    const mind = new MindElixir({
      el: container,
      direction: MindElixir.SIDE,
      editable: false,
      contextMenu: false,
      toolBar: false,
      keypress: false,
      allowUndo: false,
    });

    mind.init(model.data);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleMapClick = (e: any) => {
      if (isSyncingRef.current) return;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let el: any = e.target;
      // Walk up to find me-tpc element
      while (el && el.tagName !== 'ME-TPC') {
        el = el.parentElement ?? null;
        if (!el || el === mind.map) return;
      }
      const nodeObj = el?.nodeObj as NodeObj | undefined;
      if (!nodeObj) return;
      const lineage = modelRef.current.lineageById[nodeObj.id] ?? buildLineageFromParent(nodeObj);
      onSelectRef.current({ topic: nodeObj.topic, lineage: [...lineage] });
    };

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (mind.map as any).addEventListener('click', handleMapClick);

    mindRef.current = mind;
    return () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (mind.map as any).removeEventListener('click', handleMapClick);
      mind.destroy();
      mindRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Refresh when node data changes
  useEffect(() => {
    mindRef.current?.refresh(model.data);
  }, [model.data]);

  // Sync external activeTopic → mind-elixir selection
  useEffect(() => {
    const mind = mindRef.current;
    const nodeId = model.topicToId[activeTopic];
    if (!mind || !nodeId) return;
    try {
      const nodeEl = mind.findEle(nodeId);
      if (!nodeEl) return;
      isSyncingRef.current = true;
      mind.selectNode(nodeEl);
      isSyncingRef.current = false;
    } catch {
      isSyncingRef.current = false;
    }
  }, [activeTopic, model.topicToId]);

  return (
    <div className="flex min-h-[520px] flex-col overflow-hidden rounded-xl border bg-card">
      <div
        ref={containerRef}
        className="flex-1 bg-background"
        aria-label="Mind map visualization"
      />
    </div>
  );
}
