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
  onDataChange?: (nodes: ReadonlyArray<MindmapNode>) => void;
}

interface MindElixirModel {
  data: MindElixirData;
  lineageById: Readonly<Record<string, string[]>>;
  topicToId: Readonly<Record<string, string>>;
}

const SYNTHETIC_ROOT_ID = '__root__';

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
  lineageById[SYNTHETIC_ROOT_ID] = ['Mind Map'];
  topicToId['Mind Map'] = SYNTHETIC_ROOT_ID;
  const children = nodes.map((n) => convertNode(n, ['Mind Map'], lineageById, topicToId));
  const nodeData = { id: SYNTHETIC_ROOT_ID, topic: 'Mind Map', children } as MindElixirData['nodeData'];
  return { data: { nodeData }, lineageById, topicToId };
}

function nodeObjToMindmapNode(node: NodeObj): MindmapNode {
  return {
    id: node.id,
    topic: node.topic,
    children: node.children?.length ? node.children.map(nodeObjToMindmapNode) : undefined,
  };
}

function mindElixirDataToNodes(data: MindElixirData): ReadonlyArray<MindmapNode> {
  const root = data.nodeData;
  if (root.id === SYNTHETIC_ROOT_ID) {
    return (root.children ?? []).map(nodeObjToMindmapNode);
  }
  return [nodeObjToMindmapNode(root)];
}

function buildLineageFromParent(node: NodeObj): string[] {
  return node.parent
    ? [...buildLineageFromParent(node.parent), node.topic]
    : [node.topic];
}

export function SimpleMindMap({ nodes, activeTopic, onSelect, onDataChange }: SimpleMindMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mindRef = useRef<MindElixirInstance | null>(null);
  const onSelectRef = useRef(onSelect);
  const onDataChangeRef = useRef(onDataChange);
  const isSyncingRef = useRef(false);
  const skipRefreshRef = useRef(false);
  const model = useMemo(() => buildModel(nodes), [nodes]);
  const modelRef = useRef(model);

  useEffect(() => { onSelectRef.current = onSelect; }, [onSelect]);
  useEffect(() => { onDataChangeRef.current = onDataChange; }, [onDataChange]);
  useEffect(() => { modelRef.current = model; }, [model]);

  // Init mind-elixir once
  useEffect(() => {
    const container = containerRef.current;
    if (!container || mindRef.current) return;

    const mind = new MindElixir({
      el: container,
      direction: MindElixir.SIDE,
      editable: true,
      contextMenu: true,
      toolBar: false,
      keypress: true,
      allowUndo: true,
    });

    mind.init(model.data);

    // Sync node selection → React state
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const handleMapClick = (e: any) => {
      if (isSyncingRef.current) return;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let el: any = e.target;
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

    // Sync structural edits (add/delete/rename/move) → React nodes state
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    mind.bus.addListener('operation', (_op: any) => {
      skipRefreshRef.current = true;
      const newNodes = mindElixirDataToNodes(mind.getData());
      onDataChangeRef.current?.(newNodes);
    });

    mindRef.current = mind;
    return () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (mind.map as any).removeEventListener('click', handleMapClick);
      mind.destroy();
      mindRef.current = null;
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Refresh when node data changes (skip if change originated from mind-elixir itself)
  useEffect(() => {
    if (skipRefreshRef.current) {
      skipRefreshRef.current = false;
      return;
    }
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
