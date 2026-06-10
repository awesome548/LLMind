'use client';

import MindElixir, {
  type MindElixirData,
  type MindElixirInstance,
  type NodeObj,
} from 'mind-elixir';
import 'mind-elixir/style.css';
import { Loader2, RotateCcw } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { MindmapNode, MindmapSelection } from '../../features/mindmap/types';
import { nodeColor, nodeTextColor } from '../../lib/node-colors';
import { ZOOM_FACTOR, ZOOM_MAX, ZOOM_MIN } from '../../lib/view-interactions';

interface SimpleMindMapProps {
  nodes: ReadonlyArray<MindmapNode>;
  activeTopic: string;
  onSelect: (selection: MindmapSelection) => void;
  onDataChange?: (nodes: ReadonlyArray<MindmapNode>) => void;
  /** Id of the node currently having children generated — shows a spinner on it. */
  generatingNodeId?: string | null;
  /** node id → pruning/composition state; rejected nodes render muted, chosen
   * (in the active candidate) render emphasized. */
  nodeStates?: Readonly<Record<string, 'rejected' | 'chosen'>>;
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
  topicToId: Record<string, string>,
  depth: number,
  branchIndex: number,
  nodeStates: Readonly<Record<string, 'rejected' | 'chosen'>>
): object {
  const lineage = [...parentLineage, node.topic];
  lineageById[node.id] = lineage;
  if (!topicToId[node.topic]) topicToId[node.topic] = node.id;

  const children = (node.children ?? []).map((child, i) =>
    // Top-level children define the branch index; descendants inherit it —
    // mirrors the design space so the same node is the same color in both.
    convertNode(child, lineage, lineageById, topicToId, depth + 1, depth === 0 ? i : branchIndex, nodeStates)
  );
  const state = nodeStates[node.id];
  const style =
    state === 'rejected'
      ? { background: '#e2e8f0', color: '#94a3b8' } // muted — pruned from the space
      : {
          background: nodeColor(branchIndex, depth),
          color: nodeTextColor(branchIndex, depth),
          ...(state === 'chosen' ? { fontWeight: '700' } : {}),
        };
  const base = { id: node.id, topic: node.topic, style };
  return children.length ? { ...base, children } : base;
}

function buildModel(
  nodes: ReadonlyArray<MindmapNode>,
  nodeStates: Readonly<Record<string, 'rejected' | 'chosen'>>
): MindElixirModel {
  const lineageById: Record<string, string[]> = {};
  const topicToId: Record<string, string> = {};

  if (nodes.length === 0) {
    lineageById['root'] = ['Mindmap'];
    topicToId['Mindmap'] = 'root';
    return { data: { nodeData: { id: 'root', topic: 'Mindmap' } }, lineageById, topicToId };
  }

  if (nodes.length === 1) {
    const nodeData = convertNode(nodes[0]!, [], lineageById, topicToId, 0, -1, nodeStates) as MindElixirData['nodeData'];
    return { data: { nodeData }, lineageById, topicToId };
  }

  // Multiple root nodes — wrap in synthetic root; each top-level node is a branch.
  lineageById[SYNTHETIC_ROOT_ID] = ['Mind Map'];
  topicToId['Mind Map'] = SYNTHETIC_ROOT_ID;
  const children = nodes.map((n, i) => convertNode(n, ['Mind Map'], lineageById, topicToId, 1, i, nodeStates));
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

const EMPTY_NODE_STATES: Readonly<Record<string, 'rejected' | 'chosen'>> = {};

export function SimpleMindMap({
  nodes,
  activeTopic,
  onSelect,
  onDataChange,
  generatingNodeId = null,
  nodeStates = EMPTY_NODE_STATES,
}: SimpleMindMapProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mindRef = useRef<MindElixirInstance | null>(null);
  const onSelectRef = useRef(onSelect);
  const onDataChangeRef = useRef(onDataChange);
  const isSyncingRef = useRef(false);
  const skipRefreshRef = useRef(false);
  const model = useMemo(() => buildModel(nodes, nodeStates), [nodes, nodeStates]);
  const modelRef = useRef(model);
  const [genPos, setGenPos] = useState<{ x: number; y: number } | null>(null);

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
      // ── Unified interactions (must match the design-space surface) ─────────
      // Left-drag pans (selection box moves to right-drag), plain wheel zooms
      // toward the cursor with the same factor and limits as the surface view.
      mouseSelectionButton: 2,
      scaleMin: ZOOM_MIN,
      scaleMax: ZOOM_MAX,
      handleWheel: (e: WheelEvent) => {
        e.preventDefault();
        const instance = mindRef.current;
        if (!instance) return;
        const factor = e.deltaY < 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
        const next = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, instance.scaleVal * factor));
        instance.scale(next, { x: e.clientX, y: e.clientY });
      },
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
      onSelectRef.current({ topic: nodeObj.topic, lineage: [...lineage], nodeId: nodeObj.id });
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

  // Position a generation spinner over the node having children generated.
  // Placement is scheduled (not synchronous) so it doesn't setState within the
  // effect body, and re-runs once in case the node is still settling.
  useEffect(() => {
    const place = () => {
      const mind = mindRef.current;
      if (!generatingNodeId || !mind) {
        setGenPos(null);
        return;
      }
      try {
        const el = mind.findEle(generatingNodeId);
        if (el) {
          const r = el.getBoundingClientRect();
          setGenPos({ x: r.left + r.width / 2, y: r.top + r.height / 2 });
        } else {
          setGenPos(null);
        }
      } catch {
        setGenPos(null);
      }
    };
    const t1 = window.setTimeout(place, 0);
    const t2 = window.setTimeout(place, 250);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
    };
  }, [generatingNodeId, model.topicToId]);

  const resetView = () => {
    const mind = mindRef.current;
    if (!mind) return;
    mind.scale(1);
    mind.toCenter();
  };

  return (
    <div className="relative flex h-full min-h-[520px] flex-col overflow-hidden bg-background">
      <div
        ref={containerRef}
        className="min-h-0 flex-1 bg-background"
        aria-label="Mind map visualization"
      />
      {/* Same control, same spot as the design-space view. */}
      <button
        type="button"
        onClick={resetView}
        className="absolute bottom-24 left-4 z-30 flex items-center gap-1.5 rounded-lg border bg-background/90 px-2.5 py-1.5 text-[10px] font-semibold text-muted-foreground shadow-sm backdrop-blur transition-colors hover:text-foreground"
        title="Reset view (scroll to zoom, drag to pan)"
      >
        <RotateCcw className="h-3 w-3" />
        Reset view
      </button>
      {genPos && (
        <div
          className="pointer-events-none fixed z-50 -translate-x-1/2 -translate-y-1/2"
          style={{ left: genPos.x, top: genPos.y }}
        >
          <span className="flex h-9 w-9 items-center justify-center rounded-full bg-background/80 shadow-lg ring-2 ring-sky-400/60 backdrop-blur">
            <Loader2 className="h-5 w-5 animate-spin text-sky-500" />
          </span>
        </div>
      )}
    </div>
  );
}
