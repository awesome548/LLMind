'use client';

import { RotateCcw } from 'lucide-react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { MindmapNode, MindmapSelection } from '@/src/features/mindmap/types';
import type { CoordMap, Surface } from '@/src/features/design-space/types';
import { nodeColor } from '@/src/lib/node-colors';

const VIEW = 1000; // SVG coordinate space (square)
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 8;
const DRAG_THRESHOLD = 4; // px of movement before a press counts as a pan
const DISCOVERED_COLOR = '#0ea5e9'; // sky — "already discovered" ring + trace lines
const EMPTY_SET: Set<string> = new Set();

interface PlacedNode {
  id: string;
  topic: string;
  depth: number;
  branchTopic: string | null;
  branchIndex: number;
  lineage: string[];
  gx: number;
  gy: number;
}

/** A transient burst drawn from the clicked cell to the nodes that were generated. */
export interface GenerationTrail {
  from: { x: number; y: number };
  to: Array<{ x: number; y: number }>;
}

interface ViewTransform {
  k: number;
  tx: number;
  ty: number;
}

interface Props {
  surface: Surface;
  nodes: ReadonlyArray<MindmapNode>;
  coords: CoordMap;
  selection: MindmapSelection;
  onSelectNode: (selection: MindmapSelection) => void;
  onGenerateAt: (x: number, y: number) => void;
  isGenerating?: boolean;
  pendingCell?: [number, number] | null;
  /** Connector from the active discovered cell to where its nodes were generated. */
  trail?: GenerationTrail | null;
  /** Cell keys ("gx,gy") that have already been used to generate — drawn hollow. */
  discovered?: Set<string>;
  /** Clicking an already-discovered (hollow) dot traces its generated nodes. */
  onShowDiscovery?: (cellKey: string) => void;
  /** Clicking empty (non-dot) space deselects. */
  onBackgroundClick?: () => void;
}

/** Walk the tree → flat placed nodes (depth, branch index/topic, snapped cell). */
function placeNodes(
  nodes: ReadonlyArray<MindmapNode>,
  coords: CoordMap,
  resolution: number
): { placed: PlacedNode[]; lineageOf: Map<string, string[]> } {
  const placed: PlacedNode[] = [];
  const lineageOf = new Map<string, string[]>();
  const seen = new Set<string>(); // guard against duplicate ids → duplicate React keys

  const walk = (node: MindmapNode, lineage: string[], depth: number, branchIndex: number) => {
    const nextLineage = [...lineage, node.topic];
    const branchTopic: string | null = nextLineage.length > 1 ? nextLineage[1] ?? null : null;
    lineageOf.set(node.id, nextLineage);

    const c = coords[node.id];
    if (c && !seen.has(node.id)) {
      seen.add(node.id);
      const gx = Math.min(resolution - 1, Math.max(0, Math.floor(c.x * resolution)));
      const gy = Math.min(resolution - 1, Math.max(0, Math.floor(c.y * resolution)));
      placed.push({
        id: node.id,
        topic: node.topic,
        depth,
        branchTopic,
        branchIndex,
        lineage: nextLineage,
        gx,
        gy,
      });
    }
    (node.children ?? []).forEach((child, i) =>
      // Top-level children (depth 0 → 1) define the branch index; descendants inherit it.
      walk(child, nextLineage, depth + 1, depth === 0 ? i : branchIndex)
    );
  };

  for (const node of nodes) walk(node, [], 0, -1);
  return { placed, lineageOf };
}

export function DesignSpaceSurface({
  surface,
  nodes,
  coords,
  selection,
  onSelectNode,
  onGenerateAt,
  isGenerating = false,
  pendingCell = null,
  trail = null,
  discovered,
  onShowDiscovery,
  onBackgroundClick,
}: Props) {
  const discoveredCells = discovered ?? EMPTY_SET;
  const R = surface.grid.resolution;
  const cell = VIEW / R;
  const containerRef = useRef<HTMLDivElement>(null);

  const [hover, setHover] = useState<{ gx: number; gy: number } | null>(null);
  const [tip, setTip] = useState<{ x: number; y: number; label: string; sub?: string } | null>(null);
  const [view, setView] = useState<ViewTransform>({ k: 1, tx: 0, ty: 0 });

  // `moved` lets us swallow the click that ends a drag (so a pan doesn't also
  // select/generate). We intentionally do NOT use setPointerCapture: capturing
  // the pointer on the container redirects pointerup/click away from the dots,
  // which would break clicking a dot to select it.
  const movedRef = useRef(false);

  const { placed, lineageOf } = useMemo(() => placeNodes(nodes, coords, R), [nodes, coords, R]);

  const occupied = useMemo(() => {
    const m = new Map<string, PlacedNode>();
    for (const p of placed) {
      const key = `${p.gx},${p.gy}`;
      if (!m.has(key)) m.set(key, p); // first wins on collision
    }
    return m;
  }, [placed]);

  const maxDensity = useMemo(() => {
    let max = 0;
    for (const row of surface.density) for (const d of row) if (d > max) max = d;
    return Math.max(1, max);
  }, [surface.density]);

  const selectedBranch = selection.lineage.length > 1 ? selection.lineage[1] : null;
  const hasBranchSelected = Boolean(selectedBranch);

  // Highlight EVERY dot whose node matches the selection by topic — there can be
  // several (same label under different branches). The exact one (lineage match)
  // is emphasized; the rest get a fainter glow so you can still see them.
  const selectionKey = selection.lineage.join('');
  const matches = useMemo(() => {
    const isExact = (p: PlacedNode) => p.lineage.join('') === selectionKey;
    const all = placed.filter((p) => p.topic === selection.topic);
    // If nothing matched by lineage (e.g. lineage drift), treat topic matches as
    // the selection so at least the right dots light up.
    const anyExact = all.some(isExact);
    return all.map((p) => ({ node: p, exact: anyExact ? isExact(p) : true }));
  }, [placed, selection.topic, selectionKey]);
  const exactMatchIds = useMemo(
    () => new Set(matches.filter((m) => m.exact).map((m) => m.node.id)),
    [matches]
  );
  const matchIds = useMemo(() => new Set(matches.map((m) => m.node.id)), [matches]);

  // ── Geometry helpers (data [0,1] → SVG, y flipped so up = larger y) ─────────
  const cx = useCallback((gx: number) => (gx + 0.5) * cell, [cell]);
  const cy = useCallback((gy: number) => VIEW - (gy + 0.5) * cell, [cell]);

  // ── Zoom (wheel, toward cursor) — native non-passive listener so we can
  //    preventDefault and stop the page from scrolling. ───────────────────────
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      setView((v) => {
        const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
        const k = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v.k * factor));
        const ratio = k / v.k;
        return { k, tx: px - (px - v.tx) * ratio, ty: py - (py - v.ty) * ratio };
      });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  // ── Pan (pointer drag via window listeners — no pointer capture) ────────────
  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const startX = e.clientX;
    const startY = e.clientY;
    const baseTx = view.tx;
    const baseTy = view.ty;
    movedRef.current = false;

    const onMove = (ev: PointerEvent) => {
      const dx = ev.clientX - startX;
      const dy = ev.clientY - startY;
      if (!movedRef.current && Math.hypot(dx, dy) > DRAG_THRESHOLD) movedRef.current = true;
      if (movedRef.current) {
        setTip(null);
        setView((v) => ({ ...v, tx: baseTx + dx, ty: baseTy + dy }));
      }
    };
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };

  // Swallow the click that terminates a drag so a pan doesn't also select/generate.
  const onClickCapture = (e: React.MouseEvent) => {
    if (movedRef.current) {
      e.stopPropagation();
      e.preventDefault();
      movedRef.current = false;
    }
  };

  const resetView = () => setView({ k: 1, tx: 0, ty: 0 });

  // ── Base lattice (heat-shaded empty dots) ────────────────────────────────
  const baseDots = useMemo(() => {
    const dots: React.ReactNode[] = [];
    for (let gy = 0; gy < R; gy++) {
      for (let gx = 0; gx < R; gx++) {
        const key = `${gx},${gy}`;
        if (occupied.has(key)) continue;
        const d = surface.density[gy]?.[gx] ?? 0;
        const heat = d / maxDensity;
        const isHover = hover?.gx === gx && hover?.gy === gy;
        const isPending = pendingCell?.[0] === gx && pendingCell?.[1] === gy;
        const isDiscovered = discoveredCells.has(key);
        const fill = d > 0 ? `rgba(244,140,43,${0.18 + 0.55 * heat})` : 'rgba(148,163,184,0.28)';
        dots.push(
          <circle
            key={`b-${gx}-${gy}`}
            cx={cx(gx)}
            cy={cy(gy)}
            r={
              isDiscovered
                ? cell * 0.36
                : isHover || isPending
                  ? cell * 0.42
                  : cell * 0.2 + heat * cell * 0.18
            }
            // Discovered cells render as a hollow ring (already explored).
            fill={isDiscovered ? 'none' : isPending ? DISCOVERED_COLOR : fill}
            stroke={isDiscovered || isHover ? DISCOVERED_COLOR : 'none'}
            strokeWidth={isDiscovered ? 2.5 : isHover ? 2 : 0}
            className="cursor-pointer transition-[r] duration-100"
            onMouseEnter={(e) => {
              setHover({ gx, gy });
              const rect = e.currentTarget.getBoundingClientRect();
              setTip({
                x: rect.left + rect.width / 2,
                y: rect.top,
                label: isDiscovered ? 'Discovered' : d > 0 ? `${d} nearby project${d > 1 ? 's' : ''}` : 'Empty space',
                sub: isDiscovered ? 'Click to trace generated nodes' : 'Click to generate here',
              });
            }}
            onMouseLeave={() => {
              setHover((h) => (h?.gx === gx && h?.gy === gy ? null : h));
              setTip(null);
            }}
            onClick={() =>
              isDiscovered
                ? onShowDiscovery?.(key)
                : onGenerateAt((gx + 0.5) / R, (gy + 0.5) / R)
            }
          />
        );
      }
    }
    return dots;
  }, [
    R,
    cell,
    occupied,
    surface.density,
    maxDensity,
    hover,
    pendingCell,
    discoveredCells,
    onGenerateAt,
    onShowDiscovery,
    cx,
    cy,
  ]);

  // ── Spinner ring around the cell being generated ─────────────────────────
  const spinner = useMemo(() => {
    if (!isGenerating || !pendingCell) return null;
    const [gx, gy] = pendingCell;
    const px = cx(gx);
    const py = cy(gy);
    const ringR = cell * 0.95;
    const circ = 2 * Math.PI * ringR;
    return (
      <g>
        <circle cx={px} cy={py} r={ringR} fill="none" stroke="rgba(14,165,233,0.18)" strokeWidth={cell * 0.18} />
        <circle
          cx={px}
          cy={py}
          r={ringR}
          fill="none"
          stroke="#0ea5e9"
          strokeWidth={cell * 0.18}
          strokeLinecap="round"
          strokeDasharray={`${circ * 0.28} ${circ * 0.72}`}
        >
          <animateTransform
            attributeName="transform"
            type="rotate"
            from={`0 ${px} ${py}`}
            to={`360 ${px} ${py}`}
            dur="0.9s"
            repeatCount="indefinite"
          />
        </circle>
      </g>
    );
  }, [isGenerating, pendingCell, cx, cy, cell]);

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full cursor-grab touch-none overflow-hidden active:cursor-grabbing"
      onPointerDown={onPointerDown}
      onClickCapture={onClickCapture}
      onClick={(e) => {
        // A click that didn't land on a dot (a <circle>) is empty space → deselect.
        if (movedRef.current) return;
        const tag = (e.target as Element).tagName?.toLowerCase();
        if (tag !== 'circle') onBackgroundClick?.();
      }}
      onMouseLeave={() => setTip(null)}
    >
      <svg
        viewBox={`0 0 ${VIEW} ${VIEW}`}
        className="h-full w-full"
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Design-space surface"
        style={{ transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.k})`, transformOrigin: '0 0' }}
      >
        <rect x={0} y={0} width={VIEW} height={VIEW} fill="transparent" />
        {baseDots}

        {/* Selection "range" glows: one per matching dot, exact emphasized. */}
        {matches.map(({ node, exact }) => {
          const color = nodeColor(node.branchIndex, node.depth);
          const gid = `ds-glow-${node.id}`;
          const inner = exact ? 0.55 : 0.22;
          const mid = exact ? 0.22 : 0.08;
          const radius = exact ? cell * 6 : cell * 4;
          return (
            <g key={`glow-${node.id}`}>
              <defs>
                <radialGradient id={gid}>
                  <stop offset="0%" stopColor={color} stopOpacity={inner} />
                  <stop offset="45%" stopColor={color} stopOpacity={mid} />
                  <stop offset="100%" stopColor={color} stopOpacity={0} />
                </radialGradient>
              </defs>
              <circle
                cx={cx(node.gx)}
                cy={cy(node.gy)}
                r={radius}
                fill={`url(#${gid})`}
                pointerEvents="none"
              />
            </g>
          );
        })}

        {/* Trace lines: from the discovered cell to where its nodes were generated.
            Endpoints are SNAPPED to the same lattice cell centers the dots use, so
            the lines land exactly on the target/source dots (not their raw coords). */}
        {trail &&
          (() => {
            const snapX = (x: number) => cx(Math.min(R - 1, Math.max(0, Math.floor(x * R))));
            const snapY = (y: number) => cy(Math.min(R - 1, Math.max(0, Math.floor(y * R))));
            const fx = snapX(trail.from.x);
            const fy = snapY(trail.from.y);
            return (
              <g pointerEvents="none">
                {/* Range glow at the clicked/discovered cell — the trace origin. */}
                <defs>
                  <radialGradient id="ds-discovery-glow">
                    <stop offset="0%" stopColor={DISCOVERED_COLOR} stopOpacity={0.4} />
                    <stop offset="45%" stopColor={DISCOVERED_COLOR} stopOpacity={0.16} />
                    <stop offset="100%" stopColor={DISCOVERED_COLOR} stopOpacity={0} />
                  </radialGradient>
                </defs>
                <circle cx={fx} cy={fy} r={cell * 5} fill="url(#ds-discovery-glow)" />
                {trail.to.map((pt, i) => {
                  const tx = snapX(pt.x);
                  const ty = snapY(pt.y);
                  return (
                    <g key={`trail-${i}`}>
                      <line
                        x1={fx}
                        y1={fy}
                        x2={tx}
                        y2={ty}
                        stroke={DISCOVERED_COLOR}
                        strokeWidth={1.75}
                        strokeLinecap="round"
                        opacity={0.7}
                      >
                        <animate attributeName="opacity" from="0" to="0.7" dur="0.35s" fill="freeze" />
                      </line>
                      <circle cx={tx} cy={ty} r={cell * 0.18} fill={DISCOVERED_COLOR} opacity={0.8} />
                    </g>
                  );
                })}
              </g>
            );
          })()}
        {spinner}

        {/* Node dots: colored by branch (shared with the mind map), snapped to the lattice. */}
        {placed.map((p) => {
          const color = nodeColor(p.branchIndex, p.depth);
          const isExact = exactMatchIds.has(p.id);
          const isMatch = matchIds.has(p.id);
          const inSelected =
            !hasBranchSelected || p.branchTopic === selectedBranch || p.depth === 0 || isMatch;
          const baseR = p.depth === 0 ? cell * 0.6 : p.depth === 1 ? cell * 0.5 : cell * 0.4;
          const r = isExact ? baseR * 1.4 : isMatch ? baseR * 1.15 : baseR;
          return (
            <g key={`n-${p.id}`}>
              <circle
                cx={cx(p.gx)}
                cy={cy(p.gy)}
                r={r}
                fill={color}
                opacity={inSelected ? 1 : 0.25}
                stroke={isExact ? '#0f172a' : isMatch ? '#475569' : 'white'}
                strokeWidth={isExact ? 2.5 : isMatch ? 2 : 1.5}
                className="cursor-pointer transition-[r] duration-100 hover:brightness-110"
                onMouseEnter={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  setTip({
                    x: rect.left + rect.width / 2,
                    y: rect.top,
                    label: p.topic,
                    sub:
                      p.depth === 0
                        ? 'Root'
                        : p.depth === 1
                          ? 'Branch'
                          : p.branchTopic
                            ? `Option in ${p.branchTopic}`
                            : 'Option',
                  });
                }}
                onMouseLeave={() => setTip(null)}
                onClick={() => onSelectNode({ topic: p.topic, lineage: lineageOf.get(p.id) ?? [p.topic] })}
              />
            </g>
          );
        })}
      </svg>

      {/* Hover tooltip — positioned at the hovered dot (viewport coords) */}
      {tip && (
        <div
          className="pointer-events-none fixed z-50 -translate-x-1/2 -translate-y-full"
          style={{ left: tip.x, top: tip.y - 10 }}
        >
          <div className="max-w-[16rem] rounded-lg border bg-background/95 px-2.5 py-1.5 shadow-lg backdrop-blur">
            <div className="truncate text-xs font-semibold text-foreground">{tip.label}</div>
            {tip.sub && <div className="mt-0.5 text-[10px] text-muted-foreground">{tip.sub}</div>}
          </div>
        </div>
      )}

      {/* Reset view + legend (bottom-left, above the navigator/dev indicator) */}
      <div className="pointer-events-none absolute bottom-24 left-4 z-30 flex flex-col items-start gap-2">
        <button
          type="button"
          onClick={resetView}
          className="pointer-events-auto flex items-center gap-1.5 rounded-lg border bg-background/90 px-2.5 py-1.5 text-[10px] font-semibold text-muted-foreground shadow-sm backdrop-blur transition-colors hover:text-foreground"
          title="Reset view (scroll to zoom, drag to pan)"
        >
          <RotateCcw className="h-3 w-3" />
          Reset view
        </button>
        <div className="flex flex-col gap-1 rounded-lg border bg-background/80 px-3 py-2 text-[10px] text-muted-foreground shadow-sm backdrop-blur">
          <div className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: 'rgba(244,140,43,0.7)' }} />
            corpus density (real projects)
          </div>
          <div className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: 'rgba(148,163,184,0.5)' }} />
            empty — click to generate
          </div>
          <div className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: nodeColor(0, 1) }} />
            taxonomy node (colored by branch)
          </div>
        </div>
      </div>
    </div>
  );
}
