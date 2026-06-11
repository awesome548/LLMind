'use client';

import { RotateCcw } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import type { MindmapNode, MindmapSelection } from '@/src/features/mindmap/types';
import type { CoordMap, GenerationTrail, Surface } from '@/src/features/design-space/types';
import type { RelevanceMap } from '@/src/features/design-space/hooks/use-relevance-query';
import { usePanZoom } from '@/src/features/design-space/hooks/use-pan-zoom';
import { nodeColor } from '@/src/lib/node-colors';
import { starPath } from '@/src/lib/svg-glyphs';

export type { GenerationTrail } from '@/src/features/design-space/types';

const VIEW = 1000; // SVG coordinate space (square)
const DISCOVERED_COLOR = '#0ea5e9'; // sky — "already discovered" ring + trace lines
const CORPUS_COLOR = 'rgba(244,140,43,0.9)'; // amber — real corpus projects
// Placement confidence below this renders dashed ("approximate"). Calibrated
// against the corpus itself: real projects at their own true coordinates score
// ~0.25 mean, so anything well under that is projection noise.
const LOW_CONFIDENCE = 0.1;
const CANDIDATE_COLOR = '#7c3aed'; // violet — composed design candidates
const EMPTY_DISCOVERED: Record<string, GenerationTrail> = {};
const EMPTY_REJECTED: ReadonlySet<string> = new Set();

/** Relevance-lens ramp: cool-to-warm heatmap (blue → green → yellow → red).
 * The cold/hot reading is immediate, and low-relevance dots stay visually
 * quiet (cool, mid-light) on the light background. */
export function relevanceColor(t: number): string {
  const hue = 240 - 230 * t;
  const sat = 70 + 20 * t;
  const light = 60 - 12 * t;
  return `hsl(${hue} ${sat}% ${light}%)`;
}

export interface CandidateMarker {
  id: string;
  name: string;
  x: number;
  y: number;
  active: boolean;
}

interface PlacedNode {
  id: string;
  topic: string;
  depth: number;
  branchTopic: string | null;
  branchIndex: number;
  lineage: string[];
  gx: number;
  gy: number;
  confidence: number | null;
}

interface Props {
  surface: Surface;
  nodes: ReadonlyArray<MindmapNode>;
  coords: CoordMap;
  selection: MindmapSelection;
  onSelectNode: (selection: MindmapSelection) => void;
  onGenerateAt: (x: number, y: number) => void;
  /** Clicking a corpus (real project) glyph — open it for inspection. */
  onSelectProject?: (projectId: string) => void;
  isGenerating?: boolean;
  pendingCell?: [number, number] | null;
  /** Connector from the active discovered cell to where its nodes were generated. */
  trail?: GenerationTrail | null;
  /** "gx,gy" cell key → its generation trail; discovered cells draw hollow. */
  discovered?: Record<string, GenerationTrail>;
  /** Clicking an already-discovered (hollow) dot traces its generated nodes. */
  onShowDiscovery?: (cellKey: string) => void;
  /** Clicking empty (non-dot) space deselects. */
  onBackgroundClick?: () => void;
  /** Composed design candidates, drawn as stars at their own embedding position. */
  candidates?: ReadonlyArray<CandidateMarker>;
  onSelectCandidate?: (candidateId: string) => void;
  /** Node ids the designer rejected — drawn dimmed. */
  rejected?: ReadonlySet<string>;
  /** Stop waiting on the running generation (shown while isGenerating). */
  onCancelGenerate?: () => void;
  /** Corpus ids of the selected node's related projects — highlighted so the
   * panel's examples are also visible as places on the map. (Similarity mode.) */
  relatedProjects?: ReadonlySet<string>;
  /** Relevance-lens painting: when set, corpus glyphs recolor by normalized
   * true-cosine relevance to the anchor, and non-anchor nodes fade. */
  relevance?: RelevanceMap | null;
  /** Node id (or `cand:` key) of the lens anchor — keeps full opacity. */
  lensAnchorId?: string | null;
  /** Anchor label for the lens legend. */
  lensAnchorLabel?: string | null;
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
        confidence: c.confidence ?? null,
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
  onSelectProject,
  isGenerating = false,
  pendingCell = null,
  trail = null,
  discovered,
  onShowDiscovery,
  onBackgroundClick,
  candidates,
  onSelectCandidate,
  rejected,
  onCancelGenerate,
  relatedProjects,
  relevance = null,
  lensAnchorId = null,
  lensAnchorLabel = null,
}: Props) {
  const discoveredCells = discovered ?? EMPTY_DISCOVERED;
  const rejectedIds = rejected ?? EMPTY_REJECTED;
  const lensActive = Boolean(relevance);
  const R = surface.grid.resolution;
  const cell = VIEW / R;

  const [hover, setHover] = useState<{ gx: number; gy: number } | null>(null);
  const [tip, setTip] = useState<{ x: number; y: number; label: string; sub?: string } | null>(null);
  // Disambiguation popover for cells where several nodes snapped to one spot.
  const [chooser, setChooser] = useState<{ x: number; y: number; nodes: PlacedNode[] } | null>(
    null
  );

  // Unified canvas interactions (shared with the axes view; mirrored by the
  // mind map). Tooltips and the viewport-fixed chooser dismiss on pan.
  const { containerRef, view, movedRef, onPointerDown, onClickCapture, resetView } = usePanZoom(
    () => {
      setTip(null);
      setChooser(null);
    }
  );

  const { placed, lineageOf } = useMemo(() => placeNodes(nodes, coords, R), [nodes, coords, R]);

  // Every node per cell — collisions get a count badge + click-to-disambiguate.
  const cellNodes = useMemo(() => {
    const m = new Map<string, PlacedNode[]>();
    for (const p of placed) {
      const key = `${p.gx},${p.gy}`;
      const list = m.get(key);
      if (list) list.push(p);
      else m.set(key, [p]);
    }
    return m;
  }, [placed]);
  const occupied = cellNodes;

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
  const selectionKey = selection.lineage.join('');
  const matches = useMemo(() => {
    // Exact identity comes from the clicked node's id when available; the
    // lineage comparison is the fallback for legacy string-only selections.
    const isExact = (p: PlacedNode) =>
      selection.nodeId ? p.id === selection.nodeId : p.lineage.join('') === selectionKey;
    const all = placed.filter((p) => p.topic === selection.topic);
    // If nothing matched (e.g. lineage drift), treat topic matches as the
    // selection so at least the right dots light up.
    const anyExact = all.some(isExact);
    return all.map((p) => ({ node: p, exact: anyExact ? isExact(p) : true }));
  }, [placed, selection.topic, selection.nodeId, selectionKey]);
  const exactMatchIds = useMemo(
    () => new Set(matches.filter((m) => m.exact).map((m) => m.node.id)),
    [matches]
  );
  const matchIds = useMemo(() => new Set(matches.map((m) => m.node.id)), [matches]);

  // ── Geometry helpers (data [0,1] → SVG, y flipped so up = larger y) ─────────
  const cx = useCallback((gx: number) => (gx + 0.5) * cell, [cell]);
  const cy = useCallback((gy: number) => VIEW - (gy + 0.5) * cell, [cell]);
  // Continuous-coordinate variants (corpus glyphs sit at their true position).
  const px = useCallback((x: number) => x * VIEW, []);
  const py = useCallback((y: number) => VIEW - y * VIEW, []);

  // Aggregate density heat is redundant while individual corpus glyphs are
  // legible; it fades in as you zoom OUT and the glyphs shrink (overview mode).
  // The lens replaces the corpus painting entirely, so the heat hides with it.
  const heatOpacity = lensActive ? 0 : Math.max(0, Math.min(1, (1 - view.k) / 0.4));

  // ── Density heat layer (non-interactive; visibility driven by zoom) ────────
  const heatDots = useMemo(() => {
    const dots: React.ReactNode[] = [];
    for (let gy = 0; gy < R; gy++) {
      for (let gx = 0; gx < R; gx++) {
        const d = surface.density[gy]?.[gx] ?? 0;
        if (d <= 0) continue;
        const heat = d / maxDensity;
        dots.push(
          <circle
            key={`h-${gx}-${gy}`}
            cx={cx(gx)}
            cy={cy(gy)}
            r={cell * 0.5 + heat * cell * 0.3}
            fill={`rgba(244,140,43,${0.15 + 0.4 * heat})`}
          />
        );
      }
    }
    return dots;
  }, [R, cell, surface.density, maxDensity, cx, cy]);

  // ── Base lattice (uniform empty dots — the generation affordance) ──────────
  // Hover/pending highlights are drawn as a SEPARATE overlay so moving the mouse
  // never rebuilds these ~2300 circles.
  const baseDots = useMemo(() => {
    const dots: React.ReactNode[] = [];
    for (let gy = 0; gy < R; gy++) {
      for (let gx = 0; gx < R; gx++) {
        const key = `${gx},${gy}`;
        if (occupied.has(key)) continue;
        const cellTrail = discoveredCells[key];
        const isDiscovered = Boolean(cellTrail);
        dots.push(
          <circle
            key={`b-${gx}-${gy}`}
            cx={cx(gx)}
            cy={cy(gy)}
            r={isDiscovered ? cell * 0.36 : cell * 0.2}
            // Discovered cells render as a hollow ring (already explored).
            fill={isDiscovered ? 'none' : 'rgba(148,163,184,0.28)'}
            stroke={isDiscovered ? DISCOVERED_COLOR : 'none'}
            strokeWidth={isDiscovered ? 2.5 : 0}
            className="cursor-pointer"
            onMouseEnter={(e) => {
              setHover({ gx, gy });
              const rect = e.currentTarget.getBoundingClientRect();
              setTip({
                x: rect.left + rect.width / 2,
                y: rect.top,
                label: isDiscovered ? 'Discovered' : 'Empty space',
                sub: isDiscovered
                  ? cellTrail?.meanDrift != null
                    ? `Click to trace · nodes landed Ø ${cellTrail.meanDrift.toFixed(2)} away`
                    : 'Click to trace generated nodes'
                  : 'Click to generate here',
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
  }, [R, cell, occupied, discoveredCells, onGenerateAt, onShowDiscovery, cx, cy]);

  // Hover/pending highlight overlay (cheap to re-render; non-interactive).
  const cellHighlights = (
    <g pointerEvents="none">
      {hover && !occupied.has(`${hover.gx},${hover.gy}`) && (
        <circle
          cx={cx(hover.gx)}
          cy={cy(hover.gy)}
          r={cell * 0.42}
          fill="none"
          stroke={DISCOVERED_COLOR}
          strokeWidth={2}
        />
      )}
      {pendingCell && (
        <circle
          cx={cx(pendingCell[0])}
          cy={cy(pendingCell[1])}
          r={cell * 0.42}
          fill={DISCOVERED_COLOR}
        />
      )}
    </g>
  );

  // ── Corpus glyphs (real projects, inspectable) ──────────────────────────────
  // Heat splats for the lens: one soft circle per corpus point, sized and
  // colored by relevance. Only the warm half glows — broad anchors score
  // mid-high everywhere, and glowing everything turns the field into a wash
  // that buries the glyphs.
  const lensGlow = useMemo(() => {
    if (!relevance) return null;
    return surface.points.map((p) => {
      const t = relevance.byId[p.id] ?? 0;
      if (t < 0.45) return null;
      const heat = (t - 0.45) / 0.55;
      return (
        <circle
          key={`lg-${p.id}`}
          cx={px(p.x)}
          cy={py(p.y)}
          r={cell * (0.7 + 1.9 * heat)}
          fill={relevanceColor(t)}
          opacity={0.15 + 0.75 * heat}
        />
      );
    });
  }, [relevance, surface.points, cell, px, py]);

  const corpusGlyphs = useMemo(() => {
    const glows: React.ReactNode[] = [];
    const glyphs = surface.points.map((p) => {
      const gx = px(p.x);
      const gy = py(p.y);
      // Lens mode: continuous relevance painting replaces the binary
      // related-projects highlight (which stays a Similarity-mode affordance).
      const t = relevance ? relevance.byId[p.id] ?? 0 : null;
      const isRelated = !relevance && (relatedProjects?.has(p.id) ?? false);
      const s = t != null ? cell * (0.16 + 0.22 * t) : cell * (isRelated ? 0.34 : 0.24);
      if (isRelated) {
        // Soft halo beneath the glyph so the panel's examples pop on the map.
        glows.push(
          <circle
            key={`crg-${p.id}`}
            cx={gx}
            cy={gy}
            r={cell * 1.4}
            fill="url(#ds-related-glow)"
            pointerEvents="none"
          />
        );
      }
      return (
        <rect
          key={`c-${p.id}`}
          x={gx - s}
          y={gy - s}
          width={s * 2}
          height={s * 2}
          transform={`rotate(45 ${gx} ${gy})`}
          fill={t != null ? relevanceColor(t) : CORPUS_COLOR}
          stroke={isRelated ? '#0f172a' : 'white'}
          strokeWidth={isRelated ? 1.8 : 0.8}
          className="cursor-pointer hover:brightness-110"
          onMouseEnter={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            setTip({
              x: rect.left + rect.width / 2,
              y: rect.top,
              label: p.name || 'Corpus project',
              sub:
                t != null
                  ? `Relevance ${(t * 100).toFixed(0)}% (relative) — click to view`
                  : isRelated
                    ? 'Related to your selection — click to view'
                    : 'Real project — click to view',
            });
          }}
          onMouseLeave={() => setTip(null)}
          onClick={(e) => {
            // Don't let the container's background-click handler deselect.
            e.stopPropagation();
            onSelectProject?.(p.id);
          }}
        />
      );
    });
    return (
      <>
        <defs>
          <radialGradient id="ds-related-glow">
            <stop offset="0%" stopColor={CORPUS_COLOR} stopOpacity={0.45} />
            <stop offset="60%" stopColor={CORPUS_COLOR} stopOpacity={0.15} />
            <stop offset="100%" stopColor={CORPUS_COLOR} stopOpacity={0} />
          </radialGradient>
        </defs>
        {glows}
        {glyphs}
      </>
    );
  }, [surface.points, cell, px, py, onSelectProject, relatedProjects, relevance]);

  // ── Spinner ring around the cell being generated ─────────────────────────
  const spinner = useMemo(() => {
    if (!isGenerating || !pendingCell) return null;
    const [gx, gy] = pendingCell;
    const sx = cx(gx);
    const sy = cy(gy);
    const ringR = cell * 0.95;
    const circ = 2 * Math.PI * ringR;
    return (
      <g>
        <circle cx={sx} cy={sy} r={ringR} fill="none" stroke="rgba(14,165,233,0.18)" strokeWidth={cell * 0.18} />
        <circle
          cx={sx}
          cy={sy}
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
            from={`0 ${sx} ${sy}`}
            to={`360 ${sx} ${sy}`}
            dur="0.9s"
            repeatCount="indefinite"
          />
        </circle>
      </g>
    );
  }, [isGenerating, pendingCell, cx, cy, cell]);

  const trustworthiness =
    typeof surface.meta?.trustworthiness === 'number' ? surface.meta.trustworthiness : null;

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full cursor-grab touch-none overflow-hidden active:cursor-grabbing"
      onPointerDown={onPointerDown}
      onClickCapture={onClickCapture}
      onClick={(e) => {
        // A click that didn't land on a dot (a <circle>; corpus <rect>s stop
        // propagation themselves) is empty space → deselect + dismiss chooser.
        if (movedRef.current) return;
        setChooser(null);
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
        {heatOpacity > 0 && (
          <g opacity={heatOpacity} pointerEvents="none">
            {heatDots}
          </g>
        )}
        {/* Lens heat-glow: blurred splats AT the real projects (kernel radius is
            small on purpose — a continuous interpolated plane would assert
            relevance values for empty regions, where the projection is least
            trustworthy; this stays anchored to actual corpus points). */}
        {relevance && (
          <>
            <defs>
              <filter id="ds-lens-blur" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation={cell * 1.2} />
              </filter>
            </defs>
            <g filter="url(#ds-lens-blur)" opacity={0.35} pointerEvents="none">
              {lensGlow}
            </g>
          </>
        )}
        {baseDots}
        {cellHighlights}

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

        {/* Corpus projects (real evidence) — between the lattice and node dots. */}
        {corpusGlyphs}

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

        {/* Node dots: colored by branch (shared with the mind map), snapped to the
            lattice. A dashed outline marks low placement confidence — the node's
            2D neighbourhood diverges from its true embedding neighbourhood. */}
        {placed.map((p) => {
          const color = nodeColor(p.branchIndex, p.depth);
          const isExact = exactMatchIds.has(p.id);
          const isMatch = matchIds.has(p.id);
          const isRejected = rejectedIds.has(p.id);
          const lowConfidence = p.confidence != null && p.confidence < LOW_CONFIDENCE;
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
                opacity={
                  lensActive && p.id !== lensAnchorId
                    ? 0.12 // the lens is about the corpus; non-anchor ideas recede
                    : isRejected
                      ? 0.15
                      : inSelected
                        ? 1
                        : 0.25
                }
                stroke={isExact ? '#0f172a' : isMatch ? '#475569' : 'white'}
                strokeWidth={isExact ? 2.5 : isMatch ? 2 : 1.5}
                strokeDasharray={lowConfidence ? '4 3' : undefined}
                className="cursor-pointer transition-[r] duration-100 hover:brightness-110"
                onMouseEnter={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  const kind =
                    p.depth === 0
                      ? 'Root'
                      : p.depth === 1
                        ? 'Branch'
                        : p.branchTopic
                          ? `Option in ${p.branchTopic}`
                          : 'Option';
                  const notes = [
                    isRejected ? 'rejected' : null,
                    lowConfidence ? 'placement approximate' : null,
                  ].filter(Boolean);
                  setTip({
                    x: rect.left + rect.width / 2,
                    y: rect.top,
                    label: p.topic,
                    sub: notes.length ? `${kind} · ${notes.join(' · ')}` : kind,
                  });
                }}
                onMouseLeave={() => setTip(null)}
                onClick={(e) => {
                  const colocated = cellNodes.get(`${p.gx},${p.gy}`) ?? [];
                  if (colocated.length > 1) {
                    // Several nodes snapped to this cell — let the user pick.
                    // Stop propagation: the container's click handler dismisses
                    // the chooser, which would cancel it in the same batch.
                    e.stopPropagation();
                    const rect = e.currentTarget.getBoundingClientRect();
                    setTip(null);
                    setChooser({
                      x: rect.left + rect.width / 2,
                      y: rect.top,
                      nodes: colocated,
                    });
                    return;
                  }
                  onSelectNode({
                    topic: p.topic,
                    lineage: lineageOf.get(p.id) ?? [p.topic],
                    nodeId: p.id,
                  });
                }}
              />
            </g>
          );
        })}

        {/* Collision badges: cells where several nodes snapped together. */}
        {[...cellNodes.entries()]
          .filter(([, list]) => list.length > 1)
          .map(([key, list]) => {
            const first = list[0]!;
            const bx = cx(first.gx) + cell * 0.45;
            const by = cy(first.gy) - cell * 0.45;
            return (
              <g key={`badge-${key}`} pointerEvents="none">
                <circle cx={bx} cy={by} r={cell * 0.3} fill="#0f172a" stroke="white" strokeWidth={1} />
                <text
                  x={bx}
                  y={by}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={cell * 0.4}
                  fill="white"
                  fontWeight={700}
                >
                  {list.length}
                </text>
              </g>
            );
          })}

        {/* Candidate designs: a composed choice-set is itself a point in the
            space — drawn as a star at the embedding of its combined text. */}
        {(candidates ?? []).map((c) => {
          const sx = px(c.x);
          const sy = py(c.y);
          const r = cell * (c.active ? 0.85 : 0.65);
          return (
            <path
              key={`cand-${c.id}`}
              d={starPath(sx, sy, r)}
              fill={CANDIDATE_COLOR}
              opacity={
                lensActive
                  ? `cand:${c.id}` === lensAnchorId
                    ? 1
                    : 0.12
                  : c.active
                    ? 1
                    : 0.55
              }
              stroke="white"
              strokeWidth={1.5}
              className="cursor-pointer hover:brightness-110"
              onMouseEnter={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                setTip({
                  x: rect.left + rect.width / 2,
                  y: rect.top,
                  label: c.name,
                  sub: c.active
                    ? 'Active candidate design'
                    : 'Candidate design — click to activate',
                });
              }}
              onMouseLeave={() => setTip(null)}
              onClick={(e) => {
                e.stopPropagation();
                onSelectCandidate?.(c.id);
              }}
            />
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

      {/* Collision chooser — pick which co-located node to select */}
      {chooser && (
        <div
          className="fixed z-50 -translate-x-1/2 -translate-y-full"
          style={{ left: chooser.x, top: chooser.y - 8 }}
          onPointerDown={(e) => e.stopPropagation()}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="max-w-[18rem] rounded-lg border bg-background/95 p-1.5 shadow-lg backdrop-blur">
            <p className="px-1.5 pb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              {chooser.nodes.length} nodes here
            </p>
            {chooser.nodes.map((n) => (
              <button
                key={n.id}
                type="button"
                className="flex w-full items-center gap-1.5 rounded-md px-1.5 py-1 text-left text-xs transition-colors hover:bg-muted"
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectNode({
                    topic: n.topic,
                    lineage: lineageOf.get(n.id) ?? [n.topic],
                    nodeId: n.id,
                  });
                  setChooser(null);
                }}
              >
                <span
                  className="h-2 w-2 shrink-0 rounded-full"
                  style={{ background: nodeColor(n.branchIndex, n.depth) }}
                />
                <span className="truncate">{n.topic}</span>
              </button>
            ))}
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
        {isGenerating && onCancelGenerate && (
          <button
            type="button"
            onClick={onCancelGenerate}
            className="pointer-events-auto flex items-center gap-1.5 rounded-lg border border-destructive/40 bg-background/90 px-2.5 py-1.5 text-[10px] font-semibold text-destructive shadow-sm backdrop-blur transition-colors hover:bg-destructive/10"
            title="Stop waiting for this generation"
          >
            Cancel generation
          </button>
        )}
        <div className="flex flex-col gap-1 rounded-lg border bg-background/80 px-3 py-2 text-[10px] text-muted-foreground shadow-sm backdrop-blur">
          {lensActive && (
            <div className="flex flex-col gap-0.5 border-b pb-1">
              <span
                className="h-2 w-full rounded-sm"
                style={{
                  background: `linear-gradient(to right, ${relevanceColor(0)}, ${relevanceColor(0.5)}, ${relevanceColor(1)})`,
                }}
              />
              <span>
                relevance to <span className="font-semibold">{lensAnchorLabel ?? 'selection'}</span>
              </span>
              {relevance && (
                <span title="Min-max normalized per query — the painting shows RELATIVE relevance">
                  relative (cos {relevance.min.toFixed(2)}–{relevance.max.toFixed(2)})
                </span>
              )}
            </div>
          )}
          <div className="flex items-center gap-1.5">
            <span
              className="inline-block h-2.5 w-2.5 rotate-45"
              style={{ background: CORPUS_COLOR }}
            />
            real project — click to view
          </div>
          {(relatedProjects?.size ?? 0) > 0 && (
            <div className="flex items-center gap-1.5">
              <span
                className="inline-block h-2.5 w-2.5 rotate-45 ring-2 ring-slate-900"
                style={{ background: CORPUS_COLOR }}
              />
              related to your selection
            </div>
          )}
          <div className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: 'rgba(148,163,184,0.5)' }} />
            empty — click to generate
          </div>
          <div className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: nodeColor(0, 1) }} />
            taxonomy node (dashed = approximate)
          </div>
          {(candidates?.length ?? 0) > 0 && (
            <div className="flex items-center gap-1.5">
              <svg viewBox="0 0 12 12" className="h-2.5 w-2.5">
                <path d={starPath(6, 6, 6)} fill={CANDIDATE_COLOR} />
              </svg>
              candidate design (your composition)
            </div>
          )}
          {trustworthiness != null && (
            <div
              className="mt-0.5 border-t pt-1"
              title="sklearn trustworthiness of the 2D layout vs the original embedding space — how much neighbourhood structure survived the projection"
            >
              layout fidelity: {trustworthiness.toFixed(2)} (trustworthiness)
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
