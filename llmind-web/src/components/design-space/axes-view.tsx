'use client';

import { Loader2, RotateCcw } from 'lucide-react';
import { useMemo, useState } from 'react';
import type {
  AxesConfig,
  AxisEndConfig,
  MindmapNode,
  MindmapSelection,
} from '@/src/features/mindmap/types';
import {
  candidateCoordKey,
  candidateEmbeddingText,
  indexNodesById,
  listAspects,
} from '@/src/features/design-space/candidate-utils';
import { useAxesQuery } from '@/src/features/design-space/hooks/use-axes-query';
import { usePanZoom } from '@/src/features/design-space/hooks/use-pan-zoom';
import { nodeColor } from '@/src/lib/node-colors';
import { starPath } from '@/src/lib/svg-glyphs';
import { useMindmapStore } from '@/src/store/mindmap-store';

const VIEW = 1000;
const MARGIN = 110; // room for pole labels around the plot
const PLOT = VIEW - 2 * MARGIN;
const CORPUS_COLOR = 'rgba(244,140,43,0.9)';
const CANDIDATE_COLOR = '#7c3aed';
const POLE_SIM_WARN = 0.85;
const AXIS_CORR_WARN = 0.6;

interface AxesViewProps {
  nodes: ReadonlyArray<MindmapNode>;
  selection: MindmapSelection;
  onSelectNode: (selection: MindmapSelection) => void;
  /** Open a corpus project in the Related Projects panel. */
  onSelectProject: (projectId: string) => void;
  descriptionByTopic: Readonly<Record<string, string>>;
}

/** Aspect/option lookup rows for the axis pickers. */
interface AspectRow {
  aspect: MindmapNode;
  branchIndex: number;
  options: MindmapNode[];
}

function defaultAxisEnd(row: AspectRow): AxisEndConfig {
  // Default poles: first vs last option — a cheap stand-in for "most distant";
  // the pole-similarity diagnostic flags bad defaults and the user can override.
  return {
    aspectId: row.aspect.id,
    poleAId: row.options[row.options.length - 1]!.id,
    poleBId: row.options[0]!.id,
  };
}

/**
 * Semantic-axes perspective: a bipolar scatterplot whose axes the designer
 * picks from their own taxonomy. Scores are exact (original embedding metric,
 * no projection), so an empty region genuinely means "no precedent is both".
 * Read-only v1: inspect and compare; generation stays in the Design Space view.
 */
export function AxesView({
  nodes,
  selection,
  onSelectNode,
  onSelectProject,
  descriptionByTopic,
}: AxesViewProps) {
  const axesConfig = useMindmapStore((s) => s.axesConfig);
  const setAxesConfig = useMindmapStore((s) => s.setAxesConfig);
  const descriptionById = useMindmapStore((s) => s.descriptionById);
  const candidates = useMindmapStore((s) => s.candidates);
  const activeCandidateId = useMindmapStore((s) => s.activeCandidateId);
  const optionState = useMindmapStore((s) => s.optionState);

  const [tip, setTip] = useState<{ x: number; y: number; label: string; sub?: string } | null>(
    null
  );
  const { containerRef, view, onPointerDown, onClickCapture, resetView } = usePanZoom(() =>
    setTip(null)
  );

  const byId = useMemo(() => indexNodesById(nodes), [nodes]);
  const aspectRows = useMemo<AspectRow[]>(
    () =>
      listAspects(nodes)
        .map((aspect, i) => ({
          aspect,
          branchIndex: i,
          options: [...(aspect.children ?? [])],
        }))
        .filter((row) => row.options.length >= 2),
    [nodes]
  );

  // Resolve the persisted config against the current tree; fall back to defaults.
  const config = useMemo<AxesConfig | null>(() => {
    if (aspectRows.length === 0) return null;
    const resolveEnd = (end: AxisEndConfig | undefined, fallback: AspectRow): AxisEndConfig => {
      if (!end) return defaultAxisEnd(fallback);
      const row = aspectRows.find((r) => r.aspect.id === end.aspectId);
      if (!row) return defaultAxisEnd(fallback);
      const hasA = row.options.some((o) => o.id === end.poleAId);
      const hasB = row.options.some((o) => o.id === end.poleBId);
      return hasA && hasB ? end : defaultAxisEnd(row);
    };
    const xRow = aspectRows[0]!;
    const yRow = aspectRows[1] ?? aspectRows[0]!;
    return {
      x: resolveEnd(axesConfig?.x, xRow),
      y: resolveEnd(axesConfig?.y, yRow),
    };
  }, [axesConfig, aspectRows]);

  const poleText = (optionId: string): string => {
    const option = byId.get(optionId);
    if (!option) return '';
    const desc = descriptionById[option.id] ?? descriptionByTopic[option.topic] ?? '';
    return desc ? `${option.topic}. ${desc}` : option.topic;
  };
  const poleTopic = (optionId: string): string => byId.get(optionId)?.topic ?? '?';

  // Items shown on the plot: the two chosen aspects' options + candidates.
  const queryParams = useMemo(() => {
    if (!config) return null;
    const xRow = aspectRows.find((r) => r.aspect.id === config.x.aspectId);
    const yRow = aspectRows.find((r) => r.aspect.id === config.y.aspectId);
    if (!xRow || !yRow) return null;
    const optionIds = new Set<string>();
    for (const row of [xRow, yRow]) for (const o of row.options) optionIds.add(o.id);
    const items: Array<{ node_id: string; text: string }> = [];
    for (const id of optionIds) {
      const text = poleText(id);
      if (text) items.push({ node_id: id, text });
    }
    for (const candidate of Object.values(candidates)) {
      // Brief-first (Part 10): the star is the design, not the choice list.
      const text = candidateEmbeddingText(candidate, nodes, descriptionByTopic, descriptionById);
      if (text) items.push({ node_id: candidateCoordKey(candidate.id), text });
    }
    const xa = poleText(config.x.poleAId);
    const xb = poleText(config.x.poleBId);
    const ya = poleText(config.y.poleAId);
    const yb = poleText(config.y.poleBId);
    if (!xa || !xb || !ya || !yb) return null;
    return { xPoleA: xa, xPoleB: xb, yPoleA: ya, yPoleB: yb, items };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config, aspectRows, candidates, nodes, descriptionByTopic, descriptionById]);

  const { data, isFetching, error } = useAxesQuery(queryParams);

  // ── Geometry: normalized [-1, 1] → SVG (y up) ───────────────────────────────
  const mapX = (v: number) => MARGIN + ((v + 1) / 2) * PLOT;
  const mapY = (v: number) => VIEW - MARGIN - ((v + 1) / 2) * PLOT;

  // Quadrant density shading (corpus counts per quadrant).
  const quadrants = useMemo(() => {
    const counts = [0, 0, 0, 0]; // [+x+y, -x+y, -x-y, +x-y]
    for (const p of data?.corpus ?? []) {
      const i = p.x >= 0 ? (p.y >= 0 ? 0 : 3) : p.y >= 0 ? 1 : 2;
      counts[i]!++;
    }
    const max = Math.max(1, ...counts);
    return counts.map((c) => c / max);
  }, [data]);

  const itemById = useMemo(() => {
    const m = new Map<string, { x: number; y: number; clipped: boolean }>();
    for (const it of data?.items ?? []) m.set(it.node_id, it);
    return m;
  }, [data]);

  const xRow = config ? aspectRows.find((r) => r.aspect.id === config.x.aspectId) : undefined;
  const yRow = config ? aspectRows.find((r) => r.aspect.id === config.y.aspectId) : undefined;

  const updateEnd = (axis: 'x' | 'y', patch: Partial<AxisEndConfig>) => {
    if (!config) return;
    let end = { ...config[axis], ...patch };
    if (patch.aspectId) {
      const row = aspectRows.find((r) => r.aspect.id === patch.aspectId);
      if (row) end = defaultAxisEnd(row);
    }
    setAxesConfig({ ...config, [axis]: end });
  };

  if (aspectRows.length === 0) {
    return (
      <div className="flex h-full w-full items-center justify-center p-8 text-center">
        <p className="max-w-md text-sm text-muted-foreground">
          The Perspectives view needs at least one aspect with two or more options.
          Generate a taxonomy first.
        </p>
      </div>
    );
  }

  const selectClass =
    'h-6 max-w-[10rem] truncate rounded-md border bg-background px-1 text-[11px]';

  const axisPicker = (axis: 'x' | 'y', row: AspectRow | undefined) => {
    if (!config || !row) return null;
    const end = config[axis];
    return (
      <div className="flex items-center gap-1">
        <span className="text-[10px] font-bold uppercase text-muted-foreground">{axis}</span>
        <select
          className={selectClass}
          value={end.aspectId}
          onChange={(e) => updateEnd(axis, { aspectId: e.target.value })}
          aria-label={`${axis} axis aspect`}
        >
          {aspectRows.map((r) => (
            <option key={r.aspect.id} value={r.aspect.id}>
              {r.aspect.topic}
            </option>
          ))}
        </select>
        <select
          className={selectClass}
          value={end.poleBId}
          onChange={(e) => updateEnd(axis, { poleBId: e.target.value })}
          aria-label={`${axis} axis negative pole`}
        >
          {row.options.map((o) => (
            <option key={o.id} value={o.id}>
              {o.topic}
            </option>
          ))}
        </select>
        <span className="text-[10px] text-muted-foreground">↔</span>
        <select
          className={selectClass}
          value={end.poleAId}
          onChange={(e) => updateEnd(axis, { poleAId: e.target.value })}
          aria-label={`${axis} axis positive pole`}
        >
          {row.options.map((o) => (
            <option key={o.id} value={o.id}>
              {o.topic}
            </option>
          ))}
        </select>
      </div>
    );
  };

  const warnings: string[] = [];
  if (data) {
    if (data.meta.x_pole_sim > POLE_SIM_WARN)
      warnings.push(
        `X poles are very similar (cos ${data.meta.x_pole_sim.toFixed(2)}) — the axis collapses; pick more contrasting options.`
      );
    if (data.meta.y_pole_sim > POLE_SIM_WARN)
      warnings.push(
        `Y poles are very similar (cos ${data.meta.y_pole_sim.toFixed(2)}) — the axis collapses; pick more contrasting options.`
      );
    if (Math.abs(data.meta.axis_corr) > AXIS_CORR_WARN)
      warnings.push(
        `The axes overlap (r ${data.meta.axis_corr.toFixed(2)}) — points hug the diagonal; these dimensions are not independent here.`
      );
  }

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full cursor-grab touch-none overflow-hidden active:cursor-grabbing"
      onPointerDown={onPointerDown}
      onClickCapture={onClickCapture}
      onMouseLeave={() => setTip(null)}
    >
      {/* Axis pickers */}
      <div className="absolute top-4 left-1/2 z-40 flex -translate-x-1/2 flex-col items-center gap-1">
        <div className="flex flex-wrap items-center gap-3 rounded-xl border bg-background/90 px-3 py-1.5 shadow-md backdrop-blur">
          {axisPicker('x', xRow)}
          <span className="h-4 w-px bg-border" />
          {axisPicker('y', yRow)}
          {isFetching && <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />}
        </div>
        {warnings.map((w) => (
          <p
            key={w}
            className="max-w-xl rounded-md bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-700"
          >
            {w}
          </p>
        ))}
        {error ? (
          <p className="rounded-md bg-destructive/10 px-2 py-0.5 text-[10px] font-medium text-destructive">
            {error instanceof Error ? error.message : 'Axes unavailable.'}
          </p>
        ) : null}
      </div>

      <svg
        viewBox={`0 0 ${VIEW} ${VIEW}`}
        className="h-full w-full"
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Semantic axes perspective"
        style={{
          transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.k})`,
          transformOrigin: '0 0',
        }}
      >
        {/* Quadrant density shading: [+x+y, -x+y, -x-y, +x-y] */}
        {config &&
          ([
            [mapX(0), MARGIN, quadrants[0]!],
            [MARGIN, MARGIN, quadrants[1]!],
            [MARGIN, mapY(0), quadrants[2]!],
            [mapX(0), mapY(0), quadrants[3]!],
          ] as const).map(([qx, qy, t], i) => (
            <rect
              key={`q-${i}`}
              x={qx}
              y={qy}
              width={PLOT / 2}
              height={PLOT / 2}
              fill={`rgba(100,116,139,${0.03 + 0.09 * t})`}
            />
          ))}

        {/* Axis cross + frame */}
        <rect
          x={MARGIN}
          y={MARGIN}
          width={PLOT}
          height={PLOT}
          fill="none"
          stroke="rgba(148,163,184,0.4)"
        />
        <line x1={mapX(0)} y1={MARGIN} x2={mapX(0)} y2={VIEW - MARGIN} stroke="rgba(148,163,184,0.6)" strokeDasharray="4 4" />
        <line x1={MARGIN} y1={mapY(0)} x2={VIEW - MARGIN} y2={mapY(0)} stroke="rgba(148,163,184,0.6)" strokeDasharray="4 4" />

        {/* Rug ticks: marginal distributions of the corpus */}
        <g stroke="rgba(244,140,43,0.45)" strokeWidth={1.5}>
          {(data?.corpus ?? []).map((p) => (
            <line
              key={`rx-${p.id}`}
              x1={mapX(p.x)}
              y1={VIEW - MARGIN + 4}
              x2={mapX(p.x)}
              y2={VIEW - MARGIN + 14}
            />
          ))}
          {(data?.corpus ?? []).map((p) => (
            <line
              key={`ry-${p.id}`}
              x1={MARGIN - 14}
              y1={mapY(p.y)}
              x2={MARGIN - 4}
              y2={mapY(p.y)}
            />
          ))}
        </g>

        {/* Pole labels */}
        {config && (
          <g className="select-none" fill="#475569" fontWeight={600}>
            <text x={VIEW - MARGIN} y={mapY(0) - 10} textAnchor="end" fontSize={17}>
              {poleTopic(config.x.poleAId)} →
            </text>
            <text x={MARGIN} y={mapY(0) - 10} textAnchor="start" fontSize={17}>
              ← {poleTopic(config.x.poleBId)}
            </text>
            <text x={mapX(0) + 10} y={MARGIN - 12} textAnchor="start" fontSize={17}>
              ↑ {poleTopic(config.y.poleAId)}
            </text>
            <text x={mapX(0) + 10} y={VIEW - MARGIN + 28} textAnchor="start" fontSize={17}>
              ↓ {poleTopic(config.y.poleBId)}
            </text>
          </g>
        )}

        {/* Corpus diamonds */}
        {(data?.corpus ?? []).map((p) => {
          const gx = mapX(p.x);
          const gy = mapY(p.y);
          const s = 7;
          return (
            <rect
              key={`c-${p.id}`}
              x={gx - s}
              y={gy - s}
              width={s * 2}
              height={s * 2}
              transform={`rotate(45 ${gx} ${gy})`}
              fill={CORPUS_COLOR}
              stroke="white"
              strokeWidth={0.8}
              className="cursor-pointer hover:brightness-110"
              onMouseEnter={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                setTip({
                  x: rect.left + rect.width / 2,
                  y: rect.top,
                  label: 'Corpus project',
                  sub: 'Real project — click to view',
                });
              }}
              onMouseLeave={() => setTip(null)}
              onClick={(e) => {
                e.stopPropagation();
                onSelectProject(p.id);
              }}
            />
          );
        })}

        {/* Options of the two chosen aspects */}
        {config &&
          [xRow, yRow]
            .filter((row): row is AspectRow => Boolean(row))
            .filter((row, i, arr) => arr.findIndex((r) => r.aspect.id === row.aspect.id) === i)
            .flatMap((row) =>
              row.options.map((option) => {
                const coord = itemById.get(option.id);
                if (!coord) return null;
                const gx = mapX(coord.x);
                const gy = mapY(coord.y);
                const isSelected = selection.nodeId === option.id;
                const isRejected = Boolean(optionState[option.id]);
                return (
                  <circle
                    key={`o-${option.id}`}
                    cx={gx}
                    cy={gy}
                    r={isSelected ? 13 : 10}
                    fill={nodeColor(row.branchIndex, 2)}
                    opacity={isRejected ? 0.2 : 1}
                    stroke={isSelected ? '#0f172a' : 'white'}
                    strokeWidth={isSelected ? 2.5 : 1.5}
                    strokeDasharray={coord.clipped ? '4 3' : undefined}
                    className="cursor-pointer hover:brightness-110"
                    onMouseEnter={(e) => {
                      const rect = e.currentTarget.getBoundingClientRect();
                      const notes = [
                        isRejected ? 'rejected' : null,
                        coord.clipped ? 'outside corpus range' : null,
                      ].filter(Boolean);
                      setTip({
                        x: rect.left + rect.width / 2,
                        y: rect.top,
                        label: option.topic,
                        sub: notes.length
                          ? `Option in ${row.aspect.topic} · ${notes.join(' · ')}`
                          : `Option in ${row.aspect.topic}`,
                      });
                    }}
                    onMouseLeave={() => setTip(null)}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectNode({
                        topic: option.topic,
                        lineage: [nodes[0]?.topic ?? '', row.aspect.topic, option.topic].filter(
                          Boolean
                        ),
                        nodeId: option.id,
                      });
                    }}
                  />
                );
              })
            )}

        {/* Candidate stars */}
        {Object.values(candidates).map((candidate) => {
          const coord = itemById.get(candidateCoordKey(candidate.id));
          if (!coord) return null;
          const gx = mapX(coord.x);
          const gy = mapY(coord.y);
          const active = candidate.id === activeCandidateId;
          return (
            <path
              key={`cand-${candidate.id}`}
              d={starPath(gx, gy, active ? 16 : 12)}
              fill={CANDIDATE_COLOR}
              opacity={active ? 1 : 0.55}
              stroke="white"
              strokeWidth={1.5}
              strokeDasharray={coord.clipped ? '4 3' : undefined}
              className="cursor-default"
              onMouseEnter={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                setTip({
                  x: rect.left + rect.width / 2,
                  y: rect.top,
                  label: candidate.name,
                  sub: coord.clipped
                    ? 'Candidate design · outside corpus range'
                    : 'Candidate design',
                });
              }}
              onMouseLeave={() => setTip(null)}
            />
          );
        })}
      </svg>

      {/* Hover tooltip */}
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

      {/* Reset + legend */}
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
            <span className="inline-block h-2.5 w-2.5 rotate-45" style={{ background: CORPUS_COLOR }} />
            real project — click to view
          </div>
          <div className="flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: nodeColor(0, 2) }} />
            option of a chosen aspect (dashed = outside corpus range)
          </div>
          <div className="flex items-center gap-1.5">
            <svg viewBox="0 0 12 12" className="h-2.5 w-2.5">
              <path d={starPath(6, 6, 6)} fill={CANDIDATE_COLOR} />
            </svg>
            candidate design
          </div>
          <div className="mt-0.5 border-t pt-1" title="Bipolar cosine scores in the original embedding metric — no projection, no distortion">
            exact by construction — scaled to this corpus
          </div>
        </div>
      </div>
    </div>
  );
}
