'use client';

import { AlertTriangle, Loader2, Trash2 } from 'lucide-react';
import { useMemo, useRef, useState } from 'react';
import type { MindmapNode } from '@/src/features/mindmap/types';
import { Button } from '@/src/components/ui/button';
import {
  candidateAlignmentAspects,
  candidateCoordKey,
  composeCandidateText,
  indexNodesById,
} from '@/src/features/design-space/candidate-utils';
import {
  buildConsistencyDefs,
  percentileOf,
  resolveRubricDefs,
  type MetricDef,
} from '@/src/features/design-space/examine-utils';
import { useAlignmentQuery } from '@/src/features/design-space/hooks/use-alignment-query';
import { useMetricsQuery } from '@/src/features/design-space/hooks/use-metrics-query';
import {
  useSteerMutation,
  type SteerResult,
} from '@/src/features/design-space/hooks/use-steer-mutation';
import type { MetricResult } from '@/src/features/design-space/types';
import { starPath } from '@/src/lib/svg-glyphs';
import { useMindmapStore } from '@/src/store/mindmap-store';
import { SteerResultCard } from './steer-result-card';

const CANDIDATE_COLOR = '#7c3aed';
const POLE_SIM_WARN = 0.85;
const CORR_WARN = 0.6;
// Strip geometry: the track SVG stretches to fill its row (only lines live in
// it — stretch-safe); the star and labels are HTML overlays positioned by
// percentage so they NEVER distort. Score -1..1 → 0..100%.
const TRACK_W = 1000;
const TRACK_H = 32;

const xOf = (score: number) => ((score + 1) / 2) * TRACK_W;
const pctOf = (score: number) => `${(((score + 1) / 2) * 100).toFixed(2)}%`;

/** One metric strip: corpus rug + the candidate's brief as a star on the line.
 * With ``onRailChange`` the track is a STEERING RAIL (Part 12 B3): click OR
 * drag sets/moves the target score; ``railTarget`` renders the requested
 * ghost, and the Steer/Cancel controls live INSIDE this card so target and
 * commitment read as one object. */
function StripRow({
  def,
  result,
  onRemove,
  railTarget,
  onRailChange,
  steering,
  onSteerCommit,
  onRailCancel,
}: {
  def: MetricDef;
  result: MetricResult | undefined;
  onRemove?: () => void;
  railTarget?: number | null;
  onRailChange?: (score: number) => void;
  steering?: boolean;
  onSteerCommit?: () => void;
  onRailCancel?: () => void;
}) {
  const item = result?.items[0];
  const pct = result && item ? percentileOf(result.corpus, item.score) : null;
  // Drag-to-aim: pointer capture keeps the ghost following even when the
  // cursor leaves the row mid-drag. A ref, not state — it drives no render,
  // and a ref stays correct when down/move land in the same frame.
  const draggingRef = useRef(false);
  const scoreAt = (e: React.PointerEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const fraction = (e.clientX - rect.left) / rect.width;
    return Math.max(-1, Math.min(1, fraction * 2 - 1));
  };
  return (
    <div className="rounded-xl border bg-background/80 px-4 py-3">
      <div className="flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <span className="truncate text-xs font-semibold">{def.label}</span>
          <span className="shrink-0 rounded-full bg-muted px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-muted-foreground">
            {def.kind === 'consistency' ? 'your choice ↔ closest alternative' : 'rubric'}
          </span>
          {def.leansAway && (
            <span
              className="flex shrink-0 items-center gap-1 rounded-full bg-amber-500/15 px-1.5 py-0.5 text-[9px] font-semibold text-amber-700"
              title="The brief reads closer to the rejected alternative than to your chosen option — edit the brief, revisit the choice, or embrace the hybrid"
            >
              <AlertTriangle className="h-2.5 w-2.5" />
              leans to the alternative
            </span>
          )}
        </div>
        {onRemove && (
          <button
            type="button"
            onClick={onRemove}
            className="text-muted-foreground transition-colors hover:text-destructive"
            title="Remove this metric from the rubric"
          >
            <Trash2 className="h-3 w-3" />
          </button>
        )}
      </div>

      {result && item ? (
        <>
          <div
            className={`relative mt-2 h-8 touch-none ${onRailChange ? 'cursor-crosshair' : ''}`}
            onPointerDown={
              onRailChange
                ? (e) => {
                    try {
                      e.currentTarget.setPointerCapture(e.pointerId);
                    } catch {
                      // A pointer released between event and capture (or a
                      // synthetic event) — the drag still works inside the row.
                    }
                    draggingRef.current = true;
                    onRailChange(scoreAt(e));
                  }
                : undefined
            }
            onPointerMove={
              onRailChange ? (e) => draggingRef.current && onRailChange(scoreAt(e)) : undefined
            }
            onPointerUp={() => {
              draggingRef.current = false;
            }}
            onPointerCancel={() => {
              draggingRef.current = false;
            }}
            title={
              onRailChange
                ? 'Click or drag along the rail to aim the brief at a score'
                : undefined
            }
          >
            {/* Track + corpus rug: lines only, so non-uniform stretch is safe. */}
            <svg
              viewBox={`0 0 ${TRACK_W} ${TRACK_H}`}
              preserveAspectRatio="none"
              className="absolute inset-0 h-full w-full"
            >
              <line x1={0} y1={TRACK_H / 2} x2={TRACK_W} y2={TRACK_H / 2} stroke="rgba(100,116,139,0.35)" strokeWidth={2} />
              <line x1={TRACK_W / 2} y1={4} x2={TRACK_W / 2} y2={TRACK_H - 4} stroke="rgba(100,116,139,0.3)" strokeWidth={1} />
              {result.corpus.map((score, i) => (
                <line
                  key={i}
                  x1={xOf(score)}
                  y1={7}
                  x2={xOf(score)}
                  y2={TRACK_H - 7}
                  stroke="rgba(244,140,43,0.28)"
                  strokeWidth={1.5}
                />
              ))}
            </svg>
            {/* The candidate's star: an undistorted overlay positioned by %.
                Clipped scores render at reduced opacity with a tooltip — a
                dashed outline at this size (20px) breaks into dots that read
                as a rendering artifact, not as a meaning. Position clamps to
                the rail so an out-of-range score can't hang off the row. */}
            <svg
              viewBox="0 0 24 24"
              className="absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2"
              style={{ left: pctOf(Math.max(-1, Math.min(1, item.score))) }}
              opacity={item.clipped ? 0.7 : 1}
            >
              {item.clipped && (
                <title>outside the corpus range — shown at the rail edge</title>
              )}
              <path
                d={starPath(12, 12, 10)}
                fill={CANDIDATE_COLOR}
                stroke="white"
                strokeWidth={1.5}
              />
            </svg>
            {/* The requested ghost: where the designer asked the brief to go. */}
            {railTarget != null && (
              <svg
                viewBox="0 0 24 24"
                className="pointer-events-none absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2"
                style={{ left: pctOf(railTarget) }}
              >
                <path
                  d={starPath(12, 12, 10)}
                  fill="none"
                  stroke={CANDIDATE_COLOR}
                  strokeWidth={1.5}
                  strokeDasharray="3 2"
                />
              </svg>
            )}
          </div>
          <div className="flex items-center justify-between gap-4 text-[10px] text-muted-foreground">
            <span className="truncate">← {def.poleBLabel}</span>
            <span className="truncate text-right">{def.poleALabel} →</span>
          </div>
          {/* The aimed target's controls live IN the card, right under the
              rail they refer to — target and commitment as one object. */}
          {railTarget != null &&
            (steering ? (
              <p className="mt-1.5 flex items-center gap-1.5 text-[11px] text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" />
                revising in language, then measuring the move…
              </p>
            ) : (
              <div className="mt-1.5 flex items-center gap-1.5 rounded-lg border border-dashed border-violet-300 bg-violet-500/5 px-2 py-1 text-[11px]">
                <span className="min-w-0 flex-1 truncate text-muted-foreground">
                  steer the brief to {railTarget >= 0 ? '+' : ''}
                  {railTarget.toFixed(2)} (toward{' '}
                  {railTarget >= 0 ? def.poleALabel : def.poleBLabel})
                </span>
                <Button size="sm" className="h-5 shrink-0 px-2 text-[10px]" onClick={onSteerCommit}>
                  Steer
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-5 shrink-0 px-2 text-[10px]"
                  onClick={onRailCancel}
                >
                  Cancel
                </Button>
              </div>
            ))}
          <p
            className="mt-0.5 text-[10px] text-muted-foreground"
            title="A percentile within the background corpus of real projects, not an absolute scale — the tick marks above are those projects' own scores"
          >
            {pct != null && !Number.isNaN(pct)
              ? `more "${def.poleALabel}" than ${(pct * 100).toFixed(0)}% of the real projects in the corpus`
              : 'no corpus distribution available'}
            {item.clipped ? ' · outside the corpus range' : ''}
          </p>
          {result.pole_sim > POLE_SIM_WARN && (
            <p className="mt-0.5 flex items-center gap-1 text-[10px] text-amber-700">
              <AlertTriangle className="h-2.5 w-2.5" />
              poles are {(result.pole_sim * 100).toFixed(0)}% similar — this metric barely separates anything
            </p>
          )}
        </>
      ) : (
        <p className="mt-1 text-[10px] text-muted-foreground">scoring…</p>
      )}
    </div>
  );
}

export function Teach({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed bg-background/60 px-4 py-6 text-center text-xs text-muted-foreground">
      {children}
    </div>
  );
}

/**
 * The examine instrument's core (Part 10 I3, extracted for Part 12 B1): the
 * active candidate's brief measured against its own commitments (one
 * consistency strip per choice) and the project's rubric metrics, with the
 * agreement headline and quality warnings. Shared between the Perspectives
 * document view and the Design Space inspector dock, so a candidate can be
 * examined without leaving the map.
 */
export function CandidateStrips({
  nodes,
  descriptionByTopic,
  onProposeQualities,
}: {
  nodes: ReadonlyArray<MindmapNode>;
  descriptionByTopic: Readonly<Record<string, string>>;
  /** C1 informing-back: an applied steer's named qualities become option
   * proposals (aspectId known for consistency/rubric strips). */
  onProposeQualities?: (
    qualities: string[],
    aspectId: string | null,
    evidence: string
  ) => void;
}) {
  const candidates = useMindmapStore((s) => s.candidates);
  const activeCandidateId = useMindmapStore((s) => s.activeCandidateId);
  const descriptionById = useMindmapStore((s) => s.descriptionById);
  const rubric = useMindmapStore((s) => s.rubric);
  const removeRubricMetric = useMindmapStore((s) => s.removeRubricMetric);
  const setCandidateBrief = useMindmapStore((s) => s.setCandidateBrief);
  const trackUsage = useMindmapStore((s) => s.trackUsage);
  const recordEvent = useMindmapStore((s) => s.recordEvent);

  const active = activeCandidateId ? candidates[activeCandidateId] ?? null : null;
  const brief = active?.brief?.trim() || null;

  // B3 strip rails: clicking a strip's track arms a target; the steer runs
  // only from the confirm chip, and its result is ALWAYS a veto card. The
  // outcome snapshots the candidate AND the steered brief, so a candidate
  // switch mid-veto can neither show the wrong "before" nor apply to the
  // wrong design.
  const [rail, setRail] = useState<{ key: string; target: number } | null>(null);
  const [steerOutcome, setSteerOutcome] = useState<{
    key: string;
    candidateId: string;
    briefBefore: string;
    result: SteerResult;
  } | null>(null);
  const [steerError, setSteerError] = useState<string | null>(null);
  const { mutateAsync: steerBrief, isPending: steering } = useSteerMutation();
  // The committed choices steering must not weaken (self-imposed constraints).
  const preserveNames = useMemo(() => {
    if (!active) return [];
    const byId = indexNodesById(nodes);
    return Object.values(active.choices)
      .map((optionId) => byId.get(optionId)?.topic)
      .filter((topic): topic is string => Boolean(topic));
  }, [active, nodes]);

  const handleSteer = async (def: MetricDef, target: number) => {
    if (!brief || !active) return;
    const steeredId = active.id;
    const steeredBrief = brief;
    setSteerError(null);
    try {
      const result = await steerBrief({
        text: steeredBrief,
        mode: 'metric',
        metric: { pole_a_text: def.poleAText, pole_b_text: def.poleBText, target_score: target },
        // The backend caps preserve at 12 — send the most recent commitments.
        preserve: preserveNames.slice(0, 12),
      });
      setSteerOutcome({ key: def.key, candidateId: steeredId, briefBefore: steeredBrief, result });
      trackUsage('steer_run');
    } catch (error) {
      setSteerError(error instanceof Error ? error.message : 'steering failed');
    } finally {
      setRail(null);
    }
  };

  const composition = useMemo(
    () => composeCandidateText(active, nodes, descriptionByTopic, descriptionById),
    [active, nodes, descriptionByTopic, descriptionById]
  );
  const alignmentAspects = useMemo(
    () => candidateAlignmentAspects(active, nodes, descriptionByTopic, descriptionById),
    [active, nodes, descriptionByTopic, descriptionById]
  );
  const { data: alignment, error: alignmentError } = useAlignmentQuery(
    brief && composition && alignmentAspects.length > 0
      ? { brief, composition, aspects: alignmentAspects }
      : null
  );

  const defs = useMemo<MetricDef[]>(
    () => [
      ...buildConsistencyDefs(
        alignment,
        active?.choices ?? {},
        nodes,
        descriptionByTopic,
        descriptionById
      ),
      ...resolveRubricDefs(rubric, nodes, descriptionByTopic, descriptionById),
    ],
    [alignment, active?.choices, rubric, nodes, descriptionByTopic, descriptionById]
  );
  const { data: metrics, error: metricsError } = useMetricsQuery(
    brief && active && defs.length > 0
      ? {
          metrics: defs.map((d) => ({ poleA: d.poleAText, poleB: d.poleBText })),
          items: [{ node_id: candidateCoordKey(active.id), text: brief }],
        }
      : null
  );

  // Cheap per-render derivations (≤12 metrics) — no memoization needed.
  // Largest divergence: the aspect where the brief most out-scores the choice.
  const worstGap = (alignment?.per_aspect ?? [])
    .map((row) => ({
      aspectId: row.aspect_id,
      gap: row.top_alternative ? row.top_alternative.score - row.chosen_score : 0,
    }))
    .filter((row) => row.gap > 0)
    .sort((a, b) => b.gap - a.gap)[0];
  const largestDivergence = worstGap
    ? defs.find((d) => d.key === `consistency:${worstGap.aspectId}`)?.label ?? null
    : null;

  // Redundancy among RUBRIC strips (consistency strips are allowed to overlap).
  const redundantPair = defs.flatMap((a, i) =>
    defs.flatMap((b, j) => {
      if (j <= i || a.kind !== 'rubric' || b.kind !== 'rubric') return [];
      const r = metrics?.corr[i]?.[j] ?? 0;
      return Math.abs(r) > CORR_WARN ? [{ a: a.label, b: b.label, r }] : [];
    })
  )[0];

  if (!active) {
    return (
      <Teach>
        Compose a candidate first (Candidate panel) — examination measures a
        candidate against your metrics.
      </Teach>
    );
  }
  if (!brief) {
    return (
      <Teach>
        Write or draft <span className="font-semibold">{active.name}</span>
        &apos;s brief in the Candidate panel — examination measures the brief
        (the actual design), not the choice list.
      </Teach>
    );
  }

  return (
    <>
      {/* Headline: concept ↔ commitments agreement */}
      {alignment && (
        <div className="rounded-xl border bg-background/80 px-4 py-3">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Concept ↔ commitments
          </p>
          <p className="mt-0.5 text-sm">
            The brief matches the composed choices{' '}
            <span className="font-semibold tabular-nums">
              {(alignment.agreement * 100).toFixed(0)}%
            </span>
            {largestDivergence ? (
              <>
                {' '}— largest divergence on{' '}
                <span className="font-semibold">{largestDivergence}</span>
              </>
            ) : (
              ' — every choice is expressed in the brief'
            )}
          </p>
        </div>
      )}
      {alignmentError && (
        <p className="text-[11px] text-destructive">{alignmentError.message}</p>
      )}
      {metricsError && (
        <p className="text-[11px] text-destructive">{metricsError.message}</p>
      )}
      {redundantPair && (
        <p className="flex items-center gap-1 text-[10px] text-amber-700">
          <AlertTriangle className="h-3 w-3" />
          &quot;{redundantPair.a}&quot; and &quot;{redundantPair.b}&quot; agree{' '}
          {(Math.abs(redundantPair.r) * 100).toFixed(0)}% of the time — redundant metrics
        </p>
      )}

      {steerError && <p className="text-[11px] text-destructive">steering failed: {steerError}</p>}

      {/* Strips — each track is a steering rail (B3) */}
      {defs.map((def, i) => (
        <div key={def.key} className="space-y-1.5">
          <StripRow
            def={def}
            result={metrics?.metrics[i]}
            onRemove={def.rubricId ? () => removeRubricMetric(def.rubricId!) : undefined}
            railTarget={rail?.key === def.key ? rail.target : null}
            onRailChange={
              steering ? undefined : (score) => setRail({ key: def.key, target: score })
            }
            steering={steering && rail?.key === def.key}
            onSteerCommit={() => rail && handleSteer(def, rail.target)}
            onRailCancel={() => setRail(null)}
          />
          {steerOutcome?.key === def.key && steerOutcome.candidateId === active?.id && (
            <SteerResultCard
              result={steerOutcome.result}
              briefBefore={steerOutcome.briefBefore}
              onApply={() => {
                setCandidateBrief(steerOutcome.candidateId, steerOutcome.result.revised_text);
                recordEvent(
                  'steer_applied',
                  `Steered "${active?.name ?? 'candidate'}" along ${def.label}`,
                  [steerOutcome.candidateId]
                );
                if (steerOutcome.result.named_qualities.length > 0) {
                  const aspectId = def.key.startsWith('consistency:')
                    ? def.key.slice('consistency:'.length)
                    : (rubric.find((m) => m.id === def.rubricId)?.aspectId ?? null);
                  onProposeQualities?.(
                    steerOutcome.result.named_qualities,
                    aspectId,
                    `steering along ${def.label}`
                  );
                }
                trackUsage('steer_applied');
                setSteerOutcome(null);
              }}
              onDiscard={() => setSteerOutcome(null)}
            />
          )}
        </div>
      ))}
      {defs.length === 0 && (
        <Teach>
          Choose options for this candidate (consistency strips appear per
          choice) or add a rubric metric in Perspectives.
        </Teach>
      )}
    </>
  );
}
