'use client';

import { AlertTriangle, Plus, Trash2 } from 'lucide-react';
import { useMemo, useState } from 'react';
import type { MindmapNode, MindmapSelection } from '@/src/features/mindmap/types';
import {
  candidateAlignmentAspects,
  candidateCoordKey,
  composeCandidateText,
  listAspects,
} from '@/src/features/design-space/candidate-utils';
import {
  buildConsistencyDefs,
  percentileOf,
  resolveRubricDefs,
  type MetricDef,
} from '@/src/features/design-space/examine-utils';
import { useAlignmentQuery } from '@/src/features/design-space/hooks/use-alignment-query';
import { useMetricsQuery } from '@/src/features/design-space/hooks/use-metrics-query';
import type { MetricResult } from '@/src/features/design-space/types';
import { starPath } from '@/src/lib/svg-glyphs';
import { useMindmapStore } from '@/src/store/mindmap-store';
import { AxesView } from './axes-view';

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

interface ExamineViewProps {
  nodes: ReadonlyArray<MindmapNode>;
  selection: MindmapSelection;
  onSelectNode: (selection: MindmapSelection) => void;
  onSelectProject: (projectId: string) => void;
  descriptionByTopic: Readonly<Record<string, string>>;
}

/** One metric strip: corpus rug + the candidate's brief as a star on the line. */
function StripRow({
  def,
  result,
  onRemove,
}: {
  def: MetricDef;
  result: MetricResult | undefined;
  onRemove?: () => void;
}) {
  const item = result?.items[0];
  const pct = result && item ? percentileOf(result.corpus, item.score) : null;
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
          <div className="relative mt-2 h-8">
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
            {/* The candidate's star: an undistorted overlay positioned by %. */}
            <svg
              viewBox="0 0 24 24"
              className="absolute top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2"
              style={{ left: pctOf(item.score) }}
            >
              <path
                d={starPath(12, 12, 10)}
                fill={CANDIDATE_COLOR}
                stroke="white"
                strokeWidth={1.5}
                strokeDasharray={item.clipped ? '3 2' : undefined}
              />
            </svg>
          </div>
          <div className="flex items-center justify-between gap-4 text-[10px] text-muted-foreground">
            <span className="truncate">← {def.poleBLabel}</span>
            <span className="truncate text-right">{def.poleALabel} →</span>
          </div>
          <p className="mt-0.5 text-[10px] text-muted-foreground">
            {pct != null && !Number.isNaN(pct)
              ? `more "${def.poleALabel}" than ${(pct * 100).toFixed(0)}% of real projects (scaled to this corpus)`
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

/**
 * Perspectives, revamped (Part 10 I3): the alignment instrument. Examine the
 * active candidate's BRIEF against (a) its own commitments — one strip per
 * aspect, chosen option vs the data-picked strongest alternative — and (b) the
 * project's saved rubric metrics. The old bipolar scatter remains as the
 * "cross two metrics" drill-down tab.
 */
export function ExamineView(props: ExamineViewProps) {
  const { nodes, descriptionByTopic } = props;
  const candidates = useMindmapStore((s) => s.candidates);
  const activeCandidateId = useMindmapStore((s) => s.activeCandidateId);
  const setActiveCandidate = useMindmapStore((s) => s.setActiveCandidate);
  const descriptionById = useMindmapStore((s) => s.descriptionById);
  const rubric = useMindmapStore((s) => s.rubric);
  const addRubricMetric = useMindmapStore((s) => s.addRubricMetric);
  const removeRubricMetric = useMindmapStore((s) => s.removeRubricMetric);

  const [tab, setTab] = useState<'strips' | 'scatter'>('strips');
  const [draft, setDraft] = useState<{ aspectId: string; poleAId: string; poleBId: string }>({
    aspectId: '',
    poleAId: '',
    poleBId: '',
  });

  const candidateList = useMemo(
    () => Object.values(candidates).sort((a, b) => a.createdAt - b.createdAt),
    [candidates]
  );
  const active = activeCandidateId ? candidates[activeCandidateId] ?? null : null;
  const brief = active?.brief?.trim() || null;

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

  const aspectRows = useMemo(
    () =>
      listAspects(nodes)
        .map((aspect) => ({ aspect, options: [...(aspect.children ?? [])] }))
        .filter((row) => row.options.length >= 2),
    [nodes]
  );
  const draftRow = aspectRows.find((r) => r.aspect.id === draft.aspectId);
  const canAdd =
    draft.aspectId && draft.poleAId && draft.poleBId && draft.poleAId !== draft.poleBId;

  if (tab === 'scatter') {
    return (
      <div className="relative h-full w-full">
        <Tabs tab={tab} setTab={setTab} />
        {/* The scatter's own picker bar anchors to the top of ITS container —
            start it below the tab switcher so both stay reachable. */}
        <div className="absolute inset-x-0 bottom-0 top-14">
          <AxesView {...props} />
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-full w-full overflow-y-auto">
      <Tabs tab={tab} setTab={setTab} />
      {/* Clear the floating Context/Candidate (left, max-w-sm) and Related
          Projects (right, max-w-md) panels — the strips are a document, not a
          pannable canvas, so they must share the layer without overlap. */}
      <div className="xl:ml-[26rem] xl:mr-[30rem]">
        <div className="mx-auto max-w-3xl space-y-3 px-6 pb-32 pt-20">
        {/* Candidate switcher */}
        {candidateList.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            {candidateList.map((candidate) => (
              <button
                key={candidate.id}
                type="button"
                onClick={() => setActiveCandidate(candidate.id)}
                className={`rounded-full border px-2.5 py-0.5 text-[11px] font-medium transition-colors ${
                  candidate.id === activeCandidateId
                    ? 'border-violet-500 bg-violet-500/10 text-violet-700'
                    : 'text-muted-foreground hover:bg-muted'
                }`}
              >
                {candidate.name}
              </button>
            ))}
          </div>
        )}

        {!active ? (
          <Teach>
            Compose a candidate first (Candidate panel) — Perspectives is where
            you examine it against your metrics.
          </Teach>
        ) : !brief ? (
          <Teach>
            Write or draft <span className="font-semibold">{active.name}</span>
            &apos;s brief in the Candidate panel — examination measures the
            brief (the actual design), not the choice list.
          </Teach>
        ) : (
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

            {/* Strips */}
            {defs.map((def, i) => (
              <StripRow
                key={def.key}
                def={def}
                result={metrics?.metrics[i]}
                onRemove={
                  def.rubricId ? () => removeRubricMetric(def.rubricId!) : undefined
                }
              />
            ))}
            {defs.length === 0 && (
              <Teach>
                Choose options for this candidate (consistency strips appear per
                choice) or add a rubric metric below.
              </Teach>
            )}
          </>
        )}

        {/* Rubric editor — the project's persistent yardstick */}
        <div className="rounded-xl border border-dashed bg-background/60 px-4 py-3">
          <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Add a rubric metric (saved with the project)
          </p>
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[11px]">
            <select
              value={draft.aspectId}
              onChange={(e) =>
                setDraft({ aspectId: e.target.value, poleAId: '', poleBId: '' })
              }
              className="rounded-md border bg-background px-1.5 py-1"
              aria-label="Metric aspect"
            >
              <option value="">aspect…</option>
              {aspectRows.map((row) => (
                <option key={row.aspect.id} value={row.aspect.id}>
                  {row.aspect.topic}
                </option>
              ))}
            </select>
            {draftRow && (
              <>
                <select
                  value={draft.poleAId}
                  onChange={(e) => setDraft((d) => ({ ...d, poleAId: e.target.value }))}
                  className="rounded-md border bg-background px-1.5 py-1"
                  aria-label="Pole A"
                >
                  <option value="">pole A…</option>
                  {draftRow.options.map((o) => (
                    <option key={o.id} value={o.id}>
                      {o.topic}
                    </option>
                  ))}
                </select>
                <span className="text-muted-foreground">↔</span>
                <select
                  value={draft.poleBId}
                  onChange={(e) => setDraft((d) => ({ ...d, poleBId: e.target.value }))}
                  className="rounded-md border bg-background px-1.5 py-1"
                  aria-label="Pole B"
                >
                  <option value="">pole B…</option>
                  {draftRow.options
                    .filter((o) => o.id !== draft.poleAId)
                    .map((o) => (
                      <option key={o.id} value={o.id}>
                        {o.topic}
                      </option>
                    ))}
                </select>
              </>
            )}
            <button
              type="button"
              disabled={!canAdd}
              onClick={() => {
                addRubricMetric({
                  id: `rm-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
                  aspectId: draft.aspectId,
                  poleAId: draft.poleAId,
                  poleBId: draft.poleBId,
                });
                setDraft({ aspectId: '', poleAId: '', poleBId: '' });
              }}
              className="flex items-center gap-1 rounded-full border px-2.5 py-1 font-medium transition-colors enabled:hover:bg-muted disabled:opacity-40"
            >
              <Plus className="h-3 w-3" />
              Add
            </button>
          </div>
        </div>

          <p className="text-[10px] text-muted-foreground">
            Strips score the brief in the original embedding metric (exact, no
            projection). Percentiles are relative to this corpus.
          </p>
        </div>
      </div>
    </div>
  );
}

function Tabs({
  tab,
  setTab,
}: {
  tab: 'strips' | 'scatter';
  setTab: (tab: 'strips' | 'scatter') => void;
}) {
  return (
    <div className="absolute top-4 left-1/2 z-50 flex -translate-x-1/2 items-center gap-1 rounded-full border bg-background/90 p-0.5 shadow-md backdrop-blur">
      <button
        type="button"
        onClick={() => setTab('strips')}
        aria-pressed={tab === 'strips'}
        className={`whitespace-nowrap rounded-full px-3 py-1 text-[11px] font-semibold transition-colors ${
          tab === 'strips' ? 'bg-violet-500/10 text-violet-700' : 'text-muted-foreground hover:text-foreground'
        }`}
      >
        Examine
      </button>
      <button
        type="button"
        onClick={() => setTab('scatter')}
        aria-pressed={tab === 'scatter'}
        className={`whitespace-nowrap rounded-full px-3 py-1 text-[11px] font-semibold transition-colors ${
          tab === 'scatter' ? 'bg-violet-500/10 text-violet-700' : 'text-muted-foreground hover:text-foreground'
        }`}
      >
        Cross two metrics
      </button>
    </div>
  );
}

function Teach({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed bg-background/60 px-4 py-6 text-center text-xs text-muted-foreground">
      {children}
    </div>
  );
}
