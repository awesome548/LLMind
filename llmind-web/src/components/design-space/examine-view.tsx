'use client';

import { Plus } from 'lucide-react';
import { useMemo, useState } from 'react';
import type { MindmapNode, MindmapSelection } from '@/src/features/mindmap/types';
import { listAspects } from '@/src/features/design-space/candidate-utils';
import { useMindmapStore } from '@/src/store/mindmap-store';
import { CandidateStrips } from './candidate-strips';
import { AxesView } from './axes-view';

interface ExamineViewProps {
  nodes: ReadonlyArray<MindmapNode>;
  selection: MindmapSelection;
  onSelectNode: (selection: MindmapSelection) => void;
  onSelectProject: (projectId: string) => void;
  descriptionByTopic: Readonly<Record<string, string>>;
  /** Open on the scatter tab (the cross-tab's "show as continuous scatter"). */
  initialTab?: 'strips' | 'scatter';
  /** C1: forwarded to the strips (applied steers propose their qualities). */
  onProposeQualities?: (
    qualities: string[],
    aspectId: string | null,
    evidence: string
  ) => void;
}

/**
 * Perspectives, revamped (Part 10 I3): the alignment instrument. Examine the
 * active candidate's BRIEF against (a) its own commitments — one strip per
 * aspect, chosen option vs the data-picked strongest alternative — and (b) the
 * project's saved rubric metrics. The strips themselves are shared with the
 * Design Space inspector dock (candidate-strips.tsx, Part 12 B1); this view
 * adds the candidate switcher, the rubric editor, and the scatter drill-down.
 */
export function ExamineView(props: ExamineViewProps) {
  const { nodes } = props;
  const candidates = useMindmapStore((s) => s.candidates);
  const activeCandidateId = useMindmapStore((s) => s.activeCandidateId);
  const setActiveCandidate = useMindmapStore((s) => s.setActiveCandidate);
  const addRubricMetric = useMindmapStore((s) => s.addRubricMetric);

  const [tab, setTab] = useState<'strips' | 'scatter'>(props.initialTab ?? 'strips');
  const [draft, setDraft] = useState<{ aspectId: string; poleAId: string; poleBId: string }>({
    aspectId: '',
    poleAId: '',
    poleBId: '',
  });

  const candidateList = useMemo(
    () => Object.values(candidates).sort((a, b) => a.createdAt - b.createdAt),
    [candidates]
  );

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
                  title={candidate.name}
                  className={`max-w-44 truncate rounded-full border px-2.5 py-0.5 text-[11px] font-medium transition-colors ${
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

          <CandidateStrips
            nodes={nodes}
            descriptionByTopic={props.descriptionByTopic}
            onProposeQualities={props.onProposeQualities}
          />

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
