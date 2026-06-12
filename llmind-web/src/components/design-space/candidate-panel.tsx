'use client';

import {
  ChevronRight,
  Download,
  Focus,
  Loader2,
  Plus,
  Scale,
  Sparkles,
  SlidersHorizontal,
  Star,
  Trash2,
  X,
} from 'lucide-react';
import { useMemo, useState } from 'react';
import { Badge } from '@/src/components/ui/badge';
import { Button } from '@/src/components/ui/button';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/src/components/ui/collapsible';
import { Input } from '@/src/components/ui/input';
import {
  candidateChoiceRows,
  candidateEmbeddingText,
} from '@/src/features/design-space/candidate-utils';
import { useCandidatePrecedentsQuery } from '@/src/features/design-space/hooks/use-candidate-precedents';
import { useDraftBriefMutation } from '@/src/features/design-space/hooks/use-draft-brief-mutation';
import {
  useSteerMutation,
  type SteerResult,
} from '@/src/features/design-space/hooks/use-steer-mutation';
import { buildExplorationMarkdown, downloadTextFile } from '@/src/lib/export-exploration';
import { useMindmapStore } from '@/src/store/mindmap-store';
import { SteerResultCard } from './steer-result-card';

interface CandidatePanelProps {
  descriptionByTopic: Readonly<Record<string, string>>;
  /** Open a corpus project in the Related Projects panel. */
  onOpenProject: (projectId: string) => void;
  onOpenCompare: () => void;
  /** Aspect currently awaiting a click-picked option (the "—" flow). */
  pendingAspectId?: string | null;
  /** Arm the pick flow: the next click on an option of this aspect (in any
   * view) fills the slot. */
  onStartPickChoice?: (aspectId: string) => void;
  onCancelPickChoice?: () => void;
  /** Turn on the relevance lens anchored to this candidate (design space). */
  onInspectRelevance?: () => void;
  /** Open the Perspectives view to examine this candidate against metrics. */
  onOpenExamine?: () => void;
  /** C1 informing-back: an applied precedent steer's named qualities become
   * option proposals (aspect unknown — the chip offers a picker). */
  onProposeQualities?: (
    qualities: string[],
    aspectId: string | null,
    evidence: string
  ) => void;
  /** Controlled collapse state (the page collapses panels in the Examine view). */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

/**
 * Compose a design: one option per aspect. The composition is itself a point in
 * the design space (the star) with its own real precedents — designs, not just
 * options, become first-class citizens of the exploration.
 */
export function CandidatePanel({
  descriptionByTopic,
  onOpenProject,
  onOpenCompare,
  pendingAspectId = null,
  onStartPickChoice,
  onCancelPickChoice,
  onInspectRelevance,
  onOpenExamine,
  onProposeQualities,
  open,
  onOpenChange,
}: CandidatePanelProps) {
  const nodes = useMindmapStore((s) => s.nodes);
  const candidates = useMindmapStore((s) => s.candidates);
  const activeCandidateId = useMindmapStore((s) => s.activeCandidateId);
  const createCandidate = useMindmapStore((s) => s.createCandidate);
  const deleteCandidate = useMindmapStore((s) => s.deleteCandidate);
  const setActiveCandidate = useMindmapStore((s) => s.setActiveCandidate);
  const renameCandidate = useMindmapStore((s) => s.renameCandidate);
  const setChoice = useMindmapStore((s) => s.setChoice);
  const setCandidateBrief = useMindmapStore((s) => s.setCandidateBrief);
  const descriptionById = useMindmapStore((s) => s.descriptionById);
  const optionState = useMindmapStore((s) => s.optionState);
  const provenance = useMindmapStore((s) => s.provenance);
  const coords = useMindmapStore((s) => s.coords);
  const discovered = useMindmapStore((s) => s.discovered);
  const trackUsage = useMindmapStore((s) => s.trackUsage);
  const recordEvent = useMindmapStore((s) => s.recordEvent);
  const events = useMindmapStore((s) => s.events);
  const reflections = useMindmapStore((s) => s.reflections);

  const candidateList = useMemo(
    () => Object.values(candidates).sort((a, b) => a.createdAt - b.createdAt),
    [candidates]
  );
  // Deletion is the one un-undoable action in the panel — guard it with a
  // two-click confirm (arms for 3s, disarms on candidate switch), never a
  // modal. Adding a rubric metric is more guarded than this was.
  const [deleteArmed, setDeleteArmed] = useState<string | null>(null);
  const active = activeCandidateId ? candidates[activeCandidateId] ?? null : null;
  const rows = useMemo(() => candidateChoiceRows(active, nodes), [active, nodes]);
  const chosenCount = rows.filter((r) => r.optionId).length;

  // Brief-first: precedents and the lens describe the actual design (Part 10).
  const candidateText = useMemo(
    () => candidateEmbeddingText(active, nodes, descriptionByTopic, descriptionById),
    [active, nodes, descriptionByTopic, descriptionById]
  );
  const { data: precedents, isFetching: precedentsLoading } =
    useCandidatePrecedentsQuery(candidateText);

  const { mutateAsync: draftBrief, isPending: drafting } = useDraftBriefMutation();

  // B3 precedent steering: pull the BRIEF toward (or push it away from) a
  // real precedent — the move is made in language, measured in embedding
  // space, and ALWAYS shown for veto before it touches the brief. The outcome
  // snapshots the candidate and the steered brief so a candidate switch
  // mid-veto can neither show the wrong "before" nor apply to the wrong design.
  const brief = active?.brief?.trim() || '';
  const { mutateAsync: steerBrief, isPending: steerPending } = useSteerMutation();
  const [steerOutcome, setSteerOutcome] = useState<{
    candidateId: string;
    briefBefore: string;
    /** Human-readable move ("pull toward …") for the event + proposals. */
    evidence: string;
    result: SteerResult;
  } | null>(null);
  const [steeringId, setSteeringId] = useState<string | null>(null);
  const [steerError, setSteerError] = useState<string | null>(null);
  const handlePrecedentSteer = async (
    p: { id: string; Name: string; Descriptions: string },
    mode: 'toward' | 'away'
  ) => {
    if (!brief || !active) return;
    const steeredId = active.id;
    const steeredBrief = brief;
    setSteeringId(p.id);
    setSteerError(null);
    try {
      const reference = `${p.Name}. ${p.Descriptions}`.slice(0, 600);
      const result = await steerBrief({
        text: steeredBrief,
        mode,
        reference: { text: reference, weight: 0.5 },
        // The backend caps preserve at 12 — send the most recent commitments.
        preserve: rows
          .filter((row) => row.optionTopic)
          .map((row) => row.optionTopic as string)
          .slice(0, 12),
      });
      setSteerOutcome({
        candidateId: steeredId,
        briefBefore: steeredBrief,
        evidence: `${mode === 'toward' ? 'pull toward' : 'push away from'} "${p.Name}"`,
        result,
      });
      trackUsage('steer_run');
    } catch (error) {
      setSteerError(error instanceof Error ? error.message : 'steering failed');
    } finally {
      setSteeringId(null);
    }
  };
  const [draftError, setDraftError] = useState<string | null>(null);
  const handleDraftBrief = async () => {
    if (!active) return;
    trackUsage('brief_draft');
    const aspects = rows
      .filter((row) => row.optionId && row.optionTopic)
      .map((row) => ({
        aspect: row.aspectTopic,
        option: row.optionTopic as string,
        desc:
          descriptionById[row.optionId as string] ??
          descriptionByTopic[row.optionTopic as string] ??
          '',
      }));
    if (aspects.length === 0) return;
    setDraftError(null);
    try {
      const { brief } = await draftBrief({ aspects });
      setCandidateBrief(active.id, brief);
    } catch (error) {
      // A silent failure looked like the button doing nothing.
      setDraftError(error instanceof Error ? error.message : 'drafting failed');
    }
  };

  const handleExport = () => {
    trackUsage('export');
    const markdown = buildExplorationMarkdown({
      nodes,
      descriptionByTopic,
      descriptionById,
      optionState,
      candidates,
      provenance,
      coords,
      discovered,
      activeCandidateId,
      events,
      reflections,
    });
    downloadTextFile(
      `design-space-exploration-${new Date().toISOString().slice(0, 10)}.md`,
      markdown
    );
  };

  return (
    <Collapsible defaultOpen={false} open={open} onOpenChange={onOpenChange}>
      <section className="overflow-hidden rounded-2xl border bg-background/90 shadow-xl backdrop-blur-md">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <Star className="h-4 w-4 text-violet-600" />
            <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              Candidate
            </h2>
            {chosenCount > 0 && (
              <Badge variant="outline" className="h-5 px-1.5 text-[10px]">
                {chosenCount} chosen
              </Badge>
            )}
          </div>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="icon" className="group h-6 w-6 rounded-full">
              <ChevronRight className="h-4 w-4 transition-transform duration-200 group-data-[state=open]:rotate-90" />
            </Button>
          </CollapsibleTrigger>
        </div>

        <CollapsibleContent>
          {/* Capped + scrollable: with many aspects, precedents, and a steer
              veto card, an unbounded panel ran under the bottom navigator
              (the Context panel above already takes up to 45dvh). */}
          <div className="max-h-[min(48dvh,520px)] space-y-3 overflow-y-auto px-4 pb-4">
            {/* Candidate switcher */}
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
              <button
                type="button"
                onClick={() => createCandidate()}
                className="flex items-center gap-1 rounded-full border border-dashed px-2.5 py-0.5 text-[11px] font-medium text-muted-foreground transition-colors hover:bg-muted"
              >
                <Plus className="h-3 w-3" />
                New
              </button>
            </div>

            {active ? (
              <>
                <div className="flex items-center gap-2">
                  <Input
                    value={active.name}
                    onChange={(e) => renameCandidate(active.id, e.target.value)}
                    className="h-7 text-xs"
                    aria-label="Candidate name"
                  />
                  {deleteArmed === active.id ? (
                    <Button
                      variant="destructive"
                      size="sm"
                      className="h-7 shrink-0 gap-1 px-2 text-[11px]"
                      onClick={() => {
                        setDeleteArmed(null);
                        deleteCandidate(active.id);
                      }}
                      title="Click again to delete this candidate and its choices (the exploration log keeps its history)"
                    >
                      <Trash2 className="h-3 w-3" />
                      Delete?
                    </Button>
                  ) : (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 shrink-0 text-muted-foreground hover:text-destructive"
                      onClick={() => {
                        const id = active.id;
                        setDeleteArmed(id);
                        window.setTimeout(
                          () => setDeleteArmed((cur) => (cur === id ? null : cur)),
                          3000
                        );
                      }}
                      title="Delete candidate (asks to confirm)"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  )}
                </div>

                {/* One choice per aspect */}
                <div className="space-y-1">
                  {rows.map((row) => (
                    <div
                      key={row.aspectId}
                      className="flex items-center justify-between gap-2 text-xs"
                    >
                      <span className="truncate text-muted-foreground">{row.aspectTopic}</span>
                      {row.optionTopic ? (
                        <span className="flex min-w-0 items-center gap-1">
                          <span className="truncate font-medium">{row.optionTopic}</span>
                          <button
                            type="button"
                            className="text-muted-foreground hover:text-destructive"
                            onClick={() => setChoice(row.aspectId, null)}
                            title="Clear choice"
                          >
                            <X className="h-3 w-3" />
                          </button>
                        </span>
                      ) : pendingAspectId === row.aspectId ? (
                        <button
                          type="button"
                          onClick={() => onCancelPickChoice?.()}
                          className="flex items-center gap-1 whitespace-nowrap rounded-full bg-violet-500/10 px-2 py-0.5 text-[10px] font-semibold text-violet-700 animate-pulse"
                          title="Click an option of this aspect in the map or space — or click here to cancel"
                        >
                          click an option… <X className="h-3 w-3" />
                        </button>
                      ) : (
                        <button
                          type="button"
                          onClick={() => onStartPickChoice?.(row.aspectId)}
                          className="rounded-full px-2 py-0.5 text-muted-foreground/60 transition-colors hover:bg-muted hover:text-foreground"
                          title={`Pick an option for ${row.aspectTopic}: click here, then click an option in the map or space`}
                        >
                          — pick
                        </button>
                      )}
                    </div>
                  ))}
                  {rows.length === 0 && (
                    <p className="text-xs text-muted-foreground">No aspects in the taxonomy yet.</p>
                  )}
                </div>
                <p className="text-[10px] leading-snug text-muted-foreground">
                  {pendingAspectId
                    ? 'Now click an option of that aspect in the mind map or design space.'
                    : 'Fill a slot via "— pick", or select an option and use "Choose" in the Context panel.'}
                </p>

                {/* The BRIEF — the design's identity layer (Part 10). It drives
                    the star, the precedents, and the Examine strips; the choices
                    above remain the commitments it is measured against. */}
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      Brief — what this design is
                    </p>
                    <button
                      type="button"
                      onClick={handleDraftBrief}
                      disabled={drafting || chosenCount === 0}
                      className="flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium text-violet-700 transition-colors enabled:hover:bg-violet-500/10 disabled:opacity-40"
                      title={
                        chosenCount === 0
                          ? 'Choose at least one option first'
                          : 'Draft a description from the choices — then edit it'
                      }
                    >
                      {drafting ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <Sparkles className="h-3 w-3" />
                      )}
                      Draft from choices
                    </button>
                  </div>
                  {draftError && (
                    <p className="text-[10px] text-destructive">drafting failed: {draftError}</p>
                  )}
                  <textarea
                    value={active.brief ?? ''}
                    onChange={(e) => setCandidateBrief(active.id, e.target.value)}
                    placeholder="Describe the actual design — what it is, how it works, what people experience. Drafting from your choices gives a starting point."
                    rows={4}
                    className="w-full resize-y rounded-md border bg-background px-2 py-1.5 text-[11px] leading-snug placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-violet-400"
                    aria-label="Candidate brief"
                  />
                  {active.brief?.trim() ? (
                    <p className="text-[10px] leading-snug text-muted-foreground">
                      The star, precedents, and Examine read this brief; your
                      choices stay the commitments it is checked against.
                    </p>
                  ) : null}
                </div>

                {/* Closest real precedents to the COMPOSED design */}
                {candidateText && (
                  <div className="space-y-1">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      Closest precedents {precedentsLoading ? '…' : ''}
                    </p>
                    {(precedents ?? []).map((p) => (
                      <div
                        key={p.id}
                        className="group flex w-full items-center justify-between gap-2 rounded-md px-1.5 py-0.5 text-[11px] transition-colors hover:bg-muted"
                      >
                        <button
                          type="button"
                          onClick={() => onOpenProject(p.id)}
                          className="min-w-0 flex-1 truncate text-left"
                        >
                          {p.Name}
                        </button>
                        {/* B3: steer the brief relative to this precedent */}
                        {brief && (
                          <span className="hidden shrink-0 items-center gap-0.5 group-hover:flex">
                            {steeringId === p.id ? (
                              <Loader2 className="h-3 w-3 animate-spin" />
                            ) : (
                              <>
                                <button
                                  type="button"
                                  disabled={steerPending}
                                  onClick={() => handlePrecedentSteer(p, 'toward')}
                                  title="Pull the brief toward this precedent (one move, shown for veto)"
                                  className="rounded border border-transparent px-1 text-[10px] text-muted-foreground hover:bg-background disabled:opacity-40"
                                >
                                  ⇢ pull
                                </button>
                                <button
                                  type="button"
                                  disabled={steerPending}
                                  onClick={() => handlePrecedentSteer(p, 'away')}
                                  title="Push the brief away from this precedent (one move, shown for veto)"
                                  className="rounded border border-transparent px-1 text-[10px] text-muted-foreground hover:bg-background disabled:opacity-40"
                                >
                                  ⇠ push
                                </button>
                              </>
                            )}
                          </span>
                        )}
                        <span className="shrink-0 tabular-nums text-muted-foreground">
                          {(p.score * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                    {steerError && (
                      <p className="text-[10px] text-destructive">steering failed: {steerError}</p>
                    )}
                    {steerOutcome && steerOutcome.candidateId === active?.id && (
                      <SteerResultCard
                        result={steerOutcome.result}
                        briefBefore={steerOutcome.briefBefore}
                        onApply={() => {
                          setCandidateBrief(steerOutcome.candidateId, steerOutcome.result.revised_text);
                          recordEvent(
                            'steer_applied',
                            `Steered "${active?.name ?? 'candidate'}" — ${steerOutcome.evidence}`,
                            [steerOutcome.candidateId]
                          );
                          if (steerOutcome.result.named_qualities.length > 0) {
                            onProposeQualities?.(
                              steerOutcome.result.named_qualities,
                              null,
                              steerOutcome.evidence
                            );
                          }
                          trackUsage('steer_applied');
                          setSteerOutcome(null);
                        }}
                        onDiscard={() => setSteerOutcome(null)}
                      />
                    )}
                  </div>
                )}
              </>
            ) : (
              <p className="text-xs text-muted-foreground">
                Create a candidate, then choose one option per aspect to compose a
                design — it appears as a star in the design space with its own real
                precedents.
              </p>
            )}

            <div className="flex flex-wrap items-center gap-2 pt-1">
              <Button
                variant="outline"
                size="sm"
                className="h-7 gap-1.5 rounded-full px-3 text-[11px]"
                onClick={onInspectRelevance}
                disabled={!candidateText}
                title={
                  candidateText
                    ? 'Color the design space by relevance to this candidate'
                    : 'Choose at least one option first'
                }
              >
                <Focus className="h-3 w-3" />
                Relevance
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-7 gap-1.5 rounded-full px-3 text-[11px]"
                onClick={onOpenExamine}
                disabled={!active}
                title={
                  active
                    ? 'Examine this candidate against your metrics (Perspectives)'
                    : 'Create a candidate first'
                }
              >
                <SlidersHorizontal className="h-3 w-3" />
                Examine
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-7 gap-1.5 rounded-full px-3 text-[11px]"
                onClick={onOpenCompare}
                disabled={candidateList.length < 2}
                title={candidateList.length < 2 ? 'Need at least two candidates' : undefined}
              >
                <Scale className="h-3 w-3" />
                Compare
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-7 gap-1.5 rounded-full px-3 text-[11px]"
                onClick={handleExport}
              >
                <Download className="h-3 w-3" />
                Export
              </Button>
            </div>
          </div>
        </CollapsibleContent>
      </section>
    </Collapsible>
  );
}
