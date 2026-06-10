'use client';

import { ChevronRight, Download, Plus, Scale, Star, Trash2, X } from 'lucide-react';
import { useMemo } from 'react';
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
  composeCandidateText,
} from '@/src/features/design-space/candidate-utils';
import { useCandidatePrecedentsQuery } from '@/src/features/design-space/hooks/use-candidate-precedents';
import { buildExplorationMarkdown, downloadTextFile } from '@/src/lib/export-exploration';
import { useMindmapStore } from '@/src/store/mindmap-store';

interface CandidatePanelProps {
  descriptionByTopic: Readonly<Record<string, string>>;
  /** Open a corpus project in the Related Projects panel. */
  onOpenProject: (projectId: string) => void;
  onOpenCompare: () => void;
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
}: CandidatePanelProps) {
  const nodes = useMindmapStore((s) => s.nodes);
  const candidates = useMindmapStore((s) => s.candidates);
  const activeCandidateId = useMindmapStore((s) => s.activeCandidateId);
  const createCandidate = useMindmapStore((s) => s.createCandidate);
  const deleteCandidate = useMindmapStore((s) => s.deleteCandidate);
  const setActiveCandidate = useMindmapStore((s) => s.setActiveCandidate);
  const renameCandidate = useMindmapStore((s) => s.renameCandidate);
  const setChoice = useMindmapStore((s) => s.setChoice);
  const descriptionById = useMindmapStore((s) => s.descriptionById);
  const optionState = useMindmapStore((s) => s.optionState);
  const provenance = useMindmapStore((s) => s.provenance);
  const coords = useMindmapStore((s) => s.coords);

  const candidateList = useMemo(
    () => Object.values(candidates).sort((a, b) => a.createdAt - b.createdAt),
    [candidates]
  );
  const active = activeCandidateId ? candidates[activeCandidateId] ?? null : null;
  const rows = useMemo(() => candidateChoiceRows(active, nodes), [active, nodes]);
  const chosenCount = rows.filter((r) => r.optionId).length;

  const candidateText = useMemo(
    () => composeCandidateText(active, nodes, descriptionByTopic, descriptionById),
    [active, nodes, descriptionByTopic, descriptionById]
  );
  const { data: precedents, isFetching: precedentsLoading } =
    useCandidatePrecedentsQuery(candidateText);

  const handleExport = () => {
    const markdown = buildExplorationMarkdown({
      nodes,
      descriptionByTopic,
      descriptionById,
      optionState,
      candidates,
      provenance,
      coords,
    });
    downloadTextFile(
      `design-space-exploration-${new Date().toISOString().slice(0, 10)}.md`,
      markdown
    );
  };

  return (
    <Collapsible defaultOpen={false}>
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
          <div className="space-y-3 px-4 pb-4">
            {/* Candidate switcher */}
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
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 shrink-0 text-muted-foreground hover:text-destructive"
                    onClick={() => deleteCandidate(active.id)}
                    title="Delete candidate"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </Button>
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
                      ) : (
                        <span className="text-muted-foreground/50">—</span>
                      )}
                    </div>
                  ))}
                  {rows.length === 0 && (
                    <p className="text-xs text-muted-foreground">No aspects in the taxonomy yet.</p>
                  )}
                </div>
                <p className="text-[10px] leading-snug text-muted-foreground">
                  Select an option (map or space) and use &ldquo;Choose&rdquo; in the
                  Context panel to fill a slot.
                </p>

                {/* Closest real precedents to the COMPOSED design */}
                {candidateText && (
                  <div className="space-y-1">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      Closest precedents {precedentsLoading ? '…' : ''}
                    </p>
                    {(precedents ?? []).map((p) => (
                      <button
                        key={p.id}
                        type="button"
                        onClick={() => onOpenProject(p.id)}
                        className="flex w-full items-center justify-between gap-2 rounded-md px-1.5 py-0.5 text-left text-[11px] transition-colors hover:bg-muted"
                      >
                        <span className="truncate">{p.Name}</span>
                        <span className="shrink-0 tabular-nums text-muted-foreground">
                          {(p.score * 100).toFixed(0)}%
                        </span>
                      </button>
                    ))}
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

            <div className="flex items-center gap-2 pt-1">
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
