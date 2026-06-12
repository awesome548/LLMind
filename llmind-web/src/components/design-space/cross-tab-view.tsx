'use client';

// The cross-tab lens (Part 12 B2): two aspects → option×option grid, each
// cell holding its annotated corpus projects (Halskov 2021 — the
// morphological view whose EMPTY cells are exact, nameable gaps) and any
// candidate committing to both options. An empty cell can be generated into:
// pole-conditioned, seeded with half-matching precedents; a kept result
// becomes a candidate skeleton (choices = the two options, brief = the desc).

import { useEffect, useMemo, useRef, useState } from 'react';
import { Loader2, ScatterChart, Sparkles, Star, Undo2 } from 'lucide-react';
import { usePanZoom } from '@/src/features/design-space/hooks/use-pan-zoom';
import { Button } from '@/src/components/ui/button';
import {
  buildCrossTabCells,
  halfMatchingExemplars,
  type CrossTabCellModel,
  type SchemaColumn,
} from '@/src/features/design-space/schema-utils';
import type { AnnotationResponse } from '@/src/features/design-space/hooks/use-annotation-query';
import {
  useGenerateCellMutation,
  type GenerateCellResult,
} from '@/src/features/design-space/hooks/use-generate-cell-mutation';
import { useMindmapStore } from '@/src/store/mindmap-store';

export interface KeepCellIdea {
  aspectAId: string;
  optionAId: string;
  aspectBId: string;
  optionBId: string;
  idea: { name: string; desc: string };
}

interface Props {
  columns: SchemaColumn[];
  annotation: AnnotationResponse | null;
  annotating: boolean;
  annotationError?: string | null;
  onOpenProject: (projectId: string) => void;
  /** Create a candidate skeleton from a kept cell idea. */
  onKeepIdea: (args: KeepCellIdea) => void;
  /** Drill down to the continuous bipolar scatter (Perspectives). */
  onShowScatter: () => void;
}

const cellKey = (aId: string, bId: string) => `${aId}:${bId}`;

export function CrossTabView({
  columns,
  annotation,
  annotating,
  annotationError,
  onOpenProject,
  onKeepIdea,
  onShowScatter,
}: Props) {
  const candidates = useMindmapStore((s) => s.candidates);
  const aspectsWithOptions = columns.filter((c) => c.options.length > 0);
  const [aId, setAId] = useState<string>('');
  const [bId, setBId] = useState<string>('');
  const aspectA =
    aspectsWithOptions.find((c) => c.id === aId) ?? aspectsWithOptions[0] ?? null;
  const aspectB =
    aspectsWithOptions.find((c) => c.id === bId && c.id !== aspectA?.id) ??
    aspectsWithOptions.find((c) => c.id !== aspectA?.id) ??
    null;

  // Anchored to its cell (pans with the grid), flipped toward the viewport
  // center when the cell sits near an edge so it never opens off-canvas.
  const [openCell, setOpenCell] = useState<{
    key: string;
    flipX: boolean;
    flipY: boolean;
  } | null>(null);
  // Generated-but-not-kept ideas, per cell — dismissable previews, the peek
  // transparency pattern (commit only from the popover). Keyed by the option
  // PAIR, so a cached idea is still about its exact gap after aspect switches.
  const [ideas, setIdeas] = useState<Record<string, GenerateCellResult>>({});
  const [generatingCell, setGeneratingCell] = useState<string | null>(null);
  const { mutateAsync: generateCell } = useGenerateCellMutation();
  const [generateError, setGenerateError] = useState<string | null>(null);
  // Cancels the client-side wait when the view unmounts mid-generation.
  const generateAbortRef = useRef<AbortController | null>(null);
  useEffect(() => () => generateAbortRef.current?.abort(), []);
  // Shared canvas grammar (pan dismisses the cell popover — it is anchored
  // to its cell, so it pans along, but closing reads cleaner mid-drag).
  const { containerRef, view, onPointerDown, onClickCapture, resetView } = usePanZoom(() =>
    setOpenCell(null)
  );

  const candidateList = useMemo(
    () => Object.values(candidates).map((c) => ({ name: c.name, choices: c.choices })),
    [candidates]
  );
  const grid = useMemo(
    () =>
      aspectA && aspectB
        ? buildCrossTabCells(aspectA, aspectB, annotation?.options ?? null, candidateList)
        : [],
    [aspectA, aspectB, annotation, candidateList]
  );

  const handleGenerate = async (cell: CrossTabCellModel) => {
    if (!aspectA || !aspectB) return;
    const key = cellKey(cell.a.id, cell.b.id);
    setGeneratingCell(key);
    setGenerateError(null);
    const controller = new AbortController();
    generateAbortRef.current = controller;
    try {
      const idea = await generateCell({
        aspect_a: aspectA.name,
        option_a: { name: cell.a.name, desc: cell.a.desc },
        aspect_b: aspectB.name,
        option_b: { name: cell.b.name, desc: cell.b.desc },
        exemplar_ids: halfMatchingExemplars(
          annotation?.options[cell.a.id],
          annotation?.options[cell.b.id]
        ),
        signal: controller.signal,
      });
      setIdeas((prev) => ({ ...prev, [key]: idea }));
    } catch (error) {
      setGenerateError(error instanceof Error ? error.message : 'generation failed');
    } finally {
      setGeneratingCell(null);
    }
  };

  if (!aspectA || !aspectB) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-sm text-muted-foreground">
        The cross-tab needs two aspects with options.
      </div>
    );
  }

  return (
    // The same canvas grammar as the schema and the maps: wheel zooms,
    // left-drag pans (nothing is ever stuck under or behind the floating
    // panels/icons), Reset view restores. The controls float as a fixed
    // pill; the grid is the panned sheet.
    <div
      ref={containerRef}
      className="relative h-full w-full cursor-grab touch-none overflow-hidden bg-background active:cursor-grabbing"
      onPointerDown={onPointerDown}
      onClickCapture={onClickCapture}
    >
      {/* Control strip — fixed overlay under the view toggle, never pans */}
      <div className="absolute inset-x-0 top-16 z-10 flex justify-center">
        <div className="flex max-w-[92%] flex-wrap items-center gap-2 rounded-full border bg-background/90 px-3 py-1 text-xs text-muted-foreground shadow-sm backdrop-blur">
          <span className="font-semibold uppercase tracking-wider">Cross-tab</span>
          <select
            value={aspectA.id}
            onChange={(e) => {
              setAId(e.target.value);
              setOpenCell(null);
            }}
            className="rounded-md border bg-background px-1.5 py-1"
            aria-label="Row aspect"
          >
            {aspectsWithOptions.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
          <span>×</span>
          <select
            value={aspectB.id}
            onChange={(e) => {
              setBId(e.target.value);
              setOpenCell(null);
            }}
            className="rounded-md border bg-background px-1.5 py-1"
            aria-label="Column aspect"
          >
            {aspectsWithOptions
              .filter((c) => c.id !== aspectA.id)
              .map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
          </select>
          {annotating ? (
            <span className="animate-pulse">annotating corpus… counts appear when done</span>
          ) : annotationError ? (
            <span className="text-red-600">annotation unavailable: {annotationError}</span>
          ) : annotation ? (
            <span title="Halskov's cross-tab: every pairing of the two aspects' options. A filled cell lists the real projects that combine both; an empty cell means nobody has — an opportunity, not an error.">
              cells = projects with BOTH options · empty = unexplored combination
            </span>
          ) : null}
          {generateError && (
            <span className="text-red-600">cell generation failed: {generateError}</span>
          )}
          <button
            type="button"
            onClick={onShowScatter}
            className="flex items-center gap-1 rounded-full border px-2 py-0.5 hover:bg-muted"
            title="The continuous version: cross two bipolar metrics in Perspectives"
          >
            <ScatterChart className="h-3 w-3" />
            show as continuous scatter
          </button>
        </div>
      </div>

      {/* bottom-24 clears the bottom navigator (same slot as the axes view) */}
      <button
        type="button"
        onClick={resetView}
        className="absolute bottom-24 left-4 z-10 flex items-center gap-1.5 rounded-full border bg-background/90 px-3 py-1 text-xs text-muted-foreground shadow-sm backdrop-blur transition-colors hover:text-foreground"
      >
        <Undo2 className="h-3 w-3" />
        Reset view
      </button>

      {/* The grid sheet — panned/zoomed as one canvas */}
      <div
        className="absolute left-0 top-0 flex w-full justify-center"
        style={{ transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.k})`, transformOrigin: '0 0' }}
      >
        <div className="select-none pb-28 pt-28">
        <table className="border-separate border-spacing-1">
          <thead>
            <tr>
              <th aria-label="corner" />
              {aspectB.options.map((b) => (
                <th
                  key={b.id}
                  className="max-w-28 px-2 pb-1 text-left align-bottom text-[11px] font-medium text-muted-foreground"
                  title={b.desc}
                >
                  <span className={b.rejected ? 'line-through opacity-50' : ''}>{b.name}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {grid.map((row) => {
              const rowOpt = row[0]?.a;
              if (!rowOpt) return null;
              return (
              <tr key={rowOpt.id}>
                <th
                  className="max-w-40 pr-2 text-left text-[11px] font-medium text-muted-foreground"
                  title={rowOpt.desc}
                >
                  <span className={rowOpt.rejected ? 'line-through opacity-50' : ''}>
                    {rowOpt.name}
                  </span>
                </th>
                {row.map((cell) => {
                  const key = cellKey(cell.a.id, cell.b.id);
                  const idea = ideas[key];
                  const empty = annotation && cell.projects.length === 0;
                  return (
                    <td key={key} className="relative">
                      <button
                        type="button"
                        onClick={(e) => {
                          if (openCell?.key === key) {
                            setOpenCell(null);
                            return;
                          }
                          const rect = e.currentTarget.getBoundingClientRect();
                          setOpenCell({
                            key,
                            flipX: rect.left > window.innerWidth * 0.55,
                            flipY: rect.top > window.innerHeight * 0.55,
                          });
                        }}
                        className={`flex h-12 w-24 flex-col items-center justify-center gap-0.5 rounded-lg border text-xs transition-colors hover:border-primary ${
                          empty ? 'border-dashed text-muted-foreground' : 'bg-card'
                        } ${idea ? 'border-violet-400' : ''}`}
                        title={`${cell.a.name} × ${cell.b.name}`}
                      >
                        {generatingCell === key ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : annotation ? (
                          <span className="font-semibold tabular-nums">
                            {cell.projects.length || (idea ? '✦' : '—')}
                          </span>
                        ) : (
                          <span className="text-muted-foreground">·</span>
                        )}
                        {cell.candidateNames.length > 0 && (
                          <span className="flex items-center gap-0.5 text-[9px] text-violet-700">
                            <Star className="h-2.5 w-2.5 fill-current" />
                            {cell.candidateNames.length}
                          </span>
                        )}
                      </button>

                      {openCell?.key === key && (
                        <div
                          className={`absolute z-30 w-72 rounded-lg border bg-popover p-3 text-xs shadow-lg ${
                            openCell.flipX ? 'right-0' : 'left-0'
                          } ${openCell.flipY ? 'bottom-full mb-1' : 'top-full mt-1'}`}
                        >
                          <p className="mb-1 font-semibold">
                            {cell.a.name} × {cell.b.name}
                          </p>
                          {cell.candidateNames.length > 0 && (
                            <p className="mb-1 text-violet-700">
                              your candidates here: {cell.candidateNames.join(', ')}
                            </p>
                          )}
                          {cell.projects.length > 0 ? (
                            <div className="max-h-40 overflow-y-auto">
                              {cell.projects.map((p) => (
                                <button
                                  key={p.id}
                                  type="button"
                                  className="block w-full rounded px-1 py-0.5 text-left hover:bg-muted"
                                  onClick={() => onOpenProject(p.id)}
                                >
                                  {p.name}
                                </button>
                              ))}
                            </div>
                          ) : annotation ? (
                            <div className="space-y-2">
                              {idea ? (
                                <div className="rounded-lg border border-violet-300 bg-violet-500/5 p-2">
                                  <p className="font-semibold">{idea.name}</p>
                                  <p className="mt-0.5 text-muted-foreground">{idea.desc}</p>
                                  <div className="mt-1.5 flex gap-1">
                                    <Button
                                      size="sm"
                                      className="h-6 text-xs"
                                      onClick={() => {
                                        onKeepIdea({
                                          aspectAId: aspectA.id,
                                          optionAId: cell.a.id,
                                          aspectBId: aspectB.id,
                                          optionBId: cell.b.id,
                                          idea: { name: idea.name, desc: idea.desc },
                                        });
                                        setIdeas((prev) => {
                                          const next = { ...prev };
                                          delete next[key];
                                          return next;
                                        });
                                        setOpenCell(null);
                                      }}
                                    >
                                      Keep as candidate
                                    </Button>
                                    <Button
                                      size="sm"
                                      variant="ghost"
                                      className="h-6 text-xs"
                                      onClick={() =>
                                        setIdeas((prev) => {
                                          const next = { ...prev };
                                          delete next[key];
                                          return next;
                                        })
                                      }
                                    >
                                      Dismiss
                                    </Button>
                                  </div>
                                </div>
                              ) : (
                                <>
                                  <p className="text-muted-foreground">
                                    No real project combines these two options — an open
                                    opportunity.
                                  </p>
                                  <Button
                                    size="sm"
                                    className="h-6 text-xs"
                                    disabled={generatingCell !== null}
                                    onClick={() => handleGenerate(cell)}
                                  >
                                    <Sparkles className="mr-1 h-3 w-3" />
                                    {generatingCell === key ? 'Generating…' : 'Generate into this gap'}
                                  </Button>
                                </>
                              )}
                            </div>
                          ) : (
                            <p className="text-muted-foreground">
                              counts need the corpus annotation — visit the Schema view or wait
                              for it to finish
                            </p>
                          )}
                        </div>
                      )}
                    </td>
                  );
                })}
              </tr>
              );
            })}
          </tbody>
        </table>

        <p className="mt-3 max-w-2xl text-[10px] text-muted-foreground">
          Counts come from the corpus annotation (evidence with receipts — click a cell).
          Generation into a gap is seeded with half-matching precedents; keeping a result
          creates a candidate committed to both options with the concept as its brief.
        </p>
        </div>
      </div>
    </div>
  );
}
