'use client';

// The design-space schema view (Part 12 A1/A3): aspects as columns, options
// as cells — the canonical representation (Halskov & Lundqvist 2021) and the
// dissertation participant's literal request ("a table where you can see
// everything at once"). Cell styling encodes the living-schema state:
// ring = chosen by the active candidate, struck = rejected, italic =
// generated/manual origin (Halskov's "informed"). Count badges carry the
// corpus annotation (evidence with receipts, never verdicts); ±facet chips
// drive the map's faceted fading (A3).

import { useState } from 'react';
import { Check, Filter, Lightbulb, Minus, Plus, RotateCcw, Undo2, X } from 'lucide-react';
import { Button } from '@/src/components/ui/button';
import type { SchemaColumn } from '@/src/features/design-space/schema-utils';
import type { AnnotationResponse } from '@/src/features/design-space/hooks/use-annotation-query';
import { usePanZoom } from '@/src/features/design-space/hooks/use-pan-zoom';

export interface SchemaFacets {
  include: ReadonlySet<string>;
  exclude: ReadonlySet<string>;
}

interface Props {
  columns: SchemaColumn[];
  annotation: AnnotationResponse | null;
  annotating: boolean;
  annotationError?: string | null;
  facets: SchemaFacets;
  onToggleFacet: (optionId: string, kind: 'include' | 'exclude') => void;
  onSelectOption: (optionId: string) => void;
  selectedOptionId?: string | null;
  /** Choose for the ACTIVE candidate — undefined disables (no candidate). */
  onChoose?: (aspectId: string, optionId: string) => void;
  onReject: (optionId: string) => void;
  onReopen: (optionId: string) => void;
  onAddOption: (aspectId: string, name: string, desc: string) => void;
  /** Open a receipt project in the Related Projects panel. */
  onOpenProject: (projectId: string) => void;
  /** Replay mode (C3): the table shows a PAST state — no mutating actions. */
  readOnly?: boolean;
  /** Replay: the ids the selected timeline step touched — outlined so every
   * scrub visibly answers ("this step was about THAT cell"). */
  highlightIds?: ReadonlySet<string>;
  /** Replay: announce the scrubbed position AT the table (Nielsen H1 — the
   * timeline strip alone is too far from where a click gets ignored). */
  replay?: { step: number; total: number; onLive: () => void } | null;
  /** The rationale layer (Part 13 L-A): per-aspect one-line "why this
   * dimension", grounded in the annotation evidence. */
  rationales?: Readonly<Record<string, string>>;
  /** The coverage probe (Part 13 L-A): N poorly-covered projects + the
   * designer-triggered "what dimension is missing?" action. */
  probe?: { count: number; running: boolean; error?: string | null; onRun: () => void } | null;
}

export function SchemaTable({
  columns,
  annotation,
  annotating,
  annotationError,
  facets,
  onToggleFacet,
  onSelectOption,
  selectedOptionId,
  onChoose,
  onReject,
  onReopen,
  onAddOption,
  onOpenProject,
  readOnly = false,
  highlightIds,
  replay,
  rationales,
  probe,
}: Props) {
  // Receipts popover: anchored to its cell (it pans with the sheet), but
  // FLIPPED toward the viewport center when the cell sits near an edge —
  // an edge-cell popover would otherwise open outside the visible canvas.
  const [receiptsFor, setReceiptsFor] = useState<{
    id: string;
    flipX: boolean;
    flipY: boolean;
  } | null>(null);
  const [addingFor, setAddingFor] = useState<string | null>(null);
  const [draftName, setDraftName] = useState('');
  const [draftDesc, setDraftDesc] = useState('');
  // The same canvas grammar as every other view: wheel zooms, left-drag pans
  // (so nothing is ever stuck under the floating panels), reset restores.
  const { containerRef, view, onPointerDown, onClickCapture, resetView } = usePanZoom(() =>
    setReceiptsFor(null)
  );

  const diag = annotation?.diagnostics;
  const facetsActive = facets.include.size > 0 || facets.exclude.size > 0;
  // Balanced multi-column packing (CSS multicol): aspects fill the vertical
  // space instead of one long clipped row. ~2 aspects per column, max 4 wide.
  const colCount = Math.min(4, Math.max(2, Math.ceil(columns.length / 2)));

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full cursor-grab touch-none overflow-hidden bg-background active:cursor-grabbing"
      onPointerDown={onPointerDown}
      onClickCapture={onClickCapture}
    >
      {/* Status strip — fixed overlay under the view toggle, never pans.
          (pointer-events-auto on the pill: its tooltips and the replay
          button need hover/click; the wrapper stays pass-through.) */}
      <div className="pointer-events-none absolute inset-x-0 top-16 z-10 flex justify-center">
        <div className="pointer-events-auto flex max-w-[90%] flex-wrap items-center gap-2 rounded-full border bg-background/90 px-3 py-1 text-xs text-muted-foreground shadow-sm backdrop-blur">
          <span
            className="font-semibold uppercase tracking-wider"
            title="Cell grammar: violet ring = chosen by the active candidate · struck out = rejected · italic = added during exploration (generated, steered, or typed in) rather than by the initial taxonomy"
          >
            Design-space schema
          </span>
          {replay && (
            <span className="flex items-center gap-1.5 rounded-full border border-amber-300 bg-amber-500/10 px-2 py-0.5 font-medium text-amber-700">
              Replay — {replay.step > 0 ? `step ${replay.step} of ${replay.total}` : 'at the start'} · read-only
              <button
                type="button"
                onClick={replay.onLive}
                className="rounded-full border border-amber-300 px-1.5 font-semibold transition-colors hover:bg-amber-500/15"
                title="Leave the replay and return to the editable present"
              >
                ▶ Back to now
              </button>
            </span>
          )}
          {annotating ? (
            <span className="animate-pulse">annotating corpus… (first run takes minutes; cached after)</span>
          ) : annotation ? (
            <span title="Mean fraction of each option's embedding shortlist the LLM accepted as genuine exemplars">
              {annotation.meta.n_projects} projects annotated · counts are evidence with receipts — click a badge
            </span>
          ) : annotationError ? (
            <span className="text-red-600">annotation unavailable: {annotationError}</span>
          ) : null}
          {facetsActive && (
            <span className="flex items-center gap-1 rounded-full border px-2 py-0.5">
              <Filter className="h-3 w-3" />
              facets active — map fades non-matching projects
            </span>
          )}
          {!readOnly && probe && probe.count > 0 && (
            <button
              type="button"
              onClick={probe.onRun}
              disabled={probe.running}
              title="Some real projects barely fit any option in this taxonomy. Ask what dimension they exemplify that the taxonomy misses — answers arrive as accept/dismiss suggestions."
              className="flex items-center gap-1 rounded-full border border-amber-300 px-2 py-0.5 font-medium text-amber-700 transition-colors enabled:hover:bg-amber-500/10 disabled:opacity-60"
            >
              <Lightbulb className="h-3 w-3" />
              {probe.running
                ? 'probing for a missing dimension…'
                : `${probe.count} project${probe.count === 1 ? ' fits' : 's fit'} poorly — probe for a missing dimension`}
            </button>
          )}
          {!readOnly && probe?.error && !probe.running && (
            <span className="text-red-600">{probe.error} — click to retry</span>
          )}
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

      {/* The schema sheet — panned/zoomed as one canvas */}
      <div
        className="absolute left-0 top-0 w-full"
        style={{ transform: `translate(${view.tx}px, ${view.ty}px) scale(${view.k})`, transformOrigin: '0 0' }}
      >
        <div
          className="mx-auto select-none pb-28 pt-28"
          style={{ width: `${colCount * 17}rem`, columns: colCount, columnGap: '1rem' }}
        >
        {columns.map((col) => (
          <div key={col.id} className="mb-4 break-inside-avoid rounded-xl border bg-card">
            <div className="border-b px-3 py-2" title={col.desc}>
              <p className="text-sm font-semibold">{col.name}</p>
              {/* The rationale layer: the system's why for this dimension,
                  grounded in annotation evidence — the study's "why these
                  seven?" answered where the question arises. */}
              {rationales?.[col.id] && (
                <p
                  className="mt-0.5 line-clamp-2 text-[10px] italic leading-snug text-muted-foreground"
                  title={`Why this dimension (AI, from corpus evidence): ${rationales[col.id]}`}
                >
                  why: {rationales[col.id]}
                </p>
              )}
            </div>
            <ul className="p-2">
              {col.options.map((opt) => {
                const rec = annotation?.options[opt.id];
                const inc = facets.include.has(opt.id);
                const exc = facets.exclude.has(opt.id);
                return (
                  <li
                    key={opt.id}
                    className={`group relative mb-1 rounded-lg border px-2 py-1.5 text-sm transition-all ${
                      selectedOptionId === opt.id ? 'border-primary' : 'border-transparent'
                    } ${opt.chosen ? 'ring-2 ring-violet-400' : ''} ${
                      opt.rejected ? 'opacity-45' : 'hover:bg-muted/60'
                    } ${opt.ghost ? 'opacity-25 grayscale' : ''} ${
                      highlightIds?.has(opt.id)
                        ? 'outline outline-2 outline-offset-1 outline-amber-400'
                        : ''
                    }`}
                  >
                    <button
                      type="button"
                      className="block w-full text-left"
                      onClick={() => onSelectOption(opt.id)}
                      title={
                        opt.ghost
                          ? 'Did not exist yet at this point of the timeline'
                          : opt.desc + (opt.rejectReason ? ` — rejected: ${opt.rejectReason}` : '')
                      }
                    >
                      <span
                        className={`${opt.rejected ? 'line-through' : ''} ${
                          opt.informed ? 'italic' : ''
                        }`}
                      >
                        {opt.name}
                      </span>
                    </button>

                    <span className="absolute right-1 top-1 flex items-center gap-0.5">
                      {rec && (
                        <button
                          type="button"
                          onClick={(e) => {
                            if (receiptsFor?.id === opt.id) {
                              setReceiptsFor(null);
                              return;
                            }
                            const rect = e.currentTarget.getBoundingClientRect();
                            setReceiptsFor({
                              id: opt.id,
                              flipX: rect.left > window.innerWidth * 0.55,
                              flipY: rect.top > window.innerHeight * 0.55,
                            });
                          }}
                          title={`${rec.count} corpus projects exemplify this — click for the list`}
                          className={`rounded-full border px-1.5 text-[10px] tabular-nums ${
                            diag?.too_broad.includes(opt.id)
                              ? 'border-amber-400 text-amber-700'
                              : diag?.unprecedented.includes(opt.id)
                                ? 'border-sky-400 text-sky-700'
                                : 'text-muted-foreground'
                          }`}
                        >
                          {rec.count}
                        </button>
                      )}
                    </span>

                    {/* Hover actions: facets ± / choose / reject-reopen */}
                    {!readOnly && (
                    <span className="mt-1 hidden items-center gap-1 group-hover:flex">
                      <IconBtn
                        active={inc}
                        title="Facet: only projects WITH this option"
                        onClick={() => onToggleFacet(opt.id, 'include')}
                      >
                        <Plus className="h-3 w-3" />
                      </IconBtn>
                      <IconBtn
                        active={exc}
                        title="Facet: only projects WITHOUT this option"
                        onClick={() => onToggleFacet(opt.id, 'exclude')}
                      >
                        <Minus className="h-3 w-3" />
                      </IconBtn>
                      {!opt.rejected && onChoose && (
                        <IconBtn title="Choose for the active candidate" onClick={() => onChoose(col.id, opt.id)}>
                          <Check className="h-3 w-3" />
                        </IconBtn>
                      )}
                      {opt.rejected ? (
                        <IconBtn title="Reopen this option" onClick={() => onReopen(opt.id)}>
                          <RotateCcw className="h-3 w-3" />
                        </IconBtn>
                      ) : (
                        <IconBtn title="Reject this option" onClick={() => onReject(opt.id)}>
                          <X className="h-3 w-3" />
                        </IconBtn>
                      )}
                    </span>
                    )}

                    {receiptsFor?.id === opt.id && rec && (
                      <div
                        className={`absolute z-30 max-h-56 w-64 overflow-y-auto rounded-lg border bg-popover p-2 text-xs shadow-lg ${
                          receiptsFor.flipX ? 'right-0' : 'left-0'
                        } ${receiptsFor.flipY ? 'bottom-full mb-1' : 'top-full mt-1'}`}
                      >
                        <p className="mb-1 font-semibold">
                          {rec.count} exemplifying project{rec.count === 1 ? '' : 's'}
                        </p>
                        {rec.projects.length === 0 && (
                          <p className="text-muted-foreground">
                            none — possibly novel territory, possibly a vague phrasing
                          </p>
                        )}
                        {rec.projects.map((p) => (
                          <button
                            key={p.id}
                            type="button"
                            className="block w-full rounded px-1 py-0.5 text-left hover:bg-muted"
                            onClick={() => {
                              onOpenProject(p.id);
                              setReceiptsFor(null);
                            }}
                          >
                            {p.name}
                          </button>
                        ))}
                      </div>
                    )}
                  </li>
                );
              })}

              {/* Manual informing: add an option (Halskov: informing the space) */}
              {readOnly ? null : addingFor === col.id ? (
                <li className="mt-1 rounded-lg border p-2">
                  <input
                    autoFocus
                    value={draftName}
                    onChange={(e) => setDraftName(e.target.value)}
                    placeholder="Option name"
                    className="mb-1 w-full rounded border px-1.5 py-1 text-xs"
                    onKeyDown={(e) => {
                      if (e.key === 'Escape') {
                        setAddingFor(null);
                        setDraftName('');
                        setDraftDesc('');
                      }
                    }}
                  />
                  <input
                    value={draftDesc}
                    onChange={(e) => setDraftDesc(e.target.value)}
                    placeholder="One-line description (drives placement + evidence)"
                    className="mb-1 w-full rounded border px-1.5 py-1 text-xs"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && draftName.trim()) {
                        onAddOption(col.id, draftName.trim(), draftDesc.trim());
                        setAddingFor(null);
                        setDraftName('');
                        setDraftDesc('');
                      }
                      if (e.key === 'Escape') {
                        setAddingFor(null);
                        setDraftName('');
                        setDraftDesc('');
                      }
                    }}
                  />
                  <div className="flex gap-1">
                    <Button
                      size="sm"
                      className="h-6 text-xs"
                      disabled={!draftName.trim()}
                      onClick={() => {
                        onAddOption(col.id, draftName.trim(), draftDesc.trim());
                        setAddingFor(null);
                        setDraftName('');
                        setDraftDesc('');
                      }}
                    >
                      Add
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="h-6 text-xs"
                      onClick={() => {
                        setAddingFor(null);
                        setDraftName('');
                        setDraftDesc('');
                      }}
                    >
                      Cancel
                    </Button>
                  </div>
                </li>
              ) : (
                <li>
                  <button
                    type="button"
                    className="mt-1 w-full rounded-lg border border-dashed px-2 py-1 text-left text-xs text-muted-foreground hover:bg-muted/60"
                    onClick={() => setAddingFor(col.id)}
                  >
                    + add option
                  </button>
                </li>
              )}
            </ul>
          </div>
        ))}
        </div>
      </div>
    </div>
  );
}

function IconBtn({
  children,
  title,
  onClick,
  active,
}: {
  children: React.ReactNode;
  title: string;
  onClick: () => void;
  active?: boolean;
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className={`rounded border p-0.5 text-muted-foreground hover:bg-muted ${
        active ? 'border-primary bg-primary/10 text-primary' : 'border-transparent'
      }`}
    >
      {children}
    </button>
  );
}
