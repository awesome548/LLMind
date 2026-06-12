'use client';

// The exploration timeline (Part 12 C3), Fusion-360 style: one marker per
// event, icon-coded by kind, with the playhead between markers. Clicking a
// marker shows the schema AS IT STOOD after that step AND opens its detail
// card — the full label, the designer's reflection if one was kept (review
// old justifications), and a "Reconsider" action for dismissed suggestions.
// Markers left of the playhead are "applied"; the rest are dimmed future.

import { useEffect, useRef, useState } from 'react';
import {
  Check,
  Flag,
  History,
  Lightbulb,
  Minus,
  MoveRight,
  NotebookPen,
  Plus,
  RotateCcw,
  Sparkles,
  Star,
  Trash2,
  X,
} from 'lucide-react';
import type {
  ExplorationEvent,
  ExplorationEventKind,
  Reflection,
} from '@/src/features/mindmap/types';
import type { OptionProposal } from './proposal-chips';

const KIND_STYLE: Record<
  ExplorationEventKind,
  { icon: React.ComponentType<{ className?: string }>; color: string; short: string }
> = {
  choose: { icon: Check, color: 'text-violet-700 border-violet-300 bg-violet-500/10', short: 'Chose' },
  unchoose: { icon: Minus, color: 'text-muted-foreground border-border bg-muted', short: 'Cleared' },
  reject: { icon: X, color: 'text-red-700 border-red-300 bg-red-500/10', short: 'Rejected' },
  reopen: { icon: RotateCcw, color: 'text-sky-700 border-sky-300 bg-sky-500/10', short: 'Reopened' },
  candidate_created: { icon: Star, color: 'text-violet-700 border-violet-300 bg-violet-500/10', short: 'New candidate' },
  candidate_deleted: { icon: Trash2, color: 'text-muted-foreground border-border bg-muted', short: 'Deleted' },
  steer_applied: { icon: MoveRight, color: 'text-indigo-700 border-indigo-300 bg-indigo-500/10', short: 'Steered' },
  cell_kept: { icon: Sparkles, color: 'text-amber-700 border-amber-300 bg-amber-500/10', short: 'Kept gap idea' },
  generated: { icon: Sparkles, color: 'text-emerald-700 border-emerald-300 bg-emerald-500/10', short: 'Generated' },
  option_added: { icon: Plus, color: 'text-emerald-700 border-emerald-300 bg-emerald-500/10', short: 'Added option' },
  // Neutral, not amber: dismissed = archived; amber stays reserved for the
  // positive "kept gap idea" so one hue never carries two opposite meanings.
  proposal_dismissed: { icon: Lightbulb, color: 'text-muted-foreground border-border bg-muted', short: 'Dismissed idea' },
  taxonomy_set: { icon: Flag, color: 'text-muted-foreground border-border bg-muted', short: 'New taxonomy' },
};

export function ReplayTimeline({
  events,
  floor,
  index,
  reflections,
  onOpen,
  onScrub,
  onLive,
  onReconsider,
}: {
  events: ReadonlyArray<ExplorationEvent>;
  /** First replayable position (after the last taxonomy change). */
  floor: number;
  /** Playhead: schema shows the state AFTER events[index-1]. null = closed. */
  index: number | null;
  reflections: Readonly<Record<string, Reflection>>;
  onOpen: () => void;
  onScrub: (index: number) => void;
  onLive: () => void;
  /** Re-offer a dismissed suggestion (the proposal travels in event.detail). */
  onReconsider: (proposal: OptionProposal) => void;
}) {
  const visible = events.slice(floor);
  const activeRef = useRef<HTMLButtonElement | null>(null);
  // The detail card follows the playhead marker; manual close sticks until
  // the playhead moves again.
  const [cardFor, setCardFor] = useState<string | null>(null);
  useEffect(() => {
    activeRef.current?.scrollIntoView({ block: 'nearest', inline: 'center' });
  }, [index]);

  if (visible.length === 0) return null;

  if (index === null) {
    return (
      <button
        type="button"
        onClick={onOpen}
        title="Scrub through everything that happened to this space, step by step"
        className="pointer-events-auto flex items-center gap-1.5 rounded-full border bg-background/90 px-3 py-1 text-xs text-muted-foreground shadow-sm backdrop-blur transition-colors hover:text-foreground"
      >
        <History className="h-3 w-3" />
        Timeline ({visible.length} steps)
      </button>
    );
  }

  // A floor that moved past a stale playhead (new taxonomy mid-replay) must
  // not strand the strip — clamp.
  const playhead = Math.max(index, floor);
  const activeEvent = playhead > floor ? events[playhead - 1]! : null;
  const cardEvent = cardFor !== null ? visible.find((e) => e.id === cardFor) ?? null : activeEvent;
  const cardReflection = cardEvent ? reflections[cardEvent.id] : undefined;
  let cardProposal: OptionProposal | null = null;
  if (cardEvent?.kind === 'proposal_dismissed' && cardEvent.detail) {
    try {
      cardProposal = JSON.parse(cardEvent.detail) as OptionProposal;
    } catch {
      cardProposal = null;
    }
  }

  return (
    <div className="pointer-events-auto flex w-[42rem] max-w-[94%] flex-col gap-1.5 rounded-xl border bg-background/95 px-4 py-2.5 shadow-lg backdrop-blur">
      <div className="flex items-center justify-between gap-2 text-[10px] uppercase tracking-wider text-muted-foreground">
        <span className="flex items-center gap-1.5 font-semibold">
          <History className="h-3 w-3" />
          Timeline — each marker is one step; click one to see the schema as it stood
        </span>
        <button
          type="button"
          onClick={onLive}
          className="rounded-full border px-2 py-0.5 text-[11px] font-medium normal-case tracking-normal text-muted-foreground transition-colors hover:text-foreground"
          title="Leave the replay and return to the present"
        >
          ▶ Back to now
        </button>
      </div>

      {/* The marker strip — horizontally scrollable like a real op timeline */}
      <div className="overflow-x-auto pb-1 pt-7">
        <div className="flex min-w-max items-center gap-1 px-1">
          {/* The "start" stop: before anything happened */}
          <button
            type="button"
            onClick={() => {
              onScrub(floor);
              setCardFor(null);
            }}
            title="Start of the current space — before anything happened"
            className={`h-3.5 w-3.5 shrink-0 rounded-full border-2 transition-all ${
              playhead === floor
                ? 'scale-125 border-violet-600 bg-violet-600'
                : 'border-muted-foreground/40 bg-background hover:border-violet-400'
            }`}
          />
          {visible.map((event, i) => {
            const globalIndex = floor + i + 1; // state AFTER this event
            const style = KIND_STYLE[event.kind];
            const Icon = style.icon;
            const applied = globalIndex <= playhead;
            const isPlayhead = globalIndex === playhead;
            return (
              <span key={event.id} className="relative flex shrink-0 items-center">
                <span className="h-px w-2 bg-border" />
                <button
                  type="button"
                  ref={isPlayhead ? activeRef : undefined}
                  onClick={() => {
                    onScrub(globalIndex);
                    setCardFor(event.id);
                  }}
                  title={`${event.label} · ${new Date(event.ts).toLocaleTimeString()}`}
                  className={`relative flex h-6 w-6 items-center justify-center rounded-md border transition-all ${style.color} ${
                    applied ? '' : 'opacity-30 grayscale'
                  } ${isPlayhead ? 'scale-125 ring-2 ring-violet-500 ring-offset-1' : 'hover:scale-110'}`}
                >
                  <Icon className="h-3.5 w-3.5" />
                  {reflections[event.id] && (
                    <NotebookPen className="absolute -right-1.5 -top-1.5 h-3 w-3 rounded-full bg-background text-violet-600" />
                  )}
                </button>
                {/* Few-words summary above the playhead marker */}
                {isPlayhead && (
                  <span className="pointer-events-none absolute -top-6 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full border bg-background px-2 py-0.5 text-[10px] font-medium shadow-sm">
                    {style.short}
                  </span>
                )}
              </span>
            );
          })}
        </div>
      </div>

      {/* Detail card for the selected step: full label, time, the kept
          reflection, and recovery for dismissed suggestions. */}
      {cardEvent ? (
        <div className="rounded-lg border bg-muted/30 px-3 py-1.5 text-[11px]">
          <p className="text-foreground">
            {cardEvent.label}
            <span className="text-muted-foreground">
              {' '}
              · {new Date(cardEvent.ts).toLocaleTimeString()}
            </span>
          </p>
          {cardReflection && (
            <p className="mt-0.5 flex items-start gap-1 text-muted-foreground">
              <NotebookPen className="mt-0.5 h-3 w-3 shrink-0 text-violet-600" />
              <span>
                “{cardReflection.text}”
                {cardReflection.edited ? '' : ' (AI draft accepted)'}
              </span>
            </p>
          )}
          {cardProposal && (
            <button
              type="button"
              onClick={() => onReconsider(cardProposal!)}
              className="mt-1 flex items-center gap-1 rounded-full border border-amber-300 px-2 py-0.5 text-[10px] font-medium text-amber-700 transition-colors hover:bg-amber-500/10"
            >
              <Lightbulb className="h-3 w-3" />
              Reconsider this suggestion
            </button>
          )}
        </div>
      ) : (
        <p className="px-1 text-[11px] text-muted-foreground">
          Start of the current space — before anything happened. Click a marker to step
          through; rings, strikes, and italics show what was committed, filtered, and
          informed by then (latest commitment per aspect, across all candidates).
        </p>
      )}
    </div>
  );
}
