'use client';

// Reflection capture (Part 12 C2), burden-inverted per Dalsgaard & Halskov:
// the system drafts the one-line "why", the designer accepts (Enter), edits
// (typing), or skips (Esc / ✕). Never modal, never required, never blocking —
// the draft streams in only if the designer hasn't started typing.
//
// The input's value is LOCAL state (the page re-rendering its canvases per
// keystroke was measurable lag) — render with key={prompt.eventId} so each
// event gets a fresh chip.

import { useState } from 'react';
import { Check, Loader2, NotebookPen, X } from 'lucide-react';

export interface ReflectionPromptState {
  eventId: string;
  label: string;
  /** Whether the AI draft has arrived (false = still drafting or failed). */
  drafted: boolean;
  /** The untouched AI draft, to detect edits on accept. */
  draftValue: string;
}

export function ReflectionChip({
  prompt,
  onAccept,
  onSkip,
}: {
  prompt: ReflectionPromptState;
  onAccept: (value: string) => void;
  onSkip: () => void;
}) {
  // null = untouched: the shown value DERIVES from the AI draft until the
  // designer types, at which point their text wins permanently (no effect,
  // no overwrite race).
  const [typed, setTyped] = useState<string | null>(null);
  const value = typed ?? (prompt.drafted ? prompt.draftValue : '');

  return (
    <div className="pointer-events-auto w-80 rounded-xl border bg-background/95 p-3 text-xs shadow-lg backdrop-blur">
      <div className="flex items-start justify-between gap-2">
        <p className="flex items-center gap-1.5 font-semibold uppercase tracking-wider text-muted-foreground">
          <NotebookPen className="h-3 w-3" />
          Why? <span className="font-normal normal-case">(optional)</span>
        </p>
        <button
          type="button"
          onClick={onSkip}
          title="Skip (Esc)"
          className="text-muted-foreground transition-colors hover:text-foreground"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
      <p className="mt-1 truncate text-muted-foreground" title={prompt.label}>
        {prompt.label}
      </p>
      <div className="mt-1.5 flex items-center gap-1.5">
        <input
          value={value}
          onChange={(e) => setTyped(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && value.trim()) onAccept(value);
            if (e.key === 'Escape') onSkip();
          }}
          placeholder={prompt.drafted ? 'your one-line why…' : 'drafting… or type your own'}
          className="w-full rounded border bg-background px-2 py-1 text-xs focus:outline-none focus:ring-1 focus:ring-violet-400"
        />
        {!prompt.drafted && value === '' ? (
          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-muted-foreground" />
        ) : (
          <button
            type="button"
            onClick={() => onAccept(value)}
            disabled={!value.trim()}
            title="Keep this reflection (Enter)"
            className="shrink-0 rounded border p-1 text-violet-700 transition-colors enabled:hover:bg-violet-500/10 disabled:opacity-40"
          >
            <Check className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    </div>
  );
}
