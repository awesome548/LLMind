'use client';

// Informing-back proposals (Part 12 C1) — the TOCHI loop: investigation
// generates candidate VOCABULARY for the space. Any instrument may emit a
// proposal; the designer accepts it into the taxonomy (with provenance) or
// dismisses it. Chips are transient page state — only ACCEPTED proposals
// touch the persisted exploration.

import { useState } from 'react';
import { Check, Lightbulb, X } from 'lucide-react';

export interface OptionProposal {
  id: string;
  /** What accepting inserts: an option under an aspect (default) or a whole
   * NEW aspect under the root (the coverage probe, Part 13 L-A). */
  kind?: 'option' | 'aspect';
  /** Aspect to add under — null when the emitter couldn't tell (the chip
   * then offers a picker). Ignored for kind 'aspect'. */
  aspectId: string | null;
  text: string;
  desc: string;
  source: 'steer' | 'cell' | 'coverage';
  /** One-line human-readable origin ("steering toward LED wall panels"). */
  evidence: string;
}

export function ProposalChips({
  proposals,
  aspects,
  onAccept,
  onDismiss,
}: {
  proposals: ReadonlyArray<OptionProposal>;
  aspects: ReadonlyArray<{ id: string; name: string }>;
  onAccept: (proposal: OptionProposal, aspectId: string) => void;
  onDismiss: (proposalId: string) => void;
}) {
  // Per-chip aspect pick for proposals whose emitter didn't know the aspect.
  const [picks, setPicks] = useState<Record<string, string>>({});

  if (proposals.length === 0) return null;
  return (
    <>
      {proposals.map((proposal) => {
        const isAspect = proposal.kind === 'aspect';
        const aspectId = proposal.aspectId ?? picks[proposal.id] ?? aspects[0]?.id ?? '';
        const aspectName = aspects.find((a) => a.id === aspectId)?.name ?? '?';
        return (
          <div
            key={proposal.id}
            className="pointer-events-auto w-80 rounded-xl border border-amber-300/60 bg-background/95 p-3 text-xs shadow-lg backdrop-blur"
          >
            <div className="flex items-start justify-between gap-2">
              <p className="flex items-center gap-1.5 font-semibold uppercase tracking-wider text-muted-foreground">
                <Lightbulb className="h-3 w-3 text-amber-600" />
                {isAspect ? 'Add as a new dimension?' : 'Add as option?'}
              </p>
              <button
                type="button"
                onClick={() => onDismiss(proposal.id)}
                title="Dismiss"
                className="text-muted-foreground transition-colors hover:text-foreground"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
            <p className="mt-1 font-medium">{proposal.text}</p>
            {proposal.desc && (
              <p className="mt-0.5 line-clamp-2 text-muted-foreground">{proposal.desc}</p>
            )}
            <p className="mt-1 text-[10px] text-muted-foreground">from {proposal.evidence}</p>
            <div className="mt-1.5 flex items-center gap-1.5">
              {isAspect ? (
                <span className="min-w-0 flex-1 truncate text-muted-foreground">
                  a new column in the schema (no options yet — add or generate them)
                </span>
              ) : proposal.aspectId === null ? (
                <select
                  value={aspectId}
                  onChange={(e) =>
                    setPicks((prev) => ({ ...prev, [proposal.id]: e.target.value }))
                  }
                  className="min-w-0 flex-1 rounded border bg-background px-1.5 py-1 text-[11px]"
                  aria-label="Aspect to add under"
                >
                  {aspects.map((a) => (
                    <option key={a.id} value={a.id}>
                      under {a.name}
                    </option>
                  ))}
                </select>
              ) : (
                <span className="min-w-0 flex-1 truncate text-muted-foreground">
                  under <span className="font-medium text-foreground">{aspectName}</span>
                </span>
              )}
              <button
                type="button"
                disabled={!isAspect && !aspectId}
                onClick={() => onAccept(proposal, aspectId)}
                title="Add to the taxonomy"
                className="shrink-0 rounded border p-1 text-amber-700 transition-colors enabled:hover:bg-amber-500/10 disabled:opacity-40"
              >
                <Check className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        );
      })}
    </>
  );
}
