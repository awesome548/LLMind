'use client';

// The steering veto card (Part 12 B3): every steer result is shown as a diff
// with its requested-vs-achieved measurement before anything commits — the
// peek transparency pattern. Shared by the strip rails (metric mode) and the
// precedent pulls (toward/away).

import { ArrowRight } from 'lucide-react';
import { Button } from '@/src/components/ui/button';
import type { SteerResult } from '@/src/features/design-space/hooks/use-steer-mutation';

export function SteerResultCard({
  result,
  briefBefore,
  onApply,
  onDiscard,
}: {
  result: SteerResult;
  briefBefore: string;
  onApply: () => void;
  onDiscard: () => void;
}) {
  const m = result.measurement;
  return (
    <div className="rounded-xl border border-violet-300 bg-violet-500/5 p-3 text-xs">
      <p className="font-semibold uppercase tracking-wider text-muted-foreground">
        Proposed revision — yours to veto
      </p>
      <p className="mt-1 whitespace-pre-wrap text-muted-foreground line-through opacity-60">
        {briefBefore}
      </p>
      <p className="mt-1 whitespace-pre-wrap">{result.revised_text}</p>

      {result.named_qualities.length > 0 && (
        <p className="mt-1.5 text-muted-foreground">
          named qualities: {result.named_qualities.join(' · ')}
        </p>
      )}

      {m ? (
        <p
          className="mt-1.5 tabular-nums text-muted-foreground"
          title="along/orthogonal are raw cosine-space magnitudes of the brief's displacement"
        >
          requested {m.requested >= 0 ? '+' : ''}
          {m.requested.toFixed(2)} <ArrowRight className="inline h-3 w-3" /> achieved{' '}
          {m.achieved >= 0 ? '+' : ''}
          {m.achieved.toFixed(2)}
          {Math.abs(m.requested - m.achieved) > 0.15 ? ' — language only moved part of the way' : ''}
          <br />
          along the asked direction {m.along.toFixed(3)} · side effects (orthogonal){' '}
          {m.orthogonal.toFixed(3)}
        </p>
      ) : (
        <p className="mt-1.5 text-muted-foreground">
          measurement unavailable (embedding service failed) — the revision itself still stands
        </p>
      )}

      <div className="mt-2 flex gap-1">
        <Button size="sm" className="h-6 text-xs" onClick={onApply}>
          Apply to brief
        </Button>
        <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={onDiscard}>
          Discard
        </Button>
      </div>
    </div>
  );
}
