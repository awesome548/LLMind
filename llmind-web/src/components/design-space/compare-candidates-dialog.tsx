'use client';

import { useMemo } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/src/components/ui/dialog';
import {
  candidateChoiceRows,
  candidateCoordKey,
  composeCandidateText,
  listAspects,
} from '@/src/features/design-space/candidate-utils';
import { useManyCandidatePrecedents } from '@/src/features/design-space/hooks/use-candidate-precedents';
import { useMindmapStore } from '@/src/store/mindmap-store';

interface CompareCandidatesDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  descriptionByTopic: Readonly<Record<string, string>>;
}

/** Side-by-side comparison of candidate designs: choices per aspect, closest
 * real precedents, and approximate pairwise distance in the design space. */
export function CompareCandidatesDialog({
  open,
  onOpenChange,
  descriptionByTopic,
}: CompareCandidatesDialogProps) {
  const nodes = useMindmapStore((s) => s.nodes);
  const candidates = useMindmapStore((s) => s.candidates);
  const descriptionById = useMindmapStore((s) => s.descriptionById);
  const coords = useMindmapStore((s) => s.coords);

  const candidateList = useMemo(
    () => Object.values(candidates).sort((a, b) => a.createdAt - b.createdAt),
    [candidates]
  );
  const aspects = useMemo(() => listAspects(nodes), [nodes]);
  const rowsByCandidate = useMemo(
    () => candidateList.map((c) => candidateChoiceRows(c, nodes)),
    [candidateList, nodes]
  );

  const texts = useMemo(
    () =>
      open
        ? candidateList.map((c) =>
            composeCandidateText(c, nodes, descriptionByTopic, descriptionById)
          )
        : candidateList.map(() => null),
    [open, candidateList, nodes, descriptionByTopic, descriptionById]
  );
  const precedentResults = useManyCandidatePrecedents(texts, 3);

  const distances = useMemo(() => {
    const out: Array<{ a: string; b: string; d: number | null }> = [];
    for (let i = 0; i < candidateList.length; i++) {
      for (let j = i + 1; j < candidateList.length; j++) {
        const ca = coords[candidateCoordKey(candidateList[i]!.id)];
        const cb = coords[candidateCoordKey(candidateList[j]!.id)];
        out.push({
          a: candidateList[i]!.name,
          b: candidateList[j]!.name,
          d: ca && cb ? Math.hypot(ca.x - cb.x, ca.y - cb.y) : null,
        });
      }
    }
    return out;
  }, [candidateList, coords]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] overflow-y-auto sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Compare candidates</DialogTitle>
          <DialogDescription>
            Choices per aspect, each design&apos;s closest real precedents, and how far
            apart the designs sit in the space.
          </DialogDescription>
        </DialogHeader>

        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-xs">
            <thead>
              <tr>
                <th className="border-b p-2 text-left font-semibold text-muted-foreground">
                  Aspect
                </th>
                {candidateList.map((c) => (
                  <th key={c.id} className="border-b p-2 text-left font-semibold">
                    {c.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {aspects.map((aspect, rowIdx) => (
                <tr key={aspect.id} className={rowIdx % 2 ? 'bg-muted/40' : undefined}>
                  <td className="p-2 text-muted-foreground">{aspect.topic}</td>
                  {rowsByCandidate.map((rows, i) => {
                    const row = rows.find((r) => r.aspectId === aspect.id);
                    return (
                      <td key={candidateList[i]!.id} className="p-2">
                        {row?.optionTopic ?? <span className="text-muted-foreground/50">—</span>}
                      </td>
                    );
                  })}
                </tr>
              ))}
              <tr>
                <td className="border-t p-2 align-top font-semibold text-muted-foreground">
                  Closest precedents
                </td>
                {candidateList.map((c, i) => {
                  const result = precedentResults[i];
                  return (
                    <td key={c.id} className="border-t p-2 align-top">
                      {result?.isFetching ? (
                        <span className="text-muted-foreground">…</span>
                      ) : (
                        <ul className="space-y-0.5">
                          {(result?.data ?? []).map((p) => (
                            <li key={p.id} className="truncate">
                              {p.Name}{' '}
                              <span className="tabular-nums text-muted-foreground">
                                {(p.score * 100).toFixed(0)}%
                              </span>
                            </li>
                          ))}
                        </ul>
                      )}
                    </td>
                  );
                })}
              </tr>
            </tbody>
          </table>
        </div>

        {distances.length > 0 && (
          <div className="space-y-1">
            <p className="text-xs font-semibold text-muted-foreground">
              Distance between designs{' '}
              <span className="font-normal">
                (in the 2D projection — approximate; see layout fidelity)
              </span>
            </p>
            <ul className="space-y-0.5 text-xs">
              {distances.map(({ a, b, d }) => (
                <li key={`${a}-${b}`}>
                  {a} ↔ {b}:{' '}
                  <span className="tabular-nums">
                    {d == null ? 'not yet placed' : d.toFixed(3)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
