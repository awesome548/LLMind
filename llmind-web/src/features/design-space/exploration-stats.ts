// Exploration statistics — the behavioural measure of "how much of the design
// space has been explored". Doubles as the study instrument (E5): the same
// numbers feed the UI strip, the markdown export, and any later analysis.

import type {
  DesignCandidate,
  MindmapNode,
  NodeProvenance,
  OptionStateEntry,
} from '@/src/features/mindmap/types';
import type { CoordMap } from '@/src/features/design-space/types';
import { candidateCoordKey, listAspects } from './candidate-utils';

export interface ExplorationStats {
  aspects: number;
  options: number;
  generatedOptions: number;
  rejectedOptions: number;
  /** Aspects with a choice in the ACTIVE candidate / total aspects. */
  chosenAspects: number;
  candidates: number;
  cellsDiscovered: number;
  /** Mean pairwise 2D distance between located candidates (≥2), else null. */
  candidateDiversity: number | null;
}

export function computeExplorationStats(input: {
  nodes: ReadonlyArray<MindmapNode>;
  coords: Readonly<CoordMap>;
  discovered: Readonly<Record<string, unknown>>;
  provenance: Readonly<Record<string, NodeProvenance>>;
  optionState: Readonly<Record<string, OptionStateEntry>>;
  candidates: Readonly<Record<string, DesignCandidate>>;
  activeCandidateId: string | null;
}): ExplorationStats {
  const { nodes, coords, discovered, provenance, optionState, candidates, activeCandidateId } =
    input;

  const aspects = listAspects(nodes);
  let options = 0;
  for (const aspect of aspects) options += aspect.children?.length ?? 0;

  const active = activeCandidateId ? candidates[activeCandidateId] : undefined;
  const chosenAspects = active
    ? aspects.filter((aspect) => Boolean(active.choices[aspect.id])).length
    : 0;

  const candidatePoints = Object.values(candidates)
    .map((candidate) => coords[candidateCoordKey(candidate.id)])
    .filter((coord): coord is NonNullable<typeof coord> => Boolean(coord));
  let candidateDiversity: number | null = null;
  if (candidatePoints.length >= 2) {
    let total = 0;
    let pairs = 0;
    for (let i = 0; i < candidatePoints.length; i++) {
      for (let j = i + 1; j < candidatePoints.length; j++) {
        total += Math.hypot(
          candidatePoints[i]!.x - candidatePoints[j]!.x,
          candidatePoints[i]!.y - candidatePoints[j]!.y
        );
        pairs++;
      }
    }
    candidateDiversity = total / pairs;
  }

  return {
    aspects: aspects.length,
    options,
    generatedOptions: Object.keys(provenance).length,
    rejectedOptions: Object.keys(optionState).length,
    chosenAspects,
    candidates: Object.keys(candidates).length,
    cellsDiscovered: Object.keys(discovered).length,
    candidateDiversity,
  };
}

/** One-line rendering shared by the UI strip and the markdown export. */
export function formatExplorationStats(stats: ExplorationStats): string {
  const parts = [
    `${stats.options} options across ${stats.aspects} aspects`,
    `${stats.generatedOptions} generated`,
    `${stats.rejectedOptions} rejected`,
    `${stats.chosenAspects}/${stats.aspects} aspects chosen`,
    `${stats.cellsDiscovered} cells explored`,
  ];
  if (stats.candidateDiversity != null) {
    parts.push(`candidate spread ${stats.candidateDiversity.toFixed(2)}`);
  }
  return parts.join(' · ');
}
