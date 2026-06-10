import type { DesignCandidate, MindmapNode } from '@/src/features/mindmap/types';

export interface AspectChoiceRow {
  aspectId: string;
  aspectTopic: string;
  optionId: string | null;
  optionTopic: string | null;
}

/** Flat id → node index over the working tree. */
export function indexNodesById(
  nodes: ReadonlyArray<MindmapNode>
): Map<string, MindmapNode> {
  const map = new Map<string, MindmapNode>();
  const walk = (node: MindmapNode) => {
    map.set(node.id, node);
    for (const child of node.children ?? []) walk(child);
  };
  for (const node of nodes) walk(node);
  return map;
}

/** The taxonomy's aspects (depth-1 nodes) in tree order. */
export function listAspects(nodes: ReadonlyArray<MindmapNode>): MindmapNode[] {
  return nodes.flatMap((root) => [...(root.children ?? [])]);
}

/** One row per aspect with the candidate's chosen option (if any). */
export function candidateChoiceRows(
  candidate: DesignCandidate | null,
  nodes: ReadonlyArray<MindmapNode>
): AspectChoiceRow[] {
  const byId = indexNodesById(nodes);
  return listAspects(nodes).map((aspect) => {
    const optionId = candidate?.choices[aspect.id] ?? null;
    const option = optionId ? byId.get(optionId) : undefined;
    return {
      aspectId: aspect.id,
      aspectTopic: aspect.topic,
      optionId: option ? option.id : null,
      optionTopic: option ? option.topic : null,
    };
  });
}

/**
 * The candidate as ONE text — the unit that gets embedded so a composed design
 * (not just its individual options) has a position and real precedents.
 * Returns null when the candidate has no resolvable choices.
 */
export function composeCandidateText(
  candidate: DesignCandidate | null,
  nodes: ReadonlyArray<MindmapNode>,
  descriptionByTopic: Readonly<Record<string, string>> = {},
  descriptionById: Readonly<Record<string, string>> = {}
): string | null {
  if (!candidate) return null;
  const byId = indexNodesById(nodes);
  const parts: string[] = [];
  for (const [aspectId, optionId] of Object.entries(candidate.choices)) {
    const aspect = byId.get(aspectId);
    const option = byId.get(optionId);
    if (!aspect || !option) continue;
    const desc = descriptionById[option.id] ?? descriptionByTopic[option.topic] ?? '';
    parts.push(
      desc
        ? `${aspect.topic}: ${option.topic} — ${desc}`
        : `${aspect.topic}: ${option.topic}`
    );
  }
  if (parts.length === 0) return null;
  return `A media architecture design combining ${parts.join('; ')}`;
}

/** Coordinate-map key for a candidate's position in the design space. */
export const candidateCoordKey = (candidateId: string) => `cand:${candidateId}`;
