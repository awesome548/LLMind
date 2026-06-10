// Exploration record export — the durable design-rationale artifact a session
// produces: the taxonomy with pruning states and reasons, candidate designs
// with their choices, and the provenance of every generated idea.

import type {
  DesignCandidate,
  MindmapNode,
  NodeProvenance,
  OptionStateEntry,
} from '@/src/features/mindmap/types';
import type { CoordMap } from '@/src/features/design-space/types';
import { candidateChoiceRows } from '@/src/features/design-space/candidate-utils';

export interface ExplorationSnapshot {
  nodes: ReadonlyArray<MindmapNode>;
  descriptionByTopic: Readonly<Record<string, string>>;
  descriptionById: Readonly<Record<string, string>>;
  optionState: Readonly<Record<string, OptionStateEntry>>;
  candidates: Readonly<Record<string, DesignCandidate>>;
  provenance: Readonly<Record<string, NodeProvenance>>;
  coords: Readonly<CoordMap>;
}

export function buildExplorationMarkdown(snapshot: ExplorationSnapshot): string {
  const {
    nodes,
    descriptionByTopic,
    descriptionById,
    optionState,
    candidates,
    provenance,
    coords,
  } = snapshot;

  const lines: string[] = [];
  lines.push(`# Design-Space Exploration`);
  lines.push('');
  lines.push(`Exported: ${new Date().toISOString()}`);
  lines.push('');

  lines.push('## Taxonomy');
  lines.push('');
  const walk = (node: MindmapNode, depth: number) => {
    const indent = '  '.repeat(depth);
    const desc = descriptionById[node.id] ?? descriptionByTopic[node.topic] ?? '';
    const state = optionState[node.id];
    const flags: string[] = [];
    if (state?.state === 'rejected') {
      flags.push(state.reason ? `REJECTED — ${state.reason}` : 'REJECTED');
    }
    const flagText = flags.length ? ` **[${flags.join('; ')}]**` : '';
    const descText = desc ? ` — ${desc}` : '';
    lines.push(`${indent}- ${node.topic}${flagText}${descText}`);
    for (const child of node.children ?? []) walk(child, depth + 1);
  };
  for (const node of nodes) walk(node, 0);
  lines.push('');

  const candidateList = Object.values(candidates).sort(
    (a, b) => a.createdAt - b.createdAt
  );
  if (candidateList.length > 0) {
    lines.push('## Candidate designs');
    lines.push('');
    for (const candidate of candidateList) {
      lines.push(`### ${candidate.name}`);
      const coord = coords[`cand:${candidate.id}`];
      if (coord) {
        lines.push(
          `Position in design space: (${coord.x.toFixed(3)}, ${coord.y.toFixed(3)})`
        );
      }
      for (const row of candidateChoiceRows(candidate, nodes)) {
        lines.push(`- ${row.aspectTopic}: ${row.optionTopic ?? '—'}`);
      }
      if (candidate.note) lines.push(`> ${candidate.note}`);
      lines.push('');
    }
  }

  const provenanceEntries = Object.entries(provenance);
  if (provenanceEntries.length > 0) {
    lines.push('## Provenance of generated ideas');
    lines.push('');
    const topicOf = new Map<string, string>();
    const collect = (node: MindmapNode) => {
      topicOf.set(node.id, node.topic);
      for (const child of node.children ?? []) collect(child);
    };
    for (const node of nodes) collect(node);

    for (const [nodeId, p] of provenanceEntries) {
      const topic = topicOf.get(nodeId);
      if (!topic) continue; // node was removed from the tree
      const seeds = p.seedProjects.map((s) => s.name).join(', ') || '(none)';
      const target = p.target
        ? ` at (${p.target.x.toFixed(3)}, ${p.target.y.toFixed(3)})`
        : '';
      lines.push(
        `- **${topic}** — ${p.source}${target}, seeded by: ${seeds} (${new Date(p.createdAt).toISOString()})`
      );
    }
    lines.push('');
  }

  return lines.join('\n');
}

export function downloadTextFile(filename: string, content: string): void {
  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
