import { describe, expect, test } from 'bun:test';
import { computeExplorationStats, formatExplorationStats } from './exploration-stats';
import type { MindmapNode } from '@/src/features/mindmap/types';

const tree: MindmapNode[] = [
  {
    id: 'root',
    topic: 'Root',
    children: [
      {
        id: 'a1',
        topic: 'Display',
        children: [
          { id: 'o1', topic: 'LED' },
          { id: 'o2', topic: 'Projection' },
        ],
      },
      { id: 'a2', topic: 'Interaction', children: [{ id: 'o3', topic: 'Gesture' }] },
    ],
  },
];

describe('computeExplorationStats', () => {
  test('counts the exploration dimensions', () => {
    const stats = computeExplorationStats({
      nodes: tree,
      coords: {
        'cand:c1': { x: 0, y: 0 },
        'cand:c2': { x: 1, y: 0 },
      },
      discovered: { '1,2': {} as never, '3,4': {} as never },
      provenance: { o2: {} as never },
      optionState: { o3: { state: 'rejected' } },
      candidates: {
        c1: { id: 'c1', name: 'A', choices: { a1: 'o1' }, createdAt: 0 },
        c2: { id: 'c2', name: 'B', choices: {}, createdAt: 1 },
      },
      activeCandidateId: 'c1',
    });
    expect(stats).toEqual({
      aspects: 2,
      options: 3,
      generatedOptions: 1,
      rejectedOptions: 1,
      chosenAspects: 1,
      candidates: 2,
      cellsDiscovered: 2,
      candidateDiversity: 1, // distance between (0,0) and (1,0)
    });
  });

  test('diversity is null with fewer than two located candidates', () => {
    const stats = computeExplorationStats({
      nodes: tree,
      coords: {},
      discovered: {},
      provenance: {},
      optionState: {},
      candidates: {},
      activeCandidateId: null,
    });
    expect(stats.candidateDiversity).toBeNull();
    expect(formatExplorationStats(stats)).toContain('0 cells explored');
  });
});
