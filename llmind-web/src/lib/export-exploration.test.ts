import { describe, expect, test } from 'bun:test';
import { buildExplorationMarkdown } from './export-exploration';
import type { MindmapNode } from '@/src/features/mindmap/types';

const tree: MindmapNode[] = [
  {
    id: 'root',
    topic: 'Root',
    children: [{ id: 'a1', topic: 'Display', children: [{ id: 'o1', topic: 'LED' }] }],
  },
];

describe('buildExplorationMarkdown', () => {
  test('records taxonomy states, candidates, provenance, and stats', () => {
    const markdown = buildExplorationMarkdown({
      nodes: tree,
      descriptionByTopic: { Display: 'How light is shown' },
      descriptionById: { o1: 'Bright panels' },
      optionState: { o1: { state: 'rejected', reason: 'too costly' } },
      candidates: {
        c1: { id: 'c1', name: 'My design', choices: { a1: 'o1' }, createdAt: 0 },
      },
      provenance: {
        o1: {
          source: 'generate-at',
          seedProjects: [{ id: '7', name: 'Seed Project' }],
          target: { x: 0.25, y: 0.75 },
          createdAt: 0,
        },
      },
      coords: { 'cand:c1': { x: 0.1, y: 0.2 } },
      discovered: { '3,4': { from: { x: 0, y: 0 }, to: [] } },
      activeCandidateId: 'c1',
    });

    expect(markdown).toContain('**[REJECTED — too costly]**');
    expect(markdown).toContain('### My design');
    expect(markdown).toContain('Position in design space: (0.100, 0.200)');
    expect(markdown).toContain('seeded by: Seed Project');
    expect(markdown).toContain('**Stats:**');
    expect(markdown).toContain('1 cells explored');
  });

  test('includes reflections attached to events (C2)', () => {
    const markdown = buildExplorationMarkdown({
      nodes: tree,
      descriptionByTopic: {},
      descriptionById: {},
      optionState: {},
      candidates: {},
      provenance: {},
      coords: {},
      discovered: {},
      activeCandidateId: null,
      events: [
        { id: 'ev-1', ts: 0, kind: 'choose', label: 'Chose "LED" for Display', refs: ['o1', 'a1', 'c1'] },
        { id: 'ev-2', ts: 1, kind: 'reject', label: 'Rejected "Laser"', refs: ['o2'] },
      ],
      reflections: {
        'ev-1': { text: 'Daylight legibility matters most here.', edited: false, ts: 5 },
      },
    });
    expect(markdown).toContain('## Reflections');
    expect(markdown).toContain('Chose "LED" for Display — “Daylight legibility matters most here.”');
    expect(markdown).toContain('*(AI draft accepted)*');
    expect(markdown).not.toContain('Rejected "Laser" —'); // no reflection → not listed
  });

  test('omits provenance for nodes no longer in the tree', () => {
    const markdown = buildExplorationMarkdown({
      nodes: tree,
      descriptionByTopic: {},
      descriptionById: {},
      optionState: {},
      candidates: {},
      provenance: {
        ghost: { source: 'generate-nodes', seedProjects: [], createdAt: 0 },
      },
      coords: {},
      discovered: {},
      activeCandidateId: null,
    });
    expect(markdown).not.toContain('ghost');
  });
});
