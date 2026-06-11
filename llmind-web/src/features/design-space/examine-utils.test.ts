import { describe, expect, test } from 'bun:test';
import type { MindmapNode } from '@/src/features/mindmap/types';
import {
  candidateAlignmentAspects,
  candidateEmbeddingText,
} from './candidate-utils';
import {
  buildConsistencyDefs,
  percentileOf,
  resolveRubricDefs,
} from './examine-utils';

const nodes: MindmapNode[] = [
  {
    id: 'root',
    topic: 'Design Aspects',
    children: [
      {
        id: 'a1',
        topic: 'Display',
        children: [
          { id: 'o1', topic: 'LED mesh' },
          { id: 'o2', topic: 'Projection' },
          { id: 'o3', topic: 'E-ink' },
        ],
      },
      { id: 'a2', topic: 'Input', children: [{ id: 'o4', topic: 'Gesture' }] },
    ],
  },
];

const candidate = {
  id: 'c1',
  name: 'C1',
  choices: { a1: 'o1', a2: 'o4' },
  brief: 'A breathing facade of LED mesh.',
  createdAt: 0,
};

describe('candidateEmbeddingText (brief-first)', () => {
  test('prefers the brief when present', () => {
    expect(candidateEmbeddingText(candidate, nodes)).toBe(
      'A breathing facade of LED mesh.'
    );
  });

  test('falls back to the composition without a brief', () => {
    expect(candidateEmbeddingText({ ...candidate, brief: undefined }, nodes)).toContain(
      'Display: LED mesh'
    );
  });
});

describe('candidateAlignmentAspects', () => {
  test('one row per chosen aspect, siblings as alternatives with embed text', () => {
    const rows = candidateAlignmentAspects(candidate, nodes, {
      Projection: 'light cast onto surfaces',
    });
    expect(rows).toHaveLength(2);
    const display = rows.find((r) => r.aspect_id === 'a1')!;
    expect(display.chosen).toEqual({ id: 'o1', text: 'LED mesh' });
    expect(display.alternatives.map((a) => a.id).sort()).toEqual(['o2', 'o3']);
    expect(display.alternatives.find((a) => a.id === 'o2')!.text).toBe(
      'Projection. light cast onto surfaces'
    );
    // The only option of Input has no alternatives.
    expect(rows.find((r) => r.aspect_id === 'a2')!.alternatives).toEqual([]);
  });
});

describe('examine strips', () => {
  test('consistency defs come from the alignment top alternatives', () => {
    const defs = buildConsistencyDefs(
      {
        agreement: 0.8,
        per_aspect: [
          {
            aspect_id: 'a1',
            chosen_score: 0.6,
            top_alternative: { id: 'o2', score: 0.7 },
            leans_away: true,
          },
          { aspect_id: 'a2', chosen_score: 0.5, top_alternative: null, leans_away: false },
        ],
      },
      candidate.choices,
      nodes
    );
    expect(defs).toHaveLength(1);
    expect(defs[0]).toMatchObject({
      key: 'consistency:a1',
      label: 'Display',
      poleALabel: 'LED mesh',
      poleBLabel: 'Projection',
      leansAway: true,
    });
  });

  test('rubric defs resolve ids and skip dangling metrics', () => {
    const defs = resolveRubricDefs(
      [
        { id: 'm1', aspectId: 'a1', poleAId: 'o1', poleBId: 'o3' },
        { id: 'm2', aspectId: 'a1', poleAId: 'o1', poleBId: 'deleted' },
      ],
      nodes
    );
    expect(defs).toHaveLength(1);
    expect(defs[0]).toMatchObject({
      key: 'rubric:m1',
      poleALabel: 'LED mesh',
      poleBLabel: 'E-ink',
      rubricId: 'm1',
    });
  });

  test('percentileOf counts the corpus at or below the score', () => {
    const corpus = [-1, -0.5, 0, 0.5, 1];
    expect(percentileOf(corpus, 0)).toBe(0.6);
    expect(percentileOf(corpus, 1)).toBe(1);
    expect(percentileOf(corpus, -2)).toBe(0);
    expect(Number.isNaN(percentileOf([], 0))).toBe(true);
  });
});
