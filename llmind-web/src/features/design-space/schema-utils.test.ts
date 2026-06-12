import { describe, expect, test } from 'bun:test';
import {
  buildCrossTabCells,
  buildSchemaColumns,
  computeFacetMatches,
  halfMatchingExemplars,
} from './schema-utils';
import type { MindmapNode } from '@/src/features/mindmap/types';

const nodes: ReadonlyArray<MindmapNode> = [
  {
    id: 'root',
    topic: 'Design Aspects',
    children: [
      {
        id: 'display',
        topic: 'Display',
        children: [
          { id: 'led', topic: 'LED' },
          { id: 'laser', topic: 'Laser' },
          { id: 'gen-1', topic: 'Fog screen' },
        ],
      },
      { id: 'context', topic: 'Context', children: [{ id: 'plaza', topic: 'Plaza' }] },
    ],
  },
];

describe('buildSchemaColumns', () => {
  test('derives columns with state styling flags', () => {
    const cols = buildSchemaColumns(
      nodes,
      { Display: 'How it shows.', LED: 'Panels.' },
      { 'gen-1': 'Generated fog.' },
      { laser: { state: 'rejected', reason: 'too costly' } },
      { display: 'led' },
      { 'gen-1': { source: 'generate-at', seedProjects: [], createdAt: 0 } }
    );
    expect(cols.map((c) => c.name)).toEqual(['Display', 'Context']);
    const [led, laser, fog] = cols[0]!.options;
    expect(led).toMatchObject({ chosen: true, rejected: false, informed: false, desc: 'Panels.' });
    expect(laser).toMatchObject({ rejected: true, rejectReason: 'too costly' });
    expect(fog).toMatchObject({ informed: true, desc: 'Generated fog.' });
  });

  test('empty tree yields no columns', () => {
    expect(buildSchemaColumns([], {}, {}, {}, {}, {})).toEqual([]);
  });
});

describe('computeFacetMatches', () => {
  const ann = {
    led: { count: 2, project_ids: ['p1', 'p2'], projects: [] },
    plaza: { count: 2, project_ids: ['p2', 'p3'], projects: [] },
  };
  const universe = ['p1', 'p2', 'p3', 'p4'];

  test('no facets → null (no fading)', () => {
    expect(computeFacetMatches(ann, [], [], universe)).toBeNull();
  });

  test('include intersects across options', () => {
    expect([...computeFacetMatches(ann, ['led', 'plaza'], [], universe)!]).toEqual(['p2']);
  });

  test('exclude removes from included set', () => {
    expect([...computeFacetMatches(ann, ['led'], ['plaza'], universe)!]).toEqual(['p1']);
  });

  test('exclude-only starts from the whole corpus', () => {
    expect([...computeFacetMatches(ann, [], ['led'], universe)!.values()].sort()).toEqual([
      'p3',
      'p4',
    ]);
  });
});

describe('buildCrossTabCells', () => {
  const opt = (id: string, name: string) => ({
    id,
    name,
    desc: '',
    chosen: false,
    rejected: false,
    informed: false,
  });
  const aspectA = { id: 'display', name: 'Display', desc: '', options: [opt('led', 'LED'), opt('laser', 'Laser')] };
  const aspectB = { id: 'context', name: 'Context', desc: '', options: [opt('plaza', 'Plaza')] };
  const ann = {
    led: { count: 2, project_ids: ['p1', 'p2'], projects: [{ id: 'p1', name: 'One' }, { id: 'p2', name: 'Two' }] },
    laser: { count: 1, project_ids: ['p4'], projects: [{ id: 'p4', name: 'Four' }] },
    plaza: { count: 2, project_ids: ['p2', 'p3'], projects: [{ id: 'p2', name: 'Two' }, { id: 'p3', name: 'Three' }] },
  };

  test('cells carry the receipt intersection', () => {
    const grid = buildCrossTabCells(aspectA, aspectB, ann, []);
    expect(grid.length).toBe(2);
    expect(grid[0]![0]!.projects).toEqual([{ id: 'p2', name: 'Two' }]);
    expect(grid[1]![0]!.projects).toEqual([]); // laser × plaza = the gap
  });

  test('candidates committing to both options land in the cell', () => {
    const grid = buildCrossTabCells(aspectA, aspectB, ann, [
      { name: 'Mine', choices: { display: 'led', context: 'plaza' } },
      { name: 'Other', choices: { display: 'laser' } },
    ]);
    expect(grid[0]![0]!.candidateNames).toEqual(['Mine']);
    expect(grid[1]![0]!.candidateNames).toEqual([]);
  });

  test('no annotation → empty cells, no candidates lost', () => {
    const grid = buildCrossTabCells(aspectA, aspectB, null, [
      { name: 'Mine', choices: { display: 'led', context: 'plaza' } },
    ]);
    expect(grid[0]![0]!.projects).toEqual([]);
    expect(grid[0]![0]!.candidateNames).toEqual(['Mine']);
  });
});

describe('halfMatchingExemplars', () => {
  const recA = { count: 3, project_ids: ['p1', 'p2', 'p5'], projects: [] };
  const recB = { count: 3, project_ids: ['p2', 'p3', 'p4'], projects: [] };

  test('interleaves A-only and B-only receipts, never the intersection', () => {
    expect(halfMatchingExemplars(recA, recB)).toEqual(['p1', 'p3', 'p5', 'p4']);
  });

  test('caps at max', () => {
    expect(halfMatchingExemplars(recA, recB, 3)).toEqual(['p1', 'p3', 'p5']);
  });

  test('missing records yield what exists', () => {
    expect(halfMatchingExemplars(undefined, recB)).toEqual(['p2', 'p3', 'p4']);
    expect(halfMatchingExemplars(undefined, undefined)).toEqual([]);
  });
});
