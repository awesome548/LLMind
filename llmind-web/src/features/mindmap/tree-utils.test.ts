import { describe, expect, test } from 'bun:test';
import {
  collectIds,
  ensureUniqueChildIds,
  findNodeByLineage,
  insertChildrenAtNode,
} from './tree-utils';
import type { MindmapNode } from './types';

const tree: MindmapNode[] = [
  {
    id: 'root',
    topic: 'Root',
    children: [
      { id: 'a1', topic: 'Display', children: [{ id: 'o1', topic: 'LED' }] },
      { id: 'a2', topic: 'Interaction' },
    ],
  },
];

describe('findNodeByLineage', () => {
  test('walks a full topic path', () => {
    expect(findNodeByLineage(tree, ['Root', 'Display', 'LED'])?.id).toBe('o1');
  });
  test('returns null for a broken path', () => {
    expect(findNodeByLineage(tree, ['Root', 'Nope'])).toBeNull();
  });
});

describe('insertChildrenAtNode', () => {
  test('inserts immutably under a nested parent', () => {
    const result = insertChildrenAtNode(tree, 'a2', [{ id: 'o2', topic: 'Gesture' }]);
    expect(result.inserted).toBe(true);
    expect(findNodeByLineage(result.nodes, ['Root', 'Interaction', 'Gesture'])?.id).toBe('o2');
    // original untouched
    expect(tree[0]!.children![1]!.children).toBeUndefined();
  });
  test('skips children whose id already exists under the parent', () => {
    const result = insertChildrenAtNode(tree, 'a1', [{ id: 'o1', topic: 'Duplicate' }]);
    expect(result.inserted).toBe(true);
    expect(result.nodes[0]!.children![0]!.children).toHaveLength(1);
  });
  test('reports a missing parent', () => {
    expect(insertChildrenAtNode(tree, 'ghost', [{ id: 'x', topic: 'X' }]).inserted).toBe(false);
  });
});

describe('ensureUniqueChildIds', () => {
  test('remaps colliding ids and reports the remap', () => {
    const { children, remap } = ensureUniqueChildIds(tree, [
      { id: 'o1', topic: 'Collides' },
      { id: 'fresh', topic: 'Fresh' },
    ]);
    expect(children[0]!.id).toBe('o1-2');
    expect(remap).toEqual({ o1: 'o1-2' });
    expect(children[1]!.id).toBe('fresh');
  });
  test('avoids collisions among the new children themselves', () => {
    const { children } = ensureUniqueChildIds(tree, [
      { id: 'o1', topic: 'A' },
      { id: 'o1', topic: 'B' },
    ]);
    expect(new Set(children.map((c) => c.id)).size).toBe(2);
  });
});

describe('collectIds', () => {
  test('collects every id in the tree', () => {
    const ids = new Set<string>();
    collectIds(tree, ids);
    expect(ids).toEqual(new Set(['root', 'a1', 'o1', 'a2']));
  });
});
