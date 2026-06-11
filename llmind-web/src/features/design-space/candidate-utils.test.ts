import { describe, expect, test } from 'bun:test';
import {
  candidateChoiceRows,
  candidateCoordKey,
  composeCandidateText,
  indexNodesById,
  listAspects,
} from './candidate-utils';
import type { DesignCandidate, MindmapNode } from '@/src/features/mindmap/types';

const tree: MindmapNode[] = [
  {
    id: 'root',
    topic: 'Root',
    children: [
      { id: 'a1', topic: 'Display', children: [{ id: 'o1', topic: 'LED' }] },
      { id: 'a2', topic: 'Interaction', children: [{ id: 'o2', topic: 'Gesture' }] },
    ],
  },
];

const candidate: DesignCandidate = {
  id: 'c1',
  name: 'Test',
  choices: { a1: 'o1' },
  createdAt: 0,
};

describe('candidate-utils', () => {
  test('listAspects returns depth-1 nodes in order', () => {
    expect(listAspects(tree).map((a) => a.id)).toEqual(['a1', 'a2']);
  });

  test('indexNodesById covers the whole tree', () => {
    expect(indexNodesById(tree).get('o2')?.topic).toBe('Gesture');
  });

  test('candidateChoiceRows resolves chosen options and leaves gaps null', () => {
    const rows = candidateChoiceRows(candidate, tree);
    expect(rows).toEqual([
      { aspectId: 'a1', aspectTopic: 'Display', optionId: 'o1', optionTopic: 'LED' },
      { aspectId: 'a2', aspectTopic: 'Interaction', optionId: null, optionTopic: null },
    ]);
  });

  test('composeCandidateText uses id-keyed descriptions over topic-keyed', () => {
    const text = composeCandidateText(candidate, tree, { LED: 'topic desc' }, { o1: 'id desc' });
    expect(text).toContain('Display: LED — id desc');
  });

  test('composeCandidateText is null with no resolvable choices', () => {
    expect(composeCandidateText({ ...candidate, choices: { ghost: 'x' } }, tree)).toBeNull();
    expect(composeCandidateText(null, tree)).toBeNull();
  });

  test('candidateCoordKey is namespaced', () => {
    expect(candidateCoordKey('c1')).toBe('cand:c1');
  });
});
