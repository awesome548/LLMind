import { describe, expect, test } from 'bun:test';
import { buildReplayOverlay } from './replay-utils';
import type { ExplorationEvent } from '@/src/features/mindmap/types';

let n = 0;
const ev = (kind: ExplorationEvent['kind'], refs: string[]): ExplorationEvent => ({
  id: `ev-${n++}`,
  ts: n,
  kind,
  label: kind,
  refs,
});

describe('buildReplayOverlay', () => {
  const log = [
    ev('choose', ['led', 'display', 'cand-1']),
    ev('reject', ['laser']),
    ev('option_added', ['fog']),
    ev('choose', ['plaza', 'context', 'cand-1']),
    ev('reject', ['led']), // rejecting clears the display choice
    ev('reopen', ['laser']),
  ];

  test('empty prefix: nothing applied, future additions are ghosts', () => {
    const o = buildReplayOverlay(log, 0);
    expect(o.optionState).toEqual({});
    expect(o.activeChoices).toEqual({});
    expect(o.informed).toEqual({});
    expect(o.notYet).toEqual({ fog: true });
  });

  test('mid-log: choice and rejection visible; later addition still a ghost', () => {
    const o = buildReplayOverlay(log, 2);
    expect(o.activeChoices).toEqual({ display: 'led' });
    expect(o.optionState).toEqual({ laser: { state: 'rejected' } });
    expect(o.notYet).toEqual({ fog: true });
  });

  test('rejecting a chosen option clears its aspect (store invariant)', () => {
    const o = buildReplayOverlay(log, 5);
    expect(o.activeChoices).toEqual({ context: 'plaza' });
    expect(o.optionState.led).toEqual({ state: 'rejected' });
    expect(o.informed).toEqual({ fog: true });
    expect(o.notYet).toEqual({});
  });

  test('reopen lifts the rejection', () => {
    const o = buildReplayOverlay(log, 6);
    expect(o.optionState.laser).toBeUndefined();
  });

  test('generated events mark all their nodes informed', () => {
    const o = buildReplayOverlay([ev('generated', ['g1', 'g2'])], 1);
    expect(o.informed).toEqual({ g1: true, g2: true });
  });

  test('a new taxonomy resets the overlay but later events still apply', () => {
    const o = buildReplayOverlay(
      [...log, ev('taxonomy_set', []), ev('choose', ['x', 'aspect-x', 'cand-2'])],
      8
    );
    expect(o.activeChoices).toEqual({ 'aspect-x': 'x' });
    expect(o.optionState).toEqual({});
    expect(o.informed).toEqual({});
  });
});
