import { beforeEach, describe, expect, test } from 'bun:test';
// No DOM here: zustand's persist middleware logs that storage is unavailable
// and no-ops, which is exactly right — these tests cover state logic only.
import { selectSessionSnapshot, useMindmapStore } from './mindmap-store';

const store = useMindmapStore;

beforeEach(() => {
  store.getState().resetMindmapStore();
});

describe('candidate invariants (E3)', () => {
  test('a rejected option cannot be chosen', () => {
    const { createCandidate, rejectOption, setChoice } = store.getState();
    createCandidate();
    rejectOption('opt-x', 'bad fit');
    setChoice('aspect-a', 'opt-x');
    const active = store.getState().candidates[store.getState().activeCandidateId!]!;
    expect(active.choices).toEqual({});
  });

  test('rejecting clears the option from every candidate', () => {
    const { createCandidate, setChoice, rejectOption } = store.getState();
    const first = createCandidate();
    setChoice('aspect-a', 'opt-x');
    const second = createCandidate();
    setChoice('aspect-a', 'opt-x');
    rejectOption('opt-x');
    expect(store.getState().candidates[first]!.choices).toEqual({});
    expect(store.getState().candidates[second]!.choices).toEqual({});
    expect(store.getState().optionState['opt-x']).toEqual({ state: 'rejected' });
  });

  test('reopen makes the option choosable again', () => {
    const { createCandidate, rejectOption, reopenOption, setChoice } = store.getState();
    createCandidate();
    rejectOption('opt-x');
    reopenOption('opt-x');
    setChoice('aspect-a', 'opt-x');
    const active = store.getState().candidates[store.getState().activeCandidateId!]!;
    expect(active.choices).toEqual({ 'aspect-a': 'opt-x' });
  });
});

describe('pruneMissingNodes (E4)', () => {
  test('drops state for deleted nodes but keeps candidate coords', () => {
    const state = store.getState();
    state.mergeCoords({
      kept: { x: 0.1, y: 0.1 },
      gone: { x: 0.2, y: 0.2 },
      'cand:c1': { x: 0.3, y: 0.3 },
    });
    state.recordProvenance({
      gone: { source: 'generate-at', seedProjects: [], createdAt: 0 },
    });
    state.mergeDescriptions({ gone: 'desc' });
    state.rejectOption('gone');
    const candidateId = state.createCandidate();
    store.getState().setChoice('kept-aspect', 'kept');

    store.getState().pruneMissingNodes(new Set(['kept', 'kept-aspect']));

    const next = store.getState();
    expect(Object.keys(next.coords).sort()).toEqual(['cand:c1', 'kept']);
    expect(next.provenance).toEqual({});
    expect(next.descriptionById).toEqual({});
    expect(next.optionState).toEqual({});
    expect(next.candidates[candidateId]!.choices).toEqual({ 'kept-aspect': 'kept' });
  });
});

describe('dual-layer candidates + rubric (Part 10)', () => {
  test('brief is set per candidate and survives renames', () => {
    const id = store.getState().createCandidate('A');
    store.getState().setCandidateBrief(id, 'A kinetic facade of recycled lenses.');
    store.getState().renameCandidate(id, 'A2');
    expect(store.getState().candidates[id]!.brief).toBe(
      'A kinetic facade of recycled lenses.'
    );
  });

  test('trail appends and caps at 10', () => {
    const id = store.getState().createCandidate();
    for (let i = 0; i < 13; i++) {
      store.getState().appendCandidateTrail(id, { x: i / 20, y: 0.5 });
    }
    const trail = store.getState().candidates[id]!.trail!;
    expect(trail).toHaveLength(10);
    expect(trail[0]).toEqual({ x: 3 / 20, y: 0.5 });
    expect(trail[9]).toEqual({ x: 12 / 20, y: 0.5 });
  });

  test('rubric metrics persist in snapshots and GC with the tree', () => {
    store.getState().addRubricMetric({
      id: 'm1', aspectId: 'a1', poleAId: 'o1', poleBId: 'o2',
    });
    store.getState().addRubricMetric({
      id: 'm2', aspectId: 'a1', poleAId: 'o1', poleBId: 'gone',
    });
    expect(selectSessionSnapshot(store.getState()).rubric).toHaveLength(2);
    store.getState().pruneMissingNodes(new Set(['a1', 'o1', 'o2']));
    expect(store.getState().rubric).toEqual([
      { id: 'm1', aspectId: 'a1', poleAId: 'o1', poleBId: 'o2' },
    ]);
    store.getState().removeRubricMetric('m1');
    expect(store.getState().rubric).toEqual([]);
  });

  test('restoring a pre-rubric session resets the rubric to default', () => {
    store.getState().addRubricMetric({
      id: 'm1', aspectId: 'a1', poleAId: 'o1', poleBId: 'o2',
    });
    const snapshot = selectSessionSnapshot(store.getState());
    // Simulate an old session file saved before the rubric slice existed.
    delete (snapshot as Partial<typeof snapshot>).rubric;
    store.getState().restoreSession(snapshot);
    expect(store.getState().rubric).toEqual([]);
  });
});

describe('exploration log + reflections (Part 12 C2/C3)', () => {
  test('discrete actions record events with kind-specific refs', () => {
    const { createCandidate, setChoice, rejectOption, reopenOption } = store.getState();
    const candidateId = createCandidate('Mine');
    setChoice('aspect-a', 'opt-x');
    rejectOption('opt-y', 'too costly');
    reopenOption('opt-y');
    const kinds = store.getState().events.map((e) => e.kind);
    expect(kinds).toEqual(['candidate_created', 'choose', 'reject', 'reopen']);
    const choose = store.getState().events[1]!;
    expect(choose.refs).toEqual(['opt-x', 'aspect-a', candidateId]);
    expect(store.getState().events[2]!.label).toContain('too costly');
  });

  test('recordEvent returns the id reflections attach to', () => {
    const id = store.getState().recordEvent('steer_applied', 'Steered "A"', ['c1']);
    store.getState().addReflection(id, '  sharper at night  ', true);
    expect(store.getState().reflections[id]).toMatchObject({
      text: 'sharper at night',
      edited: true,
    });
  });

  test('the log caps at 500 events and GCs reflections of dropped events', () => {
    const firstId = store.getState().recordEvent('option_added', 'Added 0', []);
    store.getState().addReflection(firstId, 'the first why', false);
    for (let i = 1; i < 510; i++) {
      store.getState().recordEvent('option_added', `Added ${i}`, []);
    }
    const events = store.getState().events;
    expect(events).toHaveLength(500);
    expect(events[events.length - 1]!.label).toBe('Added 509');
    // The first event fell off the front — its reflection must not leak.
    expect(store.getState().reflections[firstId]).toBeUndefined();
  });

  test('restoreSession drops reflections for events the snapshot lacks', () => {
    const id = store.getState().recordEvent('choose', 'Chose X', ['o', 'a', 'c']);
    store.getState().addReflection(id, 'why', false);
    const snapshot = { ...selectSessionSnapshot(store.getState()), events: [] };
    store.getState().restoreSession(snapshot);
    expect(store.getState().reflections).toEqual({});
  });

  test('events + reflections round-trip in snapshots, defaults-first', () => {
    const id = store.getState().recordEvent('choose', 'Chose "LED"', ['o1', 'a1', 'c1']);
    store.getState().addReflection(id, 'why', false);
    const snapshot = selectSessionSnapshot(store.getState());
    expect(snapshot.events).toHaveLength(1);
    expect(snapshot.reflections[id]).toBeDefined();
    // A pre-C session file (no events/reflections) restores to empty, not stale.
    const old = { ...snapshot } as Partial<typeof snapshot>;
    delete old.events;
    delete old.reflections;
    store.getState().restoreSession(old as typeof snapshot);
    expect(store.getState().events).toEqual([]);
    expect(store.getState().reflections).toEqual({});
  });
});

describe('usage + session', () => {
  test('store actions count their own usage', () => {
    const { createCandidate, setChoice, rejectOption, trackUsage } = store.getState();
    createCandidate();
    setChoice('a', 'o');
    rejectOption('other');
    trackUsage('peek');
    trackUsage('peek');
    expect(store.getState().usage).toEqual({
      candidate_created: 1,
      choice_set: 1,
      option_rejected: 1,
      peek: 2,
    });
  });

  test('restoreSession replaces the exploration wholesale', () => {
    store.getState().createCandidate('Before');
    const snapshot = {
      ...selectSessionSnapshot(store.getState()),
      candidates: {},
      activeCandidateId: null,
      selectedTopic: 'Restored',
      usage: { session_load: 1 },
    };
    store.getState().restoreSession(snapshot);
    const next = store.getState();
    expect(next.candidates).toEqual({});
    expect(next.selectedTopic).toBe('Restored');
    expect(next.usage).toEqual({ session_load: 1 });
  });
});
