import { describe, expect, test } from 'bun:test';
import { buildSessionFile, parseSessionFile, sanitizeSnapshot } from './session-io';
import type { SessionSnapshot } from '@/src/store/mindmap-store';

const snapshot = {
  contextText: 'Mindmap',
  contextDescription: '',
  selectedTopic: 'LED',
  taxonomy: null,
  nodes: [{ id: 'root', topic: 'Root' }],
  coords: { root: { x: 0.5, y: 0.5 } },
  discovered: {},
  provenance: {},
  descriptionById: {},
  candidates: {},
  activeCandidateId: null,
  optionState: {},
  axesConfig: null,
  usage: { peek: 3 },
} as unknown as SessionSnapshot;

describe('session-io', () => {
  test('round-trips a snapshot with no warnings', () => {
    const { snapshot: restored, warnings } = parseSessionFile(buildSessionFile(snapshot));
    expect(restored).toEqual(snapshot);
    expect(warnings).toEqual([]);
  });

  test('rejects non-JSON', () => {
    expect(() => parseSessionFile('not json')).toThrow('Not a valid JSON file.');
  });

  test('rejects foreign JSON', () => {
    expect(() => parseSessionFile('{"format":"other"}')).toThrow('Not an LLMind session file.');
  });

  test('rejects newer versions', () => {
    const file = JSON.parse(buildSessionFile(snapshot));
    file.version = 99;
    expect(() => parseSessionFile(JSON.stringify(file))).toThrow('newer');
  });

  test('rejects files without exploration state', () => {
    const file = JSON.parse(buildSessionFile(snapshot));
    file.state = { nodes: 'nope' };
    expect(() => parseSessionFile(JSON.stringify(file))).toThrow('missing its exploration state');
  });

  test('resets a null slice to its default and warns (the M-E3 crash vector)', () => {
    const { snapshot: out, warnings } = sanitizeSnapshot({
      nodes: [{ id: 'root', topic: 'Root' }],
      coords: null,
      events: null,
      candidates: [],
    });
    expect(out.coords).toEqual({});
    expect(out.events).toEqual([]);
    expect(out.candidates).toEqual({});
    expect(warnings.length).toBe(3);
  });

  test('drops malformed candidate entries but keeps good ones', () => {
    const { snapshot: out, warnings } = sanitizeSnapshot({
      nodes: [],
      candidates: {
        good: { id: 'good', name: 'A', choices: {} },
        bad: null,
      },
    });
    expect(Object.keys(out.candidates)).toEqual(['good']);
    expect(warnings.some((w) => w.includes('candidate'))).toBe(true);
  });

  test('repairs a candidate whose choices is null (the persisted-crash vector)', () => {
    const { snapshot: out, warnings } = sanitizeSnapshot({
      nodes: [],
      candidates: { c1: { id: 'c1', name: 'A', choices: null } },
    });
    // Kept (not dropped) with choices coerced to {} — Object.values(choices) is safe.
    expect(out.candidates.c1).toBeDefined();
    expect(out.candidates.c1!.choices).toEqual({});
    expect(warnings.some((w) => w.includes('candidate'))).toBe(true);
  });

  test('clears an active id that names no candidate', () => {
    const { snapshot: out, warnings } = sanitizeSnapshot({
      nodes: [],
      candidates: {},
      activeCandidateId: 'ghost',
    });
    expect(out.activeCandidateId).toBeNull();
    expect(warnings.some((w) => w.includes('active candidate'))).toBe(true);
  });

  test('does not throw when candidates is absent but an active id is present', () => {
    const { snapshot: out } = sanitizeSnapshot({
      nodes: [],
      activeCandidateId: 'ghost',
    });
    expect(out.activeCandidateId).toBeNull();
  });

  test('resets a non-string participantId (the export-crash vector)', () => {
    const { snapshot: out, warnings } = sanitizeSnapshot({
      nodes: [],
      participantId: 42,
    });
    expect(out.participantId).toBe('');
    expect(warnings.some((w) => w.includes('participantId'))).toBe(true);
  });

  test('leaves absent slices absent (store supplies their defaults)', () => {
    const { snapshot: out } = sanitizeSnapshot({ nodes: [] });
    expect('coords' in out).toBe(false);
    expect('rubric' in out).toBe(false);
  });
});
