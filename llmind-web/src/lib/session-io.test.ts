import { describe, expect, test } from 'bun:test';
import { buildSessionFile, parseSessionFile } from './session-io';
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
  test('round-trips a snapshot', () => {
    const restored = parseSessionFile(buildSessionFile(snapshot));
    expect(restored).toEqual(snapshot);
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
});
