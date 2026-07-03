import { describe, expect, test } from 'bun:test';
import { buildStudyBundle, studyBundleFilename } from './study-bundle';
import type { SessionSnapshot } from '@/src/store/mindmap-store';
import type { ExplorationStats } from '@/src/features/design-space/exploration-stats';

const session = { nodes: [], participantId: 'P2' } as unknown as SessionSnapshot;
const stats = { aspects: 6, options: 20 } as unknown as ExplorationStats;

describe('study-bundle', () => {
  test('builds a parseable bundle with all parts', () => {
    const parsed = JSON.parse(
      buildStudyBundle({ participantId: 'P2', session, markdown: '# Record', stats })
    );
    expect(parsed.format).toBe('llmind-study-bundle');
    expect(parsed.participantId).toBe('P2');
    expect(parsed.markdown).toBe('# Record');
    expect(parsed.session).toEqual(session);
    expect(parsed.stats).toEqual(stats);
    expect(typeof parsed.exportedAt).toBe('string');
  });

  test('filename is filesystem-safe', () => {
    expect(studyBundleFilename('P 2/x', '2026-07-03')).toBe('llmind-bundle-P-2-x-2026-07-03.json');
    expect(studyBundleFilename('', '2026-07-03')).toBe('llmind-bundle-anon-2026-07-03.json');
    expect(studyBundleFilename('///', '2026-07-03')).toBe('llmind-bundle-anon-2026-07-03.json');
  });
});
