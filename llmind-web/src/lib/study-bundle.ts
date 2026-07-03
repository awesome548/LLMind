// Study bundle: one file capturing everything a pilot/study session produces —
// the machine-restorable session (which already carries the event log, usage
// counters, and reflections), the human-readable markdown record, and the
// computed exploration stats — tagged with the participant id (ITERATION-M
// M-E12). One download instead of three, named for the participant.

import type { SessionSnapshot } from '@/src/store/mindmap-store';
import type { ExplorationStats } from '@/src/features/design-space/exploration-stats';

const BUNDLE_FORMAT = 'llmind-study-bundle';
const BUNDLE_VERSION = 1;

export interface StudyBundle {
  format: typeof BUNDLE_FORMAT;
  version: number;
  participantId: string;
  exportedAt: string;
  /** Machine-restorable exploration — the event log, usage, and reflections
   * live inside this snapshot; load it back via the normal session loader. */
  session: SessionSnapshot;
  /** Human-readable exploration record (the same markdown the Export button
   * produces). */
  markdown: string;
  /** Computed exploration statistics — a study measure. */
  stats: ExplorationStats;
}

export function buildStudyBundle(input: {
  participantId: string;
  session: SessionSnapshot;
  markdown: string;
  stats: ExplorationStats;
}): string {
  const bundle: StudyBundle = {
    format: BUNDLE_FORMAT,
    version: BUNDLE_VERSION,
    participantId: input.participantId,
    exportedAt: new Date().toISOString(),
    session: input.session,
    markdown: input.markdown,
    stats: input.stats,
  };
  return JSON.stringify(bundle, null, 2);
}

/** Filesystem-safe bundle filename: `llmind-bundle-<participant>-<date>.json`. */
export function studyBundleFilename(participantId: string, isoDate: string): string {
  const safe =
    (participantId || 'anon').replace(/[^a-zA-Z0-9_-]+/g, '-').replace(/^-+|-+$/g, '') ||
    'anon';
  return `llmind-bundle-${safe}-${isoDate}.json`;
}
