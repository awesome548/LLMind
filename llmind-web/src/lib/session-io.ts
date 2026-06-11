// Session save/load: the full exploration state as a portable JSON file.
// Enables study capture, crash recovery, and sharing — the markdown export is
// the human-readable record; this is the machine-restorable one.

import type { SessionSnapshot } from '@/src/store/mindmap-store';

const SESSION_FORMAT = 'llmind-session';
const SESSION_VERSION = 1;

interface SessionFile {
  format: typeof SESSION_FORMAT;
  version: number;
  exportedAt: string;
  state: SessionSnapshot;
}

export function buildSessionFile(state: SessionSnapshot): string {
  const file: SessionFile = {
    format: SESSION_FORMAT,
    version: SESSION_VERSION,
    exportedAt: new Date().toISOString(),
    state,
  };
  return JSON.stringify(file, null, 2);
}

/** Parse + validate a session file; throws with a readable message on mismatch. */
export function parseSessionFile(json: string): SessionSnapshot {
  let file: unknown;
  try {
    file = JSON.parse(json);
  } catch {
    throw new Error('Not a valid JSON file.');
  }
  if (
    typeof file !== 'object' ||
    file === null ||
    (file as SessionFile).format !== SESSION_FORMAT
  ) {
    throw new Error('Not an LLMind session file.');
  }
  const { version, state } = file as SessionFile;
  if (version > SESSION_VERSION) {
    throw new Error(`Session file version ${version} is newer than this app supports.`);
  }
  if (typeof state !== 'object' || state === null || !Array.isArray(state.nodes)) {
    throw new Error('Session file is missing its exploration state.');
  }
  return state;
}
