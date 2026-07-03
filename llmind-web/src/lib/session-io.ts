// Session save/load: the full exploration state as a portable JSON file.
// Enables study capture, crash recovery, and sharing — the markdown export is
// the human-readable record; this is the machine-restorable one.
//
// This module is the TRUST BOUNDARY for imported state: a corrupt or
// hand-edited file must degrade to defaults with a warning, never crash the
// app at render (session files are the study's capture format — ITERATION-M
// M-E3). Every slice is type-checked here; the store's restoreSession then
// merges the sanitized snapshot over initial-state defaults.

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

// ── Slice validation ─────────────────────────────────────────────────────────
// One predicate + default factory per persisted slice. `nodes` is validated
// separately (it is a hard requirement — a file without it is not a session).
// Keys ABSENT from the file are left absent: the store's restoreSession supplies
// their defaults via `{ ...initialState, ...snapshot }`, so a slice added after
// a file was saved resets correctly. Only keys that are PRESENT-but-malformed
// are reset here (those are what override the default in the spread).

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);
const isString = (v: unknown): v is string => typeof v === 'string';

interface SliceSpec {
  ok: (v: unknown) => boolean;
  def: () => unknown;
}

const SLICE_SPEC: Record<string, SliceSpec> = {
  contextText: { ok: isString, def: () => 'Mindmap' },
  contextDescription: { ok: isString, def: () => '' },
  selectedTopic: { ok: isString, def: () => '' },
  taxonomy: { ok: (v) => v === null || isPlainObject(v), def: () => null },
  projectBrief: { ok: isString, def: () => '' },
  participantId: { ok: isString, def: () => '' },
  coords: { ok: isPlainObject, def: () => ({}) },
  discovered: { ok: isPlainObject, def: () => ({}) },
  provenance: { ok: isPlainObject, def: () => ({}) },
  descriptionById: { ok: isPlainObject, def: () => ({}) },
  candidates: { ok: isPlainObject, def: () => ({}) },
  activeCandidateId: { ok: (v) => v === null || isString(v), def: () => null },
  optionState: { ok: isPlainObject, def: () => ({}) },
  axesConfig: { ok: (v) => v === null || isPlainObject(v), def: () => null },
  rubric: { ok: Array.isArray, def: () => [] },
  usage: { ok: isPlainObject, def: () => ({}) },
  events: { ok: Array.isArray, def: () => [] },
  reflections: { ok: isPlainObject, def: () => ({}) },
};

export interface ParsedSession {
  snapshot: SessionSnapshot;
  /** Human-readable notes about slices that were malformed and reset — shown
   * to the designer so a silent reset never looks like data loss. */
  warnings: string[];
}

/** Reset any present-but-malformed slice to its default; return warnings.
 * Exported for direct testing; `parseSessionFile` calls it after format checks. */
export function sanitizeSnapshot(state: Record<string, unknown>): ParsedSession {
  const warnings: string[] = [];
  const snapshot: Record<string, unknown> = { ...state };

  for (const [key, spec] of Object.entries(SLICE_SPEC)) {
    if (key in snapshot && !spec.ok(snapshot[key])) {
      snapshot[key] = spec.def();
      warnings.push(`"${key}" was malformed and was reset to its default.`);
    }
  }

  // Candidates are iterated with property access all over the UI: `.id` and
  // especially `.choices` (`Object.values(c.choices)`, `c.choices[aspectId]`),
  // which crash at RENDER on a null/non-object value — a crash that would
  // persist to localStorage and defeat the error boundary's reload. So drop an
  // entry that isn't a well-formed object with a string id, and REPAIR a
  // salvageable candidate whose `choices` is malformed (to `{}`) rather than
  // discard the whole thing.
  if (isPlainObject(snapshot.candidates)) {
    const entries = Object.entries(snapshot.candidates);
    const repaired: Record<string, unknown> = {};
    let changed = false;
    for (const [key, c] of entries) {
      if (!isPlainObject(c) || !isString((c as { id?: unknown }).id)) {
        changed = true; // unsalvageable — drop it
        continue;
      }
      if (isPlainObject((c as { choices?: unknown }).choices)) {
        repaired[key] = c;
      } else {
        repaired[key] = { ...(c as Record<string, unknown>), choices: {} };
        changed = true;
      }
    }
    if (changed) {
      snapshot.candidates = repaired;
      warnings.push('Malformed candidate data was repaired or dropped.');
    }
  }

  // Invariant: the active id must name an existing candidate. Coerce to an
  // object first — `x in undefined` throws (candidates may be absent from the
  // file, in which case the store default `{}` applies).
  const candidates = isPlainObject(snapshot.candidates) ? snapshot.candidates : {};
  if (
    isString(snapshot.activeCandidateId) &&
    !(snapshot.activeCandidateId in candidates)
  ) {
    snapshot.activeCandidateId = null;
    warnings.push('The active candidate no longer exists and was cleared.');
  }

  return { snapshot: snapshot as SessionSnapshot, warnings };
}

/** Parse + validate a session file. Throws on a wrong-format/version/missing-
 * nodes file (not a session at all); otherwise returns the sanitized snapshot
 * plus any warnings about slices that were reset. */
export function parseSessionFile(json: string): ParsedSession {
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
  return sanitizeSnapshot(state as Record<string, unknown>);
}
