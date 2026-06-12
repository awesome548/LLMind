// Schema replay (Part 12 C3) — pure derivation of the schema's state at a
// point in the exploration log. Halskov's dynamics reading: which options had
// been informed (added), filtered (rejected), or committed to by then. One
// honest simplification, surfaced in the UI: `activeChoices` is the LATEST
// commitment per aspect across ALL candidates — the log records who chose,
// but the replay table shows one ring per aspect.

import type { ExplorationEvent } from '@/src/features/mindmap/types';

export interface ReplayOverlay {
  optionState: Record<string, { state: 'rejected' }>;
  activeChoices: Record<string, string>;
  /** Options informed (generated/added) AT OR BEFORE the playhead. */
  informed: Record<string, true>;
  /** Options whose informing event lies AFTER the playhead — they did not
   * exist yet at this moment (rendered as ghosts; the current tree cannot
   * un-grow them, but it can say so). */
  notYet: Record<string, true>;
}

export function buildReplayOverlay(
  events: ReadonlyArray<ExplorationEvent>,
  upto: number
): ReplayOverlay {
  const bound = Math.max(0, upto);
  const rejected = new Set<string>();
  const choices: Record<string, string> = {};
  const informed = new Set<string>();
  const notYet = new Set<string>();

  // Full pass: classify every informing event as past (≤ playhead) or future
  // (> playhead). taxonomy_set resets BOTH — a new space starts clean.
  events.forEach((event, i) => {
    if (event.kind === 'taxonomy_set') {
      if (i < bound) {
        informed.clear();
      }
      // A future taxonomy_set means everything after it belongs to a space
      // the playhead never reaches — those ids stay simply unknown, which
      // buildSchemaColumns ignores; clearing notYet keeps them out.
      notYet.clear();
      return;
    }
    if (event.kind === 'option_added' || event.kind === 'generated') {
      for (const id of event.refs) {
        if (i < bound) informed.add(id);
        else if (!informed.has(id)) notYet.add(id);
      }
    }
  });

  // Prefix pass: the commitment/filter state as of the playhead.
  for (const event of events.slice(0, bound)) {
    const refs = event.refs;
    switch (event.kind) {
      case 'choose': {
        const [optionId, aspectId] = refs;
        if (optionId && aspectId) choices[aspectId] = optionId;
        break;
      }
      case 'unchoose': {
        const [aspectId] = refs;
        if (aspectId) delete choices[aspectId];
        break;
      }
      case 'reject': {
        const [id] = refs;
        if (id) {
          rejected.add(id);
          // Mirrors the store invariant: rejecting clears the choice.
          for (const [aspectId, optionId] of Object.entries(choices)) {
            if (optionId === id) delete choices[aspectId];
          }
        }
        break;
      }
      case 'reopen': {
        const [id] = refs;
        if (id) rejected.delete(id);
        break;
      }
      case 'taxonomy_set': {
        // A new taxonomy is a new space: schema state resets, history stays.
        rejected.clear();
        for (const key of Object.keys(choices)) delete choices[key];
        break;
      }
      default:
        break;
    }
  }

  return {
    optionState: Object.fromEntries(
      [...rejected].map((id) => [id, { state: 'rejected' as const }])
    ),
    activeChoices: choices,
    informed: Object.fromEntries([...informed].map((id) => [id, true as const])),
    notYet: Object.fromEntries([...notYet].map((id) => [id, true as const])),
  };
}
