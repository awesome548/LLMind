// Schema-table view models (Part 12 A1/A3) — pure, unit-tested.
//
// The design-space schema (Halskov & Lundqvist 2021; Halskov 2021) is the
// canonical representation: aspects as columns, options as cells, styled by
// state. These helpers derive the table from the working tree + store state,
// and compute faceted corpus matches from the annotation (A3).

import type { MindmapNode, NodeProvenance, OptionStateEntry } from '@/src/features/mindmap/types';

export interface SchemaOptionCell {
  id: string;
  name: string;
  desc: string;
  /** Chosen by the ACTIVE candidate (ring). */
  chosen: boolean;
  /** Rejected (struck + dimmed). */
  rejected: boolean;
  rejectReason?: string;
  /** Generated/manual origin — Halskov's "informed" italics. */
  informed: boolean;
  /** Replay only: this option did not exist yet at the playhead (ghosted —
   * the current tree cannot un-grow, but it can say so). */
  ghost?: boolean;
}

export interface SchemaColumn {
  id: string;
  name: string;
  desc: string;
  options: SchemaOptionCell[];
}

/** Aspects = the root's children; options = each aspect's children. */
export function buildSchemaColumns(
  nodes: ReadonlyArray<MindmapNode>,
  descriptionByTopic: Readonly<Record<string, string>>,
  descriptionById: Readonly<Record<string, string>>,
  optionState: Readonly<Record<string, OptionStateEntry>>,
  activeChoices: Readonly<Record<string, string>>,
  provenance: Readonly<Record<string, NodeProvenance>>,
  ghostIds?: ReadonlySet<string>
): SchemaColumn[] {
  const root = nodes[0];
  if (!root?.children) return [];
  const chosenIds = new Set(Object.values(activeChoices));
  return root.children.map((aspect) => ({
    id: aspect.id,
    name: aspect.topic,
    desc: descriptionByTopic[aspect.topic] ?? '',
    options: (aspect.children ?? []).map((option) => {
      const state = optionState[option.id];
      return {
        id: option.id,
        name: option.topic,
        desc: descriptionById[option.id] ?? descriptionByTopic[option.topic] ?? '',
        chosen: chosenIds.has(option.id),
        rejected: state?.state === 'rejected',
        ...(state?.reason ? { rejectReason: state.reason } : {}),
        informed: Boolean(provenance[option.id]),
        ...(ghostIds?.has(option.id) ? { ghost: true } : {}),
      };
    }),
  }));
}

export interface AnnotationOptionRecord {
  count: number;
  project_ids: string[];
  projects: Array<{ id: string; name: string }>;
}

// ── Cross-tab lens (Part 12 B2) ──────────────────────────────────────────────

export interface CrossTabCellModel {
  a: SchemaOptionCell;
  b: SchemaOptionCell;
  /** Corpus projects annotated with BOTH options (Halskov's cross-tab cell). */
  projects: Array<{ id: string; name: string }>;
  /** Names of candidates whose choices include both options. */
  candidateNames: string[];
}

/**
 * Option×option grid between two aspects: rows = aspect A's options, cols =
 * aspect B's options; each cell carries the annotated-project intersection
 * and the candidates committing to both. Empty cell = exact, nameable gap.
 */
export function buildCrossTabCells(
  aspectA: SchemaColumn,
  aspectB: SchemaColumn,
  annotation: Readonly<Record<string, AnnotationOptionRecord>> | null,
  candidates: ReadonlyArray<{ name: string; choices: Readonly<Record<string, string>> }>
): CrossTabCellModel[][] {
  return aspectA.options.map((a) =>
    aspectB.options.map((b) => {
      const recA = annotation?.[a.id];
      const bIds = new Set(annotation?.[b.id]?.project_ids ?? []);
      const chosenBy = candidates.filter((c) => {
        const values = new Set(Object.values(c.choices));
        return values.has(a.id) && values.has(b.id);
      });
      return {
        a,
        b,
        projects: (recA?.projects ?? []).filter((p) => bIds.has(p.id)),
        candidateNames: chosenBy.map((c) => c.name),
      };
    })
  );
}

/**
 * Seeds for generating into an empty cell: receipts that satisfy ONE of the
 * two options (never both — the cell is empty), interleaved A/B so both poles
 * stay represented, in shortlist order. Pure. The default cap matches what
 * the backend actually consumes (`cell.MAX_EXEMPLARS` — the 4k-window budget).
 */
export function halfMatchingExemplars(
  recA: AnnotationOptionRecord | undefined,
  recB: AnnotationOptionRecord | undefined,
  max = 8
): string[] {
  const aIds = recA?.project_ids ?? [];
  const bIds = recB?.project_ids ?? [];
  const aSet = new Set(aIds);
  const bSet = new Set(bIds);
  const aOnly = aIds.filter((id) => !bSet.has(id));
  const bOnly = bIds.filter((id) => !aSet.has(id));
  const out: string[] = [];
  for (let i = 0; out.length < max && (i < aOnly.length || i < bOnly.length); i++) {
    const a = aOnly[i];
    if (a !== undefined) out.push(a);
    const b = bOnly[i];
    if (out.length < max && b !== undefined) out.push(b);
  }
  return out;
}

/**
 * Faceted corpus matching (Halskov's ± search): a project matches when it is
 * annotated with EVERY included option and NO excluded option. Returns null
 * when no facets are active (= no fading).
 */
export function computeFacetMatches(
  options: Readonly<Record<string, AnnotationOptionRecord>>,
  include: ReadonlyArray<string>,
  exclude: ReadonlyArray<string>,
  universe: ReadonlyArray<string>
): Set<string> | null {
  if (include.length === 0 && exclude.length === 0) return null;
  const sets = include.map((oid) => new Set(options[oid]?.project_ids ?? []));
  let matched: Set<string>;
  if (sets.length > 0) {
    matched = new Set(sets[0]);
    for (const s of sets.slice(1)) {
      for (const id of matched) if (!s.has(id)) matched.delete(id);
    }
  } else {
    // Exclude-only: start from the WHOLE corpus (Halskov's "−" = every
    // project that does not share the option).
    matched = new Set(universe);
  }
  for (const oid of exclude) {
    for (const id of options[oid]?.project_ids ?? []) matched.delete(id);
  }
  return matched;
}
