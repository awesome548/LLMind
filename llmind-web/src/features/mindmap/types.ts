import type { MindmapProjectSchema } from '@/src/types/api-aliases';

export interface MindmapNode {
  id: string;
  topic: string;
  children?: ReadonlyArray<MindmapNode>;
}

export interface MindmapSelection {
  topic: string;
  lineage: string[];
  /** Exact node identity when the selection came from a node click — avoids
   * topic-string ambiguity when the same label exists under several branches. */
  nodeId?: string;
}

export interface FlattenedMindmapNode {
  id: string;
  topic: string;
  depth: number;
  lineage: string[];
}

export type TopicProjectsMap = Readonly<Record<string, ReadonlyArray<MindmapProjectSchema>>>;

/** Where a generated node came from — kept so every idea can cite its precedents. */
export interface NodeProvenance {
  /** What informed this node into the space (Part 12 C1: accepted proposals
   * carry their emitter; 'manual' = designer-typed in the schema view;
   * 'coverage' = an accepted missing-dimension proposal — Part 13 L-A). */
  source: 'generate-at' | 'generate-nodes' | 'manual' | 'steer' | 'cell' | 'coverage';
  /** Corpus projects that seeded the generation (id null for non-corpus rows). */
  seedProjects: Array<{ id: string | null; name: string }>;
  /** The design-space location the user clicked (generate-at only). */
  target?: { x: number; y: number };
  createdAt: number;
}

/** A candidate design with two layers (ITERATION-PLAN Part 10): the BRIEF is
 * what the design *is* (designer prose, the primary embedding); the CHOICES are
 * what it *commits to* (one option per aspect). Their divergence is measured,
 * never silently reconciled. */
export interface DesignCandidate {
  id: string;
  name: string;
  /** aspect node id → chosen option node id (radio semantics per aspect). */
  choices: Record<string, string>;
  /** The identity layer: the designer's project-style description. */
  brief?: string;
  /** Previous star positions (brief edits move the star) — the design's
   * trajectory through precedent space, capped. */
  trail?: Array<{ x: number; y: number }>;
  note?: string;
  createdAt: number;
}

/** One saved Perspectives metric: a bipolar dimension between two option poles
 * of one aspect — part of the project's persistent examination rubric. */
export interface RubricMetric {
  id: string;
  aspectId: string;
  poleAId: string;
  poleBId: string;
}

/** What can happen to the living schema (Part 12 C3). */
export type ExplorationEventKind =
  | 'choose'
  | 'unchoose'
  | 'reject'
  | 'reopen'
  | 'candidate_created'
  | 'candidate_deleted'
  | 'steer_applied'
  | 'cell_kept'
  | 'generated'
  | 'option_added'
  | 'proposal_dismissed'
  | 'taxonomy_set';

/** One entry of the append-only exploration log (persisted, capped). The
 * label is composed AT RECORD TIME — refs may dangle after deletions, the
 * label stays meaningful (PRT: process capture outlives its objects). */
export interface ExplorationEvent {
  id: string;
  ts: number;
  kind: ExplorationEventKind;
  label: string;
  /** Referenced ids, kind-specific: choose = [optionId, aspectId, candidateId];
   * unchoose = [aspectId, candidateId]; reject/reopen/option_added = [nodeId];
   * generated = [...nodeIds]; candidate_* / steer_applied / cell_kept =
   * [candidateId]. */
  refs: string[];
  /** Kind-specific JSON payload — currently only `proposal_dismissed` uses it
   * (the full proposal, so the timeline can offer "Reconsider"). */
  detail?: string;
}

/** A designer's one-line rationale attached to an event (Part 12 C2). */
export interface Reflection {
  text: string;
  /** True when the designer edited (or wrote) it rather than accepting the
   * AI draft verbatim — the study cares about the difference. */
  edited: boolean;
  ts: number;
}

/** Pruning state for an option ("open" is the implicit default; "chosen" is
 * derived from candidates). Rejections carry the designer's rationale. */
export interface OptionStateEntry {
  state: 'rejected';
  reason?: string;
}

/** One semantic axis of the Perspectives view: a bipolar dimension between two
 * option "poles" of one aspect. */
export interface AxisEndConfig {
  aspectId: string;
  poleAId: string;
  poleBId: string;
}

export interface AxesConfig {
  x: AxisEndConfig;
  y: AxisEndConfig;
}
