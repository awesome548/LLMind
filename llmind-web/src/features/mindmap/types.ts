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
  source: 'generate-at' | 'generate-nodes';
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
