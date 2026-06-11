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

/** A candidate design: one chosen option per aspect — a point in the
 * morphological design space (not just a catalog entry). */
export interface DesignCandidate {
  id: string;
  name: string;
  /** aspect node id → chosen option node id (radio semantics per aspect). */
  choices: Record<string, string>;
  note?: string;
  createdAt: number;
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
