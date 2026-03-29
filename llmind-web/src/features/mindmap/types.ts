import type { MindmapProjectSchema } from '@/src/types/openapi';

export interface MindmapNode {
  id: string;
  topic: string;
  children?: ReadonlyArray<MindmapNode>;
}

export interface MindmapSelection {
  topic: string;
  lineage: string[];
}

export interface FlattenedMindmapNode {
  id: string;
  topic: string;
  depth: number;
  lineage: string[];
}

export type TopicProjectsMap = Readonly<Record<string, ReadonlyArray<MindmapProjectSchema>>>;
