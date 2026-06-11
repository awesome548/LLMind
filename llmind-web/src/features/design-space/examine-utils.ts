// Pure helpers for the Examine (Perspectives) strips — ITERATION-PLAN Part 10 I3.

import type { MindmapNode, RubricMetric } from '@/src/features/mindmap/types';
import type { AlignmentResponse } from './types';
import { indexNodesById, optionEmbeddingText } from './candidate-utils';

/** One strip: a bipolar metric the candidate's brief is scored along. */
export interface MetricDef {
  /** Stable key — also the join key into the /metrics response order. */
  key: string;
  kind: 'consistency' | 'rubric';
  /** Strip title (the aspect's topic). */
  label: string;
  poleALabel: string;
  poleBLabel: string;
  poleAText: string;
  poleBText: string;
  /** Consistency strips: the brief leans toward the rejected alternative. */
  leansAway?: boolean;
  /** Rubric strips: the saved metric's id (for removal). */
  rubricId?: string;
}

/**
 * Consistency strips, one per aspect with a choice AND a data-picked strongest
 * competitor: pole A = the chosen option, pole B = the alternative the brief is
 * most similar to (from the alignment response).
 */
export function buildConsistencyDefs(
  alignment: AlignmentResponse | null | undefined,
  choices: Readonly<Record<string, string>>,
  nodes: ReadonlyArray<MindmapNode>,
  descriptionByTopic: Readonly<Record<string, string>> = {},
  descriptionById: Readonly<Record<string, string>> = {}
): MetricDef[] {
  if (!alignment) return [];
  const byId = indexNodesById(nodes);
  const defs: MetricDef[] = [];
  for (const row of alignment.per_aspect) {
    if (!row.top_alternative) continue;
    const aspect = byId.get(row.aspect_id);
    const chosen = byId.get(choices[row.aspect_id] ?? '');
    const alternative = byId.get(row.top_alternative.id);
    if (!aspect || !chosen || !alternative) continue;
    defs.push({
      key: `consistency:${row.aspect_id}`,
      kind: 'consistency',
      label: aspect.topic,
      poleALabel: chosen.topic,
      poleBLabel: alternative.topic,
      poleAText: optionEmbeddingText(chosen, descriptionByTopic, descriptionById),
      poleBText: optionEmbeddingText(alternative, descriptionByTopic, descriptionById),
      leansAway: row.leans_away,
    });
  }
  return defs;
}

/** Rubric strips from the saved metrics; unresolvable ids are skipped. */
export function resolveRubricDefs(
  rubric: ReadonlyArray<RubricMetric>,
  nodes: ReadonlyArray<MindmapNode>,
  descriptionByTopic: Readonly<Record<string, string>> = {},
  descriptionById: Readonly<Record<string, string>> = {}
): MetricDef[] {
  const byId = indexNodesById(nodes);
  const defs: MetricDef[] = [];
  for (const metric of rubric) {
    const aspect = byId.get(metric.aspectId);
    const poleA = byId.get(metric.poleAId);
    const poleB = byId.get(metric.poleBId);
    if (!aspect || !poleA || !poleB) continue;
    defs.push({
      key: `rubric:${metric.id}`,
      kind: 'rubric',
      label: aspect.topic,
      poleALabel: poleA.topic,
      poleBLabel: poleB.topic,
      poleAText: optionEmbeddingText(poleA, descriptionByTopic, descriptionById),
      poleBText: optionEmbeddingText(poleB, descriptionByTopic, descriptionById),
      rubricId: metric.id,
    });
  }
  return defs;
}

/**
 * Percentile of `score` within the corpus distribution: the fraction of corpus
 * projects scoring at or below it. The basis of the strip's plain-language
 * sentence ("more ⟨pole A⟩ than NN% of real projects — scaled to this corpus").
 */
export function percentileOf(corpus: ReadonlyArray<number>, score: number): number {
  if (corpus.length === 0) return NaN;
  let below = 0;
  for (const value of corpus) if (value <= score) below += 1;
  return below / corpus.length;
}
