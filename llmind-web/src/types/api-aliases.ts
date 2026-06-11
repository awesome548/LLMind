// Stable aliases over the auto-generated `openapi.ts`.
//
// Regen-safe: `bunx openapi-typescript ... -o src/types/openapi.ts` rewrites
// openapi.ts wholesale; this file is never touched by the generator, so app
// code imports from here instead of from openapi.ts directly.
//
// It also hand-declares the shapes of ASYNC JOB RESULTS. The long-running
// endpoints (generate-nodes, generate-at) return `{job_id}` and deliver their
// real payload via `GET /api/jobs/{id}`, which FastAPI types as `unknown` in
// openapi.json — so those response shapes cannot be generated and are kept
// here, mirroring the backend's Pydantic models.

import type { components } from './openapi';

// ── Generated component schemas ───────────────────────────────────────────────

export type MindmapProjectSchema = components['schemas']['RelatedProject'];
export type FetchRelatedProjectsRequestSchema =
  components['schemas']['FetchRelatedProjectsRequest'];
export type FetchRelatedProjectsResponseSchema =
  components['schemas']['FetchRelatedProjectsResponse'];
export type GenerateNodesRequestSchema = components['schemas']['GenerateNodesRequest'];
export type TaxonomyNodeInputSchema =
  components['schemas']['backend__related_projects__router__TaxonomyNodeInput'];
export type FocusNodeInputSchema = components['schemas']['FocusNodeInput'];
export type BackendModeSchema = components['schemas']['BackendMode'];
export type GenerateTaxonomyRequestSchema = components['schemas']['GenerateTaxonomyRequest'];
export type GenerateTaxonomyResponseSchema = components['schemas']['GenerateTaxonomyResponse'];
export type CorpusProjectResponseSchema = components['schemas']['CorpusProjectResponse'];

// ── Async job results (hand-written; mirror backend Pydantic models) ──────────

/** Job result of POST /api/related-projects/generate-nodes
 * (backend: `GenerateNodesResponse` in backend/related_projects/router.py). */
export interface GeneratedNodeSchema {
  node_id: string;
  topic: string;
  /** Project-style description (2-4 sentences) — embedded for placement and retrieval. */
  desc?: string;
  parent_node: string;
}

export interface GenerateNodesResponseSchema {
  parent_id: string;
  options: Record<string, string>;
  nodes: GeneratedNodeSchema[];
  related_projects: MindmapProjectSchema[];
}
