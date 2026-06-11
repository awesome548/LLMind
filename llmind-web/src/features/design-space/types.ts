// Hand-written types for the /api/projection/* endpoints.
//
// These are intentionally NOT in the auto-generated openapi.ts. Regenerate that
// file (see FRONTEND.md) once the backend is running to fold them in; until then
// this module is the single source of truth for design-space payload shapes.

export interface SurfacePoint {
  id: string;
  kind: 'project';
  x: number;
  y: number;
  z?: number;
  cell: [number, number];
  cluster?: number;
  name?: string;
}

export interface SurfaceMeta {
  /** sklearn trustworthiness of the 2D layout w.r.t. the original embedding space. */
  trustworthiness?: number | null;
  trust_neighbors?: number;
  n_reference?: number;
  [key: string]: unknown;
}

export interface Surface {
  version: number;
  dims: number;
  grid: { resolution: number };
  bounds: { min: number; max: number };
  density: number[][]; // [row=gy][col=gx] occupancy counts
  points: SurfacePoint[];
  meta: SurfaceMeta;
}

export interface LocatedPoint {
  node_id: string;
  x: number;
  y: number;
  z?: number;
  /** Jaccard overlap of true vs 2D neighbourhood — how much to trust the position. */
  confidence?: number | null;
  /** Only meaningful on the backend's no-corpus fallback transform path; the
   * primary evidence-anchored placement never leaves the corpus footprint. */
  clipped?: boolean;
  /** Corpus-support percentile in the ORIGINAL metric — how much corpus evidence
   * exists for this point, against the corpus's own self-support baseline. */
  support?: number | null;
}

export interface LocateResponse {
  points: LocatedPoint[];
}

export interface SeedNeighbour {
  id: string;
  Name: string;
  Descriptions: string;
  Details: string;
  Image?: string | null;
  x: number;
  y: number;
}

export interface GeneratedNodeWithCoord {
  node_id: string;
  topic: string;
  /** Project-style description (2-4 sentences) — embedded for placement, shown as context. */
  desc?: string;
  parent_node: string;
  x?: number;
  y?: number;
  z?: number;
  /** Distance from the clicked location to where this node actually landed. */
  drift?: number;
}

export interface GenerateAtResponse {
  parent_id: string;
  options: Record<string, string>;
  nodes: GeneratedNodeWithCoord[];
  related_projects: unknown[];
  coords: LocatedPoint[];
  seed_neighbours: SeedNeighbour[];
  target: { x: number; y: number };
  mean_drift?: number | null;
}

/** node.id → continuous coordinate in the frozen design space ([0,1] — placement
 * is a convex combination of corpus positions, so it never leaves the footprint). */
export type CoordMap = Record<
  string,
  {
    x: number;
    y: number;
    z?: number;
    confidence?: number | null;
    support?: number | null;
  }
>;

// ── Semantic-axes perspective (/api/projection/axes) ──────────────────────────

export interface AxesPoint {
  id: string;
  x: number;
  y: number;
}

export interface AxesItemPoint {
  node_id: string;
  x: number;
  y: number;
  /** Raw score fell outside the corpus range (rendered as an edge marker). */
  clipped: boolean;
}

export interface AxesMeta {
  /** cos(pole_a, pole_b) per axis — near 1.0 means the axis collapses. */
  x_pole_sim: number;
  y_pole_sim: number;
  /** Pearson r of corpus x vs y scores — near ±1 means redundant axes. */
  axis_corr: number;
}

export interface AxesResponse {
  corpus: AxesPoint[];
  items: AxesItemPoint[];
  meta: AxesMeta;
}

// ── Candidate alignment (/api/candidates/alignment) ───────────────────────────

export interface AlignmentAspectResult {
  aspect_id: string;
  /** cos(brief, chosen option) — how strongly the brief expresses the commitment. */
  chosen_score: number;
  /** The competitor the brief is most similar to (null when no alternatives). */
  top_alternative: { id: string; score: number } | null;
  /** True when the brief leans toward the alternative over the chosen option. */
  leans_away: boolean;
}

export interface AlignmentResponse {
  /** cos(brief, composition) — overall concept↔commitments agreement. */
  agreement: number;
  per_aspect: AlignmentAspectResult[];
}

// ── Metric strips (/api/projection/metrics) ───────────────────────────────────

export interface MetricItemPoint {
  node_id: string;
  score: number;
  clipped: boolean;
}

export interface MetricResult {
  /** Full corpus score distribution in [-1, 1] — rug + percentile basis. */
  corpus: number[];
  items: MetricItemPoint[];
  /** cos(pole_a, pole_b) — near 1.0 means the metric collapses. */
  pole_sim: number;
}

export interface MetricsResponse {
  metrics: MetricResult[];
  /** Pairwise corpus-score correlations; |r| near 1 → redundant metrics. */
  corr: number[][];
}

/** A connector drawn from a generation's clicked cell to the nodes it produced. */
export interface GenerationTrail {
  from: { x: number; y: number };
  to: Array<{ x: number; y: number }>;
  /** Mean distance from the click to where the generated nodes landed. */
  meanDrift?: number | null;
}
