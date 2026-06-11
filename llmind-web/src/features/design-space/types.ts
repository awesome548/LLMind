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
  /** One-sentence description — embedded for placement, shown as context. */
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

/** node.id → continuous [0,1] coordinate in the frozen design space. */
export type CoordMap = Record<
  string,
  { x: number; y: number; z?: number; confidence?: number | null }
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

/** A connector drawn from a generation's clicked cell to the nodes it produced. */
export interface GenerationTrail {
  from: { x: number; y: number };
  to: Array<{ x: number; y: number }>;
  /** Mean distance from the click to where the generated nodes landed. */
  meanDrift?: number | null;
}
