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

export interface Surface {
  version: number;
  dims: number;
  grid: { resolution: number };
  bounds: { min: number; max: number };
  density: number[][]; // [row=gy][col=gx] occupancy counts
  points: SurfacePoint[];
  meta: Record<string, unknown>;
}

export interface LocatedPoint {
  node_id: string;
  x: number;
  y: number;
  z?: number;
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
  parent_node: string;
  x?: number;
  y?: number;
  z?: number;
}

export interface GenerateAtResponse {
  parent_id: string;
  options: Record<string, string>;
  nodes: GeneratedNodeWithCoord[];
  related_projects: unknown[];
  coords: LocatedPoint[];
  seed_neighbours: SeedNeighbour[];
  target: { x: number; y: number };
}

/** node.id → continuous [0,1] coordinate in the frozen design space. */
export type CoordMap = Record<string, { x: number; y: number; z?: number }>;
