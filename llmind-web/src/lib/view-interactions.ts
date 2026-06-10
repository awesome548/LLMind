// Shared interaction constants for the two canvas views (mind map + design
// space). Both views must feel identical: plain wheel zooms toward the cursor
// by ZOOM_FACTOR per tick within [ZOOM_MIN, ZOOM_MAX]; left-drag pans.

export const ZOOM_MIN = 0.5;
export const ZOOM_MAX = 8;
export const ZOOM_FACTOR = 1.12;
