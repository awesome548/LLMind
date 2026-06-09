// Shared node colors for the mind map and the design space, so the same node is
// the same color in both views. Each top-level branch gets a distinct hue;
// nodes within a branch share that hue (a family) and vary lightness by depth.

export const ROOT_NODE_COLOR = '#334155'; // slate-700

// Hand-picked, well-separated hues → clearly different branches.
const BRANCH_HUES = [210, 28, 145, 280, 48, 330, 12, 175, 95, 255, 190, 70];

export function branchHue(branchIndex: number): number {
  if (branchIndex < 0) return 215;
  return BRANCH_HUES[branchIndex % BRANCH_HUES.length]!;
}

/** Fill color for a node. `branchIndex < 0` or `depth === 0` ⇒ root. */
export function nodeColor(branchIndex: number, depth: number): string {
  if (branchIndex < 0 || depth === 0) return ROOT_NODE_COLOR;
  const hue = branchHue(branchIndex);
  // Aspect (depth 1): darker/saturated. Option (depth 2+): lighter tint.
  if (depth === 1) return `hsl(${hue} 60% 45%)`;
  return `hsl(${hue} 55% 72%)`;
}

/** Readable text color over `nodeColor` for the same inputs. */
export function nodeTextColor(branchIndex: number, depth: number): string {
  if (branchIndex < 0 || depth <= 1) return '#ffffff';
  return '#0f172a'; // dark text on the lighter option tint
}
