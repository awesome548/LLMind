// Shared SVG glyph helpers for the canvas views.

/** 5-point star path centred on (cx, cy) with outer radius r. */
export function starPath(cx: number, cy: number, r: number): string {
  const inner = r * 0.45;
  const points: string[] = [];
  for (let i = 0; i < 10; i++) {
    const radius = i % 2 === 0 ? r : inner;
    const angle = -Math.PI / 2 + (i * Math.PI) / 5;
    points.push(`${cx + radius * Math.cos(angle)},${cy + radius * Math.sin(angle)}`);
  }
  return `M${points.join('L')}Z`;
}
