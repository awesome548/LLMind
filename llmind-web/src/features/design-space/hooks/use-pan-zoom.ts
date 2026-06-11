import { useEffect, useRef, useState } from 'react';
import { ZOOM_FACTOR, ZOOM_MAX, ZOOM_MIN } from '@/src/lib/view-interactions';

export interface ViewTransform {
  k: number;
  tx: number;
  ty: number;
}

const DRAG_THRESHOLD = 4; // px of movement before a press counts as a pan

/**
 * The unified canvas interaction grammar (shared by the design-space surface
 * and the axes view; the mind map mirrors it via mind-elixir options): plain
 * wheel zooms toward the cursor, left-drag pans, a drag-ending click is
 * swallowed so a pan never also selects.
 *
 * `onPanMove` fires when a drag crosses the threshold — dismiss tooltips and
 * viewport-fixed popovers there.
 */
export function usePanZoom(onPanMove?: () => void) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<ViewTransform>({ k: 1, tx: 0, ty: 0 });
  // Lets click handlers distinguish a real click from the click that ends a
  // drag. Exposed because views also consult it in their own handlers.
  const movedRef = useRef(false);
  const onPanMoveRef = useRef(onPanMove);
  useEffect(() => {
    onPanMoveRef.current = onPanMove;
  });

  // Zoom (wheel, toward cursor) — native non-passive listener so we can
  // preventDefault and stop the page from scrolling.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const wx = e.clientX - rect.left;
      const wy = e.clientY - rect.top;
      setView((v) => {
        const factor = e.deltaY < 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
        const k = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, v.k * factor));
        const ratio = k / v.k;
        return { k, tx: wx - (wx - v.tx) * ratio, ty: wy - (wy - v.ty) * ratio };
      });
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  // Pan (pointer drag via window listeners — no pointer capture: capturing on
  // the container would redirect pointerup/click away from the dots and break
  // clicking a dot to select it).
  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const startX = e.clientX;
    const startY = e.clientY;
    const baseTx = view.tx;
    const baseTy = view.ty;
    movedRef.current = false;

    const onMove = (ev: PointerEvent) => {
      const dx = ev.clientX - startX;
      const dy = ev.clientY - startY;
      if (!movedRef.current && Math.hypot(dx, dy) > DRAG_THRESHOLD) movedRef.current = true;
      if (movedRef.current) {
        onPanMoveRef.current?.();
        setView((v) => ({ ...v, tx: baseTx + dx, ty: baseTy + dy }));
      }
    };
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  };

  // Swallow the click that terminates a drag so a pan doesn't also select.
  const onClickCapture = (e: React.MouseEvent) => {
    if (movedRef.current) {
      e.stopPropagation();
      e.preventDefault();
      movedRef.current = false;
    }
  };

  const resetView = () => setView({ k: 1, tx: 0, ty: 0 });

  return { containerRef, view, movedRef, onPointerDown, onClickCapture, resetView };
}
