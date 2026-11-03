/** Install image-only viewport controls; no calculation or result state is changed. */
export function installViewNavigation({ getElement, updateVectorLengthLegend }) {
  let zoom = 1;
  const zoomStops = [1, 1.5, 2, 3, 4, 6, 8];
  const $ = getElement;

  function setZoom(value) {
    const view = $('image-viewport'), scene = $('scene');
    const previousWidth = Math.max(1, scene.scrollWidth);
    const previousHeight = Math.max(1, scene.scrollHeight);
    const centreX = (view.scrollLeft + view.clientWidth / 2) / previousWidth;
    const centreY = (view.scrollTop + view.clientHeight / 2) / previousHeight;
    zoom = Math.max(1, Math.min(8, value));
    scene.style.width = `${zoom * 100}%`;
    $('view-zoom-label').textContent = `${Math.round(zoom * 100)}%`;
    requestAnimationFrame(() => {
      view.scrollLeft = centreX * view.scrollWidth - view.clientWidth / 2;
      view.scrollTop = centreY * view.scrollHeight - view.clientHeight / 2;
      updateVectorLengthLegend();
    });
  }

  function stepZoom(direction) {
    const stops = direction > 0 ? zoomStops : [...zoomStops].reverse();
    const next = stops.find(value => direction > 0
      ? value > zoom + 1e-6
      : value < zoom - 1e-6);
    setZoom(next ?? (direction > 0 ? 8 : 1));
  }

  function pan(dx, dy) {
    const view = $('image-viewport');
    const step = Math.max(80, Math.round(Math.min(view.clientWidth, view.clientHeight) * 0.65));
    view.scrollBy({ left: dx * step, top: dy * step, behavior: 'smooth' });
  }

  function reset() {
    zoom = 1;
    $('scene').style.width = '100%';
    $('view-zoom-label').textContent = '100%';
    const view = $('image-viewport');
    requestAnimationFrame(() => {
      view.scrollTo({
        left: Math.max(0, (view.scrollWidth - view.clientWidth) / 2),
        top: Math.max(0, (view.scrollHeight - view.clientHeight) / 2),
        behavior: 'smooth',
      });
      updateVectorLengthLegend();
    });
  }

  $('view-zoom-out').onclick = () => stepZoom(-1);
  $('view-zoom-in').onclick = () => stepZoom(1);
  $('view-pan-up').onclick = () => pan(0, -1);
  $('view-pan-left').onclick = () => pan(-1, 0);
  $('view-pan-down').onclick = () => pan(0, 1);
  $('view-pan-right').onclick = () => pan(1, 0);
  $('view-reset').onclick = reset;

  if (typeof ResizeObserver === 'function') {
    let pendingFrame = null;
    const observer = new ResizeObserver(() => {
      if (pendingFrame !== null) return;
      pendingFrame = requestAnimationFrame(() => {
        pendingFrame = null;
        updateVectorLengthLegend();
      });
    });
    observer.observe($('image-viewport'));
    window.addEventListener('pagehide', () => {
      observer.disconnect();
      if (pendingFrame !== null) cancelAnimationFrame(pendingFrame);
    });
  }

  return reset;
}
