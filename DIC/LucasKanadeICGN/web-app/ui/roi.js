/** ROI work happens only while paused, in full reference coordinates. */
export function installRoi(canvas, shape, onChange) {
  let points = [], active = false, anchor;
  const context = canvas.getContext('2d');
  const point = (event) => { const box = canvas.getBoundingClientRect();
    return [(event.clientX - box.left) * canvas.width / box.width,
      (event.clientY - box.top) * canvas.height / box.height]; };
  function paint() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!points.length) return;
    context.beginPath(); context.moveTo(...points[0]); points.slice(1).forEach((p) => context.lineTo(...p));
    context.closePath(); context.fillStyle = '#ffffff35'; context.fill();
    context.strokeStyle = '#f1d75b'; context.lineWidth = Math.max(2, canvas.width / 400); context.stroke();
  }
  function finish() { active = false; canvas.classList.remove('drawing'); onChange(points.length >= 3); paint(); }
  canvas.addEventListener('pointerdown', (event) => {
    if (!active) return; event.preventDefault();
    if (shape.value === 'rectangle') { anchor = point(event); points = [anchor, anchor, anchor, anchor]; canvas.setPointerCapture(event.pointerId); }
    else { points.push(point(event)); paint(); }
  });
  canvas.addEventListener('pointermove', (event) => {
    if (!active || !anchor || shape.value !== 'rectangle') return;
    const p = point(event); points = [anchor, [p[0], anchor[1]], p, [anchor[0], p[1]]]; paint();
  });
  canvas.addEventListener('pointerup', () => { if (anchor) { anchor = null; finish(); } });
  canvas.addEventListener('dblclick', (event) => { if (active) { event.preventDefault(); finish(); } });
  return {
    start(width, height) { canvas.width = width; canvas.height = height; points = []; active = true; canvas.classList.add('drawing'); paint(); },
    finish,
    clear() { points = []; active = false; canvas.classList.remove('drawing'); paint(); },
    mask() {
      if (points.length < 3) throw new Error('Draw a complete polygon or rectangle');
      const mask = new Uint8Array(canvas.width * canvas.height);
      // Pixel centres: no antialiasing threshold or hidden erosion.
      for (let y = 0; y < canvas.height; y++) for (let x = 0; x < canvas.width; x++) {
        let inside = false;
        for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
          const a = points[i], b = points[j];
          if ((a[1] > y + .5) !== (b[1] > y + .5)
              && x + .5 < (b[0] - a[0]) * (y + .5 - a[1]) / (b[1] - a[1]) + a[0]) inside = !inside;
        }
        mask[y * canvas.width + x] = +inside;
      }
      return mask;
    },
  };
}
