import { bounds, compileRoi, transformShape, validateShapes } from './roi-geometry.js';
import { intersectMasks } from './mask-image.js';

/** Persistent vector editor. Rasterization happens only while paused. */
export function installRoi(canvas, tool, onChange) {
  const ctx = canvas.getContext('2d');
  let shapes = [], selected = null, active = false, draft = null, drag = null, serial = 0;
  let compiled = { mask: null, shapes: [] }, preview = null, history = [];
  let baseMask = null;
  const getPoint = e => { const b = canvas.getBoundingClientRect(); return [
    Math.max(0, Math.min(canvas.width, (e.clientX - b.left) * canvas.width / b.width)),
    Math.max(0, Math.min(canvas.height, (e.clientY - b.top) * canvas.height / b.height))]; };
  const current = () => shapes.find(s => s.id === selected);
  const checkpoint = () => { history.push({shapes:structuredClone(shapes),baseMask}); if (history.length > 50) history.shift(); };
  const notify = () => onChange({shapes:compiled.shapes,selected,editing:active,canUndo:history.length>0,hasBaseMask:!!baseMask});
  function path(s) {
    ctx.beginPath();
    if (s.type === 'polygon') { ctx.moveTo(...s.points[0]); s.points.slice(1).forEach(p => ctx.lineTo(...p)); ctx.closePath(); }
    else if (s.type === 'rectangle') ctx.rect(s.x, s.y, s.width, s.height);
    else ctx.ellipse(s.x + s.width / 2, s.y + s.height / 2, Math.abs(s.width / 2), Math.abs(s.height / 2), 0, 0, Math.PI * 2);
  }
  function handles(s) {
    if (s.type === 'polygon') return s.points;
    const [x, y, w, h] = bounds(s); return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
  }
  function paint() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (active && preview) ctx.drawImage(preview, 0, 0);
    for (const s of [...shapes, ...(draft ? [draft] : [])]) {
      const record = compiled.shapes.find(p => p.id === s.id);
      path(s); ctx.strokeStyle = s.id === selected ? '#f9cc45' : record?.role === 'hole' ? '#fc8271' : '#62ecce';
      ctx.lineWidth = canvas.width / 500; ctx.stroke();
      if (active && s.id === selected) {
        const size = canvas.width / 100;
        handles(s).forEach(([x, y]) => { ctx.fillStyle = '#fff'; ctx.fillRect(x - size / 2, y - size / 2, size, size); ctx.strokeRect(x - size / 2, y - size / 2, size, size); });
      }
    }
  }
  function update() {
    compiled = compileRoi(shapes, canvas.width, canvas.height); preview = null;
    if(baseMask)compiled.mask=intersectMasks(baseMask,compiled.mask);
    if (active && compiled.mask) {
      preview = document.createElement('canvas'); preview.width = canvas.width; preview.height = canvas.height;
      const pc = preview.getContext('2d'), image = pc.createImageData(canvas.width, canvas.height);
      for (let i = 0; i < compiled.mask.length; i++) { const kept = compiled.mask[i], k = i * 4; image.data[k] = kept ? 220 : 50; image.data[k + 1] = kept ? 255 : 15; image.data[k + 2] = kept ? 242 : 20; image.data[k + 3] = kept ? 35 : 160; }
      pc.putImageData(image, 0, 0);
    }
    paint(); notify();
  }
  function finish() {
    if (draft) {
      if (draft.type === 'polygon' && draft.points.length < 3) throw new Error('Add at least three polygon vertices');
      checkpoint(); shapes.push(draft); selected = draft.id; draft = null;
    }
    drag = null; update();
  }
  function hit(s, p) {
    if (s.type === 'polygon') { path(s); return ctx.isPointInPath(...p); }
    const [x, y, w, h] = bounds(s);
    return s.type === 'rectangle' ? p[0] >= x && p[0] <= x + w && p[1] >= y && p[1] <= y + h
      : ((p[0] - x - w / 2) / (w / 2)) ** 2 + ((p[1] - y - h / 2) / (h / 2)) ** 2 <= 1;
  }
  canvas.addEventListener('pointerdown', e => {
    if (!active || e.button > 0) return; e.preventDefault(); const p = getPoint(e);
    if (tool.value === 'select') {
      const s = current(), tolerance = canvas.width / canvas.getBoundingClientRect().width * 12;
      const handle = s ? handles(s).findIndex(q => Math.hypot(q[0] - p[0], q[1] - p[1]) < tolerance) : -1;
      if (handle < 0) selected = [...shapes].reverse().find(s => hit(s, p))?.id ?? null;
      if (current()) { checkpoint(); drag = { point: p, original: structuredClone(current()), handle }; canvas.setPointerCapture(e.pointerId); }
      paint(); notify();
    } else if (tool.value === 'polygon') {
      if (!draft) draft = { id: `region-${++serial}`, label: `Polygon ${serial}`, type: 'polygon', points: [] };
      if (e.detail < 2) draft.points.push(p); paint();
    } else {
      draft = { id: `region-${++serial}`, label: `${tool.value} ${serial}`, type: tool.value, x: p[0], y: p[1], width: 0, height: 0 };
      drag = { point: p }; canvas.setPointerCapture(e.pointerId);
    }
  });
  canvas.addEventListener('pointermove', e => {
    if (!active || !drag) return;
    const p = getPoint(e), [ax, ay] = drag.point;
    if (draft) {
      let w = Math.abs(p[0] - ax), h = Math.abs(p[1] - ay);
      if (draft.type === 'circle') w = h = Math.min(w, h);
      Object.assign(draft, { x: p[0] < ax ? ax - w : ax, y: p[1] < ay ? ay - h : ay, width: w, height: h });
    } else if (drag.original) {
      const original = drag.original, index = shapes.findIndex(s => s.id === selected), b = bounds(original);
      if (original.type === 'polygon' && drag.handle >= 0) { shapes[index] = structuredClone(original); shapes[index].points[drag.handle] = p; }
      else if (drag.handle >= 0) {
        const opposite = [[b[0] + b[2], b[1] + b[3]], [b[0], b[1] + b[3]], [b[0], b[1]], [b[0] + b[2], b[1]]][drag.handle];
        let w = Math.max(1, Math.abs(p[0] - opposite[0])), h = Math.max(1, Math.abs(p[1] - opposite[1]));
        if (original.type === 'circle') w = h = Math.min(w, h);
        shapes[index] = transformShape(original, [p[0] < opposite[0] ? opposite[0] - w : opposite[0], p[1] < opposite[1] ? opposite[1] - h : opposite[1], w, h]);
      } else shapes[index] = transformShape(original, [Math.max(0, Math.min(canvas.width - b[2], b[0] + p[0] - ax)), Math.max(0, Math.min(canvas.height - b[3], b[1] + p[1] - ay)), b[2], b[3]]);
    }
    paint();
  });
  canvas.addEventListener('pointerup', () => {
    if (!drag) return;
    if (draft && (draft.width < 1 || draft.height < 1)) draft = null;
    finish(); tool.value = 'select';
  });
  canvas.addEventListener('pointercancel', () => { draft = null; drag = null; update(); });
  canvas.addEventListener('dblclick', e => { if (active && draft?.type === 'polygon') { e.preventDefault(); finish(); tool.value = 'select'; } });
  return {
    start() { active = true; canvas.classList.add('drawing'); update(); },
    stop() { if (draft) finish(); active = false; preview = null; canvas.classList.remove('drawing'); paint(); notify(); },
    finish,
    reset(width, height, preset = [], raster = null) { if(raster&&(!(raster instanceof Uint8Array)||raster.length!==width*height||raster.some(v=>v>1)))throw new Error('Invalid imported image mask');canvas.width = width; canvas.height = height;baseMask=raster?.slice()??null; shapes = validateShapes(preset); selected = shapes[0]?.id ?? null; draft = drag = null; history = []; serial += preset.length; active = false; canvas.classList.remove('drawing'); update(); },
    select(id) { selected = id; tool.value = 'select'; paint(); notify(); },
    edit(box) { if (!current()) return; checkpoint(); shapes[shapes.findIndex(s => s.id === selected)] = transformShape(current(), box); update(); },
    remove() { if (!selected) return; checkpoint(); shapes = shapes.filter(s => s.id !== selected); selected = shapes.at(-1)?.id ?? null; update(); },
    undo() { if (!history.length) return; const previous=history.pop();shapes = previous.shapes;baseMask=previous.baseMask; selected = shapes.at(-1)?.id ?? null; draft = drag = null; update(); },
    clear() { checkpoint(); shapes = [];baseMask=null; selected = null; draft = drag = null; update(); },
    mask() { if (draft) finish(); return compiled.mask?.slice() ?? null; },
    shapes() { return structuredClone(shapes); },
  };
}
