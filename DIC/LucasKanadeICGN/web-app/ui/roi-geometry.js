/** Pixel-centre geometry, independent of canvas antialiasing and drawing order. */
export function bounds(shape) {
  if (shape.type !== 'polygon') return [shape.x, shape.y, shape.width, shape.height];
  const x = shape.points.map(p => p[0]), y = shape.points.map(p => p[1]);
  return [Math.min(...x), Math.min(...y), Math.max(...x) - Math.min(...x), Math.max(...y) - Math.min(...y)];
}
export function validateShapes(shapes) {
  const ids = new Set();
  return shapes.map((source) => {
    const s = structuredClone(source);
    if (!['rectangle', 'ellipse', 'circle', 'polygon'].includes(s.type)) throw new Error('Unknown ROI shape');
    if (typeof s.id !== 'string' || ids.has(s.id)) throw new Error('ROI shape IDs must be unique strings');
    ids.add(s.id);
    if (s.type === 'polygon') {
      if (!Array.isArray(s.points) || s.points.length < 3 || !s.points.every(p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite))) throw new Error('A polygon needs three finite points');
    } else if (![s.x, s.y, s.width, s.height].every(Number.isFinite) || s.width <= 0 || s.height <= 0) throw new Error('Shape bounds must be finite with positive sizes');
    if (s.type === 'circle' && s.width !== s.height) throw new Error('A circle must have equal width and height');
    return s;
  });
}
/** Half-open spans of selected pixel centres on each scanline. */
function footprint(shape, width, height) {
  const rows = Array.from({ length: height }, () => []), [x, y, w, h] = bounds(shape);
  let area = 0;
  for (let row = Math.max(0, Math.ceil(y - .5)); row < Math.min(height, Math.ceil(y + h - .5)); row++) {
    let crossings;
    if (shape.type === 'rectangle') crossings = [x, x + w];
    else if (shape.type === 'ellipse' || shape.type === 'circle') {
      const dy = (row + .5 - y - h / 2) / (h / 2);
      if (Math.abs(dy) >= 1) continue;
      const dx = w / 2 * Math.sqrt(1 - dy * dy);
      // Strict ellipse interior: boundary pixels are outside.
      let left = Math.max(0, Math.floor(x + w / 2 - dx - .5) + 1);
      let right = Math.min(width, Math.ceil(x + w / 2 + dx - .5));
      // Correct sqrt rounding at exact boundary pixels with the implicit conic.
      const rx = w / 2, ry = h / 2, py = row + .5 - y - ry;
      const inside = col => ((col + .5 - x - rx) * ry) ** 2 + (py * rx) ** 2 < (rx * ry) ** 2;
      while (left < right && !inside(left)) left++;
      while (left > 0 && inside(left - 1)) left--;
      while (right < width && inside(right)) right++;
      while (right > left && !inside(right - 1)) right--;
      if (right > left) { rows[row].push([left, right]); area += right - left; }
      continue;
    } else {
      crossings = [];
      for (let i = 0, j = shape.points.length - 1; i < shape.points.length; j = i++) {
        const a = shape.points[i], b = shape.points[j], py = row + .5;
        if ((a[1] > py) !== (b[1] > py)) crossings.push(a[0] + (py - a[1]) * (b[0] - a[0]) / (b[1] - a[1]));
      }
      crossings.sort((a, b) => a - b);
    }
    for (let i = 0; i + 1 < crossings.length; i += 2) {
      const left = Math.max(0, Math.ceil(crossings[i] - .5)), right = Math.min(width, Math.ceil(crossings[i + 1] - .5));
      if (right > left) { rows[row].push([left, right]); area += right - left; }
    }
  }
  return { rows, area };
}
function contains(parent, child) {
  if (parent.area <= child.area || !child.area) return false;
  return child.rows.every((spans, y) => spans.every(([a, b]) => parent.rows[y].some(([c, d]) => c <= a && d >= b)));
}
/** Roots unite. Exact pixel containment alternates domain / hole / island.
 * Equal footprints are duplicates (union), never arbitrary parent/child ties.
 */
export function compileRoi(shapes, width, height) {
  if (![width, height].every(v => Number.isInteger(v) && v > 0)) throw new Error('Invalid ROI image dimensions');
  const owned = validateShapes(shapes), plans = owned.map(s => footprint(s, width, height));
  const aliases = plans.map((p, i) => plans.findIndex((q, j) => j < i && q.area === p.area && q.rows.every((r, y) => r.length === p.rows[y].length && r.every((v, k) => v[0] === p.rows[y][k][0] && v[1] === p.rows[y][k][1]))));
  const parents = plans.map((child, i) => {
    let parent = -1;
    plans.forEach((candidate, j) => {
      if (j !== i && aliases[j] < 0 && contains(candidate, child) && (parent < 0 || candidate.area < plans[parent].area)) parent = j;
    });
    return parent;
  });
  aliases.forEach((a, i) => { if (a >= 0) parents[i] = parents[a]; });
  const depth = i => parents[i] < 0 ? 0 : depth(parents[i]) + 1;
  const descriptions = owned.map((s, i) => ({ ...s, parent: parents[i] < 0 ? null : owned[parents[i]].id,
    depth: depth(i), role: depth(i) % 2 ? 'hole' : 'domain', pixels: plans[i].area, duplicateOf: aliases[i] < 0 ? null : owned[aliases[i]].id }));
  if (!owned.length) return { mask: null, shapes: descriptions };
  const mask = new Uint8Array(width * height);
  // Unite outer domains first, then subtract all holes, then restore islands.
  // An overlapping outer domain must not silently fill somebody else's hole.
  const order = plans.map((_, i) => i).filter(i => aliases[i] < 0).sort((a, b) => depth(a) - depth(b));
  for (const i of order) {
    const value = depth(i) % 2 ? 0 : 1;
    plans[i].rows.forEach((spans, y) => spans.forEach(([a, b]) => mask.fill(value, y * width + a, y * width + b)));
  }
  return { mask, shapes: descriptions };
}
export function transformShape(shape, box) {
  const [x, y, w, h] = bounds(shape), [nx, ny, nw, nh] = box, next = structuredClone(shape);
  if (![nx, ny, nw, nh].every(Number.isFinite) || nw <= 0 || nh <= 0) throw new Error('Use finite coordinates and positive sizes');
  if (shape.type === 'polygon') next.points = shape.points.map(([px, py]) => [nx + (px - x) * nw / (w || 1), ny + (py - y) * nh / (h || 1)]);
  else Object.assign(next, { x: nx, y: ny, width: nw, height: shape.type === 'circle' ? nw : nh });
  return next;
}
export function platePreset(width = 3840, height = 2160) {
  // +0.5 reconciles editor centres with the archived integer-index hole mask.
  const sx = width / 3840, sy = height / 2160;
  return [{ id: 'plate', label: 'Plate observation window', type: 'rectangle', x: 0, y: 0, width, height },
    { id: 'hole', label: 'Traction-free hole', type: sx === sy ? 'circle' : 'ellipse',
      x: (1920.5 - 112) * sx, y: (1080.5 - 112) * sy, width: 224 * sx, height: 224 * sy }];
}
