/** Static full-window geometry, using the native conservative box-link rule. */
function prefix(mask, width, height) {
  const p = new Uint32Array((width + 1) * (height + 1));
  for (let y = 0; y < height; y++) {
    let row = 0;
    for (let x = 0; x < width; x++) {
      row += mask[y * width + x] ? 0 : 1;
      p[(y + 1) * (width + 1) + x + 1] = row + p[y * (width + 1) + x + 1];
    }
  }
  return (x0, y0, x1, y1) => p[y1 * (width + 1) + x1] - p[y0 * (width + 1) + x1]
    - p[y1 * (width + 1) + x0] + p[y0 * (width + 1) + x0] === 0;
}
export function validateMask(mask, width, height, policy = 'full', model = 'affine') {
  if (!['full', 'partial'].includes(policy)) throw new Error("roiPolicy must be 'full' or 'partial'");
  if (mask == null) return null;
  if (!['translation', 'affine'].includes(model)) throw new Error('ROI supports only translation and affine motion models');
  if (!(mask instanceof Uint8Array) || mask.length !== width * height || mask.some((v) => v > 1)) {
    throw new Error('roiMask must contain exactly width * height boolean bytes (0 or 1)');
  }
  return mask.slice();
}
const reflect = (x, n) => x < 0 ? -x : x >= n ? 2 * n - 2 - x : x;
export function reduceMask(mask, width, height) {
  const w = Math.floor(width / 2), h = Math.floor(height / 2), values = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let allowed = 1;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      allowed &= mask[reflect(2 * y + dy, height) * width + reflect(2 * x + dx, width)];
    }
    values[y * w + x] = allowed;
  }
  return { mask: values, width: w, height: h };
}
function linkTest(box, width, height) {
  return (ax, ay, bx, by) => box(Math.max(0, Math.ceil(Math.min(ax, bx) - .5)),
    Math.max(0, Math.ceil(Math.min(ay, by) - .5)),
    Math.min(width, Math.floor(Math.max(ax, bx) + .5) + 1),
    Math.min(height, Math.floor(Math.max(ay, by) + .5) + 1));
}
export function createLinkTest(mask, width, height) {
  return linkTest(prefix(mask, width, height), width, height);
}
function centralSupport(mask, width, x0, y0, size) {
  const center = size >> 1;
  const seeds = size % 2
    ? [[center, center]]
    : [[center - 1, center - 1], [center, center - 1], [center - 1, center], [center, center]];
  if (seeds.some(([x, y]) => !mask[(y0 + y) * width + x0 + x])) return null;
  const support = new Uint8Array(size * size), queue = new Uint32Array(size * size);
  let head = 0, tail = 0;
  for (const [x, y] of seeds) {
    const index = y * size + x;
    if (!support[index]) { support[index] = 1; queue[tail++] = index; }
  }
  while (head < tail) {
    const index = queue[head++], x = index % size, y = Math.floor(index / size);
    for (const [nx, ny] of [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]) {
      if (nx < 0 || ny < 0 || nx >= size || ny >= size) continue;
      const next = ny * size + nx;
      if (support[next] || !mask[(y0 + ny) * width + x0 + nx]) continue;
      support[next] = 1; queue[tail++] = next;
    }
  }
  return support;
}
export function planRoi(mask, width, height, size, stride, medianFilterSize = 3, policy = 'full') {
  if (![3, 5, 7, 9].includes(medianFilterSize)) {
    throw new Error('medianFilterSize must be 3, 5, 7, or 9');
  }
  const gx = 1 + Math.floor((width - size) / stride), gy = 1 + Math.floor((height - size) / stride);
  if (!['full', 'partial'].includes(policy)) throw new Error("roiPolicy must be 'full' or 'partial'");
  const selected = new Uint8Array(gx * gy), neighbours = new Uint32Array(gx * gy);
  const box = prefix(mask, width, height), clear = linkTest(box, width, height);
  const center = (size - 1) / 2;
  for (let y = 0; y < gy; y++) for (let x = 0; x < gx; x++) {
    selected[y * gx + x] = +box(x * stride, y * stride, x * stride + size, y * stride + size);
  }
  const fullSelected = selected.slice();
  const partial = new Uint8Array(gx * gy), partialIndices = [], supportRows = [], supportBoxes = [];
  const coverage = new Float32Array(gx * gy);
  for (let i = 0; i < selected.length; i++) if (selected[i]) coverage[i] = 1;
  if (policy === 'partial') {
    const wordsPerSupport = Math.ceil(size * size / 32);
    for (let i = 0; i < selected.length; i++) {
      if (fullSelected[i]) continue;
      const x0 = (i % gx) * stride, y0 = Math.floor(i / gx) * stride;
      const support = centralSupport(mask, width, x0, y0, size);
      if (!support) continue;
      let count = 0, minX = size, minY = size, maxX = -1, maxY = -1;
      const words = new Uint32Array(wordsPerSupport);
      for (let p = 0; p < support.length; p++) if (support[p]) {
        count++;
        const px = p % size, py = Math.floor(p / size);
        minX = Math.min(minX, px); minY = Math.min(minY, py);
        maxX = Math.max(maxX, px); maxY = Math.max(maxY, py);
        words[p >>> 5] |= 1 << (p & 31);
      }
      if (!count) continue;
      selected[i] = 1; partial[i] = 1; partialIndices.push(i);
      for (const word of words) supportRows.push(word);
      supportBoxes.push(minX, minY, maxX, maxY);
      coverage[i] = count / (size * size);
    }
  }
  for (let y = 0; y < gy; y++) for (let x = 0; x < gx; x++) {
    if (!selected[y * gx + x]) continue;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const nx = x + dx, ny = y + dy;
      if (nx >= 0 && nx < gx && ny >= 0 && ny < gy && selected[ny * gx + nx]
          && clear(center + x * stride, center + y * stride, center + nx * stride, center + ny * stride)) {
        neighbours[y * gx + x] |= 1 << ((dy + 1) * 3 + dx + 1);
      }
    }
  }
  let medianNeighbourWords = null;
  if (medianFilterSize > 3) {
    medianNeighbourWords = new Float32Array(gx * gy * 4);
    const radius = medianFilterSize >> 1;
    for (let y = 0; y < gy; y++) for (let x = 0; x < gx; x++) {
      const source = y * gx + x;
      if (!selected[source]) continue;
      for (let dy = -radius; dy <= radius; dy++) for (let dx = -radius; dx <= radius; dx++) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= gx || ny < 0 || ny >= gy || !selected[ny * gx + nx]) continue;
        if (!clear(center + x * stride, center + y * stride,
          center + nx * stride, center + ny * stride)) continue;
        const bitIndex = (dy + 4) * 9 + dx + 4;
        const word = bitIndex / 24 | 0;
        medianNeighbourWords[source * 4 + word] += 2 ** (bitIndex % 24);
      }
    }
  }
  return { mask, width, height, gridWidth: gx, gridHeight: gy, selected, neighbours,
    medianNeighbourWords,
    partial, partialIndices: Uint32Array.from(partialIndices),
    supportWords: Uint32Array.from(supportRows), supportBoxes: Float32Array.from(supportBoxes),
    coverage, policy, fullCount: fullSelected.reduce((a, b) => a + b, 0),
    count: selected.reduce((a, b) => a + b, 0) };
}
export function seedPermissions(source, destination, sourceScale, destinationScale, size, stride,
  mask, width, height, clear = null) {
  const bits = new Uint32Array(destination.selected.length), center = size / 2;
  clear ??= createLinkTest(mask, width, height);
  const pixelCenter = (size - 1) / 2;
  for (let y = 0; y < destination.gridHeight; y++) for (let x = 0; x < destination.gridWidth; x++) {
    const index = y * destination.gridWidth + x;
    if (!destination.selected[index]) continue;
    bits[index] = 16;
    const sx = (destinationScale * (center + x * stride) + (sourceScale - destinationScale) / 2) / sourceScale;
    const sy = (destinationScale * (center + y * stride) + (sourceScale - destinationScale) / 2) / sourceScale;
    const lx = Math.min(Math.max(0, Math.floor((sx - center) / stride)), Math.max(0, source.gridWidth - 2));
    const ly = Math.min(Math.max(0, Math.floor((sy - center) / stride)), Math.max(0, source.gridHeight - 2));
    for (let corner = 0; corner < 4; corner++) {
      const nx = lx + (corner & 1), ny = ly + (corner >> 1);
      if (nx < source.gridWidth && ny < source.gridHeight && source.selected[ny * source.gridWidth + nx]
          && clear(destinationScale * (pixelCenter + x * stride), destinationScale * (pixelCenter + y * stride),
            sourceScale * (pixelCenter + nx * stride), sourceScale * (pixelCenter + ny * stride))) bits[index] |= 1 << corner;
    }
    if (sourceScale > destinationScale && (bits[index] & 15) === 0
        && source.selected.length < 2 ** 27 - 1) {
      let best = Infinity, chosen = 0;
      for (let dy = -2; dy <= 3; dy++) for (let dx = -2; dx <= 3; dx++) {
        const nx = lx + dx, ny = ly + dy;
        if (nx < 0 || ny < 0 || nx >= source.gridWidth || ny >= source.gridHeight
            || !source.selected[ny * source.gridWidth + nx]) continue;
        const distance = (sx - (center + nx * stride)) ** 2 + (sy - (center + ny * stride)) ** 2;
        if (distance < best && clear(destinationScale * (pixelCenter + x * stride),
          destinationScale * (pixelCenter + y * stride), sourceScale * (pixelCenter + nx * stride),
          sourceScale * (pixelCenter + ny * stride))) {
          best = distance; chosen = ny * source.gridWidth + nx + 1;
        }
      }
      bits[index] |= chosen << 5;
    }
  }
  return bits;
}

export async function hashMask(mask, width, height) {
  const prefix = new TextEncoder().encode('live-dic-roi-v1\0');
  const bytes = new Uint8Array(prefix.length + 16 + mask.length); bytes.set(prefix);
  const dimensions = new DataView(bytes.buffer, prefix.length, 16);
  dimensions.setBigUint64(0, BigInt(height), true); dimensions.setBigUint64(8, BigInt(width), true);
  bytes.set(mask, prefix.length + 16);
  const digest = await crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, '0')).join('');
}
