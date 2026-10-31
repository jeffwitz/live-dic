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
  if (policy !== 'full') throw new Error('partial ROI is not supported in the browser engine');
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
function linkTest(mask, width, height) {
  const box = prefix(mask, width, height);
  return (ax, ay, bx, by) => box(Math.max(0, Math.ceil(Math.min(ax, bx) - .5)),
    Math.max(0, Math.ceil(Math.min(ay, by) - .5)),
    Math.min(width, Math.floor(Math.max(ax, bx) + .5) + 1),
    Math.min(height, Math.floor(Math.max(ay, by) + .5) + 1));
}
export function planRoi(mask, width, height, size, stride) {
  const gx = 1 + Math.floor((width - size) / stride), gy = 1 + Math.floor((height - size) / stride);
  const selected = new Uint8Array(gx * gy), neighbours = new Uint32Array(gx * gy);
  const box = prefix(mask, width, height), clear = linkTest(mask, width, height);
  const center = (size - 1) / 2;
  for (let y = 0; y < gy; y++) for (let x = 0; x < gx; x++) {
    selected[y * gx + x] = +box(x * stride, y * stride, x * stride + size, y * stride + size);
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
  return { mask, width, height, gridWidth: gx, gridHeight: gy, selected, neighbours,
    count: selected.reduce((a, b) => a + b, 0) };
}
export function seedPermissions(source, destination, sourceScale, destinationScale, size, stride, mask, width, height) {
  const bits = new Uint32Array(destination.selected.length), center = size / 2;
  const clear = linkTest(mask, width, height), pixelCenter = (size - 1) / 2;
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
