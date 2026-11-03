import { sequentialColour } from '../render/colormaps.js';

const PERCEPTUAL_SEQUENTIAL_MAPS = new Set(['viridis', 'cividis', 'plasma', 'inferno', 'magma']);

const contourPaths = new WeakMap();
const percentileCache = new WeakMap();

/** Cached percentile limits for a derived field and its validity population. */
export function percentileBounds(analysis, component, values, valid, minimum, maximum) {
  let components = percentileCache.get(analysis);
  if (!components) {
    components = new Map();
    percentileCache.set(analysis, components);
  }
  let sorted = components.get(component);
  if (!sorted) {
    sorted = Array.from(values).filter((value, index) => valid[index] && Number.isFinite(value))
      .sort((a, b) => a - b);
    components.set(component, sorted);
  }
  if (!sorted.length) return [null, null];
  const quantile = percentile => {
    const position = (sorted.length - 1) * percentile / 100;
    const index = Math.floor(position);
    return sorted[index] + (position - index)
      * (sorted[Math.min(sorted.length - 1, index + 1)] - sorted[index]);
  };
  return [quantile(minimum), quantile(maximum)];
}

/** Map one scalar to the selected display palette and clipping interval. */
export function fieldColour(value, low, high, palette) {
  const t = Math.max(0, Math.min(1, (value - low) / (high - low)));
  if (PERCEPTUAL_SEQUENTIAL_MAPS.has(palette)) return sequentialColour(palette, t);
  if (palette === 'gray') return Array(3).fill(Math.round(24 + 220 * t));
  if (palette === 'sequential') {
    const stops = [[21, 57, 99], [31, 132, 138], [241, 206, 95]];
    const position = t * 2;
    const index = Math.min(1, Math.floor(position));
    const fraction = position - index;
    return stops[index].map((value, channel) =>
      Math.round(value + (stops[index + 1][channel] - value) * fraction));
  }
  const fraction = Math.max(0, Math.min(1, value < 0
    ? -value / Math.max(1e-12, -low)
    : value / Math.max(1e-12, high)));
  const endpoint = value < 0 ? [39, 91, 171] : [187, 56, 54];
  return endpoint.map(channel => Math.round(247 + (channel - 247) * fraction));
}

/** Draw the scalar grid, ROI contour and optional displacement vectors. */
export function renderFieldRaster({
  canvas,
  overlay,
  analysis,
  values,
  valid,
  diagnostic,
  component,
  low,
  high,
  palette,
  statusColours,
  showContour,
  mask,
  showVectors,
  vectorStep,
  vectorGain,
}) {
  const field = analysis.field;
  const [gridWidth, gridHeight] = field.grid;
  const colour = value => fieldColour(value, low, high, palette);
  const small = typeof OffscreenCanvas === 'function'
    ? new OffscreenCanvas(gridWidth, gridHeight)
    : Object.assign(document.createElement('canvas'), { width: gridWidth, height: gridHeight });
  const smallContext = small.getContext('2d');
  const image = smallContext.createImageData(gridWidth, gridHeight);
  for (let index = 0; index < values.length; index++) {
    let rgb = [71, 86, 96];
    if (component === 'status') rgb = statusColours[Math.round(values[index])] ?? rgb;
    else if (valid[index]) rgb = colour(values[index]);
    else if (diagnostic?.valid[index]) rgb = colour(diagnostic.values[index]);
    image.data.set([...rgb, 255], index * 4);
  }
  smallContext.putImageData(image, 0, 0);

  if (canvas.width !== field.width || canvas.height !== field.height) {
    canvas.width = field.width;
    canvas.height = field.height;
  }
  if (overlay.width !== field.width || overlay.height !== field.height) {
    overlay.width = field.width;
    overlay.height = field.height;
  }
  const context = canvas.getContext('2d');
  context.fillStyle = '#263b47';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.imageSmoothingEnabled = false;
  const offset = (field.subsetSize - field.subsetStride) / 2;
  context.drawImage(small, offset, offset, gridWidth * field.subsetStride, gridHeight * field.subsetStride);

  if (showContour && mask) {
    if (!contourPaths.has(mask)) {
      const path = new Path2D(), width = field.width, height = field.height;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const index = y * width + x;
          if (!mask[index]) continue;
          if (x === 0 || !mask[index - 1]) { path.moveTo(x, y); path.lineTo(x, y + 1); }
          if (x === width - 1 || !mask[index + 1]) { path.moveTo(x + 1, y); path.lineTo(x + 1, y + 1); }
          if (y === 0 || !mask[index - width]) { path.moveTo(x, y); path.lineTo(x + 1, y); }
          if (y === height - 1 || !mask[index + width]) { path.moveTo(x, y + 1); path.lineTo(x + 1, y + 1); }
        }
      }
      contourPaths.set(mask, path);
    }
    context.strokeStyle = '#ffffff';
    context.lineWidth = Math.max(1, field.width / 1200);
    context.stroke(contourPaths.get(mask));
  }

  if (showVectors) {
    const path = new Path2D();
    for (let y = 0; y < gridHeight; y += vectorStep) {
      for (let x = 0; x < gridWidth; x += vectorStep) {
        const index = y * gridWidth + x;
        if (!analysis.valid[index]) continue;
        const ax = x * field.subsetStride + field.subsetSize / 2;
        const ay = y * field.subsetStride + field.subsetSize / 2;
        const dx = analysis.fields.u[index] / analysis.recipe.unitsPerPixel * vectorGain;
        const dy = analysis.fields.v[index] / analysis.recipe.unitsPerPixel * vectorGain;
        const length = Math.hypot(dx, dy);
        if (length < 0.01) continue;
        const bx = ax + dx, by = ay + dy, head = Math.min(length * 0.4, field.width / 300);
        const ux = dx / length, uy = dy / length;
        path.moveTo(ax, ay);
        path.lineTo(bx, by);
        path.moveTo(bx - head * ux + head * 0.45 * uy, by - head * uy - head * 0.45 * ux);
        path.lineTo(bx, by);
        path.lineTo(bx - head * ux - head * 0.45 * uy, by - head * uy + head * 0.45 * ux);
      }
    }
    context.strokeStyle = '#173b4c';
    context.lineWidth = Math.max(3, field.width / 700);
    context.stroke(path);
    context.strokeStyle = '#fff';
    context.lineWidth = Math.max(1, field.width / 1800);
    context.stroke(path);
  }
}
