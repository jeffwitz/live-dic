/*
 * Coarse-to-fine, mask-aware global affine initialization for sparse pairs.
 * This CPU preprocessing step provides seeds to the unchanged local GPU solver.
 */

const DEFAULT_LEVELS = 5;
const MIN_LEVELS = 5;
const MAX_LEVELS = 8;
const MAX_SAMPLES = 100000;
const MAX_ITERATIONS = 24;
function mirror(index, size) {
  if (index < 0) return -index;
  if (index >= size) return 2 * size - 2 - index;
  return index;
}

function reduceImage(values, width, height) {
  const nextWidth = Math.floor(width / 2), nextHeight = Math.floor(height / 2);
  const horizontal = new Float32Array(nextWidth * height), next = new Uint8Array(nextWidth * nextHeight);
  for (let y = 0; y < height; y++) for (let x = 0; x < nextWidth; x++) {
    const sourceX = x * 2;
    horizontal[y * nextWidth + x] = (
      values[y * width + mirror(sourceX - 2, width)]
      + 4 * values[y * width + mirror(sourceX - 1, width)]
      + 6 * values[y * width + sourceX]
      + 4 * values[y * width + mirror(sourceX + 1, width)]
      + values[y * width + mirror(sourceX + 2, width)]) / 16;
  }
  for (let y = 0; y < nextHeight; y++) for (let x = 0; x < nextWidth; x++) {
    const sourceY = y * 2;
    next[y * nextWidth + x] = Math.round((
      horizontal[mirror(sourceY - 2, height) * nextWidth + x]
      + 4 * horizontal[mirror(sourceY - 1, height) * nextWidth + x]
      + 6 * horizontal[sourceY * nextWidth + x]
      + 4 * horizontal[mirror(sourceY + 1, height) * nextWidth + x]
      + horizontal[mirror(sourceY + 2, height) * nextWidth + x]) / 16);
  }
  return { values: next, width: nextWidth, height: nextHeight };
}

function reduceMask(mask, width, height) {
  const nextWidth = Math.floor(width / 2), nextHeight = Math.floor(height / 2);
  const horizontal = new Uint8Array(nextWidth * height), next = new Uint8Array(nextWidth * nextHeight);
  for (let y = 0; y < height; y++) for (let x = 0; x < nextWidth; x++) {
    const sourceX = x * 2;
    horizontal[y * nextWidth + x] = mask[y * width + mirror(sourceX - 2, width)]
      && mask[y * width + mirror(sourceX - 1, width)] && mask[y * width + sourceX]
      && mask[y * width + mirror(sourceX + 1, width)] && mask[y * width + mirror(sourceX + 2, width)] ? 1 : 0;
  }
  for (let y = 0; y < nextHeight; y++) for (let x = 0; x < nextWidth; x++) {
    const sourceY = y * 2;
    next[y * nextWidth + x] = horizontal[mirror(sourceY - 2, height) * nextWidth + x]
      && horizontal[mirror(sourceY - 1, height) * nextWidth + x]
      && horizontal[sourceY * nextWidth + x]
      && horizontal[mirror(sourceY + 1, height) * nextWidth + x]
      && horizontal[mirror(sourceY + 2, height) * nextWidth + x] ? 1 : 0;
  }
  return { mask: next, width: nextWidth, height: nextHeight };
}

function sample(values, width, height, x, y) {
  if (x < 0 || y < 0 || x >= width - 1 || y >= height - 1) return null;
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const i = y0 * width + x0;
  return (values[i] * (1 - fx) + values[i + 1] * fx) * (1 - fy)
    + (values[i + width] * (1 - fx) + values[i + width + 1] * fx) * fy;
}

function solve6(matrix, vector) {
  const a = matrix.map((row, i) => [...row, vector[i]]);
  for (let col = 0; col < 6; col++) {
    let pivot = col;
    for (let row = col + 1; row < 6; row++) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    if (Math.abs(a[pivot][col]) < 1e-12) return null;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const divisor = a[col][col];
    for (let j = col; j < 7; j++) a[col][j] /= divisor;
    for (let row = 0; row < 6; row++) if (row !== col) {
      const factor = a[row][col];
      for (let j = col; j < 7; j++) a[row][j] -= factor * a[col][j];
    }
  }
  return a.map(row => row[6]);
}

function makeSamples(reference, mask, width, height) {
  let left = width, top = height, right = 0, bottom = 0;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (mask[y * width + x]) {
    left = Math.min(left, x); top = Math.min(top, y);
    right = Math.max(right, x + 1); bottom = Math.max(bottom, y + 1);
  }
  const area = Math.max(0, right - left) * Math.max(0, bottom - top);
  const step = Math.max(1, Math.ceil(Math.sqrt(area / MAX_SAMPLES)));
  let count = 0;
  for (let y = top + Math.floor(step / 2); y < bottom; y += step) {
    for (let x = left + Math.floor(step / 2); x < right; x += step) {
      if (mask[y * width + x]) count++;
    }
  }
  if (count < 500) throw new Error('Global affine initialization needs at least 500 masked image samples');
  const coordinates = new Uint16Array(count * 2), values = new Uint8Array(count);
  let cursor = 0;
  for (let y = top + Math.floor(step / 2); y < bottom; y += step) {
    for (let x = left + Math.floor(step / 2); x < right; x += step) {
      const index = y * width + x;
      if (!mask[index]) continue;
      coordinates[2 * cursor] = x; coordinates[2 * cursor + 1] = y; values[cursor] = reference[index]; cursor++;
    }
  }
  return { coordinates, values };
}

function evaluate(points, target, width, height, params, robust = false) {
  const cx = (width - 1) * 0.5, cy = (height - 1) * 0.5;
  let refSum = 0, targetSum = 0, count = 0;
  const valid = [];
  for (let point = 0; point < points.values.length; point++) {
    const x = points.coordinates[2 * point], y = points.coordinates[2 * point + 1], ref = points.values[point];
    const dx = x - cx, dy = y - cy;
    const tx = x + params[0] + params[2] * dx + params[3] * dy;
    const ty = y + params[1] + params[4] * dx + params[5] * dy;
    const value = sample(target, width, height, tx, ty);
    if (value === null) continue;
    valid.push([x, y, ref, tx, ty, value]); refSum += ref; targetSum += value; count++;
  }
  if (count < Math.max(400, points.values.length * 0.2)) return null;
  const refMean = refSum / count, targetMean = targetSum / count;
  let covariance = 0, refEnergy = 0, targetEnergy = 0;
  for (const [, , ref, , , value] of valid) {
    const a = ref - refMean, b = value - targetMean;
    covariance += a * b; refEnergy += a * a; targetEnergy += b * b;
  }
  if (refEnergy < 1e-8 || targetEnergy < 1e-8) return null;
  const correlation = covariance / Math.sqrt(refEnergy * targetEnergy);
  const gain = covariance / targetEnergy;
  const residualScale = Math.sqrt(refEnergy / count);
  const residuals = new Float64Array(count);
  let robustCost = 0;
  for (let i = 0; i < count; i++) {
    const r = (valid[i][2] - refMean) - gain * (valid[i][5] - targetMean);
    residuals[i] = r / Math.max(residualScale, 1e-6);
    const abs = Math.abs(residuals[i]), cutoff = 1.5;
    robustCost += robust && abs > cutoff ? cutoff * (abs - cutoff * 0.5) : 0.5 * abs * abs;
  }
  return { valid, correlation, gain, refMean, targetMean, residuals, residualScale, robustCost: robustCost / count, count };
}

function fitLevel(points, target, width, height, initial) {
  const params = initial.slice();
  let current = evaluate(points, target, width, height, params, true);
  if (!current) throw new Error('Global affine initialization has insufficient image overlap');
  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    const hessian = Array.from({ length: 6 }, () => new Float64Array(6));
    const gradient = new Float64Array(6);
    for (let i = 0; i < current.valid.length; i++) {
      const [x, y, , tx, ty] = current.valid[i];
      const residual = current.residuals[i], abs = Math.abs(residual);
      const weight = abs <= 1.5 ? 1 : 1.5 / abs;
      if (tx < 0.5 || ty < 0.5 || tx >= width - 1.5 || ty >= height - 1.5) continue;
      const gx = (sample(target, width, height, tx + 0.5, ty) - sample(target, width, height, tx - 0.5, ty));
      const gy = (sample(target, width, height, tx, ty + 0.5) - sample(target, width, height, tx, ty - 0.5));
      if (!Number.isFinite(gx) || !Number.isFinite(gy)) continue;
      const dx = x - (width - 1) * 0.5, dy = y - (height - 1) * 0.5;
      const derivativeScale = current.gain / Math.max(current.residualScale, 1e-6);
      const j = [gx, gy, gx * dx, gx * dy, gy * dx, gy * dy].map(value => -value * derivativeScale);
      for (let row = 0; row < 6; row++) {
        gradient[row] += weight * j[row] * residual;
        for (let col = row; col < 6; col++) hessian[row][col] += weight * j[row] * j[col];
      }
    }
    for (let row = 0; row < 6; row++) for (let col = 0; col < row; col++) hessian[row][col] = hessian[col][row];
    const delta = solve6(hessian, gradient.map(value => -value));
    if (!delta) throw new Error('Global affine fit is ill-conditioned on the selected mask');
    delta[0] = Math.max(-8, Math.min(8, delta[0]));
    delta[1] = Math.max(-8, Math.min(8, delta[1]));
    for (let i = 2; i < 6; i++) delta[i] = Math.max(-0.05, Math.min(0.05, delta[i]));
    let accepted = false;
    for (const alpha of [1, 0.5, 0.25, 0.125, 0.0625]) {
      const candidate = params.map((value, index) => value + alpha * delta[index]);
      const tested = evaluate(points, target, width, height, candidate, true);
      if (tested && tested.robustCost < current.robustCost) {
        params.splice(0, 6, ...candidate); current = tested; accepted = true; break;
      }
    }
    if (!accepted || Math.max(...delta.map(Math.abs)) < 1e-3) break;
  }
  return { params, score: current.correlation, sampleCount: points.values.length, overlap: current.count / points.values.length };
}

export class GlobalAffineInitializer {
  constructor(reference, width, height, mask = null, levels = DEFAULT_LEVELS) {
    if (!(reference instanceof Uint8Array) || reference.length !== width * height) {
      throw new Error('Global affine initialization requires grayscale reference bytes');
    }
    if (!Number.isInteger(levels) || levels < MIN_LEVELS || levels > MAX_LEVELS) {
      throw new Error(`Global affine levels must be an integer between ${MIN_LEVELS} and ${MAX_LEVELS}`);
    }
    if (mask && (!(mask instanceof Uint8Array || mask instanceof Uint8ClampedArray) || mask.length !== reference.length)) {
      throw new Error('Global affine initialization mask must match the reference dimensions');
    }
    this.width = width; this.height = height;
    const fullMask = mask ?? new Uint8Array(reference.length).fill(1);
    this.levels = [{ image: { values: reference.slice(), width, height }, mask: fullMask }];
    for (let i = 1; i < levels; i++) {
      const previous = this.levels[i - 1];
      if (Math.min(previous.image.width, previous.image.height) < 32) {
        throw new Error(`The image is too small for ${levels} global affine levels`);
      }
      this.levels.push({
        image: reduceImage(previous.image.values, previous.image.width, previous.image.height),
        ...reduceMask(previous.mask, previous.image.width, previous.image.height),
      });
    }
    if (this.levels.length !== levels) throw new Error(`The image is too small for ${levels} global affine levels`);
    this.levels.forEach(level => { level.samples = makeSamples(level.image.values, level.mask, level.image.width, level.image.height); });
  }

  estimate(target) {
    if (!(target instanceof Uint8Array) || target.length !== this.width * this.height) {
      throw new Error('Global affine target must be grayscale bytes matching the reference');
    }
    const targets = [{ values: target, width: this.width, height: this.height }];
    for (let i = 1; i < this.levels.length; i++) {
      const previous = targets[i - 1]; targets.push(reduceImage(previous.values, previous.width, previous.height));
    }
    let params = [0, 0, 0, 0, 0, 0], final = null;
    const diagnostics = [];
    for (let index = this.levels.length - 1; index >= 0; index--) {
      const targetLevel = targets[index];
      if (index !== this.levels.length - 1) { params[0] *= 2; params[1] *= 2; }
      const points = this.levels[index].samples;
      final = fitLevel(points, targetLevel.values, targetLevel.width, targetLevel.height, params);
      params = final.params;
      diagnostics.push({ level: index, score: final.score, overlap: final.overlap, params: [...params] });
    }
    const [b00, b01, b10, b11] = params.slice(2);
    const determinant = (1 + b00) * (1 + b11) - b01 * b10;
    if (final.score < 0.2 || final.overlap < 0.25 || determinant <= 0.1
        || Math.max(Math.abs(b00), Math.abs(b01), Math.abs(b10), Math.abs(b11)) > 0.5) {
      throw new Error(`Global affine initialization is ambiguous (score ${final.score.toFixed(3)}, overlap ${final.overlap.toFixed(3)})`);
    }
    return {
      dx: params[0], dy: params[1],
      linear: [params[2], params[3], params[4], params[5]],
      score: final.score, overlap: final.overlap, sampleCount: final.sampleCount,
      levels: this.levels.length,
      diagnostics,
    };
  }
}
