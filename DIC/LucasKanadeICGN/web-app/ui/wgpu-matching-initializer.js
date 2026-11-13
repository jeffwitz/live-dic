import { computeCPM } from '../vendor/wgpu-matching/cpm.js';

const MAX_SEEDS = 512;
const INLIER_THRESHOLD_PX = 3;
const MIN_INLIERS = 12;

/** Read the engine's canonical BT.709 grayscale bytes for an image source. */
export async function readMatchingPixels(engine, source) {
  if (!engine.inputs?.[0]) throw new Error('The DIC image converter is not prepared');
  return engine.inputs[0].referencePixels(source);
}

function selectedSeeds({ width, height, grid, subsetSize, subsetStride, roiSelected, roiMask, matchingSupport }) {
  const [gridWidth, gridHeight] = grid;
  const candidates = [];
  const half = (matchingSupport - 1) * 0.5;
  for (let index = 0; index < gridWidth * gridHeight; index++) {
    if (roiSelected && !roiSelected[index]) continue;
    const gx = index % gridWidth, gy = Math.floor(index / gridWidth);
    const x = gx * subsetStride + (subsetSize - 1) * 0.5;
    const y = gy * subsetStride + (subsetSize - 1) * 0.5;
    if (x < half || y < half || x > width - 1 - half || y > height - 1 - half) continue;
    candidates.push({ x, y });
  }
  if (!candidates.length) throw new Error('WGPU-Matching found no seed centres inside the selected ROI');
  const validSupport = ({ x, y }) => {
    const left = Math.ceil(x - half), right = Math.floor(x + half);
    const top = Math.ceil(y - half), bottom = Math.floor(y + half);
    for (let py = top; py <= bottom; py++) {
      for (let px = left; px <= right; px++) {
        if (!roiMask[py * width + px]) return false;
      }
    }
    return true;
  };
  if (!roiMask) {
    const count = Math.min(MAX_SEEDS, candidates.length);
    return Array.from({ length: count }, (_, i) => candidates[Math.floor((i + 0.5) * candidates.length / count)]);
  }
  let sampleCount = Math.min(candidates.length, MAX_SEEDS * 4), valid = [];
  while (true) {
    const sample = Array.from({ length: sampleCount }, (_, i) =>
      candidates[Math.floor((i + 0.5) * candidates.length / sampleCount)]);
    valid = sample.filter(validSupport);
    if (valid.length >= MIN_INLIERS || sampleCount === candidates.length) break;
    sampleCount = Math.min(candidates.length, sampleCount * 2);
  }
  if (valid.length < MIN_INLIERS) {
    throw new Error(`WGPU-Matching found only ${valid.length} seed supports fully inside the ROI`);
  }
  const count = Math.min(MAX_SEEDS, valid.length);
  return Array.from({ length: count }, (_, i) => valid[Math.floor((i + 0.5) * valid.length / count)]);
}

function solve3(matrix, vector) {
  const a = matrix.map((row, i) => [...row, vector[i]]);
  for (let column = 0; column < 3; column++) {
    let pivot = column;
    for (let row = column + 1; row < 3; row++) {
      if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
    }
    if (Math.abs(a[pivot][column]) < 1e-8) return null;
    [a[column], a[pivot]] = [a[pivot], a[column]];
    const divisor = a[column][column];
    for (let j = column; j < 4; j++) a[column][j] /= divisor;
    for (let row = 0; row < 3; row++) {
      if (row === column) continue;
      const factor = a[row][column];
      for (let j = column; j < 4; j++) a[row][j] -= factor * a[column][j];
    }
  }
  return [a[0][3], a[1][3], a[2][3]];
}

function affineFit(matches, width, height) {
  const cx = (width - 1) * 0.5, cy = (height - 1) * 0.5;
  const scale = Math.max(width, height);
  const normal = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const rhsX = [0, 0, 0], rhsY = [0, 0, 0];
  for (const match of matches) {
    const row = [1, (match.x - cx) / scale, (match.y - cy) / scale];
    for (let i = 0; i < 3; i++) {
      rhsX[i] += row[i] * match.dx;
      rhsY[i] += row[i] * match.dy;
      for (let j = 0; j < 3; j++) normal[i][j] += row[i] * row[j];
    }
  }
  const x = solve3(normal, rhsX), y = solve3(normal, rhsY);
  if (!x || !y) return null;
  return { dx: x[0], dy: y[0], linear: [x[1] / scale, x[2] / scale, y[1] / scale, y[2] / scale] };
}

function fitRobustAffine(matches, width, height) {
  const usable = matches.filter(match => match.usable && match.reciprocal
    && Number.isFinite(match.dx) && Number.isFinite(match.dy) && Number.isFinite(match.score));
  if (usable.length < MIN_INLIERS) {
    throw new Error(`WGPU-Matching returned only ${usable.length} usable matches; at least ${MIN_INLIERS} are required`);
  }
  let state = 0x6d2b79f5, best = null;
  const next = () => { state = (Math.imul(state, 1664525) + 1013904223) >>> 0; return state; };
  for (let trial = 0; trial < 256; trial++) {
    const sample = [];
    while (sample.length < 3) {
      const point = usable[next() % usable.length];
      if (!sample.includes(point)) sample.push(point);
    }
    const model = affineFit(sample, width, height);
    if (!model) continue;
    const inliers = usable.filter(match => {
      const dx = model.dx + model.linear[0] * (match.x - (width - 1) * 0.5)
        + model.linear[1] * (match.y - (height - 1) * 0.5);
      const dy = model.dy + model.linear[2] * (match.x - (width - 1) * 0.5)
        + model.linear[3] * (match.y - (height - 1) * 0.5);
      return Math.hypot(match.dx - dx, match.dy - dy) <= INLIER_THRESHOLD_PX;
    });
    const squared = inliers.map(match => {
      const dx = model.dx + model.linear[0] * (match.x - (width - 1) * 0.5)
        + model.linear[1] * (match.y - (height - 1) * 0.5);
      const dy = model.dy + model.linear[2] * (match.x - (width - 1) * 0.5)
        + model.linear[3] * (match.y - (height - 1) * 0.5);
      return (match.dx - dx) ** 2 + (match.dy - dy) ** 2;
    });
    const rmse = squared.length ? Math.sqrt(squared.reduce((sum, value) => sum + value, 0) / squared.length) : Infinity;
    if (!best || inliers.length > best.inliers.length
        || (inliers.length === best.inliers.length && rmse < best.rmse)) best = { inliers, rmse };
  }
  if (!best || best.inliers.length < MIN_INLIERS) {
    throw new Error(`WGPU-Matching affine consensus has fewer than ${MIN_INLIERS} inliers`);
  }
  const estimate = affineFit(best.inliers, width, height);
  if (!estimate) throw new Error('WGPU-Matching affine consensus is geometrically degenerate');
  const residuals = best.inliers.map(match => {
    const dx = estimate.dx + estimate.linear[0] * (match.x - (width - 1) * 0.5)
      + estimate.linear[1] * (match.y - (height - 1) * 0.5);
    const dy = estimate.dy + estimate.linear[2] * (match.x - (width - 1) * 0.5)
      + estimate.linear[3] * (match.y - (height - 1) * 0.5);
    return Math.hypot(match.dx - dx, match.dy - dy);
  });
  const rmse = Math.sqrt(residuals.reduce((sum, value) => sum + value * value, 0) / residuals.length);
  if (!Number.isFinite(rmse) || rmse > INLIER_THRESHOLD_PX) {
    throw new Error(`WGPU-Matching affine consensus residual is too high (${rmse.toFixed(2)} px)`);
  }
  return {
    ...estimate,
    score: best.inliers.reduce((sum, match) => sum + match.score, 0) / best.inliers.length,
    overlap: null,
    sampleCount: best.inliers.length,
    levels: 0,
    diagnostics: { algorithm: 'CPM-HOG WebGPU + deterministic affine RANSAC', requestedSeeds: matches.length,
      usableMatches: usable.length, inliers: best.inliers.length, affineRmsePx: rmse },
  };
}

export async function initializeWithWgpuMatching(engine, referencePixels, targetPixels,
  { width, height, grid, subsetSize, subsetStride, roiSelected, roiMask, onProgress = () => {} }) {
  const matchingSupportPx = 32;
  const seeds = selectedSeeds({ width, height, grid, subsetSize, subsetStride,
    roiSelected, roiMask, matchingSupportPx });
  if (seeds.length < MIN_INLIERS) {
    throw new Error(`WGPU-Matching needs at least ${MIN_INLIERS} spatially distributed seed centres; found ${seeds.length}`);
  }
  const result = await computeCPM(engine.device, referencePixels, targetPixels, width, height, {
    seedPoints: seeds,
    patchSize: matchingSupportPx,
    cpmMatchingPatchSizePx: matchingSupportPx,
  }, onProgress);
  const estimate = fitRobustAffine(result.matches, width, height);
  estimate.levels = result.levels;
  estimate.diagnostics = { ...estimate.diagnostics, matchingAlgorithm: result.algorithm,
    matchingLevels: result.levels, matchingSupportPx, workGridStridePx: result.workGridStridePx,
    workGridUsableCount: result.workGridUsableCount, matchingMs: result.matchingMs,
    descriptorMs: result.descriptorMs, pyramidMs: result.pyramidMs };
  return estimate;
}
