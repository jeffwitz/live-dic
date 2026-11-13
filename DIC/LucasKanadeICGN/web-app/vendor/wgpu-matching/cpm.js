import { createDeepMatchingHogDescriptors } from "./deepmatching.js";
import { applyBrowserDefaults } from "./matcher-contracts.js";

const BU = globalThis.GPUBufferUsage;
const MM = globalThis.GPUMapMode;

async function loadMatcher(device) {
  const response = await fetch(new URL("./python/src/wgpu_matching/shaders/cpm-match.wgsl", import.meta.url));
  if (!response.ok) throw new Error("Could not load CPM WGSL shader (" + response.status + ").");
  const module = device.createShaderModule({ label: "CPM PatchMatch HOG", code: await response.text() });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
  if (errors.length) {
    throw new Error(errors.map(message => "cpm-match.wgsl:" + message.lineNum + ":" +
      message.linePos + " " + message.message).join("\n"));
  }
  const search = await device.createComputePipelineAsync({
    label: "CPM sparse HOG PatchMatch",
    layout: "auto",
    compute: { module, entryPoint: "patchmatch_step" },
  });
  const oracle = await device.createComputePipelineAsync({
    label: "CPM analytic oracle cost",
    layout: "auto",
    compute: { module, entryPoint: "evaluate_oracle" },
  });
  return { search, oracle };
}

async function buildDescriptorPyramid(device, hog, levelCount) {
  const response = await fetch(new URL("./python/src/wgpu_matching/shaders/cpm-pyramid.wgsl", import.meta.url));
  if (!response.ok) throw new Error("Could not load CPM pyramid WGSL shader (" + response.status + ").");
  const module = device.createShaderModule({ label: "CPM pooled-HOG pyramid", code: await response.text() });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
  if (errors.length) {
    throw new Error(errors.map(message => "cpm-pyramid.wgsl:" + message.lineNum + ":" +
      message.linePos + " " + message.message).join("\n"));
  }
  const pipeline = await device.createComputePipelineAsync({
    label: "CPM HOG pyramid downsample",
    layout: "auto",
    compute: { module, entryPoint: "downsample_hog" },
  });
  const usage = globalThis.GPUTextureUsage.STORAGE_BINDING | globalThis.GPUTextureUsage.TEXTURE_BINDING;
  const levels = [hog];
  const ownedTextures = [];
  try {
  for (let level = 1; level < levelCount; level++) {
    const source = levels[level - 1];
    const width = Math.floor(source.width / 2);
    const height = Math.floor(source.height / 2);
    if (width < 1 || height < 1) break;
    const createSet = label => ({
      width, height,
      first: device.createTexture({ label: label + " HOG 1-4", size: [width, height],
        format: "rgba16float", usage }),
      second: device.createTexture({ label: label + " HOG 5-8", size: [width, height],
        format: "rgba16float", usage }),
    });
    const reference = createSet("CPM level " + level + " reference");
    const target = createSet("CPM level " + level + " target");
    ownedTextures.push(reference.first, reference.second, target.first, target.second);
    const params = storage(device, 16, BU.UNIFORM | BU.COPY_DST, "CPM pyramid dimensions " + level);
    device.queue.writeBuffer(params, 0, new Uint32Array([source.width, source.height, width, height]));
    const bind = (input, output) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: input.first.createView() },
        { binding: 1, resource: input.second.createView() },
        { binding: 2, resource: output.first.createView() },
        { binding: 3, resource: output.second.createView() },
        { binding: 4, resource: { buffer: params } },
      ],
    });
    const encoder = device.createCommandEncoder({ label: "CPM HOG pyramid level " + level });
    for (const [imageLabel, input, output] of [
      ["reference", source.reference, reference], ["target", source.target, target],
    ]) {
      const pass = encoder.beginComputePass({ label: "CPM HOG pyramid " + imageLabel });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind(input, output));
      pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      pass.end();
    }
    device.queue.submit([encoder.finish()]);
    params.destroy();
    levels.push({ width, height, reference, target });
  }
  return { levels, ownedTextures };
  } catch (error) {
    try { await device.queue.onSubmittedWorkDone(); } catch {}
    for (const texture of ownedTextures) texture.destroy();
    throw error;
  }
}

function storage(device, size, usage, label) {
  return device.createBuffer({ label, size: Math.max(16, size), usage });
}

function makeParams(device, values, label) {
  const data = new Uint32Array(16);
  data.set(values.map(value => value >>> 0));
  const buffer = storage(device, data.byteLength, BU.UNIFORM | BU.COPY_DST, label);
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function bindStep(device, pipeline, sourceSet, targetSet, points, previous, next, params) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: sourceSet.first.createView() },
      { binding: 1, resource: sourceSet.second.createView() },
      { binding: 2, resource: targetSet.first.createView() },
      { binding: 3, resource: targetSet.second.createView() },
      { binding: 4, resource: { buffer: points } },
      { binding: 5, resource: { buffer: previous } },
      { binding: 6, resource: { buffer: next } },
      { binding: 7, resource: { buffer: params } },
    ],
  });
}

async function runDirection(device, pipeline, levels, pointBuffers, gridWidth, gridHeight,
  matchingSupportPx, baseScale, radiusPx, seed, settings, label, diagnosticIndices = []) {
  const count = gridWidth * gridHeight;
  const stateBytes = count * 16;
  const previous = storage(device, stateBytes, BU.STORAGE | BU.COPY_SRC, label + " states A");
  const next = storage(device, stateBytes, BU.STORAGE | BU.COPY_SRC, label + " states B");
  const readback = storage(device, stateBytes, BU.COPY_DST | BU.MAP_READ, label + " readback");
  const paramsToDestroy = [];
  const groups = [];
  const levelEnds = [];
  let diagnosticReadback = null;
  try {
    let readState = previous;
    let writeState = next;
    let iteration = 0;
    for (let levelIndex = levels.length - 1; levelIndex >= 0; levelIndex--) {
      const level = levels[levelIndex];
      const levelScale = baseScale * (2 ** levelIndex);
      const patchSize = Math.max(2, Math.round(matchingSupportPx / levelScale));
      const searchRadius = radiusPx > 0
        ? Math.floor(radiusPx / levelScale) : Math.max(level.width, level.height);
      const randomRadiusAtLevel = levelIndex === levels.length - 1
        ? searchRadius
        : Math.max(1, Math.min(searchRadius,
          Math.ceil(settings.cpmLocalSearchRadiusPx * (2 ** levelIndex) / levelScale)));
      const append = (initialize, scalePrevious, randomRadius) => {
        const params = makeParams(device, [
          gridWidth, gridHeight, level.width, level.height, patchSize,
          searchRadius, seed >>> 0, iteration, randomRadius, settings.cpmRandomTrials,
          initialize ? 1 : 0, scalePrevious ? 1 : 0,
          settings.cpmInitialHypotheses ?? settings.cpmRandomTrials, 0, 0, 0,
        ], label + " level " + levelIndex + " pass " + iteration);
        paramsToDestroy.push(params);
        groups.push(bindStep(device, pipeline, level.reference, level.target,
          pointBuffers[levelIndex], readState, writeState, params));
        [readState, writeState] = [writeState, readState];
        iteration++;
      };
      if (levelIndex === levels.length - 1) {
        append(true, false, searchRadius);
        for (let pass = 0; pass < settings.cpmPropagationIterations; pass++) {
          append(false, false, Math.max(1, Math.ceil(searchRadius / (2 ** pass))));
        }
      } else {
        for (let pass = 0; pass < settings.cpmPropagationIterations; pass++) {
          const radius = Math.max(1, Math.min(searchRadius,
            Math.ceil(randomRadiusAtLevel / (2 ** pass))));
          append(false, pass === 0, radius);
        }
      }
      levelEnds.push({ levelIndex, afterGroupCount: groups.length, stateBuffer: readState });
    }

    const snapshotBytes = levels.length * diagnosticIndices.length * 16;
    if (snapshotBytes) {
      diagnosticReadback = storage(device, snapshotBytes, BU.COPY_DST | BU.MAP_READ,
        label + " per-level diagnostic readback");
    }
    const encoder = device.createCommandEncoder({ label: label + " coarse-to-fine ping-pong passes" });
    for (let groupIndex = 0; groupIndex < groups.length; groupIndex++) {
      const group = groups[groupIndex];
      const pass = encoder.beginComputePass({ label: label + " PatchMatch pass" });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(count / 64));
      pass.end();
      for (const end of levelEnds) {
        if (end.afterGroupCount !== groupIndex + 1 || !diagnosticReadback) continue;
        diagnosticIndices.forEach((stateIndex, sampleIndex) => {
          encoder.copyBufferToBuffer(end.stateBuffer, stateIndex * 16, diagnosticReadback,
            (end.levelIndex * diagnosticIndices.length + sampleIndex) * 16, 16);
        });
      }
    }
    encoder.copyBufferToBuffer(readState, 0, readback, 0, stateBytes);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(MM.READ);
    const raw = new ArrayBuffer(stateBytes);
    new Uint8Array(raw).set(new Uint8Array(readback.getMappedRange()));
    readback.unmap();
    let levelSnapshots = null;
    if (diagnosticReadback) {
      await diagnosticReadback.mapAsync(MM.READ);
      const snapshotRaw = new ArrayBuffer(snapshotBytes);
      new Uint8Array(snapshotRaw).set(new Uint8Array(diagnosticReadback.getMappedRange()));
      diagnosticReadback.unmap();
      const ints = new Int32Array(snapshotRaw);
      const floats = new Float32Array(snapshotRaw);
      levelSnapshots = levels.map((_, levelIndex) => diagnosticIndices.map((__, sampleIndex) => {
        const offset = (levelIndex * diagnosticIndices.length + sampleIndex) * 4;
        return { dx: ints[offset], dy: ints[offset + 1], cost: floats[offset + 2], valid: ints[offset + 3] !== 0 };
      }));
    }
    return { integers: new Int32Array(raw), floats: new Float32Array(raw), count, levelSnapshots };
  } finally {
    for (const resource of [previous, next, readback, diagnosticReadback, ...paramsToDestroy]) {
      if (resource) resource.destroy();
    }
  }
}

async function evaluateOracleLevels(device, pipeline, levels, pointBuffers, centers,
  gridWidth, gridHeight, matchingSupportPx, baseScale, affine, sampleIndices) {
  const count = centers.length;
  const stateBytes = count * 16;
  const sampleCount = sampleIndices.length;
  const readback = storage(device, levels.length * sampleCount * 16, BU.COPY_DST | BU.MAP_READ,
    "CPM oracle per-level cost readback");
  const resources = [readback];
  try {
    const encoder = device.createCommandEncoder({ label: "CPM analytic oracle costs" });
    for (let levelIndex = 0; levelIndex < levels.length; levelIndex++) {
      const level = levels[levelIndex];
      const levelScale = baseScale * (2 ** levelIndex);
      const candidateValues = new Int32Array(count * 4);
      let radius = 0;
      centers.forEach((center, index) => {
        const sourceX = Math.round(center.x / levelScale);
        const sourceY = Math.round(center.y / levelScale);
        const targetX = Math.round((affine[0] * center.x + affine[1] * center.y + affine[2]) / levelScale);
        const targetY = Math.round((affine[3] * center.x + affine[4] * center.y + affine[5]) / levelScale);
        const dx = targetX - sourceX;
        const dy = targetY - sourceY;
        candidateValues[index * 4] = dx;
        candidateValues[index * 4 + 1] = dy;
        radius = Math.max(radius, Math.abs(dx), Math.abs(dy));
      });
      const candidates = storage(device, stateBytes, BU.STORAGE | BU.COPY_DST,
        "CPM oracle displacement candidates " + levelIndex);
      const output = storage(device, stateBytes, BU.STORAGE | BU.COPY_SRC,
        "CPM oracle evaluated costs " + levelIndex);
      const params = makeParams(device, [
        gridWidth, gridHeight, level.width, level.height,
        Math.max(2, Math.round(matchingSupportPx / levelScale)), radius,
        0, levelIndex, 0, 0, 0, 0, 0, 0, 0, 0,
      ], "CPM oracle parameters " + levelIndex);
      resources.push(candidates, output, params);
      device.queue.writeBuffer(candidates, 0, candidateValues);
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: level.reference.first.createView() },
          { binding: 1, resource: level.reference.second.createView() },
          { binding: 2, resource: level.target.first.createView() },
          { binding: 3, resource: level.target.second.createView() },
          { binding: 4, resource: { buffer: pointBuffers[levelIndex] } },
          { binding: 5, resource: { buffer: candidates } },
          { binding: 6, resource: { buffer: output } },
          { binding: 7, resource: { buffer: params } },
        ],
      });
      const pass = encoder.beginComputePass({ label: "CPM analytic truth cost at level " + levelIndex });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(Math.ceil(count / 64));
      pass.end();
      sampleIndices.forEach((stateIndex, sampleIndex) => {
        encoder.copyBufferToBuffer(output, stateIndex * 16, readback,
          (levelIndex * sampleCount + sampleIndex) * 16, 16);
      });
    }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(MM.READ);
    const raw = new ArrayBuffer(levels.length * sampleCount * 16);
    new Uint8Array(raw).set(new Uint8Array(readback.getMappedRange()));
    readback.unmap();
    const ints = new Int32Array(raw);
    const floats = new Float32Array(raw);
    return levels.map((_, levelIndex) => sampleIndices.map((__, sampleIndex) => {
      const offset = (levelIndex * sampleCount + sampleIndex) * 4;
      return { dx: ints[offset], dy: ints[offset + 1], cost: floats[offset + 2], valid: ints[offset + 3] !== 0 };
    }));
  } finally {
    for (const resource of resources) resource.destroy();
  }
}

function regularCenters(width, height, patchSize, stride) {
  const half = (patchSize - 1) / 2;
  const centers = [];
  for (let y = half; y <= height - 1 - half; y += stride) {
    for (let x = half; x <= width - 1 - half; x += stride) centers.push({ x, y });
  }
  return centers;
}

function nearestIndex(centers, x, y, stride) {
  const estimate = Math.max(0, Math.min(centers.length - 1,
    Math.round((y - centers[0].y) / stride) * Math.max(1, Math.round((centers.at(-1).x - centers[0].x) / stride + 1)) +
    Math.round((x - centers[0].x) / stride)));
  let best = estimate;
  let bestDistance = Infinity;
  for (let index = Math.max(0, estimate - 2); index <= Math.min(centers.length - 1, estimate + 2); index++) {
    const distance = (centers[index].x - x) ** 2 + (centers[index].y - y) ** 2;
    if (distance < bestDistance) { bestDistance = distance; best = index; }
  }
  return { index: best, distance: Math.sqrt(bestDistance) };
}

function interpolateState(states, centers, gridWidth, gridHeight, x, y, stride) {
  const origin = centers[0];
  const gridX = (x - origin.x) / stride;
  const gridY = (y - origin.y) / stride;
  if (gridX < 0 || gridY < 0 || gridX > gridWidth - 1 || gridY > gridHeight - 1) return null;
  const x0 = Math.floor(gridX), y0 = Math.floor(gridY);
  const x1 = Math.min(gridWidth - 1, x0 + 1), y1 = Math.min(gridHeight - 1, y0 + 1);
  const tx = gridX - x0, ty = gridY - y0;
  const corners = [
    [y0 * gridWidth + x0, (1 - tx) * (1 - ty)],
    [y0 * gridWidth + x1, tx * (1 - ty)],
    [y1 * gridWidth + x0, (1 - tx) * ty],
    [y1 * gridWidth + x1, tx * ty],
  ];
  let dx = 0, dy = 0;
  for (const [index, weight] of corners) {
    if (weight === 0) continue;
    const offset = index * 4;
    if (states.integers[offset + 3] === 0) return null;
    dx += states.integers[offset] * weight;
    dy += states.integers[offset + 1] * weight;
  }
  return { dx, dy };
}

function solve3x3(matrix, rhs) {
  const rows = matrix.map((row, index) => [...row, rhs[index]]);
  for (let column = 0; column < 3; column++) {
    let pivot = column;
    for (let row = column + 1; row < 3; row++) {
      if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) pivot = row;
    }
    if (Math.abs(rows[pivot][column]) < 1e-9) return null;
    [rows[column], rows[pivot]] = [rows[pivot], rows[column]];
    const scale = rows[column][column];
    for (let entry = column; entry < 4; entry++) rows[column][entry] /= scale;
    for (let row = 0; row < 3; row++) {
      if (row === column) continue;
      const factor = rows[row][column];
      for (let entry = column; entry < 4; entry++) rows[row][entry] -= factor * rows[column][entry];
    }
  }
  return rows.map(row => row[3]);
}

function fitLocalAffine(workMatches, x, y, radiusPx) {
  const candidates = workMatches.filter(match => match.usable
    && Math.hypot(match.x - x, match.y - y) <= radiusPx);
  if (candidates.length < 6) return null;
  let inliers = candidates;
  let coefficientsX = null, coefficientsY = null;
  for (let iteration = 0; iteration < 3; iteration++) {
    const matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const rhsX = [0, 0, 0], rhsY = [0, 0, 0];
    for (const match of inliers) {
      const localX = match.x - x, localY = match.y - y;
      const basis = [1, localX, localY];
      const distanceRatio = Math.hypot(localX, localY) / radiusPx;
      const weight = 1 / (1 + distanceRatio * distanceRatio);
      for (let row = 0; row < 3; row++) {
        rhsX[row] += weight * basis[row] * match.dx;
        rhsY[row] += weight * basis[row] * match.dy;
        for (let column = 0; column < 3; column++) {
          matrix[row][column] += weight * basis[row] * basis[column];
        }
      }
    }
    coefficientsX = solve3x3(matrix, rhsX);
    coefficientsY = solve3x3(matrix, rhsY);
    if (!coefficientsX || !coefficientsY) return null;
    const residuals = candidates.map(match => ({
      match,
      error: Math.hypot(match.dx - (coefficientsX[0] + coefficientsX[1] * (match.x - x)
        + coefficientsX[2] * (match.y - y)), match.dy - (coefficientsY[0]
        + coefficientsY[1] * (match.x - x) + coefficientsY[2] * (match.y - y))),
    }));
    const sortedErrors = residuals.map(row => row.error).sort((a, b) => a - b);
    const medianError = sortedErrors[Math.floor(sortedErrors.length / 2)];
    const cutoff = Math.max(2, 2.5 * medianError);
    const nextInliers = residuals.filter(row => row.error <= cutoff).map(row => row.match);
    if (nextInliers.length < 6 || nextInliers.length === inliers.length || iteration === 2) break;
    inliers = nextInliers;
  }
  if (!coefficientsX || !coefficientsY || inliers.length < 6) return null;
  const squared = inliers.map(match => {
    const predictedX = coefficientsX[0] + coefficientsX[1] * (match.x - x)
      + coefficientsX[2] * (match.y - y);
    const predictedY = coefficientsY[0] + coefficientsY[1] * (match.x - x)
      + coefficientsY[2] * (match.y - y);
    return (match.dx - predictedX) ** 2 + (match.dy - predictedY) ** 2;
  });
  return {
    coefficientsX,
    coefficientsY,
    supportCount: inliers.length,
    candidateCount: candidates.length,
    meanScore: inliers.reduce((sum, match) => sum + match.score, 0) / inliers.length,
    reciprocalErrorPx: Math.max(...inliers.map(match => match.reciprocalErrorPx)),
    rmsePx: Math.sqrt(squared.reduce((sum, value) => sum + value, 0) / squared.length),
  };
}

/**
 * CPM-HOG prototype: deterministic sparse-grid PatchMatch with reciprocal
 * verification. The output points are directly evaluated work-grid locations;
 * arbitrary requested seeds are mapped to their nearest measured grid point.
 */
export async function computeCPM(device, referencePixels, targetPixels,
  width, height, settings, onProgress = () => {}) {
  settings = await applyBrowserDefaults(settings ?? {}, "cpm");
  if (!BU || !MM) throw new Error("WebGPU buffer constants are unavailable.");
  const scale = 2 ** (settings.downscaleLevels ?? 2);
  const patchSizePx = settings.patchSize || 32;
  const matchingSupportPx = settings.cpmMatchingPatchSizePx ?? Math.min(32, patchSizePx);
  if (!Number.isInteger(matchingSupportPx) || matchingSupportPx < 8 || matchingSupportPx > 128) {
    throw new Error("CPM matching support must be an integer from 8 to 128 px.");
  }
  const patchSize = Math.max(2, Math.round(matchingSupportPx / scale));
  const workStride = settings.cpmGridStride || 16;
  const radiusPx = settings.searchRadiusPx || 0;
  if (!Number.isInteger(workStride) || workStride < 4 || workStride > 256) {
    throw new Error("CPM work-grid stride must be an integer from 4 to 256 px.");
  }
  if (!Number.isInteger(settings.cpmLevels) || settings.cpmLevels < 1 || settings.cpmLevels > 6) {
    throw new Error("CPM pyramid levels must be from 1 to 6.");
  }
  if (!Number.isInteger(settings.cpmPropagationIterations) || settings.cpmPropagationIterations < 1
      || settings.cpmPropagationIterations > 16 || !Number.isInteger(settings.cpmRandomTrials)
      || settings.cpmRandomTrials < 1 || settings.cpmRandomTrials > 16
      || !Number.isInteger(settings.cpmInitialHypotheses ?? settings.cpmRandomTrials)
      || (settings.cpmInitialHypotheses ?? settings.cpmRandomTrials) < 1
      || (settings.cpmInitialHypotheses ?? settings.cpmRandomTrials) > 64) {
    throw new Error("CPM propagation iterations and random trials must be from 1 to 16; initial hypotheses must be from 1 to 64.");
  }
  if (!Number.isInteger(radiusPx) || radiusPx < 0) throw new Error("CPM search radius must be non-negative.");
  const localSearchRadiusPx = settings.cpmLocalSearchRadiusPx ?? 16;
  if (!Number.isFinite(localSearchRadiusPx) || localSearchRadiusPx < 1 || localSearchRadiusPx > 512) {
    throw new Error("CPM local refinement radius must be from 1 to 512 px.");
  }
  const affineRadiusPx = settings.cpmAffineRadiusPx ?? 2 * workStride;
  if (!Number.isFinite(affineRadiusPx) || affineRadiusPx < workStride || affineRadiusPx > 512) {
    throw new Error("CPM local affine radius must be at least the work-grid stride and at most 512 px.");
  }
  const seeds = settings.seedPoints?.length ? settings.seedPoints : regularCenters(width, height,
    patchSizePx, settings.stride || workStride);
  if (!seeds.length) throw new Error("There are no requested DIC seed centers.");
  if (seeds.length > 512) throw new Error("CPM currently limits requested DIC seeds to 512.");
  for (let index = 0; index < seeds.length; index++) {
    const seed = seeds[index];
    if (!Number.isFinite(seed.x) || !Number.isFinite(seed.y)
        || seed.x < (patchSizePx - 1) / 2 || seed.y < (patchSizePx - 1) / 2
        || seed.x > width - 1 - (patchSizePx - 1) / 2
        || seed.y > height - 1 - (patchSizePx - 1) / 2) {
      throw new Error("CPM seed " + (index + 1) + " does not fit a full DIC patch in the reference image.");
    }
  }
  const centers = regularCenters(width, height, matchingSupportPx, workStride);
  if (!centers.length) throw new Error("The CPM work grid does not fit the DIC patch in the image.");
  if (centers.length > 200000) throw new Error("CPM work grid exceeds the current 200000-point safety limit.");

  let hog = null;
  let descriptorPyramid = null;
  const pointBuffers = [];
  let pipeline = null;
  try {
    onProgress("CPM-HOG · building shared INRIA-style GPU descriptors…");
    const descriptorStart = performance.now();
    hog = await createDeepMatchingHogDescriptors(device, referencePixels, targetPixels,
      width, height, Math.log2(scale), onProgress);
    const descriptorMs = performance.now() - descriptorStart;
    if (patchSize > hog.width || patchSize > hog.height) {
      throw new Error("CPM patch is larger than the reduced descriptor image.");
    }
    const maxLevelsFromPatch = Math.max(1, Math.floor(Math.log2(patchSize / 4)) + 1);
    const requestedLevels = settings.cpmLevels;
    const effectiveLevels = Math.min(requestedLevels, maxLevelsFromPatch);
    const pyramidStart = performance.now();
    descriptorPyramid = await buildDescriptorPyramid(device, hog, effectiveLevels);
    await device.queue.onSubmittedWorkDone();
    const pyramidMs = performance.now() - pyramidStart;
    const levels = descriptorPyramid.levels;
    if (levels.length < effectiveLevels) {
      onProgress("CPM · pyramid depth reduced because the descriptor image became too small.");
    }
    pipeline = await loadMatcher(device);
    if (!Number.isSafeInteger(centers.length * 16)
        || centers.length * 16 > device.limits.maxBufferSize
        || centers.length * 16 > device.limits.maxStorageBufferBindingSize
        || centers.length * 8 > device.limits.maxStorageBufferBindingSize
        || Math.ceil(centers.length / 64) > device.limits.maxComputeWorkgroupsPerDimension) {
      throw new Error("CPM state buffers exceed this adapter's WebGPU buffer limit.");
    }
    // The propagation lattice belongs to the matching support, not the larger
    // DIC window used to choose requested output seeds.
    const gridWidth = 1 + Math.floor((width - matchingSupportPx) / workStride);
    const gridHeight = 1 + Math.floor((height - matchingSupportPx) / workStride);
    if (gridWidth * gridHeight !== centers.length) {
      throw new Error("CPM work-grid coordinates do not match its propagation lattice.");
    }
    const perLevelBytes = centers.length * 8;
    for (let levelIndex = 0; levelIndex < levels.length; levelIndex++) {
      const levelScale = scale * (2 ** levelIndex);
      const coordinates = new Int32Array(centers.length * 2);
      centers.forEach((point, index) => {
        coordinates[2 * index] = Math.round(point.x / levelScale);
        coordinates[2 * index + 1] = Math.round(point.y / levelScale);
      });
      const pointBuffer = storage(device, perLevelBytes, BU.STORAGE | BU.COPY_DST,
        "CPM source centers at pyramid level " + levelIndex);
      device.queue.writeBuffer(pointBuffer, 0, coordinates);
      pointBuffers.push(pointBuffer);
    }
    const diagnosticIndices = seeds.map(seed => nearestIndex(centers, seed.x, seed.y, workStride).index);
    const patchmatchStart = performance.now();
    const runSettings = { ...settings, cpmLocalSearchRadiusPx: localSearchRadiusPx };
    const forward = await runDirection(device, pipeline.search, levels, pointBuffers,
      gridWidth, gridHeight, matchingSupportPx, scale, radiusPx, settings.cpmSeed >>> 0, runSettings,
      "CPM reference to target", diagnosticIndices);
    const backward = await runDirection(device, pipeline.search, levels.map(level => ({
      width: level.width, height: level.height,
      reference: level.target, target: level.reference,
    })), pointBuffers, gridWidth, gridHeight, matchingSupportPx, scale, radiusPx,
      settings.cpmSeed >>> 0, runSettings,
      "CPM target to reference");
    const matchingMs = performance.now() - patchmatchStart;

    let oracleDiagnostics = null;
    if (settings.cpmOracleAffine != null) {
      const affine = settings.cpmOracleAffine;
      if (!Array.isArray(affine) || affine.length !== 6 || !affine.every(Number.isFinite)) {
        throw new Error("CPM oracle affine must be [a00, a01, tx, a10, a11, ty].");
      }
      const oracleLevels = await evaluateOracleLevels(device, pipeline.oracle, levels, pointBuffers,
        centers, gridWidth, gridHeight, matchingSupportPx, scale, affine, diagnosticIndices);
      oracleDiagnostics = {
        definition: "At each measured work-grid center and pyramid level, compare the HOG cosine cost of the best PatchMatch state with the analytically transformed integer displacement.",
        affineSourceToTarget: affine,
        perSeed: seeds.map((seed, seedIndex) => {
          const workIndex = diagnosticIndices[seedIndex];
          return {
            requestedX: seed.x,
            requestedY: seed.y,
            measuredX: centers[workIndex].x,
            measuredY: centers[workIndex].y,
            levels: levels.map((_, levelIndex) => {
              const levelScale = scale * (2 ** levelIndex);
              const found = forward.levelSnapshots[levelIndex][seedIndex];
              const oracle = oracleLevels[levelIndex][seedIndex];
              return {
                level: levelIndex,
                descriptorScalePx: levelScale,
                foundDxPx: found.dx * levelScale,
                foundDyPx: found.dy * levelScale,
                foundCost: found.valid ? found.cost : null,
                foundValid: found.valid,
                oracleDxPx: oracle.dx * levelScale,
                oracleDyPx: oracle.dy * levelScale,
                oracleCost: oracle.valid ? oracle.cost : null,
                oracleValid: oracle.valid,
                oracleMinusFoundCost: found.valid && oracle.valid ? oracle.cost - found.cost : null,
              };
            }),
          };
        }),
      };
    }

    const reciprocalThreshold = settings.cpmReciprocalThresholdPx ?? 8;
    const workMatches = centers.map((center, index) => {
      const offset = index * 4;
      const valid = forward.integers[offset + 3] !== 0;
      const dx = valid ? forward.integers[offset] * scale : 0;
      const dy = valid ? forward.integers[offset + 1] * scale : 0;
      const score = 1 - Math.max(0, Math.min(1, forward.floats[offset + 2]));
      const inverse = valid ? interpolateState(backward, centers, gridWidth, gridHeight,
        center.x + dx, center.y + dy, workStride) : null;
      const reciprocalError = inverse ? Math.hypot(dx + inverse.dx * scale, dy + inverse.dy * scale) : Infinity;
      const reciprocal = Boolean(inverse) && reciprocalError <= reciprocalThreshold;
      return {
        x: center.x, y: center.y, dx, dy, score, valid, reciprocal,
        reciprocalErrorPx: reciprocalError,
        directlyEvaluated: true,
        usable: valid && reciprocal && score >= (settings.cpmMinimumSimilarity ?? 0.0),
      };
    });
    const workGridReciprocalCount = workMatches.filter(match => match.reciprocal).length;
    const workGridUsableCount = workMatches.filter(match => match.usable).length;
    const matches = seeds.map(requested => {
      const nearest = nearestIndex(centers, requested.x, requested.y, workStride);
      const measured = workMatches[nearest.index];
      const fit = fitLocalAffine(workMatches, requested.x, requested.y, affineRadiusPx);
      const usable = Boolean(fit);
      const dx = usable ? fit.coefficientsX[0] : measured.dx;
      const dy = usable ? fit.coefficientsY[0] : measured.dy;
      return {
        x: requested.x,
        y: requested.y,
        requestedX: requested.x,
        requestedY: requested.y,
        dx,
        dy,
        targetX: requested.x + dx,
        targetY: requested.y + dy,
        score: fit?.meanScore ?? measured.score,
        reciprocal: usable,
        reciprocalErrorPx: fit?.reciprocalErrorPx
          ?? (Number.isFinite(measured.reciprocalErrorPx) ? measured.reciprocalErrorPx : null),
        gridDistancePx: nearest.distance,
        measuredX: measured.x,
        measuredY: measured.y,
        measuredDx: measured.dx,
        measuredDy: measured.dy,
        directlyEvaluated: false,
        projectionMethod: usable ? "local-affine-fit" : "nearest-grid-fallback-rejected",
        affineSupportCount: fit?.supportCount ?? 0,
        affineCandidateCount: fit?.candidateCount ?? 0,
        affineFitRmsePx: fit?.rmsePx ?? null,
        affineCenterX: usable ? requested.x : null,
        affineCenterY: usable ? requested.y : null,
        localAffine: usable ? {
          centerX: requested.x,
          centerY: requested.y,
          parameters: [
            requested.x + fit.coefficientsX[0], 1 + fit.coefficientsX[1], fit.coefficientsX[2],
            requested.y + fit.coefficientsY[0], fit.coefficientsY[1], 1 + fit.coefficientsY[2],
          ],
        } : null,
        usable,
      };
    });
    return {
      algorithm: "CPM-WebGPU",
      width, height, patchSize: patchSizePx, stride: settings.stride || workStride,
      matchingSupportPx,
      affineRadiusPx,
      scale, levels: levels.length, requestedLevels, count: matches.length, matches,
      seedMode: settings.seedPoints?.length ? "manual" : "grid",
      gridWidth, gridHeight, workGridStridePx: workStride,
      workGridPointCount: centers.length, searchRadiusPx: radiusPx,
      workGridReciprocalCount, workGridUsableCount,
      randomTrials: settings.cpmRandomTrials,
      initialHypotheses: settings.cpmInitialHypotheses ?? settings.cpmRandomTrials,
      initializationStrategy: "deterministic stratified square displacement hypotheses at the coarsest level",
      propagationIterations: settings.cpmPropagationIterations,
      localSearchRadiusPx,
      seed: settings.cpmSeed >>> 0,
      oracleDiagnostics,
      descriptorMs, pyramidMs, matchingMs,
      methodVariant: "CPM-inspired sparse-grid PatchMatch using a separate rigid HOG support, mean-pooled INRIA-style descriptors, cosine patch cost, stratified coarse initialization, and synchronous four-neighbor propagation",
    };
  } finally {
    for (const pointBuffer of pointBuffers) pointBuffer.destroy();
    if (descriptorPyramid) {
      for (const texture of descriptorPyramid.ownedTextures) texture.destroy();
    }
    if (hog) hog.destroy();
  }
}
