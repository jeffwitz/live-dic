import { applyBrowserDefaults } from "./matcher-contracts.js";

const bufferUsage = globalThis.GPUBufferUsage;
const textureUsage = globalThis.GPUTextureUsage;
const mapMode = globalThis.GPUMapMode;

async function loadShader(device, filename, substitutions = {}) {
  const response = await fetch(new URL(`./python/src/wgpu_matching/shaders/${filename}`, import.meta.url));
  if (!response.ok) throw new Error("Chargement du noyau " + filename + " impossible (" + response.status + ").");
  let code = await response.text();
  for (const [name, value] of Object.entries(substitutions)) {
    code = code.replaceAll("{{" + name + "}}", value);
  }
  const module = device.createShaderModule({ label: filename, code });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
  if (errors.length) {
    throw new Error(errors.map(message =>
      filename + ":" + message.lineNum + ":" + message.linePos + " " + message.message).join("\n"));
  }
  return module;
}

function ownTexture(device, resources, width, height, format, usage, label) {
  const texture = device.createTexture({
    label,
    size: { width, height, depthOrArrayLayers: 1 },
    format,
    usage,
  });
  resources.add(texture);
  return texture;
}

function ownBuffer(device, resources, size, usage, label) {
  if (!Number.isSafeInteger(size) || size <= 0) throw new Error("Taille de tampon invalide pour " + label + ".");
  const buffer = device.createBuffer({ label, size, usage });
  resources.add(buffer);
  return buffer;
}

function uniform(device, resources, words, label) {
  const buffer = ownBuffer(device, resources, words.byteLength,
    bufferUsage.UNIFORM | bufferUsage.COPY_DST, label);
  device.queue.writeBuffer(buffer, 0, words);
  return buffer;
}

function makePipeline(device, module, entryPoint, label) {
  return device.createComputePipelineAsync({
    label,
    layout: "auto",
    compute: { module, entryPoint },
  });
}

function makeCorrelationPipeline(device, module, entryPoint) {
  const visibility = globalThis.GPUShaderStage.COMPUTE;
  const entries = [];
  for (let binding = 0; binding < 4; binding++) {
    entries.push({
      binding,
      visibility,
      texture: { sampleType: "float", viewDimension: "2d" },
    });
  }
  entries.push(
    { binding: 6, visibility, buffer: { type: "read-only-storage" } },
    { binding: 7, visibility, buffer: { type: "storage" } },
    { binding: 8, visibility, buffer: { type: "uniform" } },
  );
  const bindGroupLayout = device.createBindGroupLayout({ label: "DeepMatching correlation layout", entries });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
  return device.createComputePipelineAsync({
    label: "DeepMatching patch correlations",
    layout,
    compute: { module, entryPoint },
  });
}

function makeDescriptorPipelines(device, descriptorModule) {
  return Promise.all([
    makePipeline(device, descriptorModule, "presmooth_horizontal", "INRIA descriptor pre-smoothing horizontal"),
    makePipeline(device, descriptorModule, "presmooth_vertical", "INRIA descriptor pre-smoothing vertical"),
    makePipeline(device, descriptorModule, "build_hog", "INRIA pixel HOG descriptor"),
    makePipeline(device, descriptorModule, "smooth_horizontal", "INRIA HOG smoothing horizontal"),
    makePipeline(device, descriptorModule, "smooth_vertical", "INRIA HOG smoothing vertical"),
    makePipeline(device, descriptorModule, "sigmoid_smooth_horizontal", "INRIA HOG sigmoid and smoothing"),
    makePipeline(device, descriptorModule, "average_pool", "INRIA descriptor average pooling"),
  ]).then(([presmoothHorizontal, presmoothVertical, buildHog, smoothHorizontal,
    smoothVertical, sigmoidSmoothHorizontal, averagePool]) =>
    ({ presmoothHorizontal, presmoothVertical, buildHog, smoothHorizontal,
      smoothVertical, sigmoidSmoothHorizontal, averagePool }));
}

function makePipelines(device, descriptorModule, matcherModule) {
  return Promise.all([
    makeDescriptorPipelines(device, descriptorModule),
    makeCorrelationPipeline(device, matcherModule, "correlate_patches"),
    makePipeline(device, matcherModule, "select_best_patch", "DeepMatching patch selection"),
  ]).then(([descriptorPipelines, correlate, select]) =>
    ({ ...descriptorPipelines, correlate, select }));
}

function releaseResource(resources, resource) {
  if (resource && resources.delete(resource)) resource.destroy();
}

function uploadGrayInputs(device, resources, pixelsA, pixelsB, width, height) {
  return [pixelsA, pixelsB].map((pixels, index) => {
    const texture = ownTexture(device, resources, width, height, "r8unorm",
      textureUsage.COPY_DST | textureUsage.TEXTURE_BINDING, "DeepMatching grayscale source " + index);
    device.queue.writeTexture(
      { texture },
      pixels,
      { bytesPerRow: width, rowsPerImage: height },
      { width, height, depthOrArrayLayers: 1 },
    );
    return { width, height, texture };
  });
}

function makeDescriptorSet(device, resources, width, height, label) {
  const usage = textureUsage.STORAGE_BINDING | textureUsage.TEXTURE_BINDING;
  return {
    width,
    height,
    first: ownTexture(device, resources, width, height, "rgba16float", usage, label + " HOG 1-4"),
    second: ownTexture(device, resources, width, height, "rgba16float", usage, label + " HOG 5-8"),
  };
}

function dispatchCompute(encoder, pipeline, bindGroup, workgroups, label) {
  const pass = encoder.beginComputePass({ label });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(...workgroups);
  pass.end();
}

async function makeDescriptors(device, resources, inputs, fullWidth, fullHeight,
  downscaleLevels, pipelines, onProgress) {
  const scale = 2 ** downscaleLevels;
  const width = Math.floor(fullWidth / scale);
  const height = Math.floor(fullHeight / scale);
  if (width < 1 || height < 1) throw new Error("La réduction des descripteurs dépasse la taille de l’image.");
  const grayFormat = device.features.has("texture-formats-tier1") ? "r8unorm" : "rgba8unorm";
  const makeSet = label => makeDescriptorSet(device, resources, width, height, label);
  const referenceSet = makeSet("Reference");
  const targetSet = makeSet("Target");
  const imageParams = makeParamsBuffer(device, resources,
    [fullWidth, fullHeight], "INRIA pixel descriptor image dimensions");
  const imageCount = inputs.length;

  for (let imageIndex = 0; imageIndex < imageCount; imageIndex++) {
    const input = inputs[imageIndex];
    const output = imageIndex === 0 ? referenceSet : targetSet;
    const grayUsage = textureUsage.STORAGE_BINDING | textureUsage.TEXTURE_BINDING;
    const grayHorizontal = ownTexture(device, resources, fullWidth, fullHeight,
      grayFormat, grayUsage, "Descriptor pre-smooth horizontal " + imageIndex);
    const graySmoothed = ownTexture(device, resources, fullWidth, fullHeight,
      grayFormat, grayUsage, "Descriptor pre-smoothed image " + imageIndex);
    const blurEncoder = device.createCommandEncoder({ label: "INRIA descriptor image pre-smoothing" });
    const horizontalGroup = makeBindGroup(device, pipelines.presmoothHorizontal, [
      { binding: 0, resource: input.texture.createView() },
      { binding: 1, resource: grayHorizontal.createView() },
      { binding: 2, resource: { buffer: imageParams } },
    ]);
    const verticalGroup = makeBindGroup(device, pipelines.presmoothVertical, [
      { binding: 0, resource: grayHorizontal.createView() },
      { binding: 1, resource: graySmoothed.createView() },
      { binding: 2, resource: { buffer: imageParams } },
    ]);
    dispatchCompute(blurEncoder, pipelines.presmoothHorizontal, horizontalGroup,
      [Math.ceil(fullWidth / 16), Math.ceil(fullHeight / 16)], "Descriptor Gaussian pre-smoothing horizontal");
    dispatchCompute(blurEncoder, pipelines.presmoothVertical, verticalGroup,
      [Math.ceil(fullWidth / 16), Math.ceil(fullHeight / 16)], "Descriptor Gaussian pre-smoothing vertical");
    device.queue.submit([blurEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    releaseResource(resources, grayHorizontal);

    const tileSide = 16;
    const halo = 4;
    const workSide = tileSide * scale + halo * 2;
    const tileBytes = workSide * workSide * 8 * Float32Array.BYTES_PER_ELEMENT;
    const maxTilesFromBinding = Math.floor(device.limits.maxStorageBufferBindingSize / tileBytes);
    const maxTilesFromBuffer = Math.floor(device.limits.maxBufferSize / tileBytes);
    const maxGroups = device.limits.maxComputeWorkgroupsPerDimension;
    const tilesX = Math.ceil(width / tileSide);
    const tilesY = Math.ceil(height / tileSide);
    const totalTiles = tilesX * tilesY;
    const workgroupsPerTileAxis = Math.ceil(workSide / 8);
    if (workgroupsPerTileAxis > maxGroups) {
      throw new Error("Une tuile de descripteur dépasse la limite de dispatch WebGPU.");
    }
    const batchSize = Math.min(32, totalTiles, maxTilesFromBinding, maxTilesFromBuffer,
      maxGroups);
    if (batchSize < 1) throw new Error("Une tuile de descripteur dépasse les limites mémoire WebGPU.");
    const scratchBytes = batchSize * tileBytes;
    const scratchA = ownBuffer(device, resources, scratchBytes,
      bufferUsage.STORAGE, "INRIA descriptor tile scratch A");
    const scratchB = ownBuffer(device, resources, scratchBytes,
      bufferUsage.STORAGE, "INRIA descriptor tile scratch B");

    for (let batchStart = 0; batchStart < totalTiles; batchStart += batchSize) {
      const actualBatchSize = Math.min(batchSize, totalTiles - batchStart);
      const config = makeParamsBuffer(device, resources, [
        fullWidth, fullHeight, width, height, scale, tileSide, workSide, halo,
        batchStart, actualBatchSize, tilesX, totalTiles,
      ], "INRIA descriptor tile batch " + imageIndex + ":" + batchStart);
      const tileEncoder = device.createCommandEncoder({ label: "INRIA HOG descriptor tile batch" });
      const hogGroup = makeBindGroup(device, pipelines.buildHog, [
        { binding: 3, resource: graySmoothed.createView() },
        { binding: 4, resource: { buffer: scratchA } },
        { binding: 6, resource: { buffer: config } },
      ]);
      const smoothHorizontalGroup = makeBindGroup(device, pipelines.smoothHorizontal, [
        { binding: 4, resource: { buffer: scratchA } },
        { binding: 5, resource: { buffer: scratchB } },
        { binding: 6, resource: { buffer: config } },
      ]);
      const smoothVerticalGroup = makeBindGroup(device, pipelines.smoothVertical, [
        { binding: 4, resource: { buffer: scratchA } },
        { binding: 5, resource: { buffer: scratchB } },
        { binding: 6, resource: { buffer: config } },
      ]);
      const postHorizontalGroup = makeBindGroup(device, pipelines.sigmoidSmoothHorizontal, [
        { binding: 4, resource: { buffer: scratchA } },
        { binding: 5, resource: { buffer: scratchB } },
        { binding: 6, resource: { buffer: config } },
      ]);
      const poolGroup = makeBindGroup(device, pipelines.averagePool, [
        { binding: 4, resource: { buffer: scratchA } },
        { binding: 6, resource: { buffer: config } },
        { binding: 7, resource: output.first.createView() },
        { binding: 8, resource: output.second.createView() },
      ]);
      const tileGroups = [Math.ceil(workSide / 8), Math.ceil(workSide / 8), actualBatchSize];
      dispatchCompute(tileEncoder, pipelines.buildHog, hogGroup, tileGroups, "INRIA pixel HOG responses");
      dispatchCompute(tileEncoder, pipelines.smoothHorizontal, smoothHorizontalGroup,
        tileGroups, "INRIA HOG mid-smoothing horizontal");
      dispatchCompute(tileEncoder, pipelines.smoothVertical, smoothVerticalGroup,
        tileGroups, "INRIA HOG mid-smoothing vertical");
      dispatchCompute(tileEncoder, pipelines.sigmoidSmoothHorizontal, postHorizontalGroup,
        tileGroups, "INRIA HOG sigmoid and post-smoothing horizontal");
      dispatchCompute(tileEncoder, pipelines.smoothVertical, smoothVerticalGroup,
        tileGroups, "INRIA HOG post-smoothing vertical");
      dispatchCompute(tileEncoder, pipelines.averagePool, poolGroup,
        [Math.ceil(tileSide / 8), Math.ceil(tileSide / 8), actualBatchSize],
        "INRIA descriptor average pooling");
      device.queue.submit([tileEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      releaseResource(resources, config);
      onProgress("Descripteur HOG INRIA · image " + (imageIndex + 1) + "/" + imageCount +
        " · tuiles " + Math.min(batchStart + actualBatchSize, totalTiles) + "/" + totalTiles + ".");
    }

    releaseResource(resources, graySmoothed);
    releaseResource(resources, scratchA);
    releaseResource(resources, scratchB);
    releaseResource(resources, input.texture);
  }

  releaseResource(resources, imageParams);
  return { width, height, reference: referenceSet, target: targetSet };
}

// Shared descriptor entry point for matchers that use the repository's validated
// INRIA-style HOG representation without running the DeepMatching hierarchy.
export async function createDeepMatchingHogDescriptors(device, referencePixels,
  targetPixels, width, height, downscaleLevels, onProgress = () => {}) {
  if (!bufferUsage || !textureUsage) {
    throw new Error("Les constantes WebGPU requises ne sont pas disponibles dans ce navigateur.");
  }
  if (referencePixels.length !== width * height || targetPixels.length !== width * height) {
    throw new Error("Les pixels reçus ne correspondent pas aux dimensions des images.");
  }
  if (width > device.limits.maxTextureDimension2D || height > device.limits.maxTextureDimension2D) {
    throw new Error("Les images dépassent la limite de texture WebGPU (" +
      device.limits.maxTextureDimension2D + " px).");
  }
  if (!Number.isInteger(downscaleLevels) || downscaleLevels < 0 || downscaleLevels > 6) {
    throw new Error("Le nombre de réductions HOG doit être compris entre 0 et 6.");
  }

  const resources = new Set();
  try {
    const grayStorageFormat = device.features.has("texture-formats-tier1") ? "r8unorm" : "rgba8unorm";
    const descriptorModule = await loadShader(device,
      "deepmatching-descriptor.wgsl", { GRAY_STORAGE_FORMAT: grayStorageFormat });
    const pipelines = await makeDescriptorPipelines(device, descriptorModule);
    const inputs = uploadGrayInputs(device, resources, referencePixels, targetPixels, width, height);
    const descriptors = await makeDescriptors(device, resources, inputs,
      width, height, downscaleLevels, pipelines, onProgress);
    await device.queue.onSubmittedWorkDone();
    let disposed = false;
    return {
      ...descriptors,
      scale: 2 ** downscaleLevels,
      destroy() {
        if (disposed) return;
        disposed = true;
        for (const resource of resources) resource.destroy();
        resources.clear();
      },
    };
  } catch (error) {
    try { await device.queue.onSubmittedWorkDone(); } catch {}
    for (const resource of resources) resource.destroy();
    throw error;
  }
}

function makeBindGroup(device, pipeline, entries) {
  return device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
}

function makeParamsBuffer(device, resources, params, label) {
  const words = new Uint32Array(16);
  for (let index = 0; index < Math.min(words.length, params.length); index++) words[index] = params[index] >>> 0;
  return uniform(device, resources, words, label);
}

export async function computeDeepMatchingDirect(device, referencePixels, targetPixels,
  width, height, settings, onProgress = () => {}) {
  if (!bufferUsage || !textureUsage || !mapMode) {
    throw new Error("Les constantes WebGPU requises ne sont pas disponibles dans ce navigateur.");
  }
  if (referencePixels.length !== width * height || targetPixels.length !== width * height) {
    throw new Error("Les pixels reçus ne correspondent pas aux dimensions des images.");
  }
  if (width > device.limits.maxTextureDimension2D || height > device.limits.maxTextureDimension2D) {
    throw new Error("Les images dépassent la limite de texture WebGPU (" +
      device.limits.maxTextureDimension2D + " px). Réduire les images avant de les charger.");
  }
  const scale = 2 ** settings.downscaleLevels;
  const scaledPatch = settings.patchSize / scale;
  const cellsSide = scaledPatch / 4;
  if (!Number.isInteger(scaledPatch) || !Number.isInteger(cellsSide)
      || cellsSide < 1 || (cellsSide & (cellsSide - 1)) !== 0) {
    throw new Error("La fenêtre réduite doit mesurer 4, 8, 16, 32 ou 64 pixels.");
  }
  if (settings.patchSize % scale !== 0
      || (!settings.seedPoints && settings.stride % scale !== 0)) {
    throw new Error("La fenêtre et le pas de grille doivent être divisibles par le facteur de réduction.");
  }
  const seedMode = settings.seedPoints && settings.seedPoints.length ? "manual" : "grid";
  const halfPatch = (settings.patchSize - 1) / 2;
  const manualTopLefts = seedMode === "manual" ? settings.seedPoints.map((point, index) => {
    const pixelX = Math.round(point.x - halfPatch);
    const pixelY = Math.round(point.y - halfPatch);
    const alignedX = Math.round(pixelX / scale) * scale;
    const alignedY = Math.round(pixelY / scale) * scale;
    if (alignedX < 0 || alignedY < 0
        || alignedX + settings.patchSize > width || alignedY + settings.patchSize > height) {
      throw new Error("Seed " + (index + 1) + " : la fenêtre sort de l’image.");
    }
    return { x: alignedX, y: alignedY };
  }) : null;
  const gridWidth = seedMode === "manual"
    ? manualTopLefts.length : 1 + Math.floor((width - settings.patchSize) / settings.stride);
  const gridHeight = seedMode === "manual"
    ? 1 : 1 + Math.floor((height - settings.patchSize) / settings.stride);
  if (gridWidth < 1 || gridHeight < 1) throw new Error("La fenêtre est plus grande que l’image.");
  const queryCount = gridWidth * gridHeight;
  if (!Number.isSafeInteger(queryCount) || queryCount < 1) throw new Error("La grille de requêtes est trop grande.");
  const sourceTopLeft = index => seedMode === "manual" ? manualTopLefts[index] : {
    x: (index % gridWidth) * settings.stride,
    y: Math.floor(index / gridWidth) * settings.stride,
  };

  const resources = new Set();
  try {
    const grayStorageFormat = device.features.has("texture-formats-tier1") ? "r8unorm" : "rgba8unorm";
    const [descriptorModule, matcherModule] = await Promise.all([
      loadShader(device, "deepmatching-descriptor.wgsl", { GRAY_STORAGE_FORMAT: grayStorageFormat }),
      loadShader(device, "deepmatching-match.wgsl"),
    ]);
    const pipelines = await makePipelines(device, descriptorModule, matcherModule);
    const inputs = uploadGrayInputs(device, resources,
      referencePixels, targetPixels, width, height);
    const descriptors = await makeDescriptors(device, resources, inputs,
      width, height, settings.downscaleLevels, pipelines, onProgress);
    await device.queue.onSubmittedWorkDone();

    const maximumTargetX = descriptors.width - scaledPatch + 1;
    const maximumTargetY = descriptors.height - scaledPatch + 1;
    const radius = settings.searchRadiusPx > 0
      ? Math.floor(settings.searchRadiusPx / scale) : 0;
    const candidateWidth = settings.searchRadiusPx > 0
      ? Math.min(maximumTargetX, 2 * radius + 1) : maximumTargetX;
    const candidateHeight = settings.searchRadiusPx > 0
      ? Math.min(maximumTargetY, 2 * radius + 1) : maximumTargetY;
    const mapArea = candidateWidth * candidateHeight;
    const estimate = queryCount * mapArea * scaledPatch * scaledPatch;
    if (!Number.isSafeInteger(estimate) || estimate > 600000000) {
      throw new Error("La corrélation des patches demanderait " +
        Math.round(estimate / 1000000) + " millions de comparaisons de descripteurs. " +
        "Augmentez la réduction, réduisez le rayon ou saisissez moins de seeds.");
    }
    const mapBytesPerQuery = mapArea * Float32Array.BYTES_PER_ELEMENT;
    const maxBatchFromMaps = Math.floor(device.limits.maxStorageBufferBindingSize / mapBytesPerQuery);
    const batchSize = Math.min(8, queryCount, maxBatchFromMaps);
    if (batchSize < 1) {
      throw new Error("La carte de recherche d’une seed dépasse la mémoire du GPU. " +
        "Augmentez la réduction ou réduisez le rayon.");
    }
    const mapBufferBytes = mapBytesPerQuery * batchSize;
    if (mapBufferBytes > device.limits.maxBufferSize) {
      throw new Error("Les cartes de recherche dépassent la mémoire disponible sur ce GPU.");
    }
    if (queryCount * 16 > device.limits.maxBufferSize) {
      throw new Error("La liste des seeds dépasse la taille maximale de lecture du GPU.");
    }
    if (Math.ceil(candidateWidth / 8) > device.limits.maxComputeWorkgroupsPerDimension
        || Math.ceil(candidateHeight / 8) > device.limits.maxComputeWorkgroupsPerDimension) {
      throw new Error("Le voisinage dépasse la limite de dispatch WebGPU. Augmentez la réduction.");
    }

    const mapsBuffer = ownBuffer(device, resources, mapBufferBytes, bufferUsage.STORAGE,
      "DeepMatching patch response maps");
    const queryBuffer = ownBuffer(device, resources, batchSize * 16,
      bufferUsage.STORAGE | bufferUsage.COPY_DST, "DeepMatching query positions");
    const resultBuffer = ownBuffer(device, resources, batchSize * 16,
      bufferUsage.STORAGE | bufferUsage.COPY_SRC, "DeepMatching patch matches");
    const readback = ownBuffer(device, resources, queryCount * 16,
      bufferUsage.COPY_DST | bufferUsage.MAP_READ, "DeepMatching result readback");

    const correlationParams = makeParamsBuffer(device, resources, [
      candidateWidth, candidateHeight, descriptors.width, descriptors.height,
      scaledPatch, batchSize, mapArea, radius,
    ], "DeepMatching patch correlation parameters");
    const correlationGroup = makeBindGroup(device, pipelines.correlate, [
      { binding: 0, resource: descriptors.reference.first.createView() },
      { binding: 1, resource: descriptors.target.first.createView() },
      { binding: 2, resource: descriptors.reference.second.createView() },
      { binding: 3, resource: descriptors.target.second.createView() },
      { binding: 6, resource: { buffer: queryBuffer } },
      { binding: 7, resource: { buffer: mapsBuffer } },
      { binding: 8, resource: { buffer: correlationParams } },
    ]);

    const selectParams = makeParamsBuffer(device, resources, [
      candidateWidth, candidateHeight, 0, mapArea,
      batchSize, 4,
    ], "DeepMatching patch selection parameters");
    const selectionGroup = makeBindGroup(device, pipelines.select, [
      { binding: 7, resource: { buffer: mapsBuffer } },
      { binding: 8, resource: { buffer: selectParams } },
      { binding: 9, resource: { buffer: resultBuffer } },
    ]);

    const resultValues = new Float32Array(queryCount * 4);
    const candidateStarts = new Array(queryCount);
    const batchCount = Math.ceil(queryCount / batchSize);
    for (let batch = 0; batch < batchCount; batch++) {
      const first = batch * batchSize;
      const actualCount = Math.min(batchSize, queryCount - first);
      const queryValues = new Uint32Array(batchSize * 4);
      for (let index = 0; index < actualCount; index++) {
        const globalIndex = first + index;
        const source = sourceTopLeft(globalIndex);
        const base = index * 4;
        const sourceX = Math.floor(source.x / scale);
        const sourceY = Math.floor(source.y / scale);
        const startX = Math.max(0, Math.min(maximumTargetX - candidateWidth,
          sourceX - Math.floor(candidateWidth / 2)));
        const startY = Math.max(0, Math.min(maximumTargetY - candidateHeight,
          sourceY - Math.floor(candidateHeight / 2)));
        queryValues[base] = sourceX;
        queryValues[base + 1] = sourceY;
        queryValues[base + 2] = startX;
        queryValues[base + 3] = startY;
        candidateStarts[globalIndex] = { x: startX, y: startY };
      }
      device.queue.writeBuffer(queryBuffer, 0, queryValues);

      const encoder = device.createCommandEncoder({ label: "DeepMatching patch batch " + (batch + 1) });
      const correlationPass = encoder.beginComputePass({ label: "Full patch descriptor correlations" });
      correlationPass.setPipeline(pipelines.correlate);
      correlationPass.setBindGroup(0, correlationGroup);
      correlationPass.dispatchWorkgroups(
        Math.ceil(candidateWidth / 8),
        Math.ceil(candidateHeight / 8),
        batchSize,
      );
      correlationPass.end();
      {
        const pass = encoder.beginComputePass({ label: "Select best and second patch response" });
        pass.setPipeline(pipelines.select);
        pass.setBindGroup(0, selectionGroup);
        pass.dispatchWorkgroups(batchSize);
        pass.end();
      }
      encoder.copyBufferToBuffer(resultBuffer, 0, readback, first * 16, actualCount * 16);
      device.queue.submit([encoder.finish()]);
      onProgress("Recherche DeepMatching : lot " + (batch + 1) + " / " + batchCount + ".");
      await device.queue.onSubmittedWorkDone();
    }

    await readback.mapAsync(mapMode.READ);
    resultValues.set(new Float32Array(readback.getMappedRange()).slice());
    readback.unmap();

    const matches = new Array(queryCount);
    for (let index = 0; index < queryCount; index++) {
      const source = sourceTopLeft(index);
      const sourceX = source.x;
      const sourceY = source.y;
      const base = index * 4;
      const mapX = resultValues[base];
      const mapY = resultValues[base + 1];
      const score = resultValues[base + 2];
      const secondScore = resultValues[base + 3];
      const candidateStart = candidateStarts[index];
      const targetX = (mapX + candidateStart.x) * scale;
      const targetY = (mapY + candidateStart.y) * scale;
      const dx = Number((targetX - sourceX).toFixed(2));
      const dy = Number((targetY - sourceY).toFixed(2));
      const sourceReducedX = Math.floor(sourceX / scale);
      const sourceReducedY = Math.floor(sourceY / scale);
      const atSearchBoundary = settings.searchRadiusPx > 0
        && (Math.abs(mapX + candidateStart.x - sourceReducedX) >= radius - 0.51
          || Math.abs(mapY + candidateStart.y - sourceReducedY) >= radius - 0.51);
      const hasAlternative = secondScore >= 0;
      const margin = hasAlternative ? score - secondScore : 0;
      const hasDicSupport = targetX >= 2 && targetY >= 2
        && targetX + settings.patchSize <= width - 2
        && targetY + settings.patchSize <= height - 2;
      const usable = hasDicSupport && !atSearchBoundary && hasAlternative && score >= settings.minimumScore
        && margin >= settings.minimumMargin;
      matches[index] = {
        x: sourceX + (settings.patchSize - 1) / 2,
        y: sourceY + (settings.patchSize - 1) / 2,
        dx, dy, score, secondScore, margin, usable,
      };
    }
    return {
      width,
      height,
      patchSize: settings.patchSize,
      stride: settings.stride,
      scale,
      matcherMode: "direct-patch-correlation",
      seedMode,
      searchRadiusPx: settings.searchRadiusPx || 0,
      gridWidth,
      gridHeight,
      count: queryCount,
      estimatedDescriptorComparisons: estimate,
      matches,
    };
  } finally {
    try { await device.queue.onSubmittedWorkDone(); } catch {}
    for (const resource of resources) resource.destroy();
  }
}

function buildSeedPyramid(seed, sourceWidth, sourceHeight, scale, patchSize, forcedRoot = null) {
  const atomSize = 4 * scale;
  const patchReduced = patchSize / scale;
  const cellsSide = patchReduced / 4;
  const rootLevel = Math.log2(cellsSide);
  const maximumRootX = Math.floor(sourceWidth / 4) * 4;
  const maximumRootY = Math.floor(sourceHeight / 4) * 4;
  const rootX = forcedRoot ? forcedRoot.x : rootLevel === 0
    ? Math.max(2, Math.min(sourceWidth - 2,
      2 + Math.round((seed.x / scale - 2) / 4) * 4))
    : Math.max(0, Math.min(maximumRootX, Math.round(seed.x / atomSize) * 4));
  const rootY = forcedRoot ? forcedRoot.y : rootLevel === 0
    ? Math.max(2, Math.min(sourceHeight - 2,
      2 + Math.round((seed.y / scale - 2) / 4) * 4))
    : Math.max(0, Math.min(maximumRootY, Math.round(seed.y / atomSize) * 4));
  const levels = Array.from({ length: rootLevel + 1 }, () => []);

  function makeNode(level, x, y) {
    const valid = level === 0
      ? x >= 2 && y >= 2 && x + 2 <= sourceWidth && y + 2 <= sourceHeight
        && x % 4 === 2 && y % 4 === 2
      : x >= 0 && y >= 0 && x <= maximumRootX && y <= maximumRootY
        && x % 4 === 0 && y % 4 === 0;
    if (!valid) return -1;

    const nodes = levels[level];
    const index = nodes.length;
    const node = { x, y, children: [-1, -1, -1, -1], norm: 0, offsetX: 0, offsetY: 0 };
    nodes.push(node);
    if (level === 0) {
      node.norm = 1;
    } else {
      const gap = 2 ** level;
      let norm = 0;
      let slot = 0;
      for (const dy of [-1, 1]) {
        for (const dx of [-1, 1]) {
          const child = makeNode(level - 1, x + dx * gap, y + dy * gap);
          node.children[slot++] = child;
          if (child >= 0) norm += levels[level - 1][child].norm;
        }
      }
      node.norm = norm;
    }
    return index;
  }

  const rootIndex = makeNode(rootLevel, rootX, rootY);
  if (rootIndex !== 0) throw new Error("Construction INRIA locale : racine invalide.");
  let closestLeaf = null;
  let closestDistance = Infinity;
  for (const leaf of levels[0]) {
    const x = leaf.x * scale;
    const y = leaf.y * scale;
    const distance = (x - seed.x) ** 2 + (y - seed.y) ** 2;
    if (distance < closestDistance) {
      closestLeaf = leaf;
      closestDistance = distance;
    }
  }
  return { levels, rootX, rootY, rootLevel, closestLeaf, atomSize };
}

function enumerateRootLattice(rootLevel, sourceWidth, sourceHeight, rootSpacing) {
  const maximumRootX = Math.floor(sourceWidth / 4) * 4;
  const maximumRootY = Math.floor(sourceHeight / 4) * 4;
  const roots = [];
  const first = rootLevel === 0 ? 2 : 0;
  const step = rootLevel === 0 ? 4 : rootSpacing;
  for (let y = first; y <= (rootLevel === 0 ? maximumRootY - 2 : maximumRootY); y += step) {
    for (let x = first; x <= (rootLevel === 0 ? maximumRootX - 2 : maximumRootX); x += step) {
      roots.push({ x, y });
    }
  }
  return roots;
}

function buildAtomicPaths(levels) {
  const rootLevel = levels.length - 1;
  const atomCount = levels[0].length;
  const paths = new Int32Array(atomCount * rootLevel * 4);
  if (!rootLevel || !atomCount) return paths;
  const path = Array.from({ length: rootLevel }, () => null);

  function visit(level, nodeIndex) {
    if (level === 0) {
      const atomBase = nodeIndex * rootLevel * 4;
      for (let step = 0; step < rootLevel; step++) {
        paths.set(path[step], atomBase + step * 4);
      }
      return;
    }
    const parent = levels[level][nodeIndex];
    for (let slot = 0; slot < 4; slot++) {
      const childIndex = parent.children[slot];
      if (childIndex < 0) continue;
      const child = levels[level - 1][childIndex];
      path[rootLevel - level] = [childIndex, slot, child.offsetX, child.offsetY];
      visit(level - 1, childIndex);
    }
  }

  visit(rootLevel, 0);
  return paths;
}

function makeHierarchicalPipelines(device, module) {
  return Promise.all([
    makePipeline(device, module, "correlate_atomic", "INRIA atomic patch correlations"),
    makePipeline(device, module, "aggregate_parent", "INRIA max-pool and sparse aggregation"),
    makePipeline(device, module, "initialize_topdown", "INRIA top-down candidate initialization"),
    makePipeline(device, module, "descend_topdown_one_level", "INRIA GPU top-down argmax"),
  ]).then(([atomic, aggregate, initializeTopdown, descendTopdown]) =>
    ({ atomic, aggregate, initializeTopdown, descendTopdown }));
}

function storageBuffer(device, resources, values, label, usage = bufferUsage.STORAGE) {
  const byteLength = Math.max(16, values.byteLength);
  const buffer = ownBuffer(device, resources, byteLength,
    usage | bufferUsage.COPY_DST, label);
  if (values.byteLength) device.queue.writeBuffer(buffer, 0, values);
  return buffer;
}

function uploadParentLevel(device, resources, lower, current, label) {
  const parentCount = current.length;
  const childCount = lower.length;
  const children = new Int32Array(parentCount * 4);
  const parentInfo = new Int32Array(parentCount * 4);
  const weights = new Float32Array(parentCount * 4);
  const borderFactors = new Float32Array(parentCount * 12);
  const childInfo = new Int32Array(childCount * 4);

  for (let index = 0; index < childCount; index++) {
    const child = lower[index];
    const pooledX = Math.floor(child.offsetX / 2);
    const pooledY = Math.floor(child.offsetY / 2);
    childInfo.set([pooledX, pooledY,
      ((child.offsetX % 2) + 2) % 2, ((child.offsetY % 2) + 2) % 2], index * 4);
  }

  for (let parentIndex = 0; parentIndex < parentCount; parentIndex++) {
    const parent = current[parentIndex];
    children.set(parent.children, parentIndex * 4);
    const available = [];
    let sumX = 0;
    let sumY = 0;
    for (let slot = 0; slot < 4; slot++) {
      const childIndex = parent.children[slot];
      if (childIndex < 0) continue;
      const dx = (slot % 2) * 2 - 1;
      const dy = Math.floor(slot / 2) * 2 - 1;
      const child = lower[childIndex];
      const offsetX = Math.floor(child.offsetX / 2);
      const offsetY = Math.floor(child.offsetY / 2);
      available.push({ slot, dx, dy, offsetX, offsetY, child });
      sumX += offsetX - dx;
      sumY += offsetY - dy;
    }
    parent.offsetX = available.length ? Math.floor(0.5 + sumX / available.length) : 0;
    parent.offsetY = available.length ? Math.floor(0.5 + sumY / available.length) : 0;
    parentInfo.set([parent.offsetX, parent.offsetY, 0, 0], parentIndex * 4);

    const sums = new Float64Array(9);
    for (const entry of available) {
      const transX = entry.dx + parent.offsetX - entry.offsetX;
      const transY = entry.dy + parent.offsetY - entry.offsetY;
      const weight = parent.norm > 0 ? entry.child.norm / parent.norm : 0;
      weights[parentIndex * 4 + entry.slot] = weight;
      for (let regionY = -1; regionY <= 1; regionY++) {
        for (let regionX = -1; regionX <= 1; regionX++) {
          if (regionX * transX <= 0 && regionY * transY <= 0) {
            sums[(regionY + 1) * 3 + regionX + 1] += weight;
          }
        }
      }
    }
    for (let regionY = 0; regionY < 3; regionY++) {
      for (let regionX = 0; regionX < 3; regionX++) {
        const sum = sums[regionY * 3 + regionX];
        borderFactors[parentIndex * 12 + regionY * 4 + regionX] = sum > 0
          ? 1 / Math.pow(sum, 0.9) : 0;
      }
    }
  }

  return {
    children: storageBuffer(device, resources, children, label + " children"),
    parentInfo: storageBuffer(device, resources, parentInfo, label + " parent offsets"),
    weights: storageBuffer(device, resources, weights, label + " child weights"),
    borderFactors: storageBuffer(device, resources, borderFactors, label + " border factors"),
    childInfo: storageBuffer(device, resources, childInfo, label + " pooled child offsets"),
  };
}

export async function computeDeepMatching(device, referencePixels, targetPixels,
  width, height, settings, onProgress = () => {}) {
  settings = await applyBrowserDefaults(settings ?? {}, "deepmatching");
  if (!bufferUsage || !textureUsage || !mapMode) {
    throw new Error("Les constantes WebGPU requises ne sont pas disponibles dans ce navigateur.");
  }
  if (referencePixels.length !== width * height || targetPixels.length !== width * height) {
    throw new Error("Les pixels reçus ne correspondent pas aux dimensions des images.");
  }
  if (width > device.limits.maxTextureDimension2D || height > device.limits.maxTextureDimension2D) {
    throw new Error("Les images dépassent la limite de texture WebGPU (" +
      device.limits.maxTextureDimension2D + " px). Réduisez les images avant de les charger.");
  }

  const scale = 2 ** settings.downscaleLevels;
  const rootPatchSize = settings.rootPatchSize || settings.patchSize;
  const scaledPatch = rootPatchSize / scale;
  const cellsSide = scaledPatch / 4;
  if (!Number.isInteger(scaledPatch) || !Number.isInteger(cellsSide)
      || cellsSide < 1 || (cellsSide & (cellsSide - 1)) !== 0) {
    throw new Error("La fenêtre réduite doit mesurer 4, 8, 16, 32 ou 64 pixels.");
  }
  if (rootPatchSize % scale !== 0) {
    throw new Error("L’échelle racine doit être divisible par le facteur de réduction.");
  }

  const seedMode = settings.seedPoints && settings.seedPoints.length ? "manual" : "grid";
  const seeds = seedMode === "manual"
    ? settings.seedPoints
    : Array.from({ length: Math.max(0, 1 + Math.floor((height - settings.patchSize) / settings.stride)) },
      (_, row) => Array.from({
        length: Math.max(0, 1 + Math.floor((width - settings.patchSize) / settings.stride)),
      }, (_, column) => ({
        x: column * settings.stride + (settings.patchSize - 1) / 2,
        y: row * settings.stride + (settings.patchSize - 1) / 2,
      }))).flat();
  if (!seeds.length) throw new Error("Aucune seed ne tient dans l’image.");
  if (seeds.length > 512) {
    throw new Error("Le mode INRIA local est limité à 512 seeds par calcul. Augmentez le pas de grille ou saisissez une sélection manuelle.");
  }
  for (let index = 0; index < seeds.length; index++) {
    const seed = seeds[index];
    if (!Number.isFinite(seed.x) || !Number.isFinite(seed.y)
        || seed.x < 0 || seed.y < 0 || seed.x >= width || seed.y >= height) {
      throw new Error("Seed " + (index + 1) + " : son centre doit se trouver dans l’image.");
    }
  }

  const resources = new Set();
  try {
    const grayStorageFormat = device.features.has("texture-formats-tier1") ? "r8unorm" : "rgba8unorm";
    const [descriptorModule, matcherModule, hierarchyModule] = await Promise.all([
      loadShader(device, "deepmatching-descriptor.wgsl", { GRAY_STORAGE_FORMAT: grayStorageFormat }),
      loadShader(device, "deepmatching-match.wgsl"),
      loadShader(device, "deepmatching-hierarchy.wgsl"),
    ]);
    const descriptorPipelines = await makePipelines(device, descriptorModule, matcherModule);
    const hierarchyPipelines = await makeHierarchicalPipelines(device, hierarchyModule);
    const inputs = uploadGrayInputs(device, resources,
      referencePixels, targetPixels, width, height);
    const descriptorStart = performance.now();
    const descriptors = await makeDescriptors(device, resources, inputs,
      width, height, settings.downscaleLevels, descriptorPipelines, onProgress);
    const descriptorMs = performance.now() - descriptorStart;

    const sourceWidth = Math.floor(descriptors.width / 4) * 4;
    const sourceHeight = Math.floor(descriptors.height / 4) * 4;
    const targetMapWidth = descriptors.width + 1;
    const targetMapHeight = descriptors.height + 1;
    const reducedRadius = settings.searchRadiusPx > 0
      ? Math.floor(settings.searchRadiusPx / scale) : 0;
    // INRIA uses ngh_rad/f; a value that rounds to zero requests an unlimited map.
    const mapWidth = reducedRadius > 0 ? Math.min(targetMapWidth, 2 * reducedRadius) : targetMapWidth;
    const mapHeight = reducedRadius > 0 ? Math.min(targetMapHeight, 2 * reducedRadius) : targetMapHeight;
    const mapArea = mapWidth * mapHeight;
    if (!Number.isSafeInteger(mapArea) || mapArea < 1) throw new Error("Carte de recherche invalide.");
    if (Math.ceil(mapWidth / 8) > device.limits.maxComputeWorkgroupsPerDimension
        || Math.ceil(mapHeight / 8) > device.limits.maxComputeWorkgroupsPerDimension) {
      throw new Error("Le voisinage dépasse la limite de dispatch WebGPU. Réduisez le rayon.");
    }

    const matches = [];
    let totalDescriptorComparisons = 0;
    const seedStates = seeds.map(seed => {
      const atomLeaf = {
        x: Math.max(2, Math.min(sourceWidth - 2,
          2 + Math.round((seed.x / scale - 2) / 4) * 4)),
        y: Math.max(2, Math.min(sourceHeight - 2,
          2 + Math.round((seed.y / scale - 2) / 4) * 4)),
      };
      return { seed, atomLeaf };
    });
    const rootCenters = enumerateRootLattice(Math.log2(cellsSide), sourceWidth,
      sourceHeight, rootPatchSize / scale);
    const rootJobCount = rootCenters.length;
    if (!rootJobCount) throw new Error("La grille INRIA ne contient aucune racine valide.");
    if (rootJobCount > 8192) {
      throw new Error("La grille INRIA demande " + rootJobCount + " racines. La limite actuelle est 8192; " +
        "augmentez la réduction ou l’échelle racine.");
    }
    const sourceBestGlobal = new Map();
    const targetBestGlobal = new Map();
    let completedRootJobs = 0;
    const hierarchyStart = performance.now();
    const rootBatchSize = 2;
    for (let batchStart = 0; batchStart < rootCenters.length; batchStart += rootBatchSize) {
      const batchEnd = Math.min(rootCenters.length, batchStart + rootBatchSize);
      const batchResources = new Set(resources);
      const batchEncoder = device.createCommandEncoder({
        label: "INRIA dense root batch " + (batchStart + 1) + "-" + batchEnd,
      });
      const batchJobs = [];
      for (let rootIndex = batchStart; rootIndex < batchEnd; rootIndex++) {
        const rootCenter = rootCenters[rootIndex];
        const pyramid = buildSeedPyramid({ x: rootCenter.x * scale, y: rootCenter.y * scale },
          sourceWidth, sourceHeight,
          scale, rootPatchSize, rootCenter);
        const levels = pyramid.levels;
        const atomCount = levels[0].length;
        const descriptorComparisons = atomCount * mapArea * 16;
        totalDescriptorComparisons += descriptorComparisons;
        if (!Number.isSafeInteger(descriptorComparisons)) {
          throw new Error("Le coût estimé de la racine dépasse la précision numérique JavaScript.");
        }
        if (atomCount > device.limits.maxComputeWorkgroupsPerDimension) {
          throw new Error("La pyramide de la racine " + (rootIndex + 1) + " dépasse la limite WebGPU de " +
            "workgroups en profondeur. Augmentez la réduction ou réduisez l’échelle racine.");
        }

        levels[0].mapWidth = mapWidth;
        levels[0].mapHeight = mapHeight;
        levels[0].mapArea = mapArea;
        for (let level = 1; level < levels.length; level++) {
          const pooledWidth = Math.floor((levels[level - 1].mapWidth + 2) / 2);
          const pooledHeight = Math.floor((levels[level - 1].mapHeight + 2) / 2);
          levels[level].mapWidth = Math.max(5, pooledWidth);
          levels[level].mapHeight = Math.max(5, pooledHeight);
          levels[level].mapArea = levels[level].mapWidth * levels[level].mapHeight;
        }

        const levelBytes = levels.map(level => Math.max(1, level.length) * level.mapArea * 4);
        const maxBuffer = Math.min(device.limits.maxBufferSize,
          device.limits.maxStorageBufferBindingSize);
        for (let level = 0; level < levels.length; level++) {
          if (!Number.isSafeInteger(levelBytes[level]) || levelBytes[level] > maxBuffer) {
            throw new Error("La carte INRIA de la racine " + (rootIndex + 1) + " dépasse la limite de tampon " +
              "WebGPU à l’échelle " + level + ". Augmentez la réduction ou réduisez rayon/fenêtre.");
          }
        }

        const gpuMaps = [];
        const queryBuffer = new Uint32Array(Math.max(1, atomCount) * 4);
        for (let atomIndex = 0; atomIndex < atomCount; atomIndex++) {
          const atom = levels[0][atomIndex];
          const valid = atom.x >= 2 && atom.y >= 2
            && atom.x + 2 <= sourceWidth && atom.y + 2 <= sourceHeight;
          const sourceStartX = valid ? atom.x - 2 : 0xffffffff;
          const sourceStartY = valid ? atom.y - 2 : 0xffffffff;
          const left = reducedRadius > 0
            ? Math.max(0, Math.min(atom.x - reducedRadius, targetMapWidth - 2 * reducedRadius)) : 0;
          const top = reducedRadius > 0
            ? Math.max(0, Math.min(atom.y - reducedRadius, targetMapHeight - 2 * reducedRadius)) : 0;
          queryBuffer.set([sourceStartX, sourceStartY, left, top], atomIndex * 4);
          atom.offsetX = left;
          atom.offsetY = top;
        }

        const atomMapBytes = Math.max(1, atomCount) * mapArea * Float32Array.BYTES_PER_ELEMENT;
        const atomMapBuffer = ownBuffer(device, resources, atomMapBytes,
          bufferUsage.STORAGE | bufferUsage.COPY_SRC, "INRIA atomic response maps " + rootIndex);
        gpuMaps.push(atomMapBuffer);

        const atomQueries = storageBuffer(device, resources, queryBuffer,
          "INRIA atomic patch queries " + rootIndex);
        const atomParams = makeParamsBuffer(device, resources, [
          mapWidth, mapHeight, descriptors.width, descriptors.height,
          atomCount, mapArea, reducedRadius, 0,
        ], "INRIA atomic response parameters " + rootIndex);
        const atomGroup = makeBindGroup(device, hierarchyPipelines.atomic, [
          { binding: 0, resource: descriptors.reference.first.createView() },
          { binding: 1, resource: descriptors.target.first.createView() },
          { binding: 2, resource: descriptors.reference.second.createView() },
          { binding: 3, resource: descriptors.target.second.createView() },
          { binding: 6, resource: { buffer: atomQueries } },
          { binding: 7, resource: { buffer: atomMapBuffer } },
          { binding: 8, resource: { buffer: atomParams } },
        ]);

        if (atomCount) {
          dispatchCompute(batchEncoder, hierarchyPipelines.atomic, atomGroup,
            [Math.ceil(mapWidth / 8), Math.ceil(mapHeight / 8), atomCount],
            "INRIA normalized atomic patch correlation");
        }

        for (let level = 1; level < levels.length; level++) {
          const lower = levels[level - 1];
          const current = levels[level];
          const parentBytes = levelBytes[level];
          const parentBuffer = ownBuffer(device, resources, parentBytes,
            bufferUsage.STORAGE | bufferUsage.COPY_SRC,
            "INRIA aggregate maps " + rootIndex + ":" + level);
          gpuMaps.push(parentBuffer);

          const metadata = uploadParentLevel(device, resources, lower, current,
            "INRIA hierarchy metadata " + rootIndex + ":" + level);
          const aggregateParams = makeParamsBuffer(device, resources, [
            lower.mapWidth, lower.mapHeight, Math.floor((lower.mapWidth + 2) / 2),
            Math.floor((lower.mapHeight + 2) / 2), current.length, current.mapArea,
            lower.mapArea, current.mapWidth, current.mapHeight,
          ], "INRIA aggregate parameters " + rootIndex + ":" + level);
          const aggregateGroup = makeBindGroup(device, hierarchyPipelines.aggregate, [
            { binding: 8, resource: { buffer: aggregateParams } },
            { binding: 10, resource: { buffer: gpuMaps[level - 1] } },
            { binding: 11, resource: { buffer: parentBuffer } },
            { binding: 12, resource: { buffer: metadata.children } },
            { binding: 13, resource: { buffer: metadata.childInfo } },
            { binding: 14, resource: { buffer: metadata.parentInfo } },
            { binding: 15, resource: { buffer: metadata.weights } },
            { binding: 16, resource: { buffer: metadata.borderFactors } },
          ]);
          if (current.length) {
            dispatchCompute(batchEncoder, hierarchyPipelines.aggregate, aggregateGroup,
              [Math.ceil(current.mapWidth / 8), Math.ceil(current.mapHeight / 8), current.length],
              "INRIA max-pooling and four-child aggregation level " + level);
          }
        }

        const rootLevel = levels.length - 1;
        const paths = buildAtomicPaths(levels);
        const pathsBuffer = storageBuffer(device, resources, paths,
          "INRIA atomic paths " + rootIndex);
        const rootArea = levels[rootLevel].mapArea;
        const candidateCount = atomCount * rootArea;
        if (!Number.isSafeInteger(candidateCount) || candidateCount > 64 *
            device.limits.maxComputeWorkgroupsPerDimension) {
          throw new Error("La descente GPU de la racine " + (rootIndex + 1) +
            " dépasse la limite de dispatch. Augmentez la réduction ou la taille racine.");
        }
        const candidateBytes = Math.max(16, candidateCount * 16);
        if (!Number.isSafeInteger(candidateBytes) || candidateBytes > maxBuffer) {
          throw new Error("Les candidats descendants de la racine " + (rootIndex + 1) +
            " dépassent la limite de tampon WebGPU.");
        }
        const searchStates = ownBuffer(device, resources, candidateBytes,
          bufferUsage.STORAGE | bufferUsage.COPY_SRC,
          "INRIA GPU top-down states " + rootIndex);
        const candidateReadback = ownBuffer(device, resources, candidateBytes,
          bufferUsage.COPY_DST | bufferUsage.MAP_READ,
          "INRIA compact candidates readback " + rootIndex);
        const root = levels[rootLevel][0];
        const initializeParams = makeParamsBuffer(device, resources, [
          levels[rootLevel].mapWidth, levels[rootLevel].mapHeight, atomCount,
          rootArea, candidateCount, root.offsetX, root.offsetY,
        ], "INRIA top-down root parameters " + rootIndex);
        const initializeGroup = makeBindGroup(device, hierarchyPipelines.initializeTopdown, [
          { binding: 8, resource: { buffer: initializeParams } },
          { binding: 20, resource: { buffer: gpuMaps[rootLevel] } },
          { binding: 22, resource: { buffer: searchStates } },
        ]);
        if (candidateCount) {
          dispatchCompute(batchEncoder, hierarchyPipelines.initializeTopdown, initializeGroup,
            [Math.ceil(candidateCount / 64)], "INRIA GPU top-down root initialization");
        }

        for (let level = rootLevel; level > 0; level--) {
          const lower = levels[level - 1];
          const step = rootLevel - level;
          const descendParams = makeParamsBuffer(device, resources, [
            lower.mapWidth, lower.mapHeight, lower.mapArea, atomCount,
            candidateCount, step, rootLevel,
          ], "INRIA top-down step parameters " + rootIndex + ":" + step);
          const descendGroup = makeBindGroup(device, hierarchyPipelines.descendTopdown, [
            { binding: 8, resource: { buffer: descendParams } },
            { binding: 21, resource: { buffer: pathsBuffer } },
            { binding: 22, resource: { buffer: searchStates } },
            { binding: 24, resource: { buffer: gpuMaps[level - 1] } },
          ]);
          dispatchCompute(batchEncoder, hierarchyPipelines.descendTopdown, descendGroup,
            [Math.ceil(candidateCount / 64)], "INRIA GPU top-down argmax level " + level);
        }
        batchEncoder.copyBufferToBuffer(searchStates, 0, candidateReadback, 0, candidateBytes);
        batchJobs.push({ pyramid, levels, atomCount, candidateCount, candidateReadback });
      }

      device.queue.submit([batchEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      for (const job of batchJobs) {
        await job.candidateReadback.mapAsync(mapMode.READ);
        const mapped = job.candidateReadback.getMappedRange();
        const values = new Float32Array(mapped);
        const coordinates = new Int32Array(mapped);
        const rootX = job.pyramid.rootX * scale;
        const rootY = job.pyramid.rootY * scale;
        for (let candidateIndex = 0; candidateIndex < job.candidateCount; candidateIndex++) {
          const score = values[candidateIndex * 4 + 2];
          if (!(score > 0)) continue;
          const atomIndex = candidateIndex % job.atomCount;
          const atom = job.levels[0][atomIndex];
          const x0 = atom.x * scale;
          const y0 = atom.y * scale;
          const x1 = coordinates[candidateIndex * 4] * scale;
          const y1 = coordinates[candidateIndex * 4 + 1] * scale;
          const sourceKey = Math.floor(y0 / (4 * scale)) + ":" +
            Math.floor(x0 / (4 * scale));
          const targetKey = Math.floor(y1 / (4 * scale)) + ":" +
            Math.floor(x1 / (4 * scale));
          const sourcePrevious = sourceBestGlobal.get(sourceKey);
          const targetPrevious = targetBestGlobal.get(targetKey);
          if ((!sourcePrevious || score > sourcePrevious.score)
              || (!targetPrevious || score > targetPrevious.score)) {
            const candidate = { x0, y0, x1, y1, score, rootX, rootY };
            if (!sourcePrevious || score > sourcePrevious.score) sourceBestGlobal.set(sourceKey, candidate);
            if (!targetPrevious || score > targetPrevious.score) targetBestGlobal.set(targetKey, candidate);
          }
        }
        job.candidateReadback.unmap();
        completedRootJobs++;
      }
      for (const resource of [...resources]) {
        if (!batchResources.has(resource)) releaseResource(resources, resource);
      }
      onProgress("DeepMatching INRIA · racine dense " + completedRootJobs + "/" + rootJobCount + ".");
    }

    for (let seedIndex = 0; seedIndex < seedStates.length; seedIndex++) {
      const state = seedStates[seedIndex];
      const seed = state.seed;
      const leaf = state.atomLeaf;
      const sourceX = leaf.x * scale;
      const sourceY = leaf.y * scale;
      const sourceKey = Math.floor(sourceY / (4 * scale)) + ":" +
        Math.floor(sourceX / (4 * scale));
      const candidate = sourceBestGlobal.get(sourceKey);
      const targetKey = candidate
        ? Math.floor(candidate.y1 / (4 * scale)) + ":" + Math.floor(candidate.x1 / (4 * scale))
        : null;
      const reciprocal = candidate && targetKey && targetBestGlobal.get(targetKey);
      const isReciprocal = !!reciprocal && reciprocal.x0 === candidate.x0
        && reciprocal.y0 === candidate.y0 && reciprocal.x1 === candidate.x1
        && reciprocal.y1 === candidate.y1 && reciprocal.score === candidate.score;
      matches.push({
        x: sourceX,
        y: sourceY,
        dx: isReciprocal ? candidate.x1 - candidate.x0 : 0,
        dy: isReciprocal ? candidate.y1 - candidate.y0 : 0,
        score: isReciprocal ? candidate.score : (candidate ? candidate.score : 0),
        secondScore: null,
        margin: null,
        usable: isReciprocal,
        reciprocal: isReciprocal,
        targetX: isReciprocal ? candidate.x1 : null,
        targetY: isReciprocal ? candidate.y1 : null,
        seedX: seed.x,
        seedY: seed.y,
        rootX: candidate ? candidate.rootX : (Math.round(leaf.x / (rootPatchSize / scale)) * rootPatchSize),
        rootY: candidate ? candidate.rootY : (Math.round(leaf.y / (rootPatchSize / scale)) * rootPatchSize),
      });
    }
    const hierarchyMs = performance.now() - hierarchyStart;
    const gridWidth = seedMode === "manual"
      ? seeds.length : 1 + Math.floor((width - settings.patchSize) / settings.stride);
    const gridHeight = seedMode === "manual"
      ? 1 : 1 + Math.floor((height - settings.patchSize) / settings.stride);
    return {
      width,
      height,
      patchSize: settings.patchSize,
      stride: settings.stride,
      scale,
      matcherMode: "inria-dense-root-grid-hierarchical-reciprocal",
      seedMode,
      searchRadiusPx: settings.searchRadiusPx || 0,
      gridWidth,
      gridHeight,
      count: matches.length,
      rootPatchSize,
      rootCenterConvention: "complete INRIA non-overlapping root grid at root patch stride",
      evaluatedRootCount: rootJobCount,
      atomicPatchSize: 4 * scale,
      levels: Math.log2(cellsSide) + 1,
      estimatedDescriptorComparisons: totalDescriptorComparisons,
      descriptorMs,
      hierarchyMs,
      matches,
    };
  } finally {
    try { await device.queue.onSubmittedWorkDone(); } catch {}
    for (const resource of resources) resource.destroy();
  }
}
