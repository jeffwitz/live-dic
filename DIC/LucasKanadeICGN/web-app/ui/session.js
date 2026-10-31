import { DICEngine, probeCapabilities } from '../web/core/engine.js';
import { LATENCY_OPTIONS } from '../web/core/device.js';
import { scorePlate } from './plate-truth.js';
import { FieldRenderer } from '../render/field.js';

function encodeMask(mask) {
  const chunks = [];
  for (let i=0;i<mask.length;i+=32768) chunks.push(String.fromCharCode(...mask.subarray(i,i+32768)));
  return btoa(chunks.join(''));
}

/** The same session runs in a worker, or on the main thread without OffscreenCanvas. */
export function createSession(send) {
  let engine, renderer, canvas, reference, referenceTexture, mask = null, capabilities, last = null;
  let showingReference = false;
  let display = { mode: 2, scale: null }, received = 0, completed = 0, exportSerial = 0;
  const ages = [], sampleLog = [];
  let lastComparison = null, referenceAdmissible = 0;
  let recording = false;
  let analysisEpoch = null;
  let activeOptions = null;
  let bitmapPresentation = false, profile = null;
  const completionTimes = [];
  const outstanding = new Set();
  function present(message = {}, transfer = []) {
    const bitmap = bitmapPresentation && canvas ? canvas.transferToImageBitmap() : null;
    send({ ...message, ...(bitmap ? { bitmap, width: canvas.width, height: canvas.height } : {}) }, bitmap ? [bitmap, ...transfer] : transfer);
  }
  async function referenceFrom(frame, options, roiMask = null) {
    const preparationStarted = performance.now();
    await Promise.all([...outstanding]);
    renderer?.destroy(); referenceTexture?.destroy(); engine?.destroy();
    const width = frame.displayWidth ?? frame.width, height = frame.displayHeight ?? frame.height;
    const { profile: profileName, ...engineOptions } = options;
    if (!['portable', 'qualified-latency'].includes(profileName)) throw new Error('Unsupported application profile');
    engine = await DICEngine.create({ ...LATENCY_OPTIONS, width, height, ...engineOptions, readback: 'none', pipelineDepth: 1 });
    engine.device.lost.then((info) => { if (info.reason !== 'destroyed') send({ type: 'error', message: `GPU device lost: ${info.message}` }); });
    engine.device.addEventListener('uncapturederror', (event) => send({ type: 'error', message: `WebGPU: ${event.error.message}` }));
    mask = roiMask; profile = options.profile ?? 'portable';
    activeOptions = structuredClone(options);
    analysisEpoch = crypto.randomUUID();
    const info = await engine.setReference(frame, { roiMask: mask });
    referenceAdmissible = [...info.referenceValid].reduce((n,v,i)=>n+ +(v && (!info.roiSelected || info.roiSelected[i])),0);
    lastComparison = null;
    renderer = canvas ? new FieldRenderer(engine.device, canvas) : null;
    referenceTexture = engine.device.createTexture({ size: [width, height], format: 'rgba8unorm',
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT });
    engine.device.queue.copyExternalImageToTexture({ source: frame }, { texture: referenceTexture }, [width, height]);
    showingReference = false;
    last = await engine.register(frame); renderer?.draw(engine, last, display);
    present({ type: 'preview' }); engine.resetTemporalState();
    ages.length = 0; sampleLog.length = 0; received = completed = 0; completionTimes.length = 0;
    return { ...info, controllerPreparationWallMs: performance.now() - preparationStarted };
  }
  async function handle(message) {
    const { type, id } = message;
    try {
      if (type === 'boot') {
        if (message.captureErrors && typeof document === 'undefined') {
          const original = console.error.bind(console);
          console.error = (...args) => { send({ type: 'error', message: args.map(String).join(' ') }); original(...args); };
        }
        bitmapPresentation = message.presentation === 'bitmap';
        canvas = message.canvas ?? (bitmapPresentation ? new OffscreenCanvas(1, 1) : null);
        capabilities = await probeCapabilities({ benchmark: message.measure !== false });
        send({ type: 'capabilities', id, capabilities, worker: typeof document === 'undefined' });
      } else if (type === 'reference') {
        reference?.close(); reference = await createImageBitmap(message.frame);
        try { const info = await referenceFrom(reference, message.options); send({ type: 'reference', id, info }); }
        finally { message.frame.close(); }
      } else if (type === 'roi') {
        if (!reference) throw new Error('Capture a reference before drawing the ROI');
        const info = await referenceFrom(reference, message.options, message.mask);
        send({ type: 'reference', id, info });
      } else if (type === 'comparison') {
        await Promise.all([...outstanding]);
        if (message.mode === 'fixed' && reference) await referenceFrom(reference,activeOptions,mask);
        else engine?.resetTemporalState();
        send({type:'comparison',id});
      } else if (type === 'recording') {
        recording = message.enabled === true;
        send({ type: 'recording', id, enabled: recording });
      } else if (type === 'frame') {
        received++;
        if (!engine || !engine.ring.idle || outstanding.size) { message.frame.close(); message.referenceFrame?.close(); send({ type: 'dropped', id }); return; }
        const available = performance.now();
        const operation = (async () => {
          try {
            let preparationMs = 0;
            if (message.referenceFrame) {
              const started = performance.now();
              const info = await engine.setReference(message.referenceFrame,{roiMask:mask});
              preparationMs = performance.now()-started;
              referenceAdmissible = [...info.referenceValid].reduce((n,v,i)=>n+ +(v && (!info.roiSelected || info.roiSelected[i])),0);
            }
            if (message.reset) engine.resetTemporalState();
            engine.setAnalyticTruth(message.truth ?? null);
            last = await engine.register(message.frame);
            last.timing.pairReferencePreparationWallMs = preparationMs;
            lastComparison = {...message.comparison, referenceAdmissible, requested:last.selected, fixedImageCoordinateRoi:true};
            const age = performance.now() - available;
            const plateScore = message.plate ? scorePlate(await engine.exportField(), message.plate, engine.reference.roiSelected ?? new Uint8Array(last.grid[0] * last.grid[1]).fill(1)) : null;
            if (plateScore) last.truthRmse = plateScore.displacementRmsePx;
            showingReference = false;
            const scale = renderer?.draw(engine, last, display);
            let record = null;
            if (recording) {
              const started = performance.now(), field = await engine.exportField();
              record = { field, metadata: { analysisEpoch, comparison: lastComparison, configuration: field.config,
                createdAt: new Date().toISOString(), frameIndex: message.frameIndex,
                recordingReadbackWallMs: performance.now()-started,
                workerArrivalToRecordedFieldMs: performance.now()-available,
                timingBoundary: 'Recorded fields include an explicit full readback; live age remains worker arrival to observable GPU field' } };
            }
            completed++; completionTimes.push(performance.now());
            while (completionTimes.length > 1 && completionTimes[0] < performance.now() - 3000) completionTimes.shift();
            ages.push(age); if (ages.length > 120) ages.shift();
            sampleLog.push({ frame: message.frameIndex, workerAvailableMs: available, captureTimestampUs: message.frame.timestamp ?? null, completedMs: performance.now(),
              ageMs: age, comparison:lastComparison, referencePreparationMs:preparationMs, gpuFieldObservableMs: last.timing.wallMs, truthRmse: last.truthRmse,
              acceptance: last.acceptance, accepted: last.accepted, meanIterations: last.meanIterations });
            if (sampleLog.length > 18000) sampleLog.shift();
            const sorted = [...ages].sort((a, b) => a - b);
            present({ type: 'result', id, completed, received, width: last.width, height: last.height,
              fps: completionTimes.length > 1 ? 1000 * (completionTimes.length - 1) / (completionTimes.at(-1) - completionTimes[0]) : 0, ageMedian: sorted[Math.floor(sorted.length * .5)],
              ageP95: sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * .95))],
              accepted: last.accepted, selected: last.selected, meanIterations: last.meanIterations,
              truthRmse: last.truthRmse, scale, counters: last.counters, timing: last.timing,
              frameIndex: message.frameIndex, plateScore, comparison:lastComparison, record },
              record ? [record.field.parameters.buffer, record.field.status.buffer, record.field.iterations.buffer] : []);
          } finally { message.frame.close(); message.referenceFrame?.close(); }
        })();
        outstanding.add(operation); try { await operation; } finally { outstanding.delete(operation); }
      } else if (type === 'show-reference') {
        await Promise.all([...outstanding]); showingReference = message.show;
        if (last) renderer?.draw(engine, last, showingReference ? { ...display, opacity: 0, imageTexture: referenceTexture } : display);
        present({ type: 'preview', id });
      } else if (type === 'display') {
        display = { ...display, ...message.display };
        await Promise.all([...outstanding]);
        if (last) { const scale = renderer?.draw(engine, last, showingReference ? { ...display, opacity: 0, imageTexture: referenceTexture } : display); present({ type: 'scale', id, scale }); }
        else send({ type: 'scale', id, scale: null });
      } else if (type === 'reset') {
        await Promise.all([...outstanding]); engine?.resetTemporalState(); send({ type: 'reset', id });
      } else if (type === 'export') {
        await Promise.all([...outstanding]);
        if (!engine) throw new Error('No field is available');
        const start = performance.now(), field = await engine.exportField();
        const metadata = { configuration: engine.reference.config, capabilities, profile, analysisEpoch, comparison:lastComparison, createdAt: new Date().toISOString(),
          exportSerial: ++exportSerial, readbackWallMs: performance.now() - start,
          width: field.width, height: field.height, grid: field.grid, subsetSize: field.subsetSize,
          subsetStride: field.subsetStride, samples: sampleLog,
          timingBoundary: 'Worker frame available to observable GPU field plus small HUD summary; full field is read only for this export',
          roi: mask ? { width: field.width, height: field.height, encoding: 'base64-uint8-boolean',
            data: encodeMask(mask) } : null };
        send({ type: 'export', id, field, metadata });
      } else if (type === 'report') send({ type: 'report', id, samples: sampleLog, capabilities });
      else if (type === 'destroy') {
        await Promise.all([...outstanding]); renderer?.destroy(); referenceTexture?.destroy(); engine?.destroy(); reference?.close();
        engine = renderer = reference = null; send({ type: 'destroy', id });
      }
    } catch (error) {
      message.frame?.close(); message.referenceFrame?.close();
      send({ type: 'error', id, message: error instanceof Error ? error.message : String(error) });
    }
  }
  return { handle };
}
