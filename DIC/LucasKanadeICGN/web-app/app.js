import { RecordingQueue } from './ui/recording-queue.js';
import { supervisionOptions } from './ui/supervision.js';
import { vectorGradient } from './render/vector-colours.js';
import { parameterOrder, parameterUnits } from './ui/postprocess.js';
import { decodeMaskImage, maskPng, maskThreshold } from './ui/mask-image.js';
import { installWorkbench } from './ui/workbench.js';
import { installWorkspace, solverOptions, solverPreparationKey } from './ui/workspace.js';
import { installRoi } from './ui/roi.js';
import { bounds, platePreset } from './ui/roi-geometry.js';
import { imageFiles, fileInfo, decodeImage, waitForMedia, seekVideo } from './ui/local-media.js';
import { createImageSequenceController } from './ui/image-sequence.js';
import { createDeviceReport } from './ui/device-report.js';
import { installViewNavigation } from './ui/view-navigation.js';
import { createCalculationProgress, imageSequenceTargetCount } from './ui/calculation-progress.js';
const $ = (id) => document.getElementById(id);
const query = new URLSearchParams(location.search), smoke = query.has('smoke');
let transport, stream, reader, manifest, source = null, paused = true, inFlight = 0, dropped = 0;
const sourceButtons = ['demo','plate','camera','open-video','open-folder','open-images'];
let loadingSource = false, sourceAbort = new AbortController(), localVideoUrl = null;
let previousCapture = null, captureBusy = false, receivedOrdinal = 0;
let roiEditing = false, appliedShapes = [], plateManifest = null;
let draftMaskSource = null, appliedMaskSource = null, appliedEditorState = null, editorWasActive = false;
let generation = 0, previousFrame = -1, referenceInfo, capabilities, sourceSettings = {}, requests = 0;
let pendingReferenceFrame = null, referencePrepared = false;
let buildInfo = null, appliedSolverKey = null;
let pipelineTimingUpdatedAt = -Infinity;
const synchronousAdmissions = [];
let recordingQueue = null, supervision = supervisionOptions(), lastPresentation = null, lastHudAt = -Infinity, pipelineStateUpdatedAt = -Infinity;
let lastVectorLegend = null, viewingInput = false, returnToInputAfterMask = false, startingCalculation = false, activeCalculationMode = null, finishingCalculation = null, calculationStartIndex = 0;
const pending = new Map(), video = $('video');
const imageSequence = createImageSequenceController({
  $,
  context: () => ({
    source,
    generation,
    signal: sourceAbort.signal,
    referenceInfo,
    sourceSettings,
    paused,
    video,
  }),
  setPaused: value => {
    paused = value;
    if (value) activeCalculationMode = null;
    else activeCalculationMode ??= workspace.mode;
    updateSessionIndicator();
  },
  pauseSource: () => pauseSource(),
  drain: () => drain(),
  decodeImage,
  fileInfo,
  processFrame: (...args) => processFrame(...args),
  captureReference: (...args) => captureReference(...args),
  showInputPreview: (...args) => showInputPreview(...args),
  frameFromVideo: (...args) => frameFromVideo(...args),
  seekVideo,
  details: () => details(),
  message: (...args) => message(...args),
});
const runImage = (...args) => imageSequence.runImage(...args);
const setImageReference = (...args) => imageSequence.setImageReference(...args);
const playImages = (...args) => imageSequence.playImages(...args);
const localSeek = (...args) => imageSequence.localSeek(...args);
const updateLocalPosition = (...args) => imageSequence.updateLocalPosition(...args);
sourceButtons.forEach(id => $(id).disabled = true);
const status = globalThis.liveDicStatus = { ready: false, completed: 0, errors: [], results: [] };
function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); }
function updateSessionIndicator() {
  const indicator = $('active-session');
  const cameraConnected = source === 'camera'
    && stream?.getVideoTracks().some(track => track.readyState === 'live');
  const calculationActive = !!source && (!paused || startingCalculation);
  indicator.hidden = !cameraConnected && !calculationActive;
  if (indicator.hidden) return;
  const runMode = activeCalculationMode ?? workspace.mode;
  const activity = startingCalculation
    ? 'Preparing'
    : `${runMode === 'live' ? 'Live' : 'Analysis'} calculation running`;
  $('active-session-label').textContent = calculationActive ? `${activity} · ${source}` : 'Camera connected · paused';
}
function send(type, data = {}, transfer = []) {
  const id = ++requests;
  const result = new Promise((resolve, reject) => { pending.set(id, { resolve, reject }); });
  transport.postMessage({ type, id, ...data }, transfer);
  return result;
}
if (smoke) {
  const original = console.error.bind(console);
  console.error = (...args) => { status.errors.push(args.map(String).join(' ')); original(...args); };
}
const options = solverOptions;
const calculationProgress = createCalculationProgress({ getElement: $ });
let workbench;
const workspace = installWorkspace({ onModeChange: mode => {
  if (mode === 'live' && workbench?.reviewing) workbench.returnLive(true);
}, onPane: name => {
  $('result-transport').hidden=name!=='view';
  if(name==='view'&&!workbench.reviewing)finishCalculation().catch(e=>message(e.message,true));
  else if(name!=='view'&&workbench.reviewing)workbench.returnLive(true);
}, shortcuts: {
  play: () => { if (workspace.pane === 'view' && !$('replay-results').disabled) $('replay-results').click(); },
  reference: () => { if (workspace.pane !== 'view' && !$('reference').disabled && !loadingSource) $('reference').click(); },
  clear: () => workbench.clearInspection(),
} });
function handleReviewMode(reviewing, restore) {
  $('reference').hidden = reviewing;
  $('input-field').hidden = reviewing || !viewingInput;
  if (!reviewing) $('field').hidden = viewingInput;
  if (restore && transport && referencePrepared) {
    send('display', { display: displayOptions() }).catch(error => message(error.message, true));
  }
  $('run-state').textContent = reviewing ? 'Saved field · reference coordinates'
    : source ? `${source} · ${paused ? 'paused' : 'running'}` : 'No source selected';
  updateSessionIndicator();
  queueMicrotask(updateVectorLengthLegend);
}
workbench = installWorkbench({
  captureCurrent: captureCurrentField,
  beforeProjectExport: async () => { pauseSource(); await drain(); $('pause').textContent='Start calculation'; },
  requestMemory: async () => ({...await send('memory'),pendingRecordedFields:recordingQueue.report()}),
  hasPendingResults: () => recordingQueue?.pending > 0,
  geometryDefaults: () => referenceInfo ? { width: referenceInfo.width, height: referenceInfo.height, subsetSize: referenceInfo.subsetSize, subsetStride: referenceInfo.subsetStride } : {},
  pause: () => { pauseSource(); $('pause').textContent = 'Start calculation'; },
  error: text => message(text, true),
  recordChange: async enabled => {
    if (enabled && recordingQueue?.failure && !workbench.storageBlocked) {
      await recordingQueue.drain(); recordingQueue.resume();
    }
    return send('recording', { enabled }).catch(e => message(e.message, true));
  },
  liveChanged: handleReviewMode,
  openResults: () => { workspace.showMode('analysis'); workspace.showPane('view'); },
});
recordingQueue = new RecordingQueue({
  sampleLimit:smoke?18000:256,
  write: async (record, queueTiming) => {
    record.metadata = {...record.metadata,recordingQueue:{...queueTiming,capacity:2}};
    await workbench.add(record.field,record.metadata,{mask:record.mask});
    updatePipelineState();
    return {failure:workbench.storageBlocked ? workbench.storage.report().warning ?? 'Disk/RAM admission failed; the field is retained in the results library' : null};
  },
  retain: async record => {
    await workbench.add(record.field,record.metadata,{mask:record.mask,ramOnly:true});
    updatePipelineState();
  },
  onFailure: error => {pauseSource();workbench.stopRecording();$('pause').textContent='Start calculation';message(`Recording paused: ${error.message}. Completed fields remain in RAM.`,true);},
});
function updatePipelineState(force = false) {
  if (!force && performance.now()-pipelineStateUpdatedAt < 1000) return;
  pipelineStateUpdatedAt=performance.now();
  if (!supervision.overlapRecording && !supervision.displayRate && !supervision.previewMaxWidth && supervision.previewEnabled) {
    $('pipeline-state').textContent='Synchronous recording and native-size presentation.'; return;
  }
  const q=recordingQueue.report(),sample=q.samples.at(-1);
  const shown=lastPresentation?.comparison?.target;
  const identity=lastPresentation?.referencePreview?'reference':shown?.ordinal??shown?.index??shown?.timeSeconds??'reference';
  $('pipeline-state').textContent=`Applied: display ${supervision.previewEnabled?`${supervision.displayRate||'every computed field'}${supervision.displayRate?' Hz':''}`:'suspended (calculation/recording continue)'}, preview ≤ ${supervision.previewMaxWidth||'native'} px. Shown input ${identity}. Recording queue ${q.pending}/${q.capacity} (calculation included), ${q.admitted} library admissions, ${q.retainedAfterFailure} retained after failure.${sample?` Latest receive → admission ${sample.receiveToAdmissionMs.toFixed(1)} ms.`:''}${q.blocked?' Admission blocked; export or retry RAM fields.':''}`;
}
function details() {
  const settings=sourceSettings.actual; $('camera-summary').hidden=source!=='camera';
  if(source==='camera'&&settings) $('camera-summary').textContent=`Actual ${settings.width} × ${settings.height} · ${settings.frameRate ?? '?'} Hz negotiated (requested ${sourceSettings.requested?.width} × ${sourceSettings.requested?.height} · ${sourceSettings.requested?.frameRate} Hz). ${sourceSettings.frameDelivery ?? 'Preparing capture'} · ${sourceSettings.deliveryState ?? 'preparing'}. Delivered/processed rates may be lower.`;
  $('details').textContent = JSON.stringify({ capabilities, acquisition: { ...sourceSettings, files: sourceSettings.files ? `${sourceSettings.files.length} file records retained in export` : undefined },
  build: buildInfo, execution: transport.isMain ? 'Main-thread fallback: OffscreenCanvas unavailable' : 'Dedicated worker',
  reference: referenceInfo ? { ...referenceInfo, prepared:referencePrepared, roiSelected: undefined, referenceValid: undefined } : null, profile: options().profile,
  requestedResolution: $('resolution').value }, null, 2); }
function updateScale(scale) {
  if (!Number.isFinite(scale)) return;
  const legend=document.querySelector('.legend');legend.hidden=lastVectorLegend?.imageOnly??Number($('component').value)===8;
  legend.classList.remove('status');legend.classList.add('magnitude');
  legend.querySelector('.legend-bar').style.background=vectorGradient(lastVectorLegend?.vectorColourMap ?? 'original');
  $('low').textContent='0 px';$('high').textContent=`${scale.toPrecision(3)} px · |u|`;
  document.querySelector('.gray-key').textContent='Grey crosses: requested but unavailable. ROI holes have no vectors.';
  updateVectorLengthLegend();
}
function updateVectorLengthLegend(){
  const key=$('live-vector-key'),legend=lastVectorLegend;
  key.hidden=workbench.reviewing||viewingInput||!legend||legend.imageOnly;
  $('live-colour-control').hidden=key.hidden;
  if(key.hidden)return;
  const factor=legend.gain*$('field').getBoundingClientRect().width/legend.imageWidth;
  if(!(factor>0))return;
  const value=72/factor,power=10**Math.floor(Math.log10(value)),fraction=value/power;
  const reference=(fraction<2?1:fraction<5?2:5)*power;
  $('vector-key-arrow').style.width=`${reference*factor}px`;
  $('vector-key-label').textContent=`${reference.toPrecision(3)} px displacement · lengths ×${legend.gain.toPrecision(3)}. Tips mark current subset centres.`;
}
async function receive({ data }) {
  if(data.type==='preview'&&data.presentation?.referencePreview){viewingInput=false;$('input-field').hidden=true;if(!workbench.reviewing)$('field').hidden=false;}
  if (data.bitmap) {
    // Assigning either size resets the canvas, even if the value is unchanged.
    // Keep the presentation surface between frames of the same sequence.
    const field = $('field');
    const width = data.presentation?.width ?? data.width, height = data.presentation?.height ?? data.height;
    if (field.width !== width) field.width = width;
    if (field.height !== height) field.height = height;
    $('field').getContext('bitmaprenderer').transferFromImageBitmap(data.bitmap);
    delete data.bitmap;
  }
  if (data.presentation) { lastPresentation = data.presentation; updatePipelineState(); }
  if (data.vectorLegend) lastVectorLegend=data.vectorLegend;
  if (data.supervision) {supervision=supervisionOptions(data.supervision);updatePipelineState(true);}
  if (data.type === 'error') {
    status.errors.push(data.message); message(data.message, true);
    if (pending.has(data.id)) { pending.get(data.id).reject(new Error(data.message)); pending.delete(data.id); }
    paused = true; $('export').disabled=true; return;
  }
  if (data.type === 'result') {
    viewingInput=false;$('input-field').hidden=true;if(!workbench.reviewing)$('field').hidden=false;
    if (data.plateScore) { sourceSettings.analyticScore = data.plateScore; details(); }
    $('export').disabled = false; workbench.setReady(true);
    const recordedReadbackMs=data.record?.metadata.recordingReadbackWallMs??0;let storageAdmissionMs=0;
    if (data.record) {
      const started=performance.now();
      const record={...data.record,mask:workbench.mask,
        workerArrivalEpochMs:data.record.metadata.workerArrivalEpochMs,
        metadata:{...data.record.metadata,acquisition:{source:sourceSettings.source},build:buildInfo,roiGeometry:roiMetadata()}};
      if (data.recordTicket !== undefined) {
        // submit() takes ownership synchronously; no next reservation exists
        // until capacity is available. Disk work continues beside the next solve.
        recordingQueue.submit(data.recordTicket,record).catch(error=>{pauseSource();message(`RAM retention failed: ${error.message}. Keep this page open.`,true);});
      } else {
        try { await workbench.add(record.field,record.metadata,{mask:record.mask}); }
        catch(error) {pauseSource();message(error.message,true);}
        storageAdmissionMs=performance.now()-started;
        synchronousAdmissions.push({admittedEpochMs:performance.timeOrigin+performance.now(),
          admissionMs:storageAdmissionMs,workerToAdmissionMs:Number.isFinite(record.workerArrivalEpochMs)?performance.timeOrigin+performance.now()-record.workerArrivalEpochMs:null});
        if(synchronousAdmissions.length>(smoke?18000:256))synchronousAdmissions.shift();
      }
      delete data.record;
    }

    if(performance.now()-pipelineTimingUpdatedAt>=1000){
      updatePipelineState();
      pipelineTimingUpdatedAt=performance.now();
      const preparationMs=data.timing?.pairReferencePreparationWallMs??0,solveMs=data.timing?.totalWallMs??data.timing?.wallMs;
      const ms=value=>Number.isFinite(value)?value.toFixed(1)+' ms':'unavailable';
      const gpuPair=data.timing?.gpuReferenceIncluded;
      $('pipeline-timing').textContent=`Latest sampled field (1 Hz): ${gpuPair?`GPU reference + observable solver call ${ms(solveMs)}`:`reference preparation ${ms(preparationMs)} · observable solver call ${ms(solveMs)}`}${recordedReadbackMs||storageAdmissionMs?` · recorded-field readback ${ms(recordedReadbackMs)} · main-thread storage/admission ${ms(storageAdmissionMs)}`:''}. ${data.comparison?.mode==='consecutive'?(gpuPair?'Consecutive mode recomputes the reference on GPU for every pair.':'Consecutive mode rebuilds the reference on CPU for every pair; fixed-reference FPS does not apply.'):'The fixed reference is reused.'}`;
    }
    const hudRate = supervision.previewEnabled ? supervision.displayRate : 1;
    const updateHud = !hudRate || performance.now()-lastHudAt>=1000/hudRate;
    if (updateHud) lastHudAt=performance.now();
    if (updateHud && !workbench.reviewing) $('run-state').textContent = source==='camera'&&sourceSettings.deliveryState==='stopped'?'Camera stopped · last completed field':`${source} · ${data.width} × ${data.height} · ${data.comparison?.mode ?? 'fixed'} · ${paused ? 'paused' : 'running'}`;
    if (data.comparison) { sourceSettings.lastComparison = data.comparison; updateLocalPosition(data.comparison.target);$('last-reference-summary').textContent=`Last pair: ${data.comparison.referenceAdmissible} numerically admissible references / ${data.selected} requested ROI windows. Every selected window remains requested.`; }
    status.completed++; status.results.push(data); if (status.results.length > (smoke ? 18000 : 120)) status.results.shift();
    if (updateHud) {
    $('fps').textContent = source === 'plate' ? 'Static pair' : data.fps.toFixed(1);
    $('age').textContent = `${data.ageMedian.toFixed(1)} / ${data.ageP95.toFixed(1)} ms`;
    if(data.globalTranslation)$('last-reference-summary').textContent+=` Fourier seed (${data.globalTranslation.dx}, ${data.globalTranslation.dy}) px · correlation ${data.globalTranslation.correlation.toFixed(3)}.`;
    if(data.globalAffineEstimate){const fit=data.globalAffineEstimate;$('last-reference-summary').textContent+=` Global affine seed: centre (${fit.dx.toFixed(2)}, ${fit.dy.toFixed(2)}) px · correlation ${fit.score.toFixed(3)} · ${fit.levels} scales · ${data.timing.globalAffineWallMs?.toFixed(1)??'—'} ms CPU.`;}
    $('acceptance').textContent = `${data.accepted} / ${data.selected}`;
    $('iterations').textContent = data.meanIterations?.toFixed(2) ?? '—';
    $('error').textContent = data.truthRmse == null ? '—' : `${data.truthRmse.toFixed(4)} px`;
    $('error').classList.toggle('metric-warning', data.truthRmse != null && data.truthRmse > .1);
    $('error').title = data.plateScore?.population ?? 'Whole accepted field, including possible wrong-basin estimates near the image boundary';
    if (!workbench.reviewing&&!viewingInput) updateScale(data.scale);
    }
  }
  if (data.type === 'scale' && !workbench.reviewing&&!viewingInput) updateScale(data.scale);
  if ((data.type==='preview'||data.type==='flushed')&&!workbench.reviewing&&!viewingInput&&data.vectorLegend)updateScale(data.vectorLegend.maximumPx);
  if (data.type === 'dropped') { dropped++; $('drops').textContent = dropped; }
  const request = pending.get(data.id);
  if (request) { pending.delete(data.id); request.resolve(data); }
}
async function init() {
  try { const response=await fetch(new URL('./build.json',import.meta.url)); if(response.ok)buildInfo=await response.json(); } catch { /* An unavailable build record is explicit in export metadata. */ }
  const canvas = $('field');
  if (typeof OffscreenCanvas !== 'undefined' && typeof Worker !== 'undefined') {
    transport = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    transport.onmessage = receive;
    transport.onerror = (error) => { status.errors.push(error.message); message(error.message, true); paused = true; for (const request of pending.values()) request.reject(new Error(error.message)); pending.clear(); };
    capabilities = (await send('boot', { presentation: 'bitmap', measure: !smoke, captureErrors: smoke })).capabilities;
  } else {
    const { createSession } = await import('./ui/session.js');
    const session = createSession((data) => queueMicrotask(() => receive({ data })));
    transport = { isMain: true, postMessage: (data) => { session.handle(data); } };
    capabilities = (await send('boot', { canvas, measure: !smoke, captureErrors: smoke })).capabilities;
  }
  details();
  $('download-device-report').disabled = false;
  if (!capabilities.supported) {
    message(`WebGPU is unavailable for correlation. ${capabilities.reasons.join('. ')} You can still open saved NPZ fields or results projects for CPU post-processing.`, true);
    workspace.showPane('view');$('placeholder').textContent='Open a saved NPZ field or results project. New correlations require WebGPU.';
    sourceButtons.forEach(id => $(id).disabled = true); return;
  }
  for (const option of $('resolution').options) {
    const [w, h] = option.value.split('x').map(Number);
    option.disabled = !capabilities.resolutions.some(([x, y]) => x === w && y === h);
  }
  sourceButtons.forEach(id => $(id).disabled = false);
  $('placeholder').textContent='Open images, a video or a camera. Try the plate demo to explore.';
  const recommended = capabilities.recommendedResolution ?? capabilities.resolutions[0];
  if (recommended) $('resolution').value = recommended.join('x');
  if (!workspace.hasSavedMode && navigator.mediaDevices?.enumerateDevices) {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      if (devices.some(device => device.kind === 'videoinput')) workspace.showMode('live', { persist: false });
    } catch { /* Camera discovery must not block startup. */ }
  }
  message(`Ready. ${capabilities.reasons.join('. ')}`); status.ready = true;
  if (smoke && query.get('case') !== 'manual') {
    // Canned diagnostic URLs explicitly request a preview calculation, without recording.
    $('record-fields').checked=false;
    await (query.get('case') === 'plate' ? startPlate() : startDemo());
    await startCalculation();
  }
}
async function drain() {
  while (inFlight) await new Promise((resolve) => { setTimeout(resolve, 10); });
  await recordingQueue.drain();
  await workbench.flush();
  if ((supervision.displayRate || !supervision.previewEnabled) && transport && referencePrepared) await send('flush');
  updatePipelineState(true);
  if (imageSequence.task) await imageSequence.task.catch(() => {});
  if (supervision.overlapRecording && imageSequence.loop) await imageSequence.loop.catch(() => {});
}
function clearPreviousCapture() { previousCapture?.frame.close(); previousCapture = null; }
function clearPendingReference() { pendingReferenceFrame?.close(); pendingReferenceFrame = null; }
async function duplicateFrame(frame) {
  return typeof frame.clone === 'function' ? frame.clone() : createImageBitmap(frame);
}
function pauseSource() { paused = true; activeCalculationMode = null; imageSequence.wakePlayback(); if (source === 'video-file') video.pause(); clearPreviousCapture(); updateSessionIndicator(); }
async function sourceChange(operation, {propagate = false} = {}) {
  if (loadingSource) return;
  loadingSource = true; document.querySelector('.workspace').inert=true; sourceButtons.forEach(id => $(id).disabled = true);
  const busyText = 'Loading input…'; message(busyText);
  try { imageSequence.invalidateSeek(); pauseSource(); await drain(); calculationProgress.finish(); await operation(); if($('message').textContent===busyText)message('Input loaded. Solver preparation waits for Start calculation.'); }
  catch (error) { if (error.name !== 'AbortError') message(error.message, true); if(propagate)throw error; }
  finally { loadingSource = false; document.querySelector('.workspace').inert=false; sourceButtons.forEach(id => $(id).disabled = !capabilities?.supported); updateSolverSettingsStatus(); }
}
async function stop() {
  calculationProgress.reset();
  workbench.returnLive(); workbench.stopRecording({preserveSelection:true});
  pauseSource(); generation++; imageSequence.invalidateSeek(); sourceAbort.abort();
  await drain(); await imageSequence.task?.catch(() => {}); await imageSequence.loop?.catch(() => {}); sourceAbort = new AbortController();
  workbench.setReady(false); workbench.setMask(null);appliedEditorState=null;roiEditing=false;roi.stop({discardDraft:true});
  await reader?.cancel(); reader = null;
  stream?.getTracks().forEach(track => track.stop()); stream = null;
  clearPendingReference(); referencePrepared = false;
  video.pause(); video.srcObject = null; video.removeAttribute('src'); video.load();
  if (localVideoUrl) URL.revokeObjectURL(localVideoUrl); localVideoUrl = null;
  imageSequence.reset(); receivedOrdinal = 0;sourceSettings={};
  dropped=0;$('drops').textContent='0';clearLiveMetrics();
  $('run-state').textContent='No source selected';$('applied-solver-summary').textContent='No reference prepared.';$('last-reference-summary').textContent='';$('placeholder').hidden=false;$('camera-summary').hidden=true;
  referenceInfo = null; appliedSolverKey = null; updateSolverSettingsStatus(); workbench.setSourceReady(false);viewingInput=false;$('input-field').hidden=true;$('input-field').width=$('input-field').height=1; ['reference','pause','reset','draw','clear-roi','export','import-mask','create-mask','save-mask'].forEach(id => $(id).disabled = true);
  $('comparison').disabled=false; source = null; roiEditing = false; $('roi-editor').hidden = true; $('plate-load-label').hidden = true; $('local-controls').hidden = true;$('solve-range-label').hidden=true; updateSessionIndicator();
}
async function frameFromVideo() {
  if (source === 'video-file' && referenceInfo && (video.videoWidth !== referenceInfo.width || video.videoHeight !== referenceInfo.height)) throw new Error('Video dimensions changed; capture a new reference. Frames are not resized.');
  const [width, height] = source === 'camera' ? [video.videoWidth, video.videoHeight] : sourceSettings.outputSize ?? [video.videoWidth, video.videoHeight];
  if (typeof VideoFrame === 'function' && width === video.videoWidth && height === video.videoHeight) {
    try { return new VideoFrame(video, { timestamp: Math.round(video.currentTime * 1e6) }); } catch { /* ImageBitmap fallback below. */ }
  }
  return createImageBitmap(video, { resizeWidth: width, resizeHeight: height });
}
function updateSolverSettingsStatus() {
  try {
    const requested=options(), pending=!!referenceInfo&&(!referencePrepared||solverPreparationKey(requested)!==appliedSolverKey);
    if(referenceInfo&&!referencePrepared)$('applied-solver-summary').textContent=`Reference staged · ${referenceInfo.width} × ${referenceInfo.height}. GPU reference and mask preparation waits for Start calculation.`;
    $('solver-summary').textContent=pending
      ? 'Reference, mask or solver settings are pending. Start calculation prepares the current setup once; editing inputs does not initialize the solver. Saved fields keep their original settings.'
      : referencePrepared?'Current setup is prepared. Saved fields retain the settings used to calculate them.':'Choose an input to stage a reference.';
    $('solver-summary').dataset.pending=String(pending);
  }catch(error){$('solver-summary').textContent=error.message;}
}
function invalidatePreparedReference(force=false) {
  let changed=force;
  try { changed ||= referencePrepared&&solverPreparationKey(options())!==appliedSolverKey; }
  catch { changed=true; }
  if(changed){
    referencePrepared=false;appliedSolverKey=null;workbench.setReady(false);
    $('export').disabled=true;clearLiveMetrics();pauseSource();$('pause').textContent='Start calculation';
  }
  updateSolverSettingsStatus();
  if(referenceInfo)details();
}
async function prepareCalculation(requested) {
  if(!referenceInfo||!pendingReferenceFrame)throw new Error('Choose a reference image first');
  if(roiEditing)throw new Error('Apply or cancel mask edits before changing solver settings');
  if(requested.initialization==='global-affine'&&!['images','video-file','demo','plate'].includes(source))
    throw new Error('Global affine initialization needs loaded image or video files; select another initialization for a live camera.');
  workbench.returnLive();workbench.setReady(false);$('export').disabled=true;
  clearLiveMetrics();$('solve-progress').textContent='Preparing the reference, mask and solver on the GPU…';calculationProgress.setPhase('Preparing reference and GPU solver…');
  const frame=await duplicateFrame(pendingReferenceFrame);
  const mask=appliedShapes.length||appliedMaskSource||appliedEditorState?.empty?roi.mask():null;
  const ownedMask=mask?.slice()??null;
  try{
    const reply=await send('reference',{frame,options:requested,sourceKind:source,mask},[frame,...(mask?[mask.buffer]:[])]);
    referenceInfo=reply.info;referencePrepared=true;appliedSolverKey=solverPreparationKey(requested);previousFrame=-1;
    workbench.setMask(ownedMask);
    await updateReferenceControls();workbench.setSourceReady(true);details();
    $('solve-progress').textContent='Preparation complete. Calculating fields…';
  }catch(error){referencePrepared=false;appliedSolverKey=null;workbench.setSourceReady(true);updateSolverSettingsStatus();throw error;}
}
async function updateReferenceControls() {
  supervision = supervisionOptions(referenceInfo.supervision); lastPresentation=null; lastHudAt=-Infinity; updatePipelineState(true);
  const cfg=referenceInfo.config,total=referenceInfo.grid[0]*referenceInfo.grid[1];
  let selected=0,valid=0;
  for(let i=0;i<total;i++)if(!referenceInfo.roiSelected||referenceInfo.roiSelected[i]){selected++;if(referenceInfo.referenceValid?.[i])valid++;}
  $('applied-solver-summary').textContent=`Applied ${cfg.motionModel} · ${cfg.interpolation} · ${cfg.pyramidLevels} level(s)\nWindow ${referenceInfo.subsetSize} px · step ${referenceInfo.subsetStride} px · ${Math.round(100*(1-referenceInfo.subsetStride/referenceInfo.subsetSize))}% overlap\nIteration limits (coarse → fine): ${cfg.iterationsPerLevel.join(' / ')}\nGrid ${referenceInfo.grid.join(' × ')} · ${selected} selected / ${total} positions\n${valid} admissible prepared references; target convergence remains to be measured.\nReference preparation: ${cfg.referencePreparation??'cpu'}. Initialization: ${cfg.initialization}.`;
  if(cfg.roi) $('applied-solver-summary').textContent += `\nROI ${cfg.roi.policy}: ${cfg.roi.selected} selected (${cfg.roi.full} complete windows, ${cfg.roi.partial} partial supports).`;
  if (cfg.spatialPropagationPasses) $('applied-solver-summary').textContent += `\nGPU seed propagation: ${cfg.spatialPropagationPasses} pass(es) maximum per level, rejected points only; stops after an unchanged pass.`;
  $('last-reference-summary').textContent='Awaiting a target field.';
  updateSolverSettingsStatus();
}
async function captureReference(frame = null) {
  workbench.returnLive();
  $('run-state').textContent='Reference selected · solver preparation waits for Start calculation';
  pauseSource(); await drain(); clearPendingReference(); workbench.setMask(null); workbench.setSourceReady(false); workbench.setReady(false); roiEditing = false; $('roi-editor').hidden = true;
  frame ??= await frameFromVideo();
  pendingReferenceFrame=frame;
  const width=frame.displayWidth??frame.width,height=frame.displayHeight??frame.height;
  referencePrepared=false;appliedSolverKey=null;clearLiveMetrics();
  if (source === 'camera') { sourceSettings.actual = stream?.getVideoTracks()[0]?.getSettings(); sourceSettings.outputSize = [video.videoWidth, video.videoHeight]; }
  if(source==='camera')sourceSettings.reference={kind:'camera',timestampUs:frame.timestamp??null,capturedAt: new Date().toISOString(),coordinateFrame:'captured reference image',timestampClock:'HTMLVideoElement media timeline'};
  referenceInfo={width,height};appliedShapes = []; draftMaskSource = appliedMaskSource = null; $('mask-import-summary').textContent='White pixels are kept; black pixels are excluded. Imported images must match the reference dimensions.';
  $('placeholder').hidden = true; $('image-viewport').style.aspectRatio = `${width}/${height}`;
  resetImageView();
  roi.reset(width,height); appliedEditorState=roi.state(); $('apply-roi').disabled = true;
  for (const id of ['reference', 'pause', 'reset', 'draw', 'clear-roi', 'export', 'import-mask', 'create-mask', 'save-mask']) $(id).disabled = false;
  $('export').disabled = true;workbench.setSourceReady(true);
  previousFrame = -1; details();updateSolverSettingsStatus();
  showInputPreview(await duplicateFrame(pendingReferenceFrame));
  message('Reference selected. Image, mask and solver setup will be prepared when Start calculation is pressed.');
}
function clearLiveMetrics() {for(const id of ['fps','age','acceptance','iterations','error'])$(id).textContent='—';pipelineTimingUpdatedAt=-Infinity;$('pipeline-timing').textContent='Awaiting a target field; stage times will appear here.';delete sourceSettings.lastComparison;$('last-reference-summary').textContent='Awaiting a target field.';}
function countDrop() { dropped++; $('drops').textContent = dropped; }
async function processFrame(frame, frameIndex = null, { force = false, reset = false, input = null, referenceFrame = null, referenceInput = null, playback = false } = {}) {
  const token = generation, mode = source === 'plate' ? 'fixed' : $('comparison').value;
  if (paused && (!force || playback)) { frame.close(); referenceFrame?.close(); return false; }
  input = { kind:source, ordinal:++receivedOrdinal, timestampUs:frame.timestamp ?? null,
    timestampClock:frame.timestamp==null?null:source==='camera'&&sourceSettings.frameDelivery==='MediaStreamTrackProcessor'?'Track-processor VideoFrame clock; origin is browser/device defined':'HTMLVideoElement media timeline', ...input };
  if (mode === 'consecutive' && !referenceFrame) {
    if (captureBusy) { frame.close(); countDrop(); return false; }
    captureBusy = true;
    let retained;
    try { retained = typeof frame.clone === 'function' ? frame.clone() : await createImageBitmap(frame); }
    catch (error) { frame.close(); throw error; }
    finally { captureBusy = false; }
    if (token !== generation || (paused && !force)) { retained.close(); frame.close(); return false; }
    const prior = previousCapture; previousCapture = { frame: retained, input };
    if (!prior) { frame.close(); return false; }
    referenceFrame = prior.frame; referenceInput = prior.input;
    if ((input.ordinal != null && referenceInput.ordinal != null && input.ordinal - referenceInput.ordinal !== 1) || (input.presentedFrames != null && referenceInput.presentedFrames != null && input.presentedFrames - referenceInput.presentedFrames !== 1)) {
      frame.close(); referenceFrame.close(); countDrop(); return false;
    }
  }
  if (inFlight >= 1) { frame.close(); referenceFrame?.close(); countDrop(); return false; }
  const wantsRecord = supervision.overlapRecording && workbench.recording;
  let recordTicket = null;
  if (wantsRecord) {
    recordTicket = source==='images' ? await recordingQueue.waitReserve() : recordingQueue.reserve();
    if (recordTicket===null || token!==generation || (paused&&(!force||playback))) {
      if(recordTicket!==null)recordingQueue.release(recordTicket);
      frame.close();referenceFrame?.close();countDrop();return false;
    }
  }
  inFlight++;
  const historyReset = reset || (frameIndex != null && frameIndex < previousFrame);
  if (frameIndex != null) previousFrame = frameIndex;
  const stretch = manifest?.stretch[frameIndex];
  const comparison = { mode, reference: mode === 'consecutive' ? referenceInput : sourceSettings.reference ?? { kind: source, role: 'captured reference' }, target: input };
  try {
    const reply = await send('frame', { frame, frameIndex, reset: historyReset, referenceFrame, comparison,
      ...(supervision.overlapRecording?{recordRequested:wantsRecord,...(recordTicket!==null?{recordTicket}:{})}:{}),
      truth: source === 'demo' && mode === 'fixed' && stretch ? [stretch[0] - 1, stretch[1] - 1] : null }, referenceFrame ? [referenceFrame, frame] : [frame]);
    const completed = reply.type === 'result';
    if (completed) calculationProgress.advance();
    return completed;
  } catch (error) {
    pauseSource(); message(error.message, true);
    if (calculationProgress.isActive()) setTimeout(() => finishCalculation().catch(e => message(e.message, true)), 0);
    return false;
  }
  finally { inFlight--; if(recordTicket!==null)recordingQueue.release(recordTicket); }
}
function scheduleVideo(token) {
  let lastPresented = 0, lastMediaTime = -1;
  const callback = async (_, metadata) => {
    if (!paused && lastPresented && metadata?.presentedFrames) { dropped += Math.max(0, metadata.presentedFrames - lastPresented - 1); $('drops').textContent = dropped; }
    if (metadata?.presentedFrames) lastPresented = metadata.presentedFrames;
    if (generation !== token) return;
    const mediaTime = metadata?.mediaTime ?? video.currentTime;
    if (!video.requestVideoFrameCallback && mediaTime === lastMediaTime) {
      requestAnimationFrame((time) => callback(time, null)); return;
    }
    lastMediaTime = mediaTime;
    if (!paused && (inFlight < 1 || $('comparison').value === 'consecutive')) {
      inFlight++; // Reserve admission before asynchronous ImageBitmap creation.
      try {
        const frame = await frameFromVideo(); inFlight--;
        if (generation !== token) { frame.close(); return; }
        const index = source === 'demo' ? Math.min(manifest.frames - 1, Math.round((metadata?.mediaTime ?? video.currentTime) * manifest.frame_rate)) : null;
        const input = source === 'video-file' ? {kind:source, file:sourceSettings.file, timeSeconds:metadata?.mediaTime ?? video.currentTime, presentedFrames:metadata?.presentedFrames ?? null} : null;
        processFrame(frame, index, {input}).catch(error => { pauseSource(); message(error.message, true); });
      } catch (error) {
        inFlight--; pauseSource(); message(error.message, true);
        if (calculationProgress.isActive()) setTimeout(() => finishCalculation().catch(e => message(e.message, true)), 0);
      }
    } else if (!paused) { dropped++; $('drops').textContent = dropped; }
    if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(callback);
    else requestAnimationFrame((time) => callback(time, null));
  };
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(callback);
  else requestAnimationFrame((time) => callback(time, null));
}
async function startImages(files) {
  const choice = imageFiles(files);
  await stop(); source = 'images'; manifest = null; imageSequence.setImages(choice.images);
  sourceSettings = { source:'local image files',ordering:'natural relative filename, lexical tie-break',files:imageSequence.images.map(fileInfo),ignored:choice.ignored };
  const select = $('image-reference'); select.replaceChildren();
  imageSequence.images.forEach((file,i) => { const option=document.createElement('option');option.value=i;option.textContent=fileInfo(file).name;select.append(option); });
  $('local-name').textContent = `${imageSequence.images.length} image(s) · ${choice.ignored.length} unsupported/non-image file(s) ignored`;
  $('local-seek').min=0; $('local-seek').max=imageSequence.images.length-1; $('local-seek').step=1;
  imageSequence.localControls(true); imageSequence.index = imageSequence.images.length>1 ? 1 : 0;
  await setImageReference(0);
  message(`Loaded ${imageSequence.images.length} local image(s). ${imageSequence.images.some(file=>/\.tiff?$/i.test(file.name))?'TIFF: unsigned 16-bit values use their high byte (fixed 0–65535 → 0–255); no automatic contrast adjustment. ':''}${choice.ignored.length ? 'Ignored files are listed in the details. ' : ''}Choose a reference and mask, then open Solve to calculate and save the sequence.`);
}
async function startLocalVideo(file) {
  await stop(); source='video-file'; manifest=null;
  localVideoUrl=URL.createObjectURL(file);
  const ready=waitForMedia(video,'loadeddata',sourceAbort.signal);video.playbackRate=Number($('video-rate').value);video.loop=false;video.preload='auto';video.src=localVideoUrl;video.load();await ready;
  sourceSettings={source:'local video',file:fileInfo(file),outputSize:[video.videoWidth,video.videoHeight],durationSeconds:Number.isFinite(video.duration)?video.duration:null,
    reference:{kind:'video-file',file:fileInfo(file),timeSeconds:video.currentTime}};
  imageSequence.localControls(false); $('local-name').textContent=file.name;
  $('local-seek').min=0;$('local-seek').max=Number.isFinite(video.duration)?video.duration:0;$('local-seek').step=.001;$('local-seek').disabled=!Number.isFinite(video.duration);
  await captureReference();
  showInputPreview(await frameFromVideo());
  $('pause').textContent='Start calculation';updateLocalPosition();details();scheduleVideo(generation);
  message('Video ready. Preview inputs and prepare the mask, then start calculation in Solve.');
}
function showInputPreview(frame) {
  workbench.returnLive();
  try {
    const canvas=$('input-field');canvas.width=frame.displayWidth??frame.width;canvas.height=frame.displayHeight??frame.height;
    canvas.getContext('2d').drawImage(frame,0,0,canvas.width,canvas.height);
  } finally {frame.close();}
  viewingInput=true;$('input-field').hidden=false;$('field').hidden=true;
  document.querySelector('.legend').hidden=true;$('live-vector-key').hidden=true;
  workbench.setReady(false);$('export').disabled=true;clearLiveMetrics();
  $('run-state').textContent='Input image preview · no target correlation';
}
async function showPendingReferencePreview() {
  if(!pendingReferenceFrame)return;
  showInputPreview(await duplicateFrame(pendingReferenceFrame));
}
async function previewPlateTarget() {
  pauseSource();await drain();
  const frame=Number($('plate-load').value),record=plateManifest.images.find(r=>r.frame===frame);
  const bitmap=await createImageBitmap(await fetch(new URL(`./demo/plate/${record.file}`,import.meta.url)).then(r=>r.blob()));
  showInputPreview(bitmap);$('pause').textContent='Start calculation';
}
async function startDemo() {
  await stop(); source = 'demo'; $('resolution').disabled = true;
  manifest = await fetch(new URL('./demo/manifest.json', import.meta.url)).then((r) => r.json());
  const reference = await createImageBitmap(await fetch(new URL('./demo/reference.png', import.meta.url)).then((r) => r.blob()));
  sourceSettings = { reference:{kind:'demo',frame:0}, source: 'derived analytic tensile demo', encoding: manifest.encoding, outputSize: [manifest.width, manifest.height] };
  $('resolution').value = `${manifest.width}x${manifest.height}`;
  if (!capabilities.resolutions.some(([w, h]) => w === manifest.width && h === manifest.height)) throw new Error('The shipped demo requires 1080p capability');
  await captureReference(reference);
  const ready=waitForMedia(video,'loadeddata',sourceAbort.signal);
  video.src = new URL('./demo/tensile.webm', import.meta.url); video.loop = true; video.playbackRate=1;
  video.load();await ready;
  // load() restores defaultPlaybackRate; apply the diagnostic rate afterwards.
  if (smoke && query.has('rate')) video.playbackRate = Number(query.get('rate'));
  video.pause();paused=true;scheduleVideo(generation);
  message('Analytic tensile demo · compressed 1080p replay. Live error includes image formation and video decoding.');
}
async function startPlate() {
  if (!capabilities.resolutions.some(([w,h]) => w === 3840 && h === 2160)) throw new Error('The original plate requires UHD device capacity');
  await stop(); manifest = null; source = 'plate'; $('comparison').disabled=true; $('comparison').value='fixed'; $('resolution').disabled = true; $('plate-load-label').hidden = false;
  plateManifest = await fetch(new URL('./demo/plate.json', import.meta.url)).then(r => r.json());
  const reference = await createImageBitmap(await fetch(new URL('./demo/plate/reference.png', import.meta.url)).then(r => r.blob()));
  sourceSettings = { reference:{kind:'plate',frame:0}, source: 'original native-UHD analytic plate', outputSize: [3840, 2160], mechanics: plateManifest.geometry };
  $('resolution').value = '3840x2160';
  await captureReference(reference);
  roi.reset(3840, 2160, platePreset()); appliedShapes = roi.shapes();
  workbench.setMask(roi.mask());workbench.setSourceReady(true);appliedEditorState=roi.state();updateSolverSettingsStatus();
  $('component').value='2';$('scale').value='auto';$('arrow-gain').value='auto';$('opacity').value='1';$('vector-colour-map').value='original';
  await send('display', { display: displayOptions() });
  await previewPlateTarget();
  // Show editable outer rectangle and circular hole on the fixed reference.
  message('Original UHD plate · live displacement vectors. Store a field and open Results for equivalent strain. Edit the mask to change the outlined domain or hole.');
}
async function runPlateTarget() {
  if (roiEditing) { message('Apply the edited mask before recalculating the plate.'); return; }
  paused = true; await drain();
  const frame = Number($('plate-load').value), record = plateManifest.images.find(r => r.frame === frame);
  const bitmap = await createImageBitmap(await fetch(new URL(`./demo/plate/${record.file}`, import.meta.url)).then(r => r.blob()));
  sourceSettings.load = { frame, primary: record.primary_MPa, secondary: record.secondary_MPa };
  const reply=await send('frame', { frame: bitmap, frameIndex: frame, reset: true, plate: sourceSettings.load, comparison:{mode:'fixed',reference:sourceSettings.reference,target:{kind:'plate',...sourceSettings.load}} }, [bitmap]);
  if(reply.type==='result')calculationProgress.advance();
  $('pause').textContent = 'Start calculation'; details();
}
async function lockCamera(track) {
  const caps = track.getCapabilities?.() ?? {}, settings = track.getSettings(), locks = {};
  for (const key of ['exposureMode', 'focusMode', 'whiteBalanceMode']) {
    if (Array.isArray(caps[key]) && caps[key].includes('manual')) {
      try { await track.applyConstraints({ advanced: [{ [key]: 'manual' }] }); locks[key] = 'manual requested'; }
      catch (error) { locks[key] = `not locked: ${error.message}`; }
    } else locks[key] = 'manual lock not exposed';
  }
  return { initialSettings: settings, actual: track.getSettings(), capabilities: caps, locks };
}
async function refreshCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) { message('Camera enumeration is not available in this browser.',true); return; }
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter(device => device.kind === 'videoinput');
  const selected = $('camera-device').value; $('camera-device').replaceChildren(new Option('Default / rear camera',''));
  devices.forEach((device,index) => $('camera-device').add(new Option(device.label || `Camera ${index+1} · permission needed for its name`, device.deviceId)));
  if ([...$('camera-device').options].some(option=>option.value===selected)) $('camera-device').value=selected;
}
$('refresh-cameras').onclick = () => refreshCameras().catch(e=>message(e.message,true));
$('camera-device').onchange = () => { if(source==='camera') sourceChange(startCamera); };
function cameraStopped(token, reason) {
  if(generation!==token||source!=='camera')return;
  pauseSource();sourceSettings.deliveryState='stopped';sourceSettings.deliveryError=reason;
  updateSessionIndicator();
  $('pause').textContent='Restart camera';if(!workbench.reviewing)$('run-state').textContent='Camera stopped · last completed field';
  details();message(`${reason} Restart camera to capture a new reference; reapply the mask afterwards.`,true);
}
async function startCamera() {
  await stop(); manifest = null; source = 'camera'; $('resolution').disabled = false;
  try {
  const [width, height] = $('resolution').value.split('x').map(Number);
  stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { ...($('camera-device').value ? {deviceId:{exact:$('camera-device').value}} : {facingMode:'environment'}), width, height, frameRate: { ideal: 60 } } });
  video.src = ''; video.srcObject = stream; await video.play();
  const track = stream.getVideoTracks()[0]; await refreshCameras();
  sourceSettings = { source: 'camera', deliveryState:'running', requested: { width, height, frameRate: 60 }, ...await lockCamera(track), outputSize: [video.videoWidth, video.videoHeight] };
  updateSessionIndicator();
  await captureReference(); paused = true; $('pause').textContent = 'Start calculation';
  updateSessionIndicator();
  const token = generation;
  if (typeof MediaStreamTrackProcessor === 'function') {
    try { reader = new MediaStreamTrackProcessor({ track }).readable.getReader(); }
    catch(error) {sourceSettings.trackProcessor=String(error);}
  }
  if(reader){
    sourceSettings.frameDelivery='MediaStreamTrackProcessor';
    const frameReader=reader;
    const consume=async()=>{
      while(generation===token){
        const item=await frameReader.read();
        if(generation!==token){item.value?.close();break;}
        if(item.done){cameraStopped(token,'Camera frame delivery ended.');break;}
        processFrame(item.value).catch(error=>cameraStopped(token,error.message));
      }
    };
    consume().catch(error=>cameraStopped(token,`Camera delivery failed: ${error.message}`));
  }else{sourceSettings.frameDelivery='Video-frame callback fallback';scheduleVideo(token);}
  track.addEventListener('ended',()=>cameraStopped(token,'Camera disconnected or permission revoked.'));
  details(); message('Camera active. Actual settings and available manual locks are shown below.');  } catch(error) { await stop(); $('run-state').textContent='Camera unavailable · choose an input'; throw error; }
}

const roi = installRoi($('roi'), $('shape'), (state) => {
  const list = $('regions'); list.replaceChildren();
  state.shapes.forEach(s => { const option = document.createElement('option'); option.value = s.id;
    option.textContent = `${'  '.repeat(s.depth)}${s.role === 'hole' ? '− Hole' : '+ Domain'} · ${s.label ?? s.type}`;
    option.selected = s.id === state.selected; list.append(option); });
  const selected = state.shapes.find(s => s.id === state.selected);
  $('shape-properties').disabled = !selected; $('delete-roi').disabled = !selected;
  $('undo-roi').disabled = !state.canUndo;
  if (selected) bounds(selected).forEach((v, i) => { $(['roi-x','roi-y','roi-width','roi-height'][i]).value = Math.round(v * 1000) / 1000; });
  $('roi-height').disabled = selected?.type === 'circle';
  const geometrySummary = state.hasBaseMask
    ? (state.shapes.length ? `Imported image mask · ${state.shapes.length} vector shape(s)` : 'Imported image mask · white pixels retained')
    : state.empty
      ? 'Blank mask · all pixels excluded'
      : state.shapes.length
        ? `${state.shapes.filter(s => s.role === 'domain').length} domain(s), ${state.shapes.filter(s => s.role === 'hole').length} hole(s)`
        : 'Full image · no mask.';
  $('roi-summary').textContent = `${geometrySummary}${roiEditing ? ' · unapplied edits' : ''}`;
  if(state.editing&&!editorWasActive&&matchMedia('(max-width:760px)').matches)requestAnimationFrame(()=>document.querySelector('.viewer').scrollIntoView({block:'start'})); editorWasActive=state.editing;
  $('apply-roi').disabled = !roiEditing; $('mask-toolbar').hidden=!roiEditing; $('apply-mask-top').disabled=!roiEditing;
});
const resetImageView = installViewNavigation({ getElement: $, updateVectorLengthLegend });
$('regions').onchange = () => roi.select($('regions').value);
$('delete-roi').onclick = () => roi.remove();
$('undo-roi').onclick = () => roi.undo();
$('add-centred').onclick = () => { try {roi.addCentred($('shape').value==='select'?'rectangle':$('shape').value);} catch(e){message(e.message,true);} };
$('apply-mask-top').onclick = () => $('apply-roi').click();
$('finish-mask-top').onclick = () => $('finish-shape').click();
$('cancel-mask-top').onclick = () => $('cancel-roi').click();
$('cancel-roi').onclick = () => sourceChange(async()=>{
  if(!appliedEditorState)return;
  roiEditing=false;roi.reset(appliedEditorState.width,appliedEditorState.height,appliedEditorState.shapes,appliedEditorState.baseMask,{empty:appliedEditorState.empty});draftMaskSource=appliedMaskSource;
  $('roi-editor').hidden=true;$('mask-import-summary').textContent=appliedMaskSource?`${appliedMaskSource.name} · applied base restored.`:'No imported image base. White pixels are kept; black pixels are excluded.';
  if(returnToInputAfterMask){if(source==='plate')await previewPlateTarget();else if(source==='images')await localSeek(imageSequence.index);else if(source==='video-file')showInputPreview(await frameFromVideo());}
  else await showPendingReferencePreview();
  message('Unapplied mask edits discarded. The existing numerical field and applied ROI are unchanged.');
});
$('finish-shape').onclick = () => { try { roi.finish(); $('shape').value = 'select'; } catch (e) { message(e.message, true); } };
$('update-shape').onclick = () => { try { roi.edit(['roi-x','roi-y','roi-width','roi-height'].map(id => Number($(id).value))); } catch (e) { message(e.message, true); } };
$('shape').onchange = () => { try { roi.finish(); } catch (e) { message(e.message, true); } };
$('open-video').onclick = () => $('video-file').click();
$('open-folder').onclick = () => $('image-folder').click();
$('open-images').onclick = () => $('image-files').click();
$('video-file').onchange = () => { const file=$('video-file').files[0];$('video-file').value='';if(file)sourceChange(()=>startLocalVideo(file)); };
for (const id of ['image-folder','image-files']) $(id).onchange = () => {const files=[...$(id).files];$(id).value='';if(files.length)sourceChange(()=>startImages(files));};
$('use-image-reference').onclick = () => sourceChange(()=>setImageReference(Number($('image-reference').value)));
$('previous-image').onclick = () => localSeek(Math.max(0,imageSequence.index-1));
$('next-image').onclick = () => localSeek(Math.min(imageSequence.images.length-1,imageSequence.index+1));
$('local-seek').onchange = () => localSeek($('local-seek').value);
$('image-rate').onchange = () => { sourceSettings.replayRateCapHz = Number($('image-rate').value); };
$('video-rate').onchange = () => {video.playbackRate=Number($('video-rate').value);sourceSettings.playbackRate=video.playbackRate;};
$('comparison').onchange = () => sourceChange(async () => {
  pauseSource();await imageSequence.task?.catch(()=>{});await drain();$('export').disabled=true;workbench.setReady(false);clearLiveMetrics();$('run-state').textContent='Comparison changed · awaiting a target field';$('pause').textContent='Start calculation';
  sourceSettings.requestedComparison=$('comparison').value;
  if(source==='images')$('image-reference-controls').hidden=$('comparison').value==='consecutive';
  invalidatePreparedReference(true);
  if(source==='images')await localSeek($('comparison').value==='consecutive'?Math.max(1,imageSequence.index):imageSequence.index);
  message($('comparison').value==='consecutive'?'Consecutive received-image pairs selected. Start calculation will prepare the current reference and mask.':'Fixed-reference comparison selected. Start calculation will restore the staged reference and mask.');
});
video.addEventListener('ended',()=>{if(source==='video-file'&&!paused&&!loadingSource)finishCalculation().catch(e=>message(e.message,true));});
$('plate').onclick = () => sourceChange(startPlate);
$('plate-load').onchange = () => sourceChange(previewPlateTarget);
$('demo').onclick = () => sourceChange(startDemo);
$('camera').onclick = () => sourceChange(startCamera);
$('reference').onclick = () => sourceChange(async () => {
  if (source === 'plate') return startPlate();
  if (source === 'images') return setImageReference(imageSequence.index);
  pauseSource(); await drain();
  await captureReference();
  if(source!=='camera')sourceSettings.reference = { kind:source, file:sourceSettings.file, timeSeconds:source === 'video-file' ? video.currentTime : null };
  paused=true;$('pause').textContent='Start calculation';
});
async function finishCalculation() {
  if(finishingCalculation)return finishingCalculation;
  const token=generation;
  const operation=(async()=>{
    pauseSource();$('pause').textContent='Start calculation';await drain();
    if(token!==generation||loadingSource)return;
    calculationProgress.finish();
    const count=workbench.report().count;
    $('solve-progress').textContent=count?`${count} saved field(s). Choose Results to select a quantity and replay.`:'No saved fields. Enable saving before calculation, or import results.';
    if(count&&!$('message').classList.contains('error'))message(`${count} saved field(s) available. Choose a quantity and use Play results or the slider; correlation is paused.`);
    if(count)await workbench.openFirst({index:calculationStartIndex});
  })();finishingCalculation=operation;
  try{await operation;}finally{if(finishingCalculation===operation)finishingCalculation=null;}
}
function calculationFrameTotal(includeCurrent) {
  if(source==='plate'||source==='video-file'&&$('solve-range').value==='single')return 1;
  if(source!=='images')return null;
  return imageSequenceTargetCount({
    imageCount:imageSequence.images.length,
    currentIndex:imageSequence.index,
    includeCurrent,
    range:$('solve-range').value,
    comparison:$('comparison').value,
  });
}
async function startCalculation() {
  if(!referenceInfo||!pendingReferenceFrame||startingCalculation||loadingSource)return;
  if(finishingCalculation){await finishingCalculation;if(!referenceInfo||!pendingReferenceFrame)return;}
  if(workbench.storage.report().readOnly){message('Close the saved session before starting a new calculation.',true);return;}
  if(workbench.storageBlocked){message('Export or retry saving the retained fields before starting another calculation.',true);return;}
  if(roiEditing){message('Apply the mask before calculation.');return;}
  if(source==='camera'&&sourceSettings.deliveryState==='stopped')return sourceChange(startCamera);
  startingCalculation=true;
  activeCalculationMode=workspace.mode;
  updateSessionIndicator();
  calculationProgress.begin('Preparing reference and GPU solver…');
  const token=generation;
  const includeCurrent=viewingInput;
  try {
    const requested=options();
    if(!referencePrepared||solverPreparationKey(requested)!==appliedSolverKey){
      await prepareCalculation(requested);
      if(token!==generation||!referencePrepared)return;
    }
    calculationStartIndex=workbench.report().count;
    workbench.returnLive();workspace.showPane('solve');
    await workbench.startRecording();await send('show-reference',{show:false});
    viewingInput=false;$('input-field').hidden=true;$('field').hidden=false;
    $('solve-progress').textContent=workbench.recording?'Calculating and saving full fields. Stop to inspect Results.':'Preview calculation only: fields are not saved. Enable saving for replay.';
    calculationProgress.setTargets(calculationFrameTotal(includeCurrent));
    if(source==='plate'){await runPlateTarget();startingCalculation=false;updateSessionIndicator();if(token===generation)await finishCalculation();return;}
    if(source==='video-file'&&$('solve-range').value==='single'){
      if($('comparison').value!=='fixed')throw new Error('A single video target needs fixed-reference comparison; use a sequence for consecutive pairs.');
      await processFrame(await frameFromVideo(),null,{force:true,reset:true,input:{kind:source,file:sourceSettings.file,timeSeconds:video.currentTime}});
      startingCalculation=false;updateSessionIndicator();if(token===generation)await finishCalculation();return;
    }
    if(source==='video-file'||source==='demo'){
      if(video.ended){await seekVideo(video,0,sourceAbort.signal);await send('reset');}
      await video.play();
    }
    paused=false;$('pause').textContent='Pause calculation';startingCalculation=false;updateSessionIndicator();
    if(source==='images'){
      if($('solve-range').value==='single'){paused=true;await runImage(imageSequence.index,true);}
      else await playImages(includeCurrent);
      if(token===generation)await finishCalculation();
    }
  } catch(error){pauseSource();calculationProgress.finish();message(error.message,true);}
  finally{startingCalculation=false;updateSessionIndicator();}
}
$('pause').onclick = () => !paused?finishCalculation():startCalculation();
$('stop-active-session').onclick = () => stop().catch(error => message(error.message, true));
$('solve-results').onclick = () => finishCalculation();
$('go-solve').onclick = () => workspace.showPane('solve');
$('reset').onclick = () => send('reset');
function roiMetadata() {
  return {version:appliedMaskSource?2:1,composition:appliedMaskSource?'imported binary image intersected with vector domains/holes':'union of roots; exact pixel-containment alternates domain/hole/island',baseImage:appliedMaskSource,shapes:appliedShapes};
}
$('mask-conversion').onchange = () => { $('mask-threshold-label').hidden=$('mask-conversion').value!=='threshold'; };
$('import-mask').onclick = () => $('mask-file').click();
$('mask-file').onchange = () => {
  const file=$('mask-file').files[0];$('mask-file').value='';if(!file)return;
  sourceChange(async()=>{
    if(!referenceInfo)throw new Error('Choose a reference image before importing a mask');
    pauseSource();await drain();workbench.returnLive();
    const conversion=$('mask-conversion').value,threshold=maskThreshold(conversion,Number($('mask-threshold').value));
    const mask=await decodeMaskImage(file,referenceInfo.width,referenceInfo.height,{threshold,signal:sourceAbort.signal});
    const hash=await crypto.subtle.digest('SHA-256',await file.arrayBuffer());
    draftMaskSource={...fileInfo(file),sha256:Array.from(new Uint8Array(hash),b=>b.toString(16).padStart(2,'0')).join(''),threshold,
      convention:threshold===null?'opaque exact black/white':`${conversion==='automatic'?'automatic':'explicit'} rounded BT.709 luminance >= ${threshold} and alpha >= 128`};
    roi.reset(referenceInfo.width,referenceInfo.height,[{id:'image-mask-domain',label:'Clip image mask',type:'rectangle',x:0,y:0,width:referenceInfo.width,height:referenceInfo.height}],mask);
    roiEditing=true;workspace.showPane('mask');await showPendingReferencePreview();$('roi-editor').hidden=false;$('shape').value='select';roi.start();
    $('mask-import-summary').textContent=`${fileInfo(file).name} · ${threshold===null?'exact binary pixels':`${conversion==='automatic'?'automatic threshold':'explicit threshold'} ${threshold}`} · vectors clip this base image. Full image removes the base.`;
    message('Mask imported for review. Add vector holes or clipping regions, then Apply mask. White retains pixels; dimensions are unchanged.');
  });
};
$('create-mask').onclick = () => sourceChange(async () => {
  if(!referenceInfo)throw new Error('Choose a reference image before creating a mask');
  returnToInputAfterMask=viewingInput;workbench.returnLive();pauseSource();await drain();
  roi.reset(referenceInfo.width,referenceInfo.height,[],null,{empty:true});draftMaskSource=null;
  $('mask-import-summary').textContent='Blank mask · all pixels start excluded. Add one or more domains to keep pixels.';
  roiEditing=true;workspace.showPane('mask');$('pause').textContent='Start calculation';
  await showPendingReferencePreview();$('roi-editor').hidden=false;$('shape').value='select';roi.start();
  message('Blank mask created. Add a rectangle, ellipse, circle or polygon for each domain to keep. Apply mask when ready.');
});
$('save-mask').onclick=async()=>{
  try{if(!referenceInfo)return;const mask=roi.mask(),blob=await maskPng(mask,referenceInfo.width,referenceInfo.height);download('live-dic-roi-edited.png',blob,'image/png');}
  catch(e){message(e.message,true);}
};
$('draw').onclick = () => sourceChange(async () => { returnToInputAfterMask=viewingInput;workspace.showPane('mask'); workbench.returnLive(); pauseSource(); await drain(); roiEditing = true; $('pause').textContent = 'Start calculation'; await showPendingReferencePreview(); $('roi-editor').hidden = false; $('shape').value = 'select'; roi.start(); });
$('apply-roi').onclick = () => sourceChange(async () => {
  try { workbench.setReady(false);$('export').disabled=true;clearLiveMetrics(); const nextShapes=roi.shapes();appliedShapes=nextShapes;workbench.setMask(null);appliedMaskSource=draftMaskSource;appliedEditorState=roi.state();invalidatePreparedReference(true);
    roiEditing = false; roi.stop(); $('roi-editor').hidden = true; $('apply-roi').disabled = true;
    message('Mask applied locally. Start calculation prepares the reference with this mask; all complete selected subsets are used.'); details();
    if (source === 'plate') await previewPlateTarget();
    else if (source === 'images') await localSeek(imageSequence.index);
    workspace.showPane('solve');
  } catch (error) { message(error.message, true); }
});
$('clear-roi').onclick = () => sourceChange(async () => { workbench.returnLive(); pauseSource(); await drain(); workbench.setMask(null); workbench.setReady(false);$('export').disabled=true;clearLiveMetrics(); roiEditing = false; roi.reset(referenceInfo.width, referenceInfo.height); appliedShapes = []; draftMaskSource = appliedMaskSource = null; $('roi-editor').hidden = true;
  invalidatePreparedReference(true);appliedEditorState=roi.state(); $('mask-import-summary').textContent='Full image · no imported base mask.'; $('pause').textContent = 'Start calculation'; details(); if (source === 'plate') await previewPlateTarget(); });
function displayOptions() { return { mode: Number($('component').value), scale: $('scale').value === 'auto' ? null : Number($('scale').value),
  opacity: Number($('opacity').value), vectorOutline: $('vector-outline').checked,
  vectorColourMap: $('vector-colour-map').value,
  arrowStep: $('arrow-step').value==='auto'?null:Number($('arrow-step').value), arrowGain: $('arrow-gain').value==='auto'?null:Number($('arrow-gain').value) }; }
async function updateLiveDisplay() {
  if (!transport || !referenceInfo) return;
  try { await send('display', { display: displayOptions() }); }
  catch (error) { message(error.message, true); }
}
for (const id of ['component','scale','opacity','arrow-step','arrow-gain','vector-outline','vector-colour-map']) {
  $(id).onchange = updateLiveDisplay;
}
function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type })), link = document.createElement('a');
  link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function captureCurrentField() {
  pauseSource(); await drain(); $('pause').textContent = 'Start calculation';
  const reply = await send('export');
  reply.metadata.build = buildInfo; reply.metadata.acquisition = structuredClone(sourceSettings); reply.metadata.roiGeometry = roiMetadata(); reply.metadata.display = displayOptions();
  return reply;
}
$('export').onclick = async () => {
  const { field, metadata } = await captureCurrentField();
  const parameters = field.config.motionModel === 'quadratic' ? 12 : 6;
  const lines = [`x,y,status,iterations,${Array.from({ length: parameters }, (_, i) => `p${i}`).join(',')}`];
  for (let i = 0; i < field.status.length; i++) lines.push([
    (i % field.grid[0]) * field.subsetStride + (field.subsetSize - 1) / 2,
    Math.floor(i / field.grid[0]) * field.subsetStride + (field.subsetSize - 1) / 2,
    field.status[i], field.iterations[i], ...field.parameters.slice(i * parameters, (i + 1) * parameters),
  ].join(','));
  metadata.parameterOrder=parameterOrder(field.config.motionModel);metadata.parameterUnits=parameterUnits(field.config.motionModel);
  metadata.acquisition = sourceSettings; metadata.roiGeometry = roiMetadata(); metadata.display = displayOptions();
  download('live-dic-field.csv', lines.join('\n'), 'text/csv');
  download('live-dic-metadata.json', JSON.stringify(metadata, null, 2), 'application/json');
};
$('resolution').onchange = () => { if (source === 'camera') sourceChange(startCamera); };
$('download-device-report').onclick = () => {
  const report = createDeviceReport({ capabilities, build: buildInfo });
  const stamp = report.generatedAt.replaceAll(':', '-');
  download(`live-dic-device-report-${stamp}.json`, JSON.stringify(report, null, 2), 'application/json');
};
window.addEventListener('pagehide', () => { workbench.destroy(); sourceAbort.abort(); clearPreviousCapture();clearPendingReference(); if (localVideoUrl) URL.revokeObjectURL(localVideoUrl); stream?.getTracks().forEach((t) => t.stop()); transport?.postMessage({ type: 'destroy' }); });
window.addEventListener('error', (event) => { status.errors.push(event.message); message(event.message, true); });
window.addEventListener('unhandledrejection', (event) => { const text = String(event.reason); status.errors.push(text); message(text, true); });
if (smoke) { globalThis.liveDicReport = () => send('report'); globalThis.liveDicPipelineReport = () => ({settings:supervision,recording:recordingQueue.report(),synchronousAdmissions:synchronousAdmissions.slice(),presentation:lastPresentation}); globalThis.liveDicWorkbench = workbench; globalThis.liveDicRoi = roi; globalThis.liveDicSourceState = () => ({source,paused,viewingInput,pane:workspace.pane,loadingSource,inFlight,index:imageSequence.index,fileCount:imageSequence.images.length,localBusy:!!imageSequence.task,loop:!!imageSequence.loop,settings:sourceSettings,comparison:$('comparison').value,videoTime:video.currentTime,videoPaused:video.paused}); }
for(const id of ['display-rate','preview-width'])$(id).onchange=()=>{
  if(!transport||!referencePrepared)return;
  const options={displayRate:$('display-rate').value==='off'?0:Number($('display-rate').value),previewEnabled:$('display-rate').value!=='off',previewMaxWidth:Number($('preview-width').value)};
  send('presentation',{options}).catch(e=>message(e.message,true));
};
$('large-shift-preset').onclick = () => {
  $('custom-solver').checked=true;$('pyramid-levels').value='6';
  $('coarse-iterations').value='20';$('fine-iterations').value='16';
  $('temporal-start').checked=false;$('fourier-start').checked=false;$('global-affine-start').checked=false;
  $('custom-solver').onchange();
};
for (const id of ['fourier-start','global-affine-start','global-affine-levels','overlap-pipeline','gpu-reference-preparation','custom-solver','motion-model','subset-size','subset-stride','interpolation','spatial-propagation','pyramid-levels','coarse-iterations','fine-iterations','outlier-policy','temporal-start','roi-policy']) $(id).onchange = () => {
  $('solver-settings').disabled = !$('custom-solver').checked;
  $('temporal-start').disabled = !$('custom-solver').checked;
  $('global-affine-levels').disabled = !$('global-affine-start').checked;
  $('subset-stride').max = $('subset-size').value;
  if (Number($('subset-stride').value) > Number($('subset-size').value)) $('subset-stride').value = $('subset-size').value;
  invalidatePreparedReference();
};
init().catch((error) => { status.errors.push(error.message); message(error.message, true); });
if ('serviceWorker' in navigator && (!smoke || query.has('offline'))) navigator.serviceWorker.register(new URL('./sw.js', import.meta.url)).catch((error) => message(`Offline installation unavailable: ${error.message}`, true));
