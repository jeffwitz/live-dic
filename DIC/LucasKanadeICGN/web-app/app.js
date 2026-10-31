import { decodeMaskImage, maskPng } from './ui/mask-image.js';
import { installWorkbench } from './ui/workbench.js';
import { installWorkspace, solverOptions } from './ui/workspace.js';
import { installRoi } from './ui/roi.js';
import { bounds, platePreset } from './ui/roi-geometry.js';
import { imageFiles, fileInfo, decodeImage, waitForMedia, seekVideo } from './ui/local-media.js';
const $ = (id) => document.getElementById(id);
const query = new URLSearchParams(location.search), smoke = query.has('smoke');
let transport, stream, reader, manifest, source = null, paused = true, inFlight = 0, dropped = 0;
const sourceButtons = ['demo','plate','camera','open-video','open-folder','open-images'];
let loadingSource = false, sourceAbort = new AbortController(), localVideoUrl = null;
let localImages = [], imageIndex = 0, imageTask = null, imageLoop = null, seekSerial = 0;
let imageDelay = null, imageWake = null;
let previousCapture = null, captureBusy = false, receivedOrdinal = 0;
let roiEditing = false, appliedShapes = [], plateManifest = null;
let draftMaskSource = null, appliedMaskSource = null, appliedEditorState = null, editorWasActive = false;
let generation = 0, previousFrame = -1, referenceInfo, capabilities, sourceSettings = {}, requests = 0;
let buildInfo = null;
const pending = new Map(), video = $('video');
sourceButtons.forEach(id => $(id).disabled = true);
const status = globalThis.liveDicStatus = { ready: false, completed: 0, errors: [], results: [] };
function message(text, error = false) { $('message').textContent = text; $('message').classList.toggle('error', error); }
function send(type, data = {}, transfer = []) {
  const id = ++requests;
  const result = new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
  transport.postMessage({ type, id, ...data }, transfer);
  return result;
}
if (smoke) {
  const original = console.error.bind(console);
  console.error = (...args) => { status.errors.push(args.map(String).join(' ')); original(...args); };
}
const options = solverOptions;
const workspace = installWorkspace({ shortcuts: {
  play: () => { if (!$('pause').disabled && !loadingSource) $('pause').click(); },
  reference: () => { if (!$('reference').disabled && !loadingSource) $('reference').click(); },
  clear: () => workbench.clearInspection(),
} });
const workbench = installWorkbench({
  captureCurrent: captureCurrentField,
  beforeProjectExport: async () => { pauseSource(); await drain(); $('pause').textContent='Play'; },
  geometryDefaults: () => referenceInfo ? { width: referenceInfo.width, height: referenceInfo.height, subsetSize: referenceInfo.subsetSize, subsetStride: referenceInfo.subsetStride } : {},
  pause: () => { pauseSource(); $('pause').textContent = 'Play'; },
  error: text => message(text, true),
  recordChange: enabled => send('recording', { enabled }).catch(e => message(e.message, true)),
  liveChanged: (reviewing,restore) => { if(restore && transport && referenceInfo) send('display',{display:displayOptions()}).catch(e=>message(e.message,true)); $('run-state').textContent = reviewing ? 'Saved field · reference coordinates' : source ? `${source} · ${paused ? 'paused' : 'running'}` : 'No source selected'; },
  openResults: () => workspace.showPane('view'),
});
function details() { $('details').textContent = JSON.stringify({ capabilities, acquisition: { ...sourceSettings, files: sourceSettings.files ? `${sourceSettings.files.length} file records retained in export` : undefined },
  build: buildInfo, execution: transport.isMain ? 'Main-thread fallback: OffscreenCanvas unavailable' : 'Dedicated worker',
  reference: referenceInfo ? { ...referenceInfo, roiSelected: undefined, referenceValid: undefined } : null, profile: $('profile').value,
  requestedResolution: $('resolution').value }, null, 2); }
function updateScale(scale) {
  if (!Number.isFinite(scale)) return;
  const mode = Number($('component').value);
  document.querySelector('.legend').classList.toggle('magnitude', mode === 2 || mode === 7);
  document.querySelector('.legend').classList.toggle('status', mode === 6);
  if (mode === 6) { $('low').textContent = 'Accepted / flagged'; $('high').textContent = 'Unconverged / rejected'; return; }
  const strain = Number($('component').value) >= 3 && Number($('component').value) <= 5;
  $('low').textContent = mode === 2 || mode === 7 ? '0' : `−${scale.toPrecision(3)}`;
  $('high').textContent = mode === 7 ? `${(scale * 1e6).toPrecision(3)} με` : `+${scale.toPrecision(3)}${strain ? ' (gradient)' : ' px'}`;
}
function receive({ data }) {
  if (data.bitmap) {
    $('field').width = data.width; $('field').height = data.height;
    $('field').getContext('bitmaprenderer').transferFromImageBitmap(data.bitmap);
    delete data.bitmap;
  }
  if (data.type === 'error') {
    status.errors.push(data.message); message(data.message, true);
    if (pending.has(data.id)) { pending.get(data.id).reject(new Error(data.message)); pending.delete(data.id); }
    paused = true; $('export').disabled=true; return;
  }
  if (data.type === 'result') {
    if (data.plateScore) { sourceSettings.analyticScore = data.plateScore; details(); }
    $('export').disabled = false; workbench.setReady(true);
    if (data.record) { try { workbench.add(data.record.field, { ...data.record.metadata, acquisition: {source:sourceSettings.source}, build: buildInfo, roiGeometry: roiMetadata() }); } catch(e) { message(e.message,true); } delete data.record; }
    if (!workbench.reviewing) $('run-state').textContent = `${source} · ${data.width} × ${data.height} · ${data.comparison?.mode ?? 'fixed'} · ${paused ? 'paused' : 'running'}`;
    if (data.comparison) { sourceSettings.lastComparison = data.comparison; updateLocalPosition(data.comparison.target); }
    status.completed++; status.results.push(data); if (status.results.length > (smoke ? 18000 : 120)) status.results.shift();
    $('fps').textContent = source === 'plate' ? 'Static pair' : data.fps.toFixed(1);
    $('age').textContent = `${data.ageMedian.toFixed(1)} / ${data.ageP95.toFixed(1)} ms`;
    $('acceptance').textContent = `${data.accepted} / ${data.selected}`;
    $('iterations').textContent = data.meanIterations?.toFixed(2) ?? '—';
    $('error').textContent = data.truthRmse == null ? '—' : `${data.truthRmse.toFixed(4)} px`;
    $('error').classList.toggle('metric-warning', data.truthRmse != null && data.truthRmse > .1);
    $('error').title = data.plateScore?.population ?? 'Whole accepted field, including possible wrong-basin estimates near the image boundary';
    if (!workbench.reviewing) updateScale(data.scale);
  }
  if (data.type === 'scale' && !workbench.reviewing) updateScale(data.scale);
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
  if (!capabilities.supported) {
    message(`WebGPU is unavailable. ${capabilities.reasons.join('. ')} Use a supported secure browser; this app has no WebGL fallback.`, true);
    sourceButtons.forEach(id => $(id).disabled = true); return;
  }
  for (const option of $('resolution').options) {
    const [w, h] = option.value.split('x').map(Number);
    option.disabled = !capabilities.resolutions.some(([x, y]) => x === w && y === h);
  }
  sourceButtons.forEach(id => $(id).disabled = false);
  $('profile').value = capabilities.recommendedProfile;
  const recommended = capabilities.recommendedResolution ?? capabilities.resolutions[0];
  if (recommended) $('resolution').value = recommended.join('x');
  message(`Ready. ${capabilities.reasons.join('. ')}`); status.ready = true;
  if (smoke) await (query.get('case') === 'plate' ? startPlate() : startDemo());
}
async function drain() {
  while (inFlight) await new Promise((resolve) => setTimeout(resolve, 10));
  if (imageTask) await imageTask.catch(() => {});
}
function clearPreviousCapture() { previousCapture?.frame.close(); previousCapture = null; }
function pauseSource() { paused = true; clearTimeout(imageDelay); imageWake?.(); imageWake=null; if (source === 'video-file') video.pause(); clearPreviousCapture(); }
async function sourceChange(operation) {
  if (loadingSource) return;
  loadingSource = true; document.querySelector('.workspace').inert=true; sourceButtons.forEach(id => $(id).disabled = true);
  const busyText = 'Preparing input / reference…'; message(busyText);
  try { seekSerial++; pauseSource(); await drain(); await operation(); if($('message').textContent===busyText)message('Ready. Preparation finished.'); }
  catch (error) { if (error.name !== 'AbortError') message(error.message, true); }
  finally { loadingSource = false; document.querySelector('.workspace').inert=false; sourceButtons.forEach(id => $(id).disabled = !capabilities?.supported); }
}
async function stop() {
  workbench.returnLive(); workbench.stopRecording();
  pauseSource(); generation++; seekSerial++; sourceAbort.abort();
  await drain(); await imageTask?.catch(() => {}); await imageLoop?.catch(() => {}); sourceAbort = new AbortController();
  workbench.setReady(false); workbench.setMask(null);appliedEditorState=null;roiEditing=false;roi.stop({discardDraft:true});
  await reader?.cancel(); reader = null;
  stream?.getTracks().forEach(track => track.stop()); stream = null;
  video.pause(); video.srcObject = null; video.removeAttribute('src'); video.load();
  if (localVideoUrl) URL.revokeObjectURL(localVideoUrl); localVideoUrl = null;
  localImages = []; imageIndex = 0; receivedOrdinal = 0;
  referenceInfo = null; ['reference','pause','reset','draw','clear-roi','export','import-mask','save-mask'].forEach(id => $(id).disabled = true);
  $('comparison').disabled=false; source = null; roiEditing = false; $('roi-editor').hidden = true; $('plate-load-label').hidden = true; $('local-controls').hidden = true;
}
async function frameFromVideo() {
  if (source === 'video-file' && referenceInfo && (video.videoWidth !== referenceInfo.width || video.videoHeight !== referenceInfo.height)) throw new Error('Video dimensions changed; capture a new reference. Frames are not resized.');
  const [width, height] = source === 'camera' ? [video.videoWidth, video.videoHeight] : sourceSettings.outputSize ?? [video.videoWidth, video.videoHeight];
  if (typeof VideoFrame === 'function' && width === video.videoWidth && height === video.videoHeight) {
    try { return new VideoFrame(video, { timestamp: Math.round(video.currentTime * 1e6) }); } catch { /* ImageBitmap fallback below. */ }
  }
  return createImageBitmap(video, { resizeWidth: width, resizeHeight: height });
}
async function updateReferenceControls() {
  const translationOnly = referenceInfo.config?.motionModel === 'translation';
  for (const option of $('component').options) if ([3,4,5,7].includes(Number(option.value))) option.disabled = translationOnly;
  if (translationOnly && [3,4,5,7].includes(Number($('component').value))) { $('component').value='2'; await send('display',{display:displayOptions()}); }
}
async function captureReference(frame = null) {
  workbench.returnLive();
  pauseSource(); await drain(); workbench.setMask(null); workbench.setReady(false); roiEditing = false; $('roi-editor').hidden = true;
  frame ??= await frameFromVideo();
  if (source === 'camera') { sourceSettings.actual = stream?.getVideoTracks()[0]?.getSettings(); sourceSettings.outputSize = [video.videoWidth, video.videoHeight]; }
  referenceInfo = (await send('reference', { frame, options: options() }, [frame])).info; appliedShapes = []; draftMaskSource = appliedMaskSource = null; $('mask-import-summary').textContent='White pixels are kept; black pixels are excluded. Imported images must match the reference dimensions.';
  $('placeholder').hidden = true; $('image-viewport').style.aspectRatio = `${referenceInfo.width}/${referenceInfo.height}`;
  roi.reset(referenceInfo.width, referenceInfo.height); appliedEditorState=roi.state(); $('apply-roi').disabled = true;
  for (const id of ['reference', 'pause', 'reset', 'draw', 'clear-roi', 'export', 'import-mask', 'save-mask']) $(id).disabled = false;
  if ($('comparison').value === 'consecutive' && source !== 'plate') $('export').disabled = true;
  await updateReferenceControls();
  previousFrame = -1; details(); message('Reference captured. Camera and results stay on this device.');
}
function countDrop() { dropped++; $('drops').textContent = dropped; }
async function processFrame(frame, frameIndex = null, { force = false, reset = false, input = null, referenceFrame = null, referenceInput = null } = {}) {
  const token = generation, mode = source === 'plate' ? 'fixed' : $('comparison').value;
  if (paused && !force) { frame.close(); referenceFrame?.close(); return false; }
  input = { kind:source, ordinal:++receivedOrdinal, timestampUs:frame.timestamp ?? null, ...input };
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
  inFlight++;
  const historyReset = reset || (frameIndex != null && frameIndex < previousFrame);
  if (frameIndex != null) previousFrame = frameIndex;
  const stretch = manifest?.stretch[frameIndex];
  const comparison = { mode, reference: mode === 'consecutive' ? referenceInput : sourceSettings.reference ?? { kind: source, role: 'captured reference' }, target: input };
  try {
    const reply = await send('frame', { frame, frameIndex, reset: historyReset, referenceFrame, comparison,
      truth: source === 'demo' && mode === 'fixed' && stretch ? [stretch[0] - 1, stretch[1] - 1] : null }, referenceFrame ? [referenceFrame, frame] : [frame]);
    return reply.type === 'result';
  } catch (error) { pauseSource(); message(error.message, true); return false; }
  finally { inFlight--; }
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
      } catch (error) { inFlight--; pauseSource(); message(error.message, true); }
    } else if (!paused) { dropped++; $('drops').textContent = dropped; }
    if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(callback);
    else requestAnimationFrame((time) => callback(time, null));
  };
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(callback);
  else requestAnimationFrame((time) => callback(time, null));
}
function updateLocalPosition(input = null) {
  if (source === 'images') {
    const index = input?.index ?? imageIndex;
    $('local-seek').value = index;
    $('local-position').textContent = `${index + 1} / ${localImages.length} · ${input?.name ?? fileInfo(localImages[index]).name}`;
  } else if (source === 'video-file') {
    const time = input?.timeSeconds ?? video.currentTime;
    $('local-seek').value = time;
    $('local-position').textContent = `${time.toFixed(3)} / ${Number.isFinite(video.duration) ? video.duration.toFixed(3) : '?'} s`;
  }
}
function localControls(images) {
  $('local-controls').hidden = false; $('image-reference-controls').hidden = !images || $('comparison').value === 'consecutive';
  $('image-step-controls').hidden = !images; $('image-rate-label').hidden = !images; $('video-rate-label').hidden = images;
  $('resolution').disabled = true;
}
async function runImage(index, reset = false) {
  if (!Number.isInteger(index) || index < 0 || index >= localImages.length) return false;
  const token = generation, signal = sourceAbort.signal;
  await imageTask?.catch(() => {});
  if (token !== generation) return false;
  const work = (async () => {
    let frame, prior;
    try {
      frame = await decodeImage(localImages[index], [referenceInfo.width,referenceInfo.height], signal);
      if ($('comparison').value === 'consecutive') {
        if (index === 0) { frame.close(); frame = null; imageIndex = 0; updateLocalPosition(); $('export').disabled = true; message('The first image has no predecessor. Choose the next image.'); return false; }
        prior = await decodeImage(localImages[index - 1], [referenceInfo.width,referenceInfo.height], signal);
      }
      if (token !== generation) return false;
      const input = { kind:'images',index,...fileInfo(localImages[index]) };
      const referenceInput = prior ? { kind:'images',index:index-1,...fileInfo(localImages[index-1]) } : null;
      const target = frame, reference = prior; frame = prior = null;
      const ok = await processFrame(target,index,{force:true,reset,input,referenceFrame:reference,referenceInput});
      if (ok) { imageIndex = index; updateLocalPosition(input); }
      return ok;
    } finally { frame?.close(); prior?.close(); }
  })();
  imageTask = work;
  try { return await work; }
  finally { if (imageTask === work) imageTask = null; }
}
async function setImageReference(index) {
  pauseSource(); await imageTask?.catch(() => {}); await imageLoop?.catch(() => {}); await drain();
  const bitmap = await decodeImage(localImages[index],null,sourceAbort.signal);
  await captureReference(bitmap);
  sourceSettings.reference = { kind:'images',index,...fileInfo(localImages[index]) };
  $('image-reference').value = index; $('pause').textContent = 'Play';
  await runImage(imageIndex,true); details();
}
async function startImages(files) {
  const choice = imageFiles(files);
  await stop(); source = 'images'; manifest = null; localImages = choice.images;
  sourceSettings = { source:'local image files',ordering:'natural relative filename, lexical tie-break',files:localImages.map(fileInfo),ignored:choice.ignored };
  const select = $('image-reference'); select.replaceChildren();
  localImages.forEach((file,i) => { const option=document.createElement('option');option.value=i;option.textContent=fileInfo(file).name;select.append(option); });
  $('local-name').textContent = `${localImages.length} image(s) · ${choice.ignored.length} unsupported/non-image file(s) ignored`;
  $('local-seek').min=0; $('local-seek').max=localImages.length-1; $('local-seek').step=1;
  localControls(true); imageIndex = localImages.length>1 ? 1 : 0;
  await setImageReference(0);
  message(`Loaded ${localImages.length} local image(s). ${choice.ignored.length ? 'Ignored files are listed in the details. ' : ''}Choose a reference, inspect the mask, then Play.`);
}
async function playImages() {
  if (imageLoop) { await imageLoop; if (imageLoop) return; }
  if (imageIndex >= localImages.length-1) imageIndex = -1;
  paused = false; $('pause').textContent = 'Pause';
  const token = generation;
  const operation = (async () => {
    while (!paused && token === generation) {
      const index = imageIndex + 1;
      if (index >= localImages.length) break;
      const start = performance.now();
      if (!await runImage(index, index === 0)) { if (index === 0 && $('comparison').value === 'consecutive') continue; break; }
      const rate = Number($('image-rate').value);
      if (!Number.isFinite(rate) || rate < 0 || rate > 240) throw new Error('Image rate must be between 0 and 240 (0 = uncapped)');
      if (rate) await new Promise(resolve => { imageWake=resolve;imageDelay=setTimeout(()=>{imageWake=null;resolve();},Math.max(0,1000/rate-(performance.now()-start))); });
    }
  })();
  imageLoop=operation;
  try { await operation; }
  catch(error) { if (error.name !== 'AbortError') message(error.message,true); }
  finally { if (imageLoop===operation) { imageLoop=null; if(token===generation){paused=true; $('pause').textContent='Play';} } }
}
async function startLocalVideo(file) {
  await stop(); source='video-file'; manifest=null;
  localVideoUrl=URL.createObjectURL(file);
  const ready=waitForMedia(video,'loadeddata',sourceAbort.signal);video.playbackRate=Number($('video-rate').value);video.loop=false;video.preload='auto';video.src=localVideoUrl;video.load();await ready;
  sourceSettings={source:'local video',file:fileInfo(file),outputSize:[video.videoWidth,video.videoHeight],durationSeconds:Number.isFinite(video.duration)?video.duration:null,
    reference:{kind:'video-file',file:fileInfo(file),timeSeconds:video.currentTime}};
  localControls(false); $('local-name').textContent=file.name;
  $('local-seek').min=0;$('local-seek').max=Number.isFinite(video.duration)?video.duration:0;$('local-seek').step=.001;$('local-seek').disabled=!Number.isFinite(video.duration);
  await captureReference();
  if ($('comparison').value==='fixed') await processFrame(await frameFromVideo(),0,{force:true,input:{kind:'video-file',file:sourceSettings.file,timeSeconds:video.currentTime}});
  $('pause').textContent='Play';updateLocalPosition();details();scheduleVideo(generation);
  message('Local video ready. Seek to a reference and capture it, or Play. No file is uploaded.');
}
async function localSeek(value) {
  const serial=++seekSerial,token=generation;pauseSource();$('pause').textContent='Play';
  await imageTask?.catch(()=>{});await drain();
  if (serial!==seekSerial || token!==generation) return;
  try {
    if (source==='images') await runImage(Number(value),true);
    else if (source==='video-file') {
      await seekVideo(video,Number(value),sourceAbort.signal); if($('comparison').value==='consecutive')$('export').disabled=true; if(serial!==seekSerial || token!==generation)return;
      await send('reset');
      if(serial!==seekSerial || token!==generation)return;
      const input={kind:source,file:sourceSettings.file,timeSeconds:video.currentTime},frame=await frameFromVideo();
      if(serial!==seekSerial || token!==generation){frame.close();return;}
      await processFrame(frame,null,{force:true,reset:true,input});
      updateLocalPosition();
    }
    details();
  } catch(error) {if(error.name!=='AbortError')message(error.message,true);updateLocalPosition();}
}
async function startDemo() {
  await stop(); source = 'demo'; $('resolution').disabled = true;
  manifest = await fetch(new URL('./demo/manifest.json', import.meta.url)).then((r) => r.json());
  const reference = await createImageBitmap(await fetch(new URL('./demo/reference.png', import.meta.url)).then((r) => r.blob()));
  sourceSettings = { reference:{kind:'demo',frame:0}, source: 'derived analytic tensile demo', encoding: manifest.encoding, outputSize: [manifest.width, manifest.height] };
  $('resolution').value = `${manifest.width}x${manifest.height}`;
  if (!capabilities.resolutions.some(([w, h]) => w === manifest.width && h === manifest.height)) throw new Error('The shipped demo requires 1080p capability');
  await captureReference(reference);
  video.src = new URL('./demo/tensile.webm', import.meta.url); video.loop = true; video.playbackRate=1;
  if (smoke && query.has('rate')) video.playbackRate = Number(query.get('rate'));
  await video.play(); paused = false; $('pause').textContent = 'Pause'; scheduleVideo(generation);
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
  const mask = roi.mask(), ownedMask = mask?.slice() ?? null; referenceInfo = (await send('roi', { mask, options: options() }, [mask.buffer])).info; workbench.setMask(ownedMask); appliedEditorState=roi.state();
  $('component').value = referenceInfo.config.motionModel === 'translation' ? '2' : '7'; $('scale').value = referenceInfo.config.motionModel === 'translation' ? '10' : '0.002'; $('arrows').checked = true; $('arrow-gain').value = '50'; $('opacity').value = '1';
  await send('display', { display: displayOptions() });
  await runPlateTarget();
  // Show editable outer rectangle and circular hole on the fixed reference.
  message('Original UHD plate · equivalent strain and displacement arrows. Edit mask to move/resize the outlined rectangle or hole, or add regions.');
}
async function runPlateTarget() {
  if (roiEditing) { message('Apply the edited mask before recalculating the plate.'); return; }
  paused = true; await drain();
  const frame = Number($('plate-load').value), record = plateManifest.images.find(r => r.frame === frame);
  const bitmap = await createImageBitmap(await fetch(new URL(`./demo/plate/${record.file}`, import.meta.url)).then(r => r.blob()));
  sourceSettings.load = { frame, primary: record.primary_MPa, secondary: record.secondary_MPa };
  await send('frame', { frame: bitmap, frameIndex: frame, reset: true, plate: sourceSettings.load, comparison:{mode:'fixed',reference:sourceSettings.reference,target:{kind:'plate',...sourceSettings.load}} }, [bitmap]);
  $('pause').textContent = 'Recalculate'; details();
}
async function lockCamera(track) {
  const caps = track.getCapabilities?.() ?? {}, settings = track.getSettings(), locks = {};
  for (const key of ['exposureMode', 'focusMode', 'whiteBalanceMode']) {
    if (Array.isArray(caps[key]) && caps[key].includes('manual')) {
      try { await track.applyConstraints({ advanced: [{ [key]: 'manual' }] }); locks[key] = 'manual requested'; }
      catch (error) { locks[key] = `not locked: ${error.message}`; }
    } else locks[key] = 'manual lock not exposed';
  }
  return { requested: settings, actual: track.getSettings(), capabilities: caps, locks };
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
async function startCamera() {
  await stop(); manifest = null; source = 'camera'; $('resolution').disabled = false;
  const [width, height] = $('resolution').value.split('x').map(Number);
  stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { ...($('camera-device').value ? {deviceId:{exact:$('camera-device').value}} : {facingMode:'environment'}), width, height, frameRate: { ideal: 60 } } });
  video.src = ''; video.srcObject = stream; await video.play();
  const track = stream.getVideoTracks()[0]; await refreshCameras();
  sourceSettings = { source: 'camera', requested: { width, height, frameRate: 60 }, ...await lockCamera(track), outputSize: [video.videoWidth, video.videoHeight] };
  await captureReference(); paused = false; $('pause').textContent = 'Pause';
  const token = generation;
  if (typeof MediaStreamTrackProcessor === 'function') {
    try { reader = new MediaStreamTrackProcessor({ track }).readable.getReader(); }
    catch (error) { sourceSettings.trackProcessor = String(error); scheduleVideo(token); details(); return; }
    const consume = async () => {
      while (generation === token) { const item = await reader.read(); if (item.done) break; processFrame(item.value).catch(error => { pauseSource(); message(error.message, true); }); }
    }; consume().catch((error) => message(error.message, true));
  } else scheduleVideo(token);
  details(); message('Camera active. Actual settings and available manual locks are shown below.');
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
  $('roi-summary').textContent = (state.hasBaseMask ? 'Image mask ∩ ' : '') + (state.shapes.length ? `${state.shapes.filter(s => s.role === 'domain').length} domain(s), ${state.shapes.filter(s => s.role === 'hole').length} hole(s)${roiEditing ? ' · unapplied edits' : ''}` : 'Full image · no mask.');
  if(state.editing&&!editorWasActive&&matchMedia('(max-width:760px)').matches)requestAnimationFrame(()=>document.querySelector('.viewer').scrollIntoView({block:'start'})); editorWasActive=state.editing;
  $('apply-roi').disabled = !roiEditing; $('mask-toolbar').hidden=!roiEditing; $('apply-mask-top').disabled=!roiEditing;
});
$('zoom').onchange = () => { $('scene').style.width = `${Number($('zoom').value)*100}%`; };
$('centre-view').onclick = () => { const view=$('image-viewport'); view.scrollLeft=(view.scrollWidth-view.clientWidth)/2; view.scrollTop=(view.scrollHeight-view.clientHeight)/2; };
$('regions').onchange = () => roi.select($('regions').value);
$('delete-roi').onclick = () => roi.remove();
$('undo-roi').onclick = () => roi.undo();
$('add-centred').onclick = () => { try {roi.addCentred($('shape').value==='select'?'rectangle':$('shape').value);} catch(e){message(e.message,true);} };
$('apply-mask-top').onclick = () => $('apply-roi').click();
$('finish-mask-top').onclick = () => $('finish-shape').click();
$('cancel-mask-top').onclick = () => $('cancel-roi').click();
$('cancel-roi').onclick = () => sourceChange(async()=>{
  if(!appliedEditorState)return;
  roiEditing=false;roi.reset(appliedEditorState.width,appliedEditorState.height,appliedEditorState.shapes,appliedEditorState.baseMask);draftMaskSource=appliedMaskSource;
  $('roi-editor').hidden=true;$('mask-import-summary').textContent=appliedMaskSource?`${appliedMaskSource.name} · applied base restored.`:'No imported image base. White pixels are kept; black pixels are excluded.';await send('show-reference',{show:false});
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
$('previous-image').onclick = () => localSeek(Math.max(0,imageIndex-1));
$('next-image').onclick = () => localSeek(Math.min(localImages.length-1,imageIndex+1));
$('local-seek').onchange = () => localSeek($('local-seek').value);
$('image-rate').onchange = () => { sourceSettings.replayRateCapHz = Number($('image-rate').value); };
$('video-rate').onchange = () => {video.playbackRate=Number($('video-rate').value);sourceSettings.playbackRate=video.playbackRate;};
$('comparison').onchange = () => sourceChange(async () => {
  pauseSource();await imageTask?.catch(()=>{});await drain();$('export').disabled=true;$('pause').textContent='Play';
  sourceSettings.requestedComparison=$('comparison').value;
  if(source==='images')$('image-reference-controls').hidden=$('comparison').value==='consecutive';
  if(referenceInfo)await send('comparison',{mode:$('comparison').value,options:options()});
  if(source==='images')await runImage($('comparison').value==='consecutive'?Math.max(1,imageIndex):imageIndex,true);
  message($('comparison').value==='consecutive'?'Consecutive received-image pairs. Reference construction is included in age; every ROI subset is requested.':'Fixed captured reference restored.');
});
video.addEventListener('ended',()=>{if(source==='video-file'){pauseSource();$('pause').textContent='Play';updateLocalPosition();}});
$('plate').onclick = () => sourceChange(startPlate);
$('plate-load').onchange = () => sourceChange(runPlateTarget);
$('demo').onclick = () => sourceChange(startDemo);
$('camera').onclick = () => sourceChange(startCamera);
$('reference').onclick = () => sourceChange(async () => {
  if (source === 'plate') return startPlate();
  if (source === 'images') return setImageReference(imageIndex);
  pauseSource(); await drain();
  await captureReference();
  sourceSettings.reference = { kind:source, file:sourceSettings.file, timeSeconds:source === 'video-file' ? video.currentTime : null };
  if (source !== 'video-file') { paused = false; $('pause').textContent = 'Pause'; }
  else $('pause').textContent = 'Play';
});
$('pause').onclick = async () => {
  workbench.returnLive();
  if (roiEditing) { message('Apply the mask before resuming.'); return; }
  if (source === 'plate') return runPlateTarget().catch(e => message(e.message,true));
  if (!paused) { pauseSource(); $('pause').textContent = 'Play'; return; }
  try {
    await send('show-reference',{show:false});
    if (source === 'images') return playImages();
    if (source === 'video-file') { if (video.ended) { await seekVideo(video,0,sourceAbort.signal); await send('reset'); } await video.play(); }
    paused = false; $('pause').textContent = 'Pause';
  } catch(error) { message(error.message,true); }
};
$('reset').onclick = () => send('reset');
function roiMetadata() {
  return {version:appliedMaskSource?2:1,composition:appliedMaskSource?'imported binary image intersected with vector domains/holes':'union of roots; exact pixel-containment alternates domain/hole/island',baseImage:appliedMaskSource,shapes:appliedShapes};
}
$('mask-conversion').onchange = () => { $('mask-threshold-label').hidden=$('mask-conversion').value==='strict'; };
$('import-mask').onclick = () => $('mask-file').click();
$('mask-file').onchange = () => {
  const file=$('mask-file').files[0];$('mask-file').value='';if(!file)return;
  sourceChange(async()=>{
    if(!referenceInfo)throw new Error('Capture a reference before importing a mask');
    pauseSource();await drain();workbench.returnLive();
    const threshold=$('mask-conversion').value==='strict'?null:Number($('mask-threshold').value);
    const mask=await decodeMaskImage(file,referenceInfo.width,referenceInfo.height,{threshold,signal:sourceAbort.signal});
    const hash=await crypto.subtle.digest('SHA-256',await file.arrayBuffer());
    draftMaskSource={...fileInfo(file),sha256:Array.from(new Uint8Array(hash),b=>b.toString(16).padStart(2,'0')).join(''),threshold,
      convention:threshold===null?'opaque exact black/white':'rounded BT.709 luminance >= threshold and alpha >= 128'};
    roi.reset(referenceInfo.width,referenceInfo.height,[{id:'image-mask-domain',label:'Clip image mask',type:'rectangle',x:0,y:0,width:referenceInfo.width,height:referenceInfo.height}],mask);
    roiEditing=true;workspace.showPane('mask');await send('show-reference',{show:true});$('roi-editor').hidden=false;$('shape').value='select';roi.start();
    $('mask-import-summary').textContent=`${fileInfo(file).name} · ${threshold===null?'exact binary pixels':`explicit threshold ${threshold}`} · vectors clip this base image. Full image removes the base.`;
    message('Mask imported for review. Add vector holes or clipping regions, then Apply mask. White retains pixels; dimensions are unchanged.');
  });
};
$('save-mask').onclick=async()=>{
  try{if(!referenceInfo)return;const mask=roi.mask(),blob=await maskPng(mask,referenceInfo.width,referenceInfo.height);download('live-dic-roi-edited.png',blob,'image/png');}
  catch(e){message(e.message,true);}
};
$('draw').onclick = () => sourceChange(async () => { workspace.showPane('mask'); workbench.returnLive(); pauseSource(); await drain(); roiEditing = true; $('pause').textContent = 'Resume'; await send('show-reference', { show: true }); $('roi-editor').hidden = false; $('shape').value = 'select'; roi.start(); });
$('apply-roi').onclick = () => sourceChange(async () => {
  try { workbench.setReady(false); const mask = roi.mask(), ownedMask = mask?.slice() ?? null; appliedShapes = roi.shapes(); referenceInfo = (await send('roi', { mask, options: options() }, mask ? [mask.buffer] : [])).info; workbench.setMask(ownedMask); appliedMaskSource = draftMaskSource; appliedEditorState=roi.state(); await updateReferenceControls();
    roiEditing = false; roi.stop(); $('roi-editor').hidden = true; $('apply-roi').disabled = true;
    message('Mask applied. Its geometry stays editable; all complete selected subsets are used.'); details();
    if (source === 'plate') await runPlateTarget();
    else if (source === 'images') await runImage(imageIndex,true);
  } catch (error) { message(error.message, true); }
});
$('clear-roi').onclick = () => sourceChange(async () => { workbench.returnLive(); pauseSource(); await drain(); workbench.setMask(null); workbench.setReady(false); roiEditing = false; roi.reset(referenceInfo.width, referenceInfo.height); appliedShapes = []; draftMaskSource = appliedMaskSource = null; $('roi-editor').hidden = true;
  referenceInfo = (await send('roi', { mask: null, options: options() })).info; await updateReferenceControls(); appliedEditorState=roi.state(); $('mask-import-summary').textContent='Full image · no imported base mask.'; $('pause').textContent = 'Resume'; details(); if (source === 'plate') await runPlateTarget(); });
function displayOptions() { return { mode: Number($('component').value), scale: $('scale').value === 'auto' ? null : Number($('scale').value),
  opacity: Number($('opacity').value), arrows: $('arrows').checked, arrowStep: Number($('arrow-step').value), arrowGain: Number($('arrow-gain').value) }; }
for (const id of ['component', 'scale', 'opacity', 'arrows', 'arrow-step', 'arrow-gain']) $(id).onchange = () => send('display', { display: displayOptions() });
function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type })), link = document.createElement('a');
  link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function captureCurrentField() {
  pauseSource(); await drain(); $('pause').textContent = 'Play';
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
  metadata.acquisition = sourceSettings; metadata.roiGeometry = roiMetadata(); metadata.display = displayOptions();
  download('live-dic-field.csv', lines.join('\n'), 'text/csv');
  download('live-dic-metadata.json', JSON.stringify(metadata, null, 2), 'application/json');
};
$('resolution').onchange = () => { if (source === 'camera') sourceChange(startCamera); };
$('profile').onchange = () => { if (referenceInfo) message('Profile change takes effect when the reference is captured again.'); };
window.addEventListener('pagehide', () => { workbench.destroy(); sourceAbort.abort(); clearPreviousCapture(); if (localVideoUrl) URL.revokeObjectURL(localVideoUrl); stream?.getTracks().forEach((t) => t.stop()); transport?.postMessage({ type: 'destroy' }); });
window.addEventListener('error', (event) => { status.errors.push(event.message); message(event.message, true); });
window.addEventListener('unhandledrejection', (event) => { const text = String(event.reason); status.errors.push(text); message(text, true); });
if (smoke) { globalThis.liveDicReport = () => send('report'); globalThis.liveDicWorkbench = workbench; globalThis.liveDicRoi = roi; globalThis.liveDicSourceState = () => ({source,paused,loadingSource,inFlight,index:imageIndex,fileCount:localImages.length,localBusy:!!imageTask,loop:!!imageLoop,settings:sourceSettings,comparison:$('comparison').value,videoTime:video.currentTime,videoPaused:video.paused}); }
for (const id of ['custom-solver','motion-model','subset-size','subset-stride','interpolation','pyramid-levels','coarse-iterations','fine-iterations','outlier-policy','temporal-start']) $(id).onchange = () => {
  $('solver-settings').disabled = !$('custom-solver').checked;
  $('subset-stride').max = $('subset-size').value;
  if (Number($('subset-stride').value) > Number($('subset-size').value)) $('subset-stride').value = $('subset-size').value;
  try { const requested = options(); $('solver-summary').textContent = $('custom-solver').checked ? `Pending: ${requested.motionModel} · ${requested.subsetSize}/${requested.subsetStride} · ${requested.pyramidLevels} level(s). Capture reference or Apply mask to apply.` : 'Preset selected for the next reference. Current fields keep their original configuration.'; }
  catch(e) { $('solver-summary').textContent = e.message; }
};
init().catch((error) => { status.errors.push(error.message); message(error.message, true); });
if ('serviceWorker' in navigator && (!smoke || query.has('offline'))) navigator.serviceWorker.register(new URL('./sw.js', import.meta.url)).catch((error) => message(`Offline installation unavailable: ${error.message}`, true));
