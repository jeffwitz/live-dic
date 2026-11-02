import { DEFAULT_RECIPE, analyzeField, componentValues as deriveComponentValues, statistics, nearestIndex, extensometer, regionSummary, validateField } from './postprocess.js';
import { fieldArchive, importFieldArchive, projectArchive, importProjectArchive } from './archive.js';
import { extractHistory, sameAnalysis } from './history.js';
import { resultFigure } from './figure.js';
import { FieldStorage, defaultCacheBytes, fieldArrayBytes, MiB } from './field-storage.js';
import { createViewCache, createAnalysisCache } from './view-cache.js';
import { sequentialColour } from '../render/colormaps.js';

const $ = id => document.getElementById(id);
const number = value => value == null || !Number.isFinite(value) ? 'unavailable' : value.toPrecision(5);
const STRAIN_COMPONENTS = new Set(['exx','eyy','exy','vonMises','principal1','principal2']);
const PERCEPTUAL_SEQUENTIAL_MAPS = new Set(['viridis','cividis','plasma','inferno','magma']);
const ARCHIVE_IMPORT_BYTES = 256 * MiB;
const encoderBytes = value => new TextEncoder().encode(JSON.stringify(value)).byteLength;
const STATUS = { '-3': 'Excluded by ROI', '-2': 'Rejected correlation', '-1': 'Unavailable reference / bounds',
  0: 'Not converged', 1: 'Accepted', 2: 'Flagged / replaced by solver policy' };
const STATUS_COLOURS = { '-3':[160,170,178],'-2':[202,51,64],'-1':[71,86,96],0:[240,193,52],1:[36,151,110],2:[238,135,45] };
const STATUS_ORDER = [-3,-2,-1,0,1,2];
const percentileCache = new WeakMap();

function percentileBounds(analysis, component, values, valid, minimum, maximum) {
  let components = percentileCache.get(analysis);
  if (!components) { components = new Map(); percentileCache.set(analysis, components); }
  let sorted = components.get(component);
  if (!sorted) {
    sorted = Array.from(values).filter((value, index) => valid[index] && Number.isFinite(value)).sort((a, b) => a - b);
    components.set(component, sorted);
  }
  if (!sorted.length) return [null, null];
  const quantile = percentile => {
    const position = (sorted.length - 1) * percentile / 100, index = Math.floor(position);
    return sorted[index] + (position - index) * (sorted[Math.min(sorted.length - 1, index + 1)] - sorted[index]);
  };
  return [quantile(minimum), quantile(maximum)];
}

function download(name, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement('a'); link.href = url; link.download = name; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
function maskFromMetadata(metadata) {
  if (!metadata.roi) return null;
  if (metadata.roi.encoding !== 'base64-uint8-boolean') throw new Error('Unsupported mask encoding');
  return Uint8Array.from(atob(metadata.roi.data), c => c.charCodeAt(0));
}
function fieldLabel(metadata, index) {
  const pair = metadata.comparison;
  const name = item => item?.name ?? (item?.frame != null ? `Frame ${item.frame}` : null) ?? (item?.index != null ? `Image ${item.index+1}` : item?.timeSeconds != null ? `${item.timeSeconds.toFixed(3)} s` : item?.ordinal != null ? `Input ${item.ordinal}` : 'capture');
  return pair ? `${index} · ${name(pair.reference)} → ${name(pair.target)}` : `${index} · ${metadata.name ?? 'Computed field'}`;
}
function recipeFromControls() {
  return { smoothing: $('post-smoothing').value, strainSource: $('strain-source').value,
    strainMeasure: $('strain-measure').value, window: Number($('post-window').value),
    excludeFlagged: $('exclude-flagged').checked, unit: $('length-unit').value,
    unitsPerPixel: $('length-unit').value === 'px' ? 1 : Number($('length-scale').value) };
}
function setRecipe(recipe) {
  $('post-smoothing').value = recipe.smoothing; $('strain-source').value = recipe.strainSource;
  $('strain-measure').value = recipe.strainMeasure; $('post-window').value = recipe.window;
  $('exclude-flagged').checked = recipe.excludeFlagged; $('length-unit').value = recipe.unit;
  $('length-scale').value = recipe.unitsPerPixel; $('length-scale').disabled = recipe.unit === 'px';
}

/** A bounded, opt-in results library; the default live path never enters this module's analysis loop. */
export function installWorkbench({ captureCurrent, pause, error, recordChange, liveChanged, openResults, requestMemory = async()=>null, geometryDefaults = () => ({}), beforeProjectExport = async () => pause(), hasPendingResults = () => false }) {
  const entries = [], canvas = $('review-field'), overlay = $('inspection'), chart = $('profile-chart');
  const viewCache = createViewCache(deriveComponentValues, statistics);
  const analysisCache = createAnalysisCache();
  const componentValues = (analysis, component) => viewCache.get(analysis, component);
  let analysisRuns = 0;
  document.querySelector('.viewer').append($('results-workbench'));
  let nextId = 1, selected = null, analysis = null, currentMask = null, reviewing = false;
  let storage = new FieldStorage(), selectionSerial = 0, selectionBusy = false, libraryTimer = null, liveReady = false, prepared = false, captureBusy = false;
  let tool = 'probe', points = [], measurements = [], activeMeasurementId = null, nextMeasurementId = 1;
  let extraction = [], computedView = false, figureLegend = null;
  let requestId = 0, workerBusy = false, pendingAnalysis = false, workerFieldId = null, debounce = null;
  let analysisJob = null, displayed = null;
  let pendingImport = null, importSerial = 0;
  let history = null, historyAbort = null, lastChartRows=[], lastChartOptions={};
  let replaying=false,replayTimer=null,replayIds=[];
  let pendingSeek = null, seeking = false;
  function stopReplay(){replaying=false;clearTimeout(replayTimer);replayTimer=null;$('replay-results').textContent='Play results';}
  // Dragging a long recording keeps one disk load and only the newest request.
  async function seekResult(id) {
    stopReplay();pendingSeek=id;
    if(seeking)return;
    seeking=true;
    try {while(pendingSeek!==null){
      if(selectionBusy){await new Promise(resolve=>setTimeout(resolve,25));continue;}
      const next=pendingSeek;pendingSeek=null;await selectEntry(next,{preserveView:true});
    }}
    finally {seeking=false;}
  }
  let projectRevision=0, downloadedRevision=0;
  const hasUnsaved=()=>entries.length>0&&projectRevision!==downloadedRevision;
  const changed=()=>{projectRevision++;};
  const beforeUnload=event=>{rememberView();if(hasUnsaved()||hasPendingResults()){event.preventDefault();event.returnValue='';}};
  window.addEventListener('beforeunload',beforeUnload);
  const contours = new WeakMap();
  let worker = typeof Worker === 'function' ? new Worker(new URL('./analysis-worker.js', import.meta.url), { type: 'module' }) : null;
  const recording = () => $('record-fields').checked;
  const activeEntry = () => entries.find(entry => entry.id === selected);
  function clearHistory() {
    historyAbort?.abort();history=null;$('spatial-view').hidden=true;$('history-state').textContent='';$('export-history').disabled=true;
  }
  function viewSettings() {
    return { component:$('review-component').value,colourScale:$('review-scale').value,
      minimum:Number($('review-min').value),maximum:Number($('review-max').value),
      minimumPercentile:Number($('review-low-percentile').value),maximumPercentile:Number($('review-high-percentile').value),
      palette:$('review-palette').value,vectors:$('review-vectors').checked,vectorStep:Number($('review-vector-step').value),
      vectorGain:Number($('review-vector-gain').value),contour:$('review-contour').checked,tool,points,
      measurements:measurements.map(item=>({...item,points:item.points.map(point=>[...point])})),activeMeasurementId };
  }
  function restoreView(view) {
    if(!view){measurements=[];activeMeasurementId=null;points=[];tool='probe';document.querySelectorAll('[data-inspect]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.inspect===tool)));return;}
    for(const [id,key] of [['review-component','component'],['review-scale','colourScale'],['review-palette','palette']])
      if([...$(id).options].some(option=>option.value===view[key]&&!option.disabled))$(id).value=view[key];
    if([view.minimum,view.maximum].every(Number.isFinite)&&view.maximum>view.minimum){$('review-min').value=view.minimum;$('review-max').value=view.maximum;}
    if([view.minimumPercentile,view.maximumPercentile].every(value=>Number.isFinite(value)&&value>=0&&value<=100)&&view.maximumPercentile>view.minimumPercentile){
      $('review-low-percentile').value=view.minimumPercentile;$('review-high-percentile').value=view.maximumPercentile;
    }
    updateColourControlVisibility();
    $('review-vectors').checked=view.vectors===true;$('review-contour').checked=view.contour===true;
    if([2,4,8].includes(view.vectorStep))$('review-vector-step').value=view.vectorStep;
    if([1,5,20,50,100].includes(view.vectorGain))$('review-vector-gain').value=view.vectorGain;
    restoreInspection(view);
  }
  function updateColourControlVisibility() {
    const mode=$('review-scale').value;
    $('percentile-limits').hidden=mode!=='robust';
    $('manual-limits').hidden=mode!=='manual';
  }
  function restoreInspection(view) {
    const field=activeEntry()?.field;
    if(view&&['probe','line','gauge','region'].includes(view.tool)&&Array.isArray(view.points)&&view.points.length<=2&&
      view.points.every(point=>Array.isArray(point)&&point.length===2&&point.every(Number.isFinite)&&
        point[0]>=0&&point[1]>=0&&point[0]<field.width&&point[1]<field.height)) {
      tool=view.tool;points=view.points.map(point=>[...point]);
      document.querySelectorAll('[data-inspect]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.inspect===tool)));
    }
    if(Array.isArray(view?.measurements)) {
      measurements=view.measurements.filter(item=>item&&typeof item.id==='string'&&typeof item.name==='string'
        &&['point','probe','line','gauge','region'].includes(item.type)&&Array.isArray(item.points)
        &&item.points.length===(['point','probe'].includes(item.type)?1:2)
        &&item.points.every(point=>Array.isArray(point)&&point.length===2&&point.every(Number.isFinite)
          &&point[0]>=0&&point[1]>=0&&point[0]<field.width&&point[1]<field.height))
        .map(item=>({...item,points:item.points.map(point=>[...point])}));
      nextMeasurementId=Math.max(1,...measurements.map(item=>Number(item.id.replace(/^m/,''))+1).filter(Number.isSafeInteger));
      activeMeasurementId=measurements.some(item=>item.id===view.activeMeasurementId)?view.activeMeasurementId:null;
      if(activeMeasurementId){const item=measurements.find(value=>value.id===activeMeasurementId);tool=item.type==='point'?'probe':item.type;points=item.points.map(point=>[...point]);document.querySelectorAll('[data-inspect]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.inspect===tool)));}
    } else { measurements=[];activeMeasurementId=null; }
  }

  function memoryBytes(extra = []) {
    const masks = new Set(); let bytes = 0;
    for (const entry of [...entries, ...extra]) {
      bytes += fieldArrayBytes(entry.field) + entry.metadataBytes;
      if (entry.mask && !masks.has(entry.mask)) { masks.add(entry.mask); bytes += entry.mask.byteLength; }
    }
    return bytes;
  }
  function normalizeMetadata(metadata) {
    return { ...metadata, samples: metadata.samples?.slice(-1), sampleScope: 'this field',
      roi: metadata.roi ? { ...metadata.roi, data: undefined, encoding: 'roi_mask.npy' } : null };
  }
  function rememberView() {
    const entry=activeEntry();if(!reviewing||!entry||!analysis)return;
    const view=viewSettings();
    if(JSON.stringify(entry.metadata.workbench?.view)!==JSON.stringify(view)){
      entry.metadata={...entry.metadata,workbench:{view}};entry.metadataBytes=encoderBytes(entry.metadata);changed();updateLibrary();
    }
  }
  function storageBlocked(){const info=storage.report();return info.unsavedFields>0&&(info.diskAvailable||info.residentBytes>info.memoryLimitBytes);}
  function updateLibrary(immediate = true) {
    if(!immediate&&!reviewing){if(!libraryTimer)libraryTimer=setTimeout(()=>{libraryTimer=null;updateLibrary();},250);return;}
    clearTimeout(libraryTimer);libraryTimer=null;
    const select = $('saved-fields'); select.replaceChildren();
    if (!entries.length) select.add(new Option('No saved fields', ''));
    else if(!reviewing)select.add(new Option('Choose a field to inspect…',''));
    const index=reviewing?entries.findIndex(entry=>entry.id===selected):-1;
    const start=Math.max(0,Math.min(entries.length-101,(index<0?entries.length-1:index)-50));
    for (const entry of entries.slice(start,start+101)) select.add(new Option(entry.label, entry.id));
    select.value = reviewing?selected??'':'';
    const info=storage.report();
    $('library-summary').textContent = `${entries.length} field(s) · ${(memoryBytes()/MiB).toFixed(1)} MiB resident / ${(info.memoryLimitBytes/MiB).toFixed(0)} MiB cache target · ${(analysisCache.report().bytes/MiB).toFixed(1)} MiB derived replay cache (32 MiB maximum) · ${info.diskFields} saved on browser disk. ${!entries.length?'No stored results.':hasUnsaved()?'Export for an independent backup; settings have unsaved changes.':'Project download requested or session settings saved; verify your backup.'} Images are not recorded.`;
    $('storage-summary').textContent=info.warning?`${info.warning}. ${info.unsavedFields} field(s) only in RAM; do not close before exporting/retrying.`:info.diskAvailable?'Disk-backed storage. Old cached fields are reloaded on demand. Browser storage is not an independent backup.':'Opening browser storage on first saved field; RAM remains bounded if disk is unavailable.';
    $('retry-storage').hidden=!info.unsavedFields&&!info.warning;$('save-session').disabled=!entries.length||!info.diskAvailable;
    $('freeze-field').disabled=!liveReady||captureBusy||storage.readOnly||storageBlocked();$('record-fields').disabled=!prepared||storage.readOnly||storageBlocked();
    $('solve-results').disabled=!entries.length;$('results-empty').hidden=!!entries.length;
    $('review-component').disabled=!entries.length;
    $('delete-field').title=storage.readOnly?'Saved sessions are read-only; close the session to start a new recording.':'';
    $('clear-fields').disabled = !entries.length; $('delete-field').disabled = !entries.length||storage.readOnly;
    $('save-project').disabled=!entries.length;
    $('previous-result').disabled=index<=0;$('next-result').disabled=index<0||index>=entries.length-1;
    $('result-seek').max=Math.max(0,entries.length-1);$('result-seek').value=Math.max(0,index);$('result-seek').disabled=!entries.length;
    $('result-order').textContent=index>=0?`${index+1} / ${entries.length} · saved order`:`${entries.length} saved fields`;
    $('result-number').max=Math.max(1,entries.length);$('result-number').value=index>=0?index+1:'';$('result-number').disabled=!entries.length;
    const base=activeEntry()??entries[0],matching=base?entries.filter(entry=>sameAnalysis(base,entry)).length:0;
    $('replay-results').disabled=matching<2;
    $('result-playback-state').textContent=!entries.length?'No calculated fields. Start calculation in Solve, or import saved results.':`${matching} compatible field(s) for playback · saved order, independent of acquisition time. Playback never runs correlation.`;
  }
  async function add(field, metadata = {}, { select = false, mask = currentMask, recipe = null, ramOnly = false } = {}) {
    validateField(field);
    // A single-field archive carries its own timing sample. The live timing log remains in direct export.
    metadata = normalizeMetadata(metadata);
    const entry = { id: nextId++, field, metadata, mask, recipe, metadataBytes: encoderBytes(metadata) };
    if (mask && mask.length !== field.width*field.height) throw new Error('Saved field mask dimensions differ');
    entry.label = fieldLabel(metadata, entry.id);
    const result=await storage.addMany([entry], {ramOnly});entries.push(entry);changed();updateLibrary(select);
    if(result.failure){$('record-fields').checked=false;recordChange(false);pause();error(`Results storage stopped: ${result.failure}. The completed field is retained in RAM. Export or retry disk storage before continuing.`);updateLibrary();}
    if (select) await selectEntry(entry.id,{preserveView:false});
    return entry;
  }
  function returnLive(restore = false) {
    pendingSeek=null;
    stopReplay();rememberView();
    historyAbort?.abort();history=null;
    reviewing = false; selectionSerial++;selectionBusy=false;storage.pin(null);storage.evict();requestId++; pendingAnalysis = false; clearTimeout(debounce);
    analysis=null;displayed=null;workerFieldId=null;lastChartRows=[];lastChartOptions={};canvas.dataset.analysisState='idle';canvas.setAttribute('aria-busy','false');worker?.postMessage({clear:true});
    $('saved-solver-summary').textContent='';
    $('results-workbench').hidden = true; canvas.hidden = true; overlay.hidden = true;
    $('field').hidden = false; $('return-live').hidden = true;
    $('roi').hidden=false;$('review-controls').hidden=true;$('live-appearance').hidden=false;$('cursor-value').hidden=true;
    document.querySelector('.legend-bar').style.background='';
    $('review-colour-controls').hidden=true;
    liveChanged(false, restore);updateLibrary();
  }
  async function selectEntry(id,{replay=false,preserveView=reviewing}={}) {
    if(!replay)stopReplay();
    const entry = entries.find(value => value.id === Number(id)); if (!entry) return;
    const previous=activeEntry(),savedView=entry.metadata.workbench?.view;
    const inspection=previous&&sameAnalysis(previous,entry)?viewSettings():savedView??(preserveView?viewSettings():null);
    rememberView();
    historyAbort?.abort();history=null;$('spatial-view').hidden=true;$('history-state').textContent='';
    pause(); selected = entry.id; analysis = null; reviewing = true; computedView = false; points = []; extraction = [];
    const token=++selectionSerial;selectionBusy=true;storage.pin(entry.id);requestId++;pendingAnalysis=false;clearTimeout(debounce);
    canvas.dataset.analysisState='pending';canvas.setAttribute('aria-busy','true');pendingLabel(`Loading field ${entry.id}`);$('review-controls').inert=true;$('review-colour-controls').inert=true;
    for(const key of ['save-npz','save-processed-csv','save-figure','measurement-inputs','apply-recipe-analysis'])$(key).disabled=true;
    try {await storage.load(entry);} catch(e){if(token===selectionSerial){selectionBusy=false;$('review-controls').inert=false;$('review-colour-controls').inert=false;canvas.dataset.analysisState='failed';canvas.setAttribute('aria-busy','false');$('result-render-state').textContent=$('analysis-state').textContent=`Unable to load field ${entry.id}. ${displayed ? `Still showing field ${displayed.id}. ` : ''}${e.message}`;error(e.message);}return;}
    if(token!==selectionSerial)return;selectionBusy=false;$('review-controls').inert=false;$('review-colour-controls').inert=false;
    $('saved-fields').value = entry.id; $('results-workbench').hidden = false;
    $('return-live').hidden = false; canvas.hidden = false; overlay.hidden = false; $('field').hidden = true;
    for(const id of ['measure-ax','measure-bx'])$(id).max=entry.field.width-1;
    for(const id of ['measure-ay','measure-by'])$(id).max=entry.field.height-1;
    $('roi').hidden=true;$('review-controls').hidden=false;$('live-appearance').hidden=true;$('recording-panel').open=false;
    updateLibrary();$('processing-batch-state').textContent='';
    for(const option of $('review-component').options)if(option.dataset.coefficient!==undefined){
      const i=Number(option.dataset.coefficient);option.disabled=i>=(entry.field.config.motionModel==='quadratic'?12:6)||(entry.field.config.motionModel==='translation'&&![0,3].includes(i));
    }
    if($('review-component').selectedOptions[0].disabled)$('review-component').value='magnitude';
    setRecipe(entry.recipe ?? { ...DEFAULT_RECIPE, strainSource: entry.field.config.motionModel === 'translation' ? 'vsg' : 'direct' });
    restoreView(inspection);
    $('compare-computed').textContent = 'Show computed field';
    liveChanged(true); openResults(); scheduleAnalysis(true);
  }
  function analysisComplete(id, result, failure, cached = false) {
    if (!cached) {
      workerBusy = false;
      if (result && analysisJob && entries.includes(analysisJob.entry)) analysisCache.set(analysisJob.key, result);
      analysisJob = null;
    }
    if (id === requestId && reviewing) {
      canvas.setAttribute('aria-busy','false');canvas.dataset.analysisState=failure?'failed':'ready';
      if (failure) { analysis = null; $('result-render-state').textContent = $('analysis-state').textContent = `Requested analysis failed. ${displayed ? `Still showing field ${displayed.id}. ` : ''}${failure}`;$('cursor-value').hidden=false;$('cursor-value').textContent=`Derived analysis unavailable: ${failure}. Stored coefficients are preserved.`; error(failure); }
      else {
        analysis = { ...result, field: activeEntry().field }; if (!computedView) {
          if(JSON.stringify(activeEntry().recipe)!==JSON.stringify(result.recipe))changed();
          activeEntry().recipe = result.recipe;updateLibrary();
        }
        $('analysis-state').textContent = cached ? 'Saved analysis reused from replay cache. No correlation or derived recalculation.' : `Analysis ${result.processingWallMs.toFixed(1)} ms, outside the live solver. Computed arrays preserved.`;
        $('processing-summary').textContent = result.recipe.smoothing === 'none' && result.recipe.strainSource === 'direct'
          ? 'Computed displacement and local model gradients. Solver smoothing, if enabled during correlation, is already part of these values.'
          : `${result.recipe.smoothing} displacement smoothing · ${result.recipe.strainSource === 'vsg' ? 'VSG strain' : 'direct gradients'} · ${result.recipe.window} × ${result.recipe.window} grid points · nominal displacement span ${result.smoothingSpanPx} px / strain span ${result.strainSpanPx} px. No filling across gaps.`;
        render();
      }
      for (const id of ['save-npz','save-processed-csv','save-figure','measurement-inputs','apply-recipe-analysis']) $(id).disabled = !analysis;
      $('apply-recipe-analysis').disabled=!analysis||computedView;
    }
    if (pendingAnalysis && reviewing) runAnalysis();
  }
  if (worker) {
    worker.onmessage = ({ data }) => analysisComplete(data.id, data.analysis, data.error);
    worker.onerror = event => {worker?.terminate();worker=null;workerFieldId=null;pendingAnalysis=false;analysisComplete(requestId,null,`Analysis worker stopped: ${event.message}. Retry a setting to use the main-thread fallback.`);};
  }
  function runAnalysis() {
    if (!reviewing || !activeEntry()) return;
    if(selectionBusy){pendingAnalysis=true;return;}
    const entry = activeEntry(), id = requestId, recipe = computedView
      ? { ...DEFAULT_RECIPE, strainSource: entry.field.config.motionModel === 'translation' ? 'vsg' : 'direct', unit: $('length-unit').value, unitsPerPixel: Number($('length-scale').value) }
      : recipeFromControls();
    const key = JSON.stringify([entry.id, recipe]), cached = analysisCache.get(key);
    if (cached) { pendingAnalysis = false; analysisComplete(id, cached, null, true); return; }
    if (workerBusy) { pendingAnalysis = true; return; }
    pendingAnalysis = false; workerBusy = true; analysisRuns++;
    analysisJob = { key, entry };
    if (worker) {
      const payload = { id, recipe };
      if (workerFieldId !== entry.id) { payload.field = entry.field; payload.mask = entry.mask; workerFieldId = entry.id; }
      worker.postMessage(payload);
    } else {
      try { analysisComplete(id, analyzeField(entry.field, recipe, entry.mask)); }
      catch (e) { analysisComplete(id, null, e.message); }
    }
  }
  function pendingLabel(action) {
    $('result-render-state').textContent = $('analysis-state').textContent = `${action}… ${displayed ? `Still showing field ${displayed.id} (${displayed.component}); inspection and exports wait for the requested analysis.` : 'Preparing the first field.'}`;
    $('cursor-value').hidden=true;
    $('probe-value').textContent='Measurements will update when the requested field is ready.';
    $('save-profile').disabled=true;
  }
  function scheduleAnalysis(immediate = false) {
    clearHistory();
    requestId++; pendingAnalysis = true; clearTimeout(debounce); analysis = null;
    canvas.dataset.analysisState='pending';canvas.setAttribute('aria-busy','true');pendingLabel(`Preparing field ${selected}`);
    for (const id of ['save-npz','save-processed-csv','save-figure','measurement-inputs','apply-recipe-analysis']) $(id).disabled = true;
    if(immediate)runAnalysis();else debounce = setTimeout(runAnalysis, 60);
  }
  function format(value, unit) { if (value == null) return 'unavailable'; return `${number(STRAIN_COMPONENTS.has($('review-component').value) ? value*1e6 : value)} ${STRAIN_COMPONENTS.has($('review-component').value) ? 'µε' : unit}`; }
  function render() {
    figureLegend = null;
    if (!analysis) return;
    const component = $('review-component').value, { values, valid, unit, source, statistics:stats } = viewCache.get(analysis, component);
    document.querySelector('.legend').hidden=false;$('live-vector-key').hidden=true;
    $('review-colour-controls').hidden=component==='status';
    document.querySelector('.gray-key').textContent='Grey: unavailable / rejected';
    $('quantity-source').hidden=!source;$('quantity-source').textContent=source?`Computed coefficient (${unit}). Workbench displacement smoothing and VSG do not change these values. The solver's original spatial policy still applies.`:'';
    const field = analysis.field, [gw,gh] = field.grid;
    const nonnegative = ['magnitude','vonMises','iterations'].includes(component);
    const scaleMode=$('review-scale').value,factor=STRAIN_COMPONENTS.has(component)?1e6:1;
    const lowPercentile=Number($('review-low-percentile').value),highPercentile=Number($('review-high-percentile').value);
    let low,high;
    if(scaleMode==='full'){low=stats.min;high=stats.max;}
    else if(scaleMode==='manual'){low=stats.min;high=stats.max;}
    else {
      if(!Number.isFinite(lowPercentile)||!Number.isFinite(highPercentile)||lowPercentile<0||lowPercentile>100||highPercentile<0||highPercentile>100||highPercentile<=lowPercentile){
        $('analysis-state').textContent='Choose percentiles from 0 to 100, with the maximum greater than the minimum.';return;
      }
      [low,high]=lowPercentile===2&&highPercentile===98?[stats.p02,stats.p98]:percentileBounds(analysis,component,values,valid,lowPercentile,highPercentile);
    }
    low??=0;high??=1;
    if(scaleMode!=='manual'&&!nonnegative){const limit=Math.max(Math.abs(low),Math.abs(high),1e-12);low=-limit;high=limit;}
    if(scaleMode==='manual'){
      low=Number($('review-min').value)/factor;high=Number($('review-max').value)/factor;
      if(!Number.isFinite(low)||!Number.isFinite(high)||high<=low){$('analysis-state').textContent='The clim maximum must be greater than its minimum, with finite values.';return;}
    } else {$('review-min').value=Number((low*factor).toPrecision(6));$('review-max').value=Number((high*factor).toPrecision(6));}
    if(high-low<Math.max(1e-14,Math.abs(high)*1e-12)){const pad=Math.max(1e-12,Math.abs(high)*.01);low-=pad;high+=pad;}
    const palette=$('review-palette').value==='auto'?(nonnegative?'sequential':'diverging'):$('review-palette').value;
    function colour(value){
      const t=Math.max(0,Math.min(1,(value-low)/(high-low)));
      if(PERCEPTUAL_SEQUENTIAL_MAPS.has(palette))return sequentialColour(palette,t);
      if(palette==='gray')return Array(3).fill(Math.round(24+220*t));
      if(palette==='sequential'){
        const stops=[[21,57,99],[31,132,138],[241,206,95]],position=t*2,index=Math.min(1,Math.floor(position)),f=position-index;
        return stops[index].map((v,k)=>Math.round(v+(stops[index+1][k]-v)*f));
      }
      const f=Math.max(0,Math.min(1,value<0?-value/Math.max(1e-12,-low):value/Math.max(1e-12,high))),endpoint=value<0?[39,91,171]:[187,56,54];
      return endpoint.map(v=>Math.round(247+(v-247)*f));
    }
    const small = typeof OffscreenCanvas==='function'?new OffscreenCanvas(gw,gh):Object.assign(document.createElement('canvas'),{width:gw,height:gh});
    const context = small.getContext('2d'), image = context.createImageData(gw, gh);
    for (let i=0;i<values.length;i++) {
      let rgb = [71,86,96];
      if (component === 'status') rgb = STATUS_COLOURS[Math.round(values[i])] ?? rgb;
      else if (valid[i]) rgb=colour(values[i]);
      image.data.set([...rgb,255],i*4);
    }
    context.putImageData(image,0,0);
    // Publish only a completed analysis. Resizing clears a canvas, so keep the
    // previous raster throughout disk/worker waits and resize in this draw task.
    if(canvas.width!==field.width||canvas.height!==field.height){canvas.width=field.width;canvas.height=field.height;}
    if(overlay.width!==field.width||overlay.height!==field.height){overlay.width=field.width;overlay.height=field.height;}
    const ctx = canvas.getContext('2d'); ctx.fillStyle = '#263b47'; ctx.fillRect(0,0,canvas.width,canvas.height); ctx.imageSmoothingEnabled = false;
    // Samples occupy a stride-sized cell centred on the actual reference subset centre.
    const offset = (field.subsetSize-field.subsetStride)/2;
    ctx.drawImage(small,offset,offset,gw*field.subsetStride,gh*field.subsetStride);
    if($('review-contour').checked){
      const mask=activeEntry().mask;
      if(mask){
        if(!contours.has(mask)){
          const path=new Path2D(),w=field.width,h=field.height;
          for(let y=0;y<h;y++)for(let x=0;x<w;x++){const i=y*w+x;if(!mask[i])continue;
            if(x===0||!mask[i-1]){path.moveTo(x,y);path.lineTo(x,y+1);}
            if(x===w-1||!mask[i+1]){path.moveTo(x+1,y);path.lineTo(x+1,y+1);}
            if(y===0||!mask[i-w]){path.moveTo(x,y);path.lineTo(x+1,y);}
            if(y===h-1||!mask[i+w]){path.moveTo(x,y+1);path.lineTo(x+1,y+1);}
          }contours.set(mask,path);
        }
        ctx.strokeStyle='#ffffff';ctx.lineWidth=Math.max(1,field.width/1200);ctx.stroke(contours.get(mask));
      }
    }
    if($('review-vectors').checked){
      const step=Number($('review-vector-step').value),gain=Number($('review-vector-gain').value),path=new Path2D();
      for(let y=0;y<gh;y+=step)for(let x=0;x<gw;x+=step){const i=y*gw+x;if(!analysis.valid[i])continue;
        const ax=x*field.subsetStride+field.subsetSize/2,ay=y*field.subsetStride+field.subsetSize/2;
        const dx=analysis.fields.u[i]/analysis.recipe.unitsPerPixel*gain,dy=analysis.fields.v[i]/analysis.recipe.unitsPerPixel*gain,length=Math.hypot(dx,dy);if(length<.01)continue;
        const bx=ax+dx,by=ay+dy,head=Math.min(length*.4,field.width/300),ux=dx/length,uy=dy/length;
        path.moveTo(ax,ay);path.lineTo(bx,by);path.moveTo(bx-head*ux+head*.45*uy,by-head*uy-head*.45*ux);path.lineTo(bx,by);path.lineTo(bx-head*ux-head*.45*uy,by-head*uy+head*.45*ux);
      }
      ctx.strokeStyle='#173b4c';ctx.lineWidth=Math.max(3,field.width/700);ctx.stroke(path);ctx.strokeStyle='#fff';ctx.lineWidth=Math.max(1,field.width/1800);ctx.stroke(path);
    }
    $('low').textContent = component==='status' ? 'Excluded / failed' : format(low,unit);
    $('high').textContent = component==='status' ? 'Accepted / flagged' : format(high,unit);
    document.querySelector('.legend').classList.toggle('status',component==='status');
    document.querySelector('.legend').classList.toggle('magnitude',nonnegative);
    document.querySelector('.legend-bar').style.background=component==='status'
      ? `linear-gradient(90deg,${STATUS_ORDER.map((code,i)=>`rgb(${STATUS_COLOURS[code].join(',')}) ${i*100/6}% ${(i+1)*100/6}%`).join(',')})`
      : `linear-gradient(90deg,${Array.from({length:11},(_,i)=>`rgb(${colour(low+(high-low)*i/10).join(',')}) ${i*10}%`).join(',')})`;
    const clipped = Array.from(values).reduce((n,v,i)=>n+ +(valid[i] && (v<low||v>high)),0);
    figureLegend = component==='status'
      ? {categories:STATUS_ORDER.map(code=>({label:`${code} · ${STATUS[code]}`,colour:`rgb(${STATUS_COLOURS[code].join(',')})`}))}
      : {low:format(low,unit),high:format(high,unit),stops:Array.from({length:11},(_,i)=>[i/10,`rgb(${colour(low+(high-low)*i/10).join(',')})`])};
    const summary = $('field-summary'); summary.replaceChildren();
    const scaleDescription=scaleMode==='robust'?`${lowPercentile}–${highPercentile}% percentiles`:scaleMode==='full'?'full range':'manual clim';
    const summaryRows=component==='status'?Object.entries(STATUS).map(([code,label])=>[label,field.status.reduce((n,status)=>n+ +(status===Number(code)),0)]):[['Valid values',`${stats.count} / ${values.length}`],['Minimum',format(stats.min,unit)],['Mean',format(stats.mean,unit)],['Maximum',format(stats.max,unit)],['Colour clipping',`${clipped} values · ${scaleDescription}`]];
    for (const [label,value] of summaryRows) {
      const box = document.createElement('div'), strong = document.createElement('strong'); box.textContent = label; strong.textContent = value; box.append(strong); summary.append(box);
    }
    displayed={id:selected,component};canvas.dataset.displayedField=String(selected);
    $('result-render-state').textContent=`Showing field ${selected} · ${$('review-component').selectedOptions[0].textContent}`;
    const entry=activeEntry();$('result-title').textContent=entry.label;
    const requested=field.roiSelected?field.roiSelected.reduce((n,v)=>n+ +!!v,0):field.status.reduce((n,v)=>n+ +(v!==-3),0);
    const accepted=field.status.reduce((n,v)=>n+ +(v>=.5),0), fraction=requested?accepted/requested:null;
    const pair=entry.metadata.comparison,referenceName=pair?.reference?.name?.split('/').at(-1),targetName=pair?.target?.name?.split('/').at(-1);
    $('saved-solver-summary').textContent=`Saved field ${selected}${referenceName&&targetName?` · ${referenceName} → ${targetName}`:''}: ${field.config.motionModel} · ${field.subsetSize}/${field.subsetStride} · initialization: ${field.config.initialization??'unspecified'} · ${accepted} / ${requested} selected subsets accepted${fraction===null?'':` (${(fraction*100).toFixed(1)}%)`}. ${fraction!==null&&fraction<.1?'Most subsets were rejected by the solver. Post-processing does not recover them. Recalculate in Solve with suitable initialization.':'Changes in Solve affect new calculations only.'}`;
    $('result-provenance').textContent=`${field.width} × ${field.height} · ${field.config.motionModel} · ${field.subsetSize}/${field.subsetStride} · solver spatial policy: ${field.config.outlierPolicy ?? 'unspecified'} · ${entry.metadata.comparison?.mode ?? 'imported'} reference. No image is stored behind this field.`;
    updateInspection();rememberView();
  }
  function pointPosition(index) {
    const f = analysis.field;return [(index%f.grid[0])*f.subsetStride+(f.subsetSize-1)/2,Math.floor(index/f.grid[0])*f.subsetStride+(f.subsetSize-1)/2];
  }
  const activeMeasurement=()=>measurements.find(item=>item.id===activeMeasurementId)??null;
  function renderMeasurementList() {
    const list=$('measurement-list');list.replaceChildren();
    if(!measurements.length){const empty=document.createElement('li');empty.className='hint';empty.textContent='No saved measurements yet.';list.append(empty);}
    for(const item of measurements) {
      const row=document.createElement('li');row.className='measurement-item';
      const select=document.createElement('button');select.type='button';select.className='measurement-select';select.setAttribute('aria-pressed',String(item.id===activeMeasurementId));
      select.textContent=`${item.name} · ${item.type==='gauge'?'Extensometer':item.type==='region'?'Zone':item.type==='line'?'Line':'Point'}${item.component?` · ${item.component}`:''}`;
      select.onclick=()=>{stopReplay();activeMeasurementId=item.id;tool=item.type==='point'?'probe':item.type;points=item.points.map(point=>[...point]);$('measure-name').value=item.name;const previous=$('review-component').value;if(item.component&&[...$('review-component').options].some(option=>option.value===item.component&&!option.disabled))$('review-component').value=item.component;document.querySelectorAll('[data-inspect]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.inspect===tool)));rememberView();if(previous!==$('review-component').value)render();else updateInspection();};
      const remove=document.createElement('button');remove.type='button';remove.className='measurement-remove';remove.setAttribute('aria-label',`Delete ${item.name}`);remove.textContent='×';
      remove.onclick=()=>{stopReplay();clearHistory();measurements=measurements.filter(value=>value.id!==item.id);if(activeMeasurementId===item.id){activeMeasurementId=null;points=[];}changed();rememberView();updateInspection();};
      row.append(select,remove);list.append(row);
    }
    $('clear-all-measurements').disabled=!measurements.length;
    $('update-measurement').disabled=!activeMeasurementId;
    $('delete-measurement').disabled=!activeMeasurementId;
    $('extract-history').disabled=!analysis||!measurements.length;
  }
  function updateInspection() {
    const ctx = overlay.getContext('2d');ctx.clearRect(0,0,overlay.width,overlay.height);extraction=[];
    renderMeasurementList();
    document.querySelectorAll('.second-endpoint').forEach(label=>label.hidden=tool==='probe');
    for(const id of ['measure-bx','measure-by']){$(id).required=tool!=='probe';$(id).disabled=tool==='probe';}
    for(const [i,ids] of [[0,['measure-ax','measure-ay']],[1,['measure-bx','measure-by']]])
      for(const [axis,id] of ids.entries())$(id).value=points[i]?.[axis]??'';
    if(analysis) {
      const f=analysis.field,colours=['#ffd166','#06d6a0','#ef476f','#118ab2','#9b5de5','#f15bb5','#00bbf9','#f9844a'];
      measurements.forEach((item,index)=>{
        const snapped=item.points.map(([x,y])=>{const i=nearestIndex(f,x,y);return i<0?null:pointPosition(i);});if(snapped.some(point=>!point))return;
        const [a,b]=snapped,selected=item.id===activeMeasurementId;ctx.strokeStyle=colours[index%colours.length];ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=Math.max(selected?4:2,f.width/(selected?500:900));ctx.setLineDash(selected?[]:[Math.max(2,f.width/1200),Math.max(2,f.width/1200)]);
        ctx.beginPath();if(item.type==='region')ctx.rect(Math.min(a[0],b[0])+.5,Math.min(a[1],b[1])+.5,Math.abs(b[0]-a[0]),Math.abs(b[1]-a[1]));else if(snapped.length===1)ctx.arc(a[0]+.5,a[1]+.5,Math.max(5,f.width/260),0,2*Math.PI);else{ctx.moveTo(a[0]+.5,a[1]+.5);ctx.lineTo(b[0]+.5,b[1]+.5);}ctx.stroke();ctx.setLineDash([]);
        for(const point of snapped){ctx.beginPath();ctx.arc(point[0]+.5,point[1]+.5,Math.max(4,f.width/350),0,2*Math.PI);ctx.fill();}
        ctx.font=`${Math.max(13,f.width/180)}px system-ui`;ctx.lineWidth=Math.max(3,f.width/1000);ctx.strokeStyle='#152b36';ctx.strokeText(item.name,a[0]+9,a[1]-9);ctx.fillStyle='#fff';ctx.fillText(item.name,a[0]+9,a[1]-9);
      });
    }
    if (!analysis || !points.length) { $('cursor-value').hidden=true;$('probe-value').textContent='Click the field above to place a measurement.'; drawChart([]);$('save-profile').disabled=true;return; }
    const component=$('review-component').value,{values,valid,unit}=componentValues(analysis,component),f=analysis.field;
    const indices=points.map(([x,y])=>nearestIndex(f,x,y)),snapped=indices.map(i=>i<0?null:pointPosition(i));
    ctx.strokeStyle='#fff';ctx.fillStyle='#153c48';ctx.lineWidth=Math.max(2,f.width/600);
    for(const point of snapped)if(point){ctx.beginPath();ctx.arc(point[0]+.5,point[1]+.5,Math.max(4,f.width/300),0,2*Math.PI);ctx.fill();ctx.stroke();}
    if(tool==='probe') {
      const i=indices[0];
      if(i<0){$('probe-value').textContent='Outside the measured grid.';$('cursor-value').hidden=true;return;}
      $('probe-value').textContent=`Subset ${i} · x=${snapped[0][0].toFixed(1)}, y=${snapped[0][1].toFixed(1)} px\n${STATUS[Math.round(f.status[i])] ?? 'Unknown status'} · ${f.iterations[i]} iterations\n${component}: ${valid[i]?format(values[i],unit):'unavailable — not a valid measurement'}`;
      extraction=[{distance:0,x:snapped[0][0],y:snapped[0][1],value:valid[i]?values[i]:null,status:f.status[i]}];
    } else if(points.length===2 && snapped.every(Boolean)) {
      const [a,b]=snapped;
      ctx.beginPath();if(tool==='region')ctx.rect(Math.min(a[0],b[0])+.5,Math.min(a[1],b[1])+.5,Math.abs(b[0]-a[0]),Math.abs(b[1]-a[1]));
      else {ctx.moveTo(a[0]+.5,a[1]+.5);ctx.lineTo(b[0]+.5,b[1]+.5);}ctx.stroke();
      if(tool==='region') {
        const summary=regionSummary(analysis,component,a,b);
        $('probe-value').textContent=`Rectangle · ${summary.count} valid ${component} values / ${summary.selected} selected / ${summary.total} grid positions\nMean ${format(summary.mean,unit)} · standard deviation ${format(summary.standardDeviation,unit)}\nMin ${format(summary.min,unit)} · max ${format(summary.max,unit)}\nEqually weighted subset centres; holes and invalid values are excluded. No area integral.`;
        if(component==='status')$('probe-value').textContent=`Rectangle · ${summary.selected} selected / ${summary.total} grid positions\n${Object.entries(summary.statusCounts).map(([status,count])=>`${STATUS[status]}: ${count}`).join('\n')}\nStatus codes are categories, not a scalar mean.`;
        extraction=component==='status'?Object.entries(summary.statusCounts).map(([status,count])=>({status:Number(status),count})):[{x0:summary.bounds[0],y0:summary.bounds[1],x1:summary.bounds[2],y1:summary.bounds[3],valid:summary.count,selected:summary.selected,total:summary.total,mean:summary.mean,standardDeviation:summary.standardDeviation,minimum:summary.min,maximum:summary.max,p02:summary.p02,p98:summary.p98}];
      } else if(tool==='gauge') {
        const gauge=extensometer(analysis,a,b);
        $('probe-value').textContent=gauge?`Gauge L₀ = ${number(gauge.referenceLength)} ${gauge.unit}\nΔL = ${number(gauge.extension)} ${gauge.unit} · ΔL/L₀ = ${number(gauge.engineeringStrain*1e6)} µε\nEndpoint distance strain; not a local tensor component.`:'The gauge needs two distinct accepted endpoints. Invalid values are never filled.';
        if(gauge)extraction=[{referenceLength:gauge.referenceLength,deformedLength:gauge.deformedLength,extension:gauge.extension,engineeringStrain:gauge.engineeringStrain}];
      } else {
        const steps=Math.max(1,Math.ceil(Math.hypot(b[0]-a[0],b[1]-a[1])/f.subsetStride*2)),seen=new Set();
        for(let n=0;n<=steps;n++){const t=n/steps,x=a[0]+t*(b[0]-a[0]),y=a[1]+t*(b[1]-a[1]),i=nearestIndex(f,x,y);if(i<0||seen.has(i))continue;seen.add(i);const p=pointPosition(i);
          extraction.push({distance:Math.hypot(p[0]-a[0],p[1]-a[1])*analysis.recipe.unitsPerPixel,x:p[0],y:p[1],value:valid[i]?values[i]:null,status:f.status[i]});}
        $('probe-value').textContent=`${extraction.length} actual subset samples · ${component} (${unit}). Invalid samples create gaps; no field interpolation.`;
      }
    } else $('probe-value').textContent='Choose the second endpoint on the field.';
    $('cursor-value').hidden=false;$('cursor-value').textContent=$('probe-value').textContent;
    if(history){
      const current=activeMeasurement(),rows=history.rows.filter(row=>row.measurementId===current?.id);
      extraction=rows;
      const sampleIndex=current?.type==='line'?Math.floor(Math.max(0,...rows.map(row=>row.sampleIndex))/2):0;
      const series=rows.filter(row=>row.sampleIndex===sampleIndex),axisKey=series.some(row=>row.mediaTimeSeconds!=null)?'mediaTimeSeconds':series.some(row=>row.sourceIndex!=null)?'sourceIndex':null;
      const origin=axisKey?series.find(row=>row[axisKey]!=null)?.[axisKey]??0:series[0]?.order??1;
      const temporal=series.map(row=>({...row,distance:axisKey&&row[axisKey]!=null?row[axisKey]-origin:row.order-origin}));
      drawChart(temporal,{axisLabel:'Acquisition time (s), frame index or saved-field order',gauge:current?.type==='gauge'});
    }
    else {
      const regionBounds=tool==='region'&&points.length===2&&snapped.every(Boolean)?[Math.min(snapped[0][0],snapped[1][0]),Math.min(snapped[0][1],snapped[1][1]),Math.max(snapped[0][0],snapped[1][0]),Math.max(snapped[0][1],snapped[1][1])]:null;
      drawChart(tool==='line'?extraction:[],regionBounds?{regionBounds}:{});
    }$('save-profile').disabled=!extraction.length;$('export-history').disabled=!history?.rows.length;
  }
  function drawChart(rows, options={}) {
    lastChartRows=rows;lastChartOptions=options;
    const w=Math.max(320,Math.round(chart.clientWidth||960)),h=w<500?240:280,ratio=Math.min(devicePixelRatio||1,2);
    if(chart.width!==Math.round(w*ratio)||chart.height!==Math.round(h*ratio)){chart.width=Math.round(w*ratio);chart.height=Math.round(h*ratio);}
    const ctx=chart.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);
    // A resize can arrive while a new worker analysis has replaced the previous one.
    if(!analysis)return;
    const finite=rows.filter(r=>r.value!=null);ctx.fillStyle='#607782';ctx.font=`${w<500?11:13}px system-ui`;
    if(!finite.length){
      if(rows.length){ctx.fillText(`No valid measurement in ${rows.length} saved samples; gaps are retained.`,24,40);return;}
      if(!analysis)return;const component=$('review-component').value,{values,valid,unit}=componentValues(analysis,component);
      const chosen=options.regionBounds?Uint8Array.from(valid,(v,i)=>{const p=pointPosition(i),[x0,y0,x1,y1]=options.regionBounds;return +(v&&p[0]>=x0&&p[0]<=x1&&p[1]>=y0&&p[1]<=y1);}):valid;
      if(component==='status'){
        const codes=[-3,-2,-1,0,1,2],counts=codes.map(code=>values.reduce((n,v,i)=>n+ +(chosen[i]&&v===code),0)),highest=Math.max(1,...counts),step=(w-80)/6;
        ctx.fillText(`${options.regionBounds?'Region status':'Status'} · categorical counts`,16,24);
        counts.forEach((count,i)=>{const x=60+i*step,height=count/highest*(h-105);ctx.fillStyle=['#687c88','#c22c23','#9e8b79','#dfb82c','#229a70','#e88a2b'][i];ctx.fillRect(x,h-65-height,step-8,height);ctx.fillStyle='#607782';ctx.fillText(String(count),x,h-72-height);ctx.fillText(String(codes[i]),x,h-45);ctx.fillText(['ROI','corr','ref','GN','ok','flag'][i],x,h-28);});return;
      }
      const stats=statistics(values,chosen);
      if(!stats.count){ctx.fillText('No valid values for this quantity.',24,40);return;}
      const bins=new Uint32Array(40),span=Math.max(1e-14,stats.max-stats.min);
      values.forEach((v,i)=>{if(chosen[i])bins[Math.min(39,Math.floor((v-stats.min)/span*40))]++;});
      const highest=Math.max(...bins,1);ctx.fillStyle='#2d8b83';bins.forEach((n,i)=>ctx.fillRect(76+i*(w-108)/40,h-48-n/highest*(h-90),(w-108)/40-1,n/highest*(h-90)));
      ctx.fillStyle='#607782';ctx.fillText(`${options.regionBounds?'Region distribution':'Distribution'} · ${stats.count} valid samples`,24,25);
      const factor=STRAIN_COMPONENTS.has($('review-component').value)?1e6:1;
      ctx.fillText(`${number(stats.min*factor)} ${factor===1e6?'µε':unit}`,76,h-16);ctx.textAlign='right';ctx.fillText(`${number(stats.max*factor)} ${factor===1e6?'µε':unit}`,w-32,h-16);ctx.textAlign='left';return;
    }
    let low=Math.min(...finite.map(r=>r.value)),high=Math.max(...finite.map(r=>r.value));const span=Math.max(1e-12,Math.abs(high)*1e-9,high-low),xmax=Math.max(...rows.map(r=>r.distance),1e-12);
    low-=.08*span;high+=.08*span;const x=v=>76+v/xmax*(w-108),y=v=>h-48-(v-low)/(high-low)*(h-82);
    const factor=options.gauge||STRAIN_COMPONENTS.has($('review-component').value)?1e6:1;
    ctx.strokeStyle='#d9e3e7';ctx.lineWidth=1;for(let i=0;i<=4;i++){const v=low+(high-low)*i/4;ctx.beginPath();ctx.moveTo(76,y(v));ctx.lineTo(w-32,y(v));ctx.stroke();ctx.fillText(number(v*factor),4,y(v)+5);}
    ctx.strokeStyle='#117c72';ctx.lineWidth=2.5;ctx.beginPath();let pen=false;for(const row of rows){if(row.value==null){pen=false;continue;}if(pen)ctx.lineTo(x(row.distance),y(row.value));else ctx.moveTo(x(row.distance),y(row.value));pen=true;}ctx.stroke();
    ctx.fillStyle='#117c72';for(const row of finite){ctx.beginPath();ctx.arc(x(row.distance),y(row.value),3,0,2*Math.PI);ctx.fill();}
    ctx.fillStyle='#607782';for(let i=0;i<=4;i++)ctx.fillText(number(xmax*i/4),x(xmax*i/4)-15,h-30);
    ctx.fillText(options.axisLabel??`Distance along snapped samples (${analysis.recipe.unit})`,w/2-160,h-10);
    ctx.fillText(`${options.gauge?'Gauge engineering strain':$('review-component').selectedOptions[0].textContent} (${factor===1e6?'µε':componentValues(analysis,$('review-component').value).unit})`,16,22);
  }
  let chartResizeFrame=null,lastChartWidth=null;
  const chartResize=typeof ResizeObserver==='function'?new ResizeObserver(entries=>{
    const width=Math.round(entries[0].contentRect.width);if(width<=0||width===lastChartWidth)return;
    lastChartWidth=width;cancelAnimationFrame(chartResizeFrame);
    chartResizeFrame=requestAnimationFrame(()=>drawChart(lastChartRows,lastChartOptions));
  }):null;chartResize?.observe(chart);
  overlay.onpointerdown = event => {
    stopReplay();clearHistory();
    if(!analysis)return;if(activeMeasurementId){activeMeasurementId=null;$('measure-name').value='';}const rect=overlay.getBoundingClientRect(),point=[(event.clientX-rect.left)/rect.width*overlay.width-.5,(event.clientY-rect.top)/rect.height*overlay.height-.5];
    if(tool==='probe'||points.length>=2)points=[];points.push(point);updateInspection();
  };
  overlay.classList.add('inspecting');
  function readMeasurementPoints() {
    const required=['measure-ax','measure-ay',...(tool==='probe'?[]:['measure-bx','measure-by'])];
    if(required.some(id=>!$(id).value.trim()))throw new Error('Enter every required coordinate before saving this measurement.');
    const positions=[['measure-ax','measure-ay'],...(tool==='probe'?[]:[['measure-bx','measure-by']])].map(ids=>ids.map(id=>Number($(id).value)));
    if(positions.some(([x,y])=>!Number.isFinite(x)||!Number.isFinite(y)||x<0||y<0||x>=analysis.field.width||y>=analysis.field.height))throw new Error('Use coordinates inside the reference image.');
    return positions;
  }
  function saveMeasurement(update=false) {
    if(!analysis)return;
    let geometry;try{geometry=readMeasurementPoints();}catch(e){$('inspection-hint').textContent=e.message;return;}
    if(update&&!activeMeasurementId){$('inspection-hint').textContent='Select a saved measurement before updating it.';return;}
    clearHistory();const old=activeMeasurement(),type=tool==='probe'?'point':tool;
    if(type==='region'&&$('review-component').value==='status'){$('inspection-hint').textContent='Choose a scalar quantity before saving a zone; solver status is categorical.';return;}
    const name=$('measure-name').value.trim()||old?.name||`${type==='gauge'?'Extensometer':type==='region'?'Zone':type==='line'?'Line':'Point'} ${update?activeMeasurementId.replace(/^m/,''):nextMeasurementId}`;
    const item={id:update?activeMeasurementId:`m${nextMeasurementId++}`,name,type,points:geometry,component:$('review-component').value};
    if(update)measurements=measurements.map(value=>value.id===activeMeasurementId?item:value);else measurements.push(item);
    activeMeasurementId=item.id;points=item.points.map(point=>[...point]);tool=type==='point'?'probe':type;
    changed();rememberView();updateInspection();
  }
  $('measurement-position').onsubmit=event=>{
    event.preventDefault();stopReplay();saveMeasurement(false);
  };
  $('update-measurement').onclick=()=>{stopReplay();saveMeasurement(true);};
  $('delete-measurement').onclick=()=>{const item=activeMeasurement();if(!item)return;measurements=measurements.filter(value=>value.id!==item.id);activeMeasurementId=null;points=[];changed();rememberView();clearHistory();updateInspection();};
  $('clear-all-measurements').onclick=()=>{if(!measurements.length)return;measurements=[];activeMeasurementId=null;points=[];changed();rememberView();clearHistory();updateInspection();};
  document.querySelectorAll('[data-inspect]').forEach(button=>button.onclick=()=>{stopReplay();clearHistory();activeMeasurementId=null;tool=button.dataset.inspect;points=[];$('measure-name').value='';document.querySelectorAll('[data-inspect]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));$('inspection-hint').textContent=tool==='probe'?'Click a subset to place a point.':tool==='region'?'Click two opposite rectangle corners; valid subset centres are equally weighted.':'Click two endpoints on the field; measurements snap to subset centres.';updateInspection();});
  $('clear-inspection').onclick=()=>{stopReplay();clearHistory();activeMeasurementId=null;points=[];updateInspection();};
  $('extract-history').onclick=async()=>{
    stopReplay();if(!analysis||!measurements.length)return;
    if(!activeMeasurementId){const item=measurements[0];activeMeasurementId=item.id;tool=item.type==='point'?'probe':item.type;points=item.points.map(point=>[...point]);updateInspection();}
    const entry=activeEntry(),candidates=entries.filter(candidate=>sameAnalysis(entry,candidate));
    const controller=new AbortController();historyAbort?.abort();historyAbort=controller;
    const recipe=analysis.recipe,component=$('review-component').value,selection=measurements.map(item=>({...item,points:item.points.map(point=>[...point])}));
    $('extract-history').disabled=true;$('cancel-history').hidden=false;
    try{
      const rows=await extractHistory(candidates,{recipe,component,measurements:selection,signal:controller.signal,loadEntry:entry=>storage.load(entry),
        progress:(done,total)=>{$('history-state').textContent=`Analysing saved field ${done} / ${total}…`;}});
      if(controller.signal.aborted)return;history={rows,recipe,component,measurements:selection,comparison:entry.metadata.comparison??null};$('spatial-view').hidden=false;$('export-history').disabled=!rows.length;
      $('history-state').textContent=`${rows.length} long-format samples · ${selection.length} measurement(s) × ${candidates.length} compatible saved fields. Missing or rejected values stay blank; unrelated fields are excluded. ${entry.metadata.comparison?.mode==='consecutive'?'Pair increments at fixed image coordinates; not cumulative material tracking.':entry.metadata.comparison?.mode==='fixed'?'Same fixed reference.':'Reference convention not recorded; no accumulated trajectory inferred.'}`;
      updateInspection();
    }catch(e){if(e.name!=='AbortError')$('history-state').textContent=e.message;}
    finally{if(historyAbort===controller){historyAbort=null;$('extract-history').disabled=false;$('cancel-history').hidden=true;}}
  };
  $('cancel-history').onclick=()=>historyAbort?.abort();
  $('spatial-view').onclick=()=>{history=null;$('spatial-view').hidden=true;updateInspection();};
  $('export-history').onclick=()=>{
    if(!history?.rows.length)return;
    const keys=['order','fieldId','sourceIndex','mediaTimeSeconds','measurementId','measurementName','measurementType','component','sampleIndex','distance','x_px','y_px','x0_px','y0_px','x1_px','y1_px','value','unit','status','referenceLength','deformedLength','extension','engineeringStrain','mean','standardDeviation','minimum','maximum','validSamples','selectedGridPositions','totalGridPositions'];
    const quote=value=>{const text=value==null?'':String(value);return /[",\r\n]/.test(text)?`"${text.replaceAll('"','""')}"`:text;};
    const csv=[keys.join(','),...history.rows.map(row=>keys.map(key=>quote(row[key])).join(','))].join('\r\n');
    download('live-dic-measurement-history.csv',csv,'text/csv;charset=utf-8');
    download('live-dic-measurement-history.json',JSON.stringify({format:'live-dic-measurement-history',version:1,recipe:history.recipe,component:history.component,
      coordinateConvention:'zero-based reference-image pixel indices; line samples snap to subset centres',comparison:history.comparison,
      measurements:history.measurements,rows:history.rows.length,missingValues:'blank in CSV; never zero-filled or interpolated'},null,2),'application/json');
  };
  $('freeze-field').onclick=async()=>{if(captureBusy||storage.readOnly||storageBlocked())return;captureBusy=true;$('freeze-field').disabled=true;try{const {field,metadata}=await captureCurrent();let mask=currentMask;if(metadata.roi&&!mask)mask=maskFromMetadata(metadata);await add(field,metadata,{select:true,mask});}catch(e){error(e.message);}finally{captureBusy=false;$('freeze-field').disabled=!liveReady||storage.readOnly||storageBlocked();}};
  $('return-live').onclick=()=>{returnLive(true);document.querySelector('[data-pane=solve]').click();};
  $('record-fields').onchange=async()=>{if(recording()&&(storage.readOnly||storageBlocked())){$('record-fields').checked=false;await recordChange(false);updateLibrary();return;}const changedMode=recordChange(recording());if(recording()){await storage.database();const persistent=await storage.persistence();$('storage-quota').textContent=persistent?'Browser persistence granted; independent exports are still required for a backup.':'Persistence not granted or unavailable: browser disk data may be evicted. Export important sequences.';}await changedMode;updateLibrary();};
  $('saved-fields').onchange=()=>seekResult(Number($('saved-fields').value));
  $('result-seek').oninput=()=>{const entry=entries[Number($('result-seek').value)];if(entry)return seekResult(entry.id);};
  $('result-number').onchange=()=>{const number=Number($('result-number').value);if(Number.isInteger(number)&&number>=1&&number<=entries.length)return seekResult(entries[number-1].id);};
  $('replay-results').onclick=()=>{
    if(replaying){stopReplay();return;}
    const base=activeEntry()??entries[0];if(!base)return;
    replayIds=entries.filter(entry=>sameAnalysis(base,entry)).map(entry=>entry.id);if(replayIds.length<2)return;
    if(!reviewing||selected===replayIds.at(-1))selectEntry(replayIds[0],{preserveView:true});
    replaying=true;$('replay-results').textContent='Pause results';
    const next=()=>{
      if(!replaying)return;
      if(workerBusy||pendingAnalysis||selectionBusy){replayTimer=setTimeout(next,25);return;}
      if(!analysis){stopReplay();return;}
      const index=replayIds.indexOf(selected),id=replayIds[index+1];
      if(!id){stopReplay();return;}
      selectEntry(id,{replay:true,preserveView:true});replayTimer=setTimeout(next,1000/Number($('replay-rate').value));
    };replayTimer=setTimeout(next,1000/Number($('replay-rate').value));
  };
  $('apply-recipe-analysis').onclick=()=>{
    stopReplay();if(!analysis||computedView)return;
    const candidates=entries.filter(entry=>sameAnalysis(activeEntry(),entry));
    for(const entry of candidates)entry.recipe={...analysis.recipe};
    changed();updateLibrary();$('processing-batch-state').textContent=`Applied recipe to ${candidates.length} matching field(s); ${entries.length-candidates.length} unrelated field(s) unchanged. Computed arrays are preserved.`;
  };
  for(const [id,delta] of [['previous-result',-1],['next-result',1]])$(id).onclick=()=>{const index=entries.findIndex(e=>e.id===selected);if(entries[index+delta])return seekResult(entries[index+delta].id);};
  $('delete-field').onclick=async()=>{stopReplay();const index=entries.findIndex(e=>e.id===Number($('saved-fields').value));if(index<0)return;const entry=entries[index];if(entry.storage?.saved&&!confirm('Permanently delete this computed field from browser disk? Export it first if needed.'))return;try{await storage.remove(entry);const active=entry.id===selected;entries.splice(index,1);analysisCache.clear();changed();if(active){returnLive(true);selected=null;}updateLibrary();}catch(e){error(e.message);}};
  async function closeLibrary(){await beforeProjectExport();await storage.flush();if(entries.length&&!confirm('Close this view? Committed disk fields remain in Saved sessions. RAM-only fields and unsaved analysis changes need an export before closing.'))return false;returnLive(true);await storage.close();analysisCache.clear();entries.length=0;selected=null;nextId=1;storage=new FieldStorage({memoryLimit:storage.memoryLimit});updateLibrary();return true;}
  $('clear-fields').onclick=()=>closeLibrary().catch(e=>error(e.message));
  $('results-memory-limit').onchange=()=>{storage.setLimit($('results-memory-limit').value==='auto'?defaultCacheBytes():Number($('results-memory-limit').value)*MiB);const report=storage.report();if(report.unsavedFields&&report.residentBytes>report.memoryLimitBytes){$('record-fields').checked=false;recordChange(false);pause();error('Cache budget is below unsaved RAM data; export or retry disk storage before continuing.');}updateLibrary();};
  $('retry-storage').onclick=async()=>{try{await beforeProjectExport();await storage.retry();updateLibrary();await refreshSessions();}catch(e){error(`Disk save failed; RAM-only fields are retained. ${e.message}`);}};
  async function refreshSessions(){
    try{const sessions=await storage.sessions();$('stored-sessions').replaceChildren();for(const session of sessions.sort((a,b)=>b.updatedAt.localeCompare(a.updatedAt)))$('stored-sessions').add(new Option(`${new Date(session.createdAt).toLocaleString()} · ${session.count} fields · ${(session.bytes/MiB).toFixed(1)} MiB`,session.id));if(!sessions.length)$('stored-sessions').add(new Option('No disk sessions',''));
      const quota=await storage.quota();$('storage-quota').textContent=`Origin storage estimate: ${quota.usage==null?'unknown':(quota.usage/MiB).toFixed(0)+' MiB used'} / ${quota.quota==null?'unknown':(quota.quota/MiB).toFixed(0)+' MiB quota'}. Persistence ${quota.persistent?'granted':'not granted; eviction is possible'}. This is not free system disk space.`;
    }catch(e){error(e.message);}
  }
  $('refresh-sessions').onclick=refreshSessions;
  $('save-session').onclick=async()=>{try{await beforeProjectExport();while(selectionBusy||workerBusy||pendingAnalysis)await new Promise(resolve=>setTimeout(resolve,20));rememberView();if(await storage.saveSettings(entries,selected)){downloadedRevision=projectRevision;$('project-state').textContent='Fields and analysis settings saved in this browser profile. Export parts for an independent backup.';}updateLibrary();await refreshSessions();}catch(e){error(e.message);}};
  $('open-session').onclick=async()=>{
    const id=$('stored-sessions').value;if(!id)return;
    try{await beforeProjectExport();await recordChange(false);$('record-fields').checked=false;const candidate=new FieldStorage({memoryLimit:storage.memoryLimit});const saved=await candidate.openSession(id);
      if(entries.length&&!confirm('Replace this view with the saved session? Current committed fields stay on disk; export RAM-only data and unsaved settings first.')){await candidate.close();return;}
      returnLive(true);await storage.close();analysisCache.clear();storage=candidate;entries.length=0;for(const entry of saved.entries)entries.push(entry);nextId=entries.reduce((max,entry)=>Math.max(max,entry.id+1),1);selected=null;changed();updateLibrary();
      const first=entries.find(entry=>entry.id===saved.selectedId)??entries[0];if(first)await selectEntry(first.id,{preserveView:false});$('project-state').textContent=`Opened ${entries.length} disk-backed fields. This session is read-only for acquisition; close this view to start a new recording.`;
    }catch(e){error(e.message);}
  };
  $('delete-session').onclick=async()=>{const id=$('stored-sessions').value;if(!id||!confirm('Permanently delete this entire session from browser disk? Other tabs using it will stop. Export a backup first.'))return;try{await storage.deleteSession(id);await refreshSessions();}catch(e){error(e.message);}};
  $('inspect-memory').onclick=async()=>{try{$('memory-report').textContent='Inspecting application allocations…';const gpu=await requestMemory();$('memory-report').textContent=JSON.stringify({cpuResults:storage.report(),cpuLibraryBytes:memoryBytes(),originStorage:await storage.quota(),gpu,
      note:'CPU cache target is not total process RAM. Analysis workers, decoded images, metadata objects and export scratch add memory. GPU values are tracked API sizes, not free VRAM. On unified GPUs the physical RAM pool is shared.'},null,2);}catch(e){error(e.message);}};
  $('import-result').onclick=()=>$('result-file').click();
  function archiveEstimate(list){const masks=new Set();let bytes=0;for(const entry of list){bytes+=entry.storage?.descriptor.bytes??0;const key=entry.storage?.maskId;if(key&&!masks.has(key)){masks.add(key);bytes+=entry.field.width*entry.field.height;}bytes+=entry.metadataBytes+4096;}return bytes;}
  async function materialize(list){const result=[];for(const entry of list){await storage.load(entry);result.push({...entry,field:entry.field,metadata:entry.metadata,mask:entry.mask});}return result;}
  let partExport=null,exportBusy=false;
  async function preparePart(){
    const selected=[];let end=partExport.offset;
    const target=Math.min(64*MiB,storage.memoryLimit/4);
    while(end<partExport.entries.length&&selected.length<128){const next=partExport.entries[end];if(selected.length&&archiveEstimate([...selected,next])>target)break;selected.push(next);end++;}
    if(archiveEstimate(selected)>200*MiB)throw new Error('One field exceeds the portable archive part size; export it separately before continuing');
    const snapshots=await materialize(selected),bytes=projectArchive(snapshots,{selectedId:snapshots[0]?.id});
    const name=`live-dic-results-part-${String(partExport.files.length+1).padStart(5,'0')}.dicproject`;
    return {bytes,name,end,fields:selected.length};
  }
  async function writePart(directory=null){
    if(exportBusy||!partExport||partExport.offset>=partExport.entries.length)return;exportBusy=true;$('download-next-part').disabled=true;$('export-parts-folder').disabled=true;
    try{const part=await preparePart();if(directory){const handle=await directory.getFileHandle(part.name,{create:true}),writer=await handle.createWritable();try{await writer.write(part.bytes);await writer.close();}catch(e){await writer.abort().catch(()=>{});throw e;}}else download(part.name,part.bytes,'application/zip');
      const hash=await crypto.subtle.digest('SHA-256',part.bytes);partExport.files.push({name:part.name,fields:part.fields,bytes:part.bytes.byteLength,sha256:Array.from(new Uint8Array(hash),value=>value.toString(16).padStart(2,'0')).join('')});partExport.offset=part.end;
      const done=partExport.offset===partExport.entries.length;$('export-parts-state').textContent=`${partExport.offset} / ${partExport.entries.length} fields exported in ${partExport.files.length} part(s). ${done?'Complete; keep every part and the index.':'Continue with the next part.'}`;
      if(done){if(directory){const handle=await directory.getFileHandle('live-dic-results-index.json',{create:true}),writer=await handle.createWritable();await writer.write(JSON.stringify(partsIndex(),null,2));await writer.close();}else $('download-parts-index').hidden=false;downloadedRevision=partExport.revision;updateLibrary();}
    }finally{exportBusy=false;$('download-next-part').disabled=!!partExport&&partExport.offset>=partExport.entries.length;$('export-parts-folder').disabled=typeof showDirectoryPicker!=='function';}
  }
  $('download-next-part').onclick=()=>writePart().catch(e=>error(e.message));
  const partsIndex=()=>({format:'live-dic-project-parts',version:1,fields:partExport.entries.length,files:partExport.files});
  $('download-parts-index').onclick=()=>{if(partExport?.files.length)download('live-dic-results-index.json',JSON.stringify(partsIndex(),null,2),'application/json');};
  $('export-parts-folder').disabled=typeof showDirectoryPicker!=='function';
  $('export-parts-folder').onclick=async()=>{if(exportBusy)return;try{const parent=await showDirectoryPicker({mode:'readwrite'}),directory=await parent.getDirectoryHandle(`live-dic-${new Date().toISOString().replace(/[:.]/g,'-')}-${crypto.randomUUID().slice(0,8)}`,{create:true});partExport.offset=0;partExport.files=[];partExport.cancelled=false;while(partExport&&partExport.offset<partExport.entries.length&&$('export-parts').open&&!partExport.cancelled)await writePart(directory);}catch(e){if(e.name!=='AbortError')error(e.message);}};
  $('close-export-parts').onclick=()=>{if(exportBusy){partExport.cancelled=true;$('export-parts-state').textContent='Stopping after the current part; computed fields remain on browser disk.';}else $('export-parts').close();};
  $('export-parts').oncancel=event=>{if(exportBusy){event.preventDefault();partExport.cancelled=true;$('export-parts-state').textContent='Stopping after the current part.';}};
  $('save-project').onclick=async()=>{
    if(exportBusy)return;stopReplay();try{
      await beforeProjectExport();
      // Snapshot the applied recipe after the last requested analysis, never its previous value.
      while(reviewing&&(workerBusy||pendingAnalysis||selectionBusy))await new Promise(resolve=>setTimeout(resolve,20));
      rememberView();
      const snapshot=[...entries];
      if(archiveEstimate(snapshot)>Math.min(128*MiB,storage.memoryLimit/2)||snapshot.length>512){partExport={entries:snapshot,offset:0,files:[],revision:projectRevision};$('export-parts-state').textContent=`${snapshot.length} fields. Export parts up to ${(Math.min(64*MiB,storage.memoryLimit/4)/MiB).toFixed(0)} MiB without loading the entire sequence.`;$('download-next-part').disabled=false;$('download-parts-index').hidden=true;$('export-parts').showModal();return;}
      const bytes=projectArchive(await materialize(snapshot),{selectedId:selected});
      download('live-dic-results.dicproject',bytes,'application/zip');
      downloadedRevision=projectRevision;updateLibrary();
      $('project-state').textContent=`Download requested: ${snapshot.length} computed fields, shared ROI masks and applied recipes. Check the downloaded file; input images are separate.`;
    }catch(e){error(e.message);}
  };
  $('open-project').onclick=()=>$('project-file').click();
  $('project-file').onchange=async()=>{
    const files=[...$('project-file').files].sort((a,b)=>a.name.localeCompare(b.name,undefined,{numeric:true}));$('project-file').value='';if(!files.length)return;
    let importedCount=0;
    try{
      await beforeProjectExport();
      for(const file of files){
      if(file.size>ARCHIVE_IMPORT_BYTES)throw new Error('Project part exceeds the 256 MiB import limit');
      const project=await importProjectArchive(await file.arrayBuffer());
      const drafts=project.records.map(record=>({...record,metadataBytes:encoderBytes(normalizeMetadata(record.metadata))}));
      await storage.database();if(!storage.backend&&memoryBytes(drafts)>storage.memoryLimit)throw new Error('Disk is unavailable and this project exceeds the RAM cache budget. This part was not imported.');
      const added=[];for(const record of project.records){const entry=await add(record.field,record.metadata,{mask:record.mask,recipe:record.recipe});added.push(entry);if(storage.backend&&!entry.storage.saved)throw new Error('Disk saving stopped during import. Already admitted fields, including the last RAM-only field, remain available; keep the source part before retrying.');}
      const previous=project.records.findIndex(record=>record.originalId===project.settings.selectedId);
      importedCount+=added.length;if(added.length)await selectEntry(added[Math.max(0,previous)].id,{preserveView:false});
      $('project-state').textContent=`Opened ${importedCount} fields; latest part ${file.name}. Current acquisition settings and source were not replaced.`;
      }
    }catch(e){error(`${e.message}${importedCount?` ${importedCount} previously imported fields remain available.`:''}`);}
  };
  async function addImported(imported, file) {
    const selection=imported.metadata.importGeometry;
    await add(imported.field,{...imported.metadata,name:imported.metadata.name??`${file.name}${selection?.frameCount>1?` · field ${selection.frameIndex+1}/${selection.frameCount}`:''}`},
      {mask:imported.roiMask,select:true,recipe:imported.recipe});
  }
  $('result-file').onchange=async()=>{
    const file=$('result-file').files[0];$('result-file').value='';if(!file)return;
    const serial=++importSerial;
    try{
      await beforeProjectExport();if(file.size>ARCHIVE_IMPORT_BYTES)throw new Error('NPZ exceeds the 256 MiB import limit');
      const bytes=await file.arrayBuffer();if(serial!==importSerial)return;
      try {const imported=await importFieldArchive(bytes);if(serial!==importSerial)return;await addImported(imported,file);}
      catch(e){
        if(e.code!=='FIELD_GEOMETRY_REQUIRED')throw e;
        pause();pendingImport={bytes,file,serial};const defaults=geometryDefaults(),geometry=e.geometry;
        for(const [id,key] of [['import-width','width'],['import-height','height'],['import-subset','subsetSize'],['import-stride','subsetStride']]){
          $(id).value=geometry[key]??defaults[key]??(key.startsWith('subset')?16:'');$(id).readOnly=geometry[key]!==undefined;
        }
        $('import-frame').max=geometry.frameCount;$('import-frame').value=1;$('import-frame-label').hidden=geometry.frameCount===1;
        $('import-geometry-note').textContent=`${file.name} · ${geometry.frameCount} field(s). Stored dimensions remain authoritative. Missing dimensions must be supplied explicitly; current-image values, if shown, must be checked. Extent is never inferred from the last grid cell.`;
        $('import-geometry-error').textContent='';$('import-geometry').showModal();
      }
    }catch(e){error(e.message);}
  };
  $('cancel-import').onclick=()=>{$('import-geometry').close();};
  $('import-geometry').onclose=()=>{pendingImport=null;importSerial++;};
  $('import-geometry-form').onsubmit=async event=>{
    event.preventDefault();if(!pendingImport)return;
    const pending=pendingImport;$('confirm-import').disabled=true;
    try{
      const geometry={width:Number($('import-width').value),height:Number($('import-height').value),subsetSize:Number($('import-subset').value),subsetStride:Number($('import-stride').value),frameIndex:Number($('import-frame').value)-1};
      const imported=await importFieldArchive(pending.bytes,geometry);
      if(pending.serial!==importSerial)return;await addImported(imported,pending.file);$('import-geometry').close();
    }catch(e){$('import-geometry-error').textContent=e.message;}
    finally{$('confirm-import').disabled=false;}
  };
  for(const id of ['post-smoothing','strain-source','post-window','strain-measure','exclude-flagged','length-unit','length-scale'])$(id).onchange=()=>{
    stopReplay();$('processing-batch-state').textContent='';
    if($('post-smoothing').value!=='none')$('strain-source').value='vsg';
    if(activeEntry()?.field.config.motionModel==='translation')$('strain-source').value='vsg';
    $('length-scale').disabled=$('length-unit').value==='px';if($('length-scale').disabled)$('length-scale').value=1;
    computedView=false;$('compare-computed').textContent='Show computed field';scheduleAnalysis();
  };
  $('reset-processing').onclick=()=>{stopReplay();setRecipe({...DEFAULT_RECIPE,strainSource:activeEntry()?.field.config.motionModel==='translation'?'vsg':'direct'});computedView=false;scheduleAnalysis();};
  $('compare-computed').onclick=()=>{stopReplay();computedView=!computedView;$('compare-computed').textContent=computedView?'Restore processing recipe':'Show computed field';scheduleAnalysis();};
  $('review-component').onchange=()=>{clearHistory();const item=activeMeasurement();if(item){item.component=$('review-component').value;changed();rememberView();renderMeasurementList();}$('review-scale').value='robust';updateColourControlVisibility();if(!reviewing&&entries.length)return selectEntry(entries[0].id,{preserveView:true});render();};
  for(const id of ['review-scale','review-min','review-max','review-low-percentile','review-high-percentile','review-palette','review-vectors','review-vector-step','review-vector-gain','review-contour'])$(id).onchange=()=>{updateColourControlVisibility();render();};
  $('save-npz').onclick=()=>{if(!analysis)return;const entry=activeEntry();download(`live-dic-${entry.id}.npz`,fieldArchive(entry.field,{...entry.metadata,workbench:{view:viewSettings()},analysis:{recipe:analysis.recipe,processingWallMs:analysis.processingWallMs}},analysis,entry.mask),'application/octet-stream');};
  $('save-processed-csv').onclick=()=>{if(!analysis)return;const {values,valid,unit}=componentValues(analysis,$('review-component').value),lines=[`x_px,y_px,${$('review-component').value}_${unit},valid,solver_status`];for(let i=0;i<values.length;i++)lines.push([...pointPosition(i),valid[i]?values[i]:'',valid[i],analysis.field.status[i]].join(','));download('live-dic-displayed-field.csv',lines.join('\n'),'text/csv');};
  $('save-profile').onclick=()=>{if(!extraction.length)return;const columns=Object.keys(extraction[0]);download('live-dic-extraction.csv',[columns.join(','),...extraction.map(row=>columns.map(k=>row[k]??'').join(','))].join('\n'),'text/csv');download('live-dic-extraction.json',JSON.stringify({recipe:analysis.recipe,view:viewSettings(),valueUnit:(history?.tool??tool)==='gauge'?'strain':componentValues(analysis,$('review-component').value).unit,coordinateUnit:'px',distanceUnit:analysis.recipe.unit,comparison:activeEntry().metadata.comparison},null,2),'application/json');};
  $('save-figure').onclick=async()=>{
    if(!analysis||!figureLegend)return;
    const {field,recipe}=analysis,component=$('review-component').value,coefficient=component.startsWith('computed-p'),categorical=component==='status';
    const details=[`${field.width} × ${field.height} reference pixels · ${field.config.motionModel} · window / step ${field.subsetSize} / ${field.subsetStride} px · grid ${field.grid.join(' × ')} · solver policy: ${field.config.outlierPolicy}`,
      categorical?'Stored solver status: workbench smoothing and VSG do not alter categories.':coefficient?'Stored model coefficient: workbench smoothing and VSG do not alter these values.':`Displacement smoothing: ${recipe.smoothing} · ${recipe.window} × ${recipe.window} grid points · strain: ${recipe.strainSource} / ${recipe.strainMeasure} · nominal strain support: ${analysis.strainSpanPx} px`,
      categorical?'Six categorical status colours; scalar colour limits do not apply.':`Length scale: ${recipe.unitsPerPixel} ${recipe.unit} / px (isotropic). Palette: ${$('review-palette').selectedOptions[0].textContent}; clim: ${figureLegend.low} to ${figureLegend.high}; ${$('review-scale').value==='robust'?`${$('review-low-percentile').value}–${$('review-high-percentile').value}% valid-value percentiles`: $('review-scale').value==='full'?'full valid-value range':'manually entered limits'}. Display settings do not alter numerical exports.`];
    if($('review-vectors').checked)details.push(`Arrows: processed displacements · magnification ×${$('review-vector-gain').value} · every ${$('review-vector-step').value} grid points.`);
    if(points.length){
      const centres=points.map(([x,y])=>{const i=nearestIndex(field,x,y);return i<0?'outside measured grid':`(${pointPosition(i).map(value=>value.toFixed(1)).join(', ')})`;}).join(' → ');
      details.push(`Measurement: ${tool} · reference pixel indices ${centres}. ${$('probe-value').textContent}`);
    }
    const figure=resultFigure({field:canvas,annotations:overlay,title:$('review-component').selectedOptions[0].textContent,label:activeEntry().label,legend:figureLegend,details});
    const blob=await new Promise(resolve=>figure.toBlob(resolve,'image/png'));
    if(blob)download('live-dic-field.png',blob,'image/png');else error('The browser could not encode the figure; computed values remain available.');
  };
  updateLibrary();
  return { add, returnLive, get mask(){return currentMask;}, get storageBlocked(){return storageBlocked();}, setMask(mask){currentMask=mask;}, get reviewing(){return reviewing;}, get recording(){return recording();},
    setReady(ready){liveReady=ready;$('freeze-field').disabled=!ready||captureBusy||storage.readOnly||storageBlocked();},
    setPrepared(ready){prepared=ready;updateLibrary();},
    async startRecording(){await $('record-fields').onchange();},
    async openFirst({index=0}={}){if(entries.length&&!reviewing)await selectEntry((entries[index]??entries[0]).id,{preserveView:true});},
    flush:()=>storage.flush(),get storage(){return storage;},
    stopRecording({preserveSelection=false}={}){const selected=$('record-fields').checked;$('record-fields').checked=false;const changed=recordChange(false);if(preserveSelection)$('record-fields').checked=selected;return changed;},
    clearInspection(){stopReplay();clearHistory();points=[];updateInspection();},
    report(){const values=analysis?viewCache.get(analysis,$('review-component').value):null;return{count:entries.length,selected,reviewing,replaying,quantity:$('review-component').value,analysisRuns,viewCache:viewCache.report(),analysisCache:analysisCache.report(),displayed,unsavedProject:hasUnsaved(),memoryBytes:memoryBytes(),storage:storage.report(),provenance:activeEntry()?{analysisEpoch:activeEntry().metadata.analysisEpoch,archiveSha256:activeEntry().metadata.archiveSha256,importGeometry:activeEntry().metadata.importGeometry}:null,recipe:analysis?.recipe??null,statistics:values?.statistics??null,extraction};},
    destroy(){analysisCache.clear();analysis=null;analysisJob=null;stopReplay();historyAbort?.abort();clearTimeout(debounce);clearTimeout(libraryTimer);chartResize?.disconnect();cancelAnimationFrame(chartResizeFrame);worker?.terminate();window.removeEventListener('beforeunload',beforeUnload);} };
}
