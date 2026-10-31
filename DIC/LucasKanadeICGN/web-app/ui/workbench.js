import { DEFAULT_RECIPE, analyzeField, componentValues, statistics, nearestIndex, extensometer, regionSummary, validateField } from './postprocess.js';
import { fieldArchive, importFieldArchive, projectArchive, importProjectArchive } from './archive.js';
import { extractHistory, sameAnalysis } from './history.js';

const $ = id => document.getElementById(id);
const number = value => value == null || !Number.isFinite(value) ? 'unavailable' : value.toPrecision(5);
const STRAIN_COMPONENTS = new Set(['exx','eyy','exy','vonMises','principal1','principal2']);
const LIMIT_BYTES = 128 * 1024 * 1024;
const encoderBytes = value => new TextEncoder().encode(JSON.stringify(value)).byteLength;
const STATUS = { '-3': 'Excluded by ROI', '-2': 'Rejected correlation', '-1': 'Unavailable reference / bounds',
  0: 'Not converged', 1: 'Accepted', 2: 'Flagged / replaced by solver policy' };

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
  const name = item => item?.name ?? (item?.frame != null ? `Frame ${item.frame}` : null) ?? (item?.index != null ? `Image ${item.index+1}` : item?.timeSeconds != null ? `${item.timeSeconds.toFixed(3)} s` : 'capture');
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
export function installWorkbench({ captureCurrent, pause, error, recordChange, liveChanged, openResults, geometryDefaults = () => ({}), beforeProjectExport = async () => pause() }) {
  const entries = [], canvas = $('review-field'), overlay = $('inspection'), chart = $('profile-chart');
  document.querySelector('.viewer').append($('results-workbench'));
  let nextId = 1, selected = null, analysis = null, currentMask = null, reviewing = false;
  let tool = 'probe', points = [], extraction = [], computedView = false;
  let requestId = 0, workerBusy = false, pendingAnalysis = false, workerFieldId = null, debounce = null;
  let pendingImport = null, importSerial = 0;
  let history = null, historyAbort = null, lastChartRows=[], lastChartOptions={};
  let replaying=false,replayTimer=null,replayIds=[];
  function stopReplay(){replaying=false;clearTimeout(replayTimer);replayTimer=null;$('replay-results').textContent='Replay matching results · 2 Hz';}
  let projectRevision=0, downloadedRevision=0;
  const hasUnsaved=()=>entries.length>0&&projectRevision!==downloadedRevision;
  const changed=()=>{projectRevision++;};
  const beforeUnload=event=>{rememberView();if(hasUnsaved()){event.preventDefault();event.returnValue='';}};
  window.addEventListener('beforeunload',beforeUnload);
  const contours = new WeakMap();
  let worker = typeof Worker === 'function' ? new Worker(new URL('./analysis-worker.js', import.meta.url), { type: 'module' }) : null;
  const recording = () => $('record-fields').checked;
  const activeEntry = () => entries.find(entry => entry.id === selected);
  function clearHistory() {
    historyAbort?.abort();history=null;$('spatial-view').hidden=true;$('history-state').textContent='';
  }
  function viewSettings() {
    return { component:$('review-component').value,colourScale:$('review-scale').value,
      minimum:Number($('review-min').value),maximum:Number($('review-max').value),
      palette:$('review-palette').value,vectors:$('review-vectors').checked,vectorStep:Number($('review-vector-step').value),
      vectorGain:Number($('review-vector-gain').value),contour:$('review-contour').checked,tool,points };
  }
  function restoreView(view) {
    if(!view)return;
    for(const [id,key] of [['review-component','component'],['review-scale','colourScale'],['review-palette','palette']])
      if([...$(id).options].some(option=>option.value===view[key]&&!option.disabled))$(id).value=view[key];
    if([view.minimum,view.maximum].every(Number.isFinite)&&view.maximum>view.minimum){$('review-min').value=view.minimum;$('review-max').value=view.maximum;}
    $('manual-limits').hidden=$('review-scale').value!=='manual';
    $('review-vectors').checked=view.vectors===true;$('review-contour').checked=view.contour===true;
    if([2,4,8].includes(view.vectorStep))$('review-vector-step').value=view.vectorStep;
    if([1,5,20,50,100].includes(view.vectorGain))$('review-vector-gain').value=view.vectorGain;
    const field=activeEntry()?.field;
    if(['probe','line','gauge','region'].includes(view.tool)&&Array.isArray(view.points)&&view.points.length<=2&&
      view.points.every(point=>Array.isArray(point)&&point.length===2&&point.every(Number.isFinite)&&
        point[0]>=0&&point[1]>=0&&point[0]<field.width&&point[1]<field.height)) {
      tool=view.tool;points=view.points.map(point=>[...point]);
      document.querySelectorAll('[data-inspect]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.inspect===tool)));
    }
  }

  function memoryBytes(extra = []) {
    const masks = new Set(); let bytes = 0;
    for (const entry of [...entries, ...extra]) {
      bytes += entry.field.parameters.byteLength + entry.field.status.byteLength + entry.field.iterations.byteLength + entry.metadataBytes;
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
  function updateLibrary() {
    const select = $('saved-fields'); select.replaceChildren();
    if (!entries.length) select.add(new Option('No saved fields', ''));
    for (const entry of entries) select.add(new Option(entry.label, entry.id));
    select.value = selected ?? entries.at(-1)?.id ?? '';
    $('library-summary').textContent = `${entries.length} field(s) · ${(memoryBytes()/1048576).toFixed(1)} / 128 MiB in this tab. ${!entries.length?'No stored results.':hasUnsaved()?'Project has changes not included in the last requested project download.':'Project download requested; check that the file was saved.'} Images are not recorded.`;
    $('clear-fields').disabled = !entries.length; $('delete-field').disabled = !entries.length;
    $('save-project').disabled=!entries.length;
    const index=entries.findIndex(entry=>entry.id===selected);
    $('previous-result').disabled=index<=0;$('next-result').disabled=index<0||index>=entries.length-1;
    $('result-seek').max=Math.max(0,entries.length-1);$('result-seek').value=Math.max(0,index);$('result-seek').disabled=!entries.length;
    $('result-order').textContent=index>=0?`${index+1} / ${entries.length} · saved order`:`${entries.length} saved fields`;
    const base=activeEntry()??entries[0];$('replay-results').disabled=!base||entries.filter(entry=>sameAnalysis(base,entry)).length<2;
  }
  function add(field, metadata = {}, { select = false, mask = currentMask, recipe = null } = {}) {
    validateField(field);
    // A single-field archive carries its own timing sample. The live timing log remains in direct export.
    metadata = normalizeMetadata(metadata);
    const entry = { id: nextId++, field, metadata, mask, recipe, metadataBytes: encoderBytes(metadata) };
    if (mask && mask.length !== field.width*field.height) throw new Error('Saved field mask dimensions differ');
    if (memoryBytes([entry]) > LIMIT_BYTES) {
      $('record-fields').checked = false; recordChange(false); pause();
      throw new Error('Results storage reached 128 MiB. Acquisition is paused. Export or delete fields before continuing; no stored result was discarded.');
    }
    entry.label = fieldLabel(metadata, entry.id); entries.push(entry); changed(); updateLibrary();
    if (select) selectEntry(entry.id);
    return entry;
  }
  function returnLive(restore = false) {
    stopReplay();rememberView();
    historyAbort?.abort();history=null;
    reviewing = false; requestId++; pendingAnalysis = false; clearTimeout(debounce);
    analysis=null;workerFieldId=null;worker?.postMessage({clear:true});
    $('results-workbench').hidden = true; canvas.hidden = true; overlay.hidden = true;
    $('field').hidden = false; $('return-live').hidden = true;
    $('roi').hidden=false;$('review-controls').hidden=true;$('live-appearance').hidden=false;$('cursor-value').hidden=true;
    document.querySelector('.legend div').style.background='';
    liveChanged(false, restore);
  }
  function selectEntry(id,{replay=false}={}) {
    if(!replay)stopReplay();
    const entry = entries.find(value => value.id === Number(id)); if (!entry) return;
    rememberView();
    historyAbort?.abort();history=null;$('spatial-view').hidden=true;$('history-state').textContent='';
    pause(); selected = entry.id; analysis = null; reviewing = true; computedView = false; points = []; extraction = [];
    $('saved-fields').value = entry.id; $('results-workbench').hidden = false;
    $('return-live').hidden = false; canvas.hidden = false; overlay.hidden = false; $('field').hidden = true;
    canvas.width = overlay.width = entry.field.width; canvas.height = overlay.height = entry.field.height;
    for(const id of ['measure-ax','measure-bx'])$(id).max=entry.field.width-1;
    for(const id of ['measure-ay','measure-by'])$(id).max=entry.field.height-1;
    $('roi').hidden=true;$('review-controls').hidden=false;$('live-appearance').hidden=true;$('recording-panel').open=false;
    updateLibrary();$('processing-batch-state').textContent='';
    $('result-title').textContent = entry.label;
    $('result-provenance').textContent = `${entry.field.width} × ${entry.field.height} · ${entry.field.config.motionModel} · ${entry.field.subsetSize}/${entry.field.subsetStride} · solver spatial policy: ${entry.field.config.outlierPolicy ?? 'unspecified'} · ${entry.metadata.comparison?.mode ?? 'imported'} reference. No image is stored behind this field.`;
    for(const option of $('review-component').options)if(option.dataset.coefficient!==undefined){
      const i=Number(option.dataset.coefficient);option.disabled=i>=(entry.field.config.motionModel==='quadratic'?12:6)||(entry.field.config.motionModel==='translation'&&![0,3].includes(i));
    }
    if($('review-component').selectedOptions[0].disabled)$('review-component').value='magnitude';
    setRecipe(entry.recipe ?? { ...DEFAULT_RECIPE, strainSource: entry.field.config.motionModel === 'translation' ? 'vsg' : 'direct' });
    restoreView(entry.metadata.workbench?.view);
    $('compare-computed').textContent = 'Show computed field';
    liveChanged(true); openResults(); scheduleAnalysis();
  }
  function analysisComplete(id, result, failure) {
    workerBusy = false;
    if (id === requestId && reviewing) {
      if (failure) { analysis = null; $('analysis-state').textContent = failure; error(failure); }
      else {
        analysis = { ...result, field: activeEntry().field }; if (!computedView) {
          if(JSON.stringify(activeEntry().recipe)!==JSON.stringify(result.recipe))changed();
          activeEntry().recipe = result.recipe;updateLibrary();
        }
        $('analysis-state').textContent = `Analysis ${result.processingWallMs.toFixed(1)} ms, outside the live solver. Computed arrays preserved.`;
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
    if (workerBusy) { pendingAnalysis = true; return; }
    pendingAnalysis = false; workerBusy = true;
    const entry = activeEntry(), id = requestId, recipe = computedView
      ? { ...DEFAULT_RECIPE, strainSource: entry.field.config.motionModel === 'translation' ? 'vsg' : 'direct', unit: $('length-unit').value, unitsPerPixel: Number($('length-scale').value) }
      : recipeFromControls();
    if (worker) {
      const payload = { id, recipe };
      if (workerFieldId !== entry.id) { payload.field = entry.field; payload.mask = entry.mask; workerFieldId = entry.id; }
      worker.postMessage(payload);
    } else {
      try { analysisComplete(id, analyzeField(entry.field, recipe, entry.mask)); }
      catch (e) { analysisComplete(id, null, e.message); }
    }
  }
  function scheduleAnalysis() {
    clearHistory();
    requestId++; pendingAnalysis = true; clearTimeout(debounce); analysis = null;
    $('analysis-state').textContent = 'Updating analysis…';
    for (const id of ['save-npz','save-processed-csv','save-figure','measurement-inputs','apply-recipe-analysis']) $(id).disabled = true;
    debounce = setTimeout(runAnalysis, 60);
  }
  function format(value, unit) { if (value == null) return 'unavailable'; return `${number(STRAIN_COMPONENTS.has($('review-component').value) ? value*1e6 : value)} ${STRAIN_COMPONENTS.has($('review-component').value) ? 'µε' : unit}`; }
  function render() {
    if (!analysis) return;
    const component = $('review-component').value, { values, valid, unit, source } = componentValues(analysis, component);
    $('quantity-source').hidden=!source;$('quantity-source').textContent=source?`Computed coefficient (${unit}). Workbench displacement smoothing and VSG do not change these values. The solver's original spatial policy still applies.`:'';
    const stats = statistics(values, valid), field = analysis.field, [gw,gh] = field.grid;
    const nonnegative = ['magnitude','vonMises','iterations'].includes(component);
    const scaleMode=$('review-scale').value,factor=STRAIN_COMPONENTS.has(component)?1e6:1;
    let low=scaleMode==='full'?stats.min:stats.p02,high=scaleMode==='full'?stats.max:stats.p98;
    low??=0;high??=1;
    if(!nonnegative){const limit=Math.max(Math.abs(low),Math.abs(high),1e-12);low=-limit;high=limit;}
    if(scaleMode==='manual'){
      low=Number($('review-min').value)/factor;high=Number($('review-max').value)/factor;
      if(!Number.isFinite(low)||!Number.isFinite(high)||high<=low){$('analysis-state').textContent='Fixed maximum must be greater than minimum, with finite limits.';return;}
    } else {$('review-min').value=Number((low*factor).toPrecision(6));$('review-max').value=Number((high*factor).toPrecision(6));}
    if(high-low<Math.max(1e-14,Math.abs(high)*1e-12)){const pad=Math.max(1e-12,Math.abs(high)*.01);low-=pad;high+=pad;}
    const palette=$('review-palette').value==='auto'?(nonnegative?'sequential':'diverging'):$('review-palette').value;
    function colour(value){
      const t=Math.max(0,Math.min(1,(value-low)/(high-low)));
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
      if (component === 'status') rgb = { '-3':[160,170,178],'-2':[202,51,64],'-1':[71,86,96],0:[240,193,52],1:[36,151,110],2:[238,135,45] }[Math.round(values[i])] ?? rgb;
      else if (valid[i]) rgb=colour(values[i]);
      image.data.set([...rgb,255],i*4);
    }
    context.putImageData(image,0,0);
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
    document.querySelector('.legend div').style.background=component==='status'?'':`linear-gradient(90deg,${Array.from({length:11},(_,i)=>`rgb(${colour(low+(high-low)*i/10).join(',')}) ${i*10}%`).join(',')})`;
    const clipped = Array.from(values).reduce((n,v,i)=>n+ +(valid[i] && (v<low||v>high)),0);
    const summary = $('field-summary'); summary.replaceChildren();
    const summaryRows=component==='status'?Object.entries(STATUS).map(([code,label])=>[label,field.status.reduce((n,status)=>n+ +(status===Number(code)),0)]):[['Valid values',`${stats.count} / ${values.length}`],['Minimum',format(stats.min,unit)],['Mean',format(stats.mean,unit)],['Maximum',format(stats.max,unit)],['Colour clipping',`${clipped} values · ${scaleMode} scale`]];
    for (const [label,value] of summaryRows) {
      const box = document.createElement('div'), strong = document.createElement('strong'); box.textContent = label; strong.textContent = value; box.append(strong); summary.append(box);
    }
    updateInspection();rememberView();
  }
  function pointPosition(index) {
    const f = analysis.field;return [(index%f.grid[0])*f.subsetStride+(f.subsetSize-1)/2,Math.floor(index/f.grid[0])*f.subsetStride+(f.subsetSize-1)/2];
  }
  function updateInspection() {
    const ctx = overlay.getContext('2d');ctx.clearRect(0,0,overlay.width,overlay.height);extraction=[];
    document.querySelectorAll('.second-endpoint').forEach(label=>label.hidden=tool==='probe');
    for(const id of ['measure-bx','measure-by']){$(id).required=tool!=='probe';$(id).disabled=tool==='probe';}
    for(const [i,ids] of [[0,['measure-ax','measure-ay']],[1,['measure-bx','measure-by']]])
      for(const [axis,id] of ids.entries())$(id).value=points[i]?.[axis]??'';
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
    if(history){extraction=history.rows;drawChart(history.rows.map(row=>({...row,distance:row.order})),{axisLabel:'Saved field order · not physical time',gauge:history.tool==='gauge'});}
    else {
      const regionBounds=tool==='region'&&points.length===2&&snapped.every(Boolean)?[Math.min(snapped[0][0],snapped[1][0]),Math.min(snapped[0][1],snapped[1][1]),Math.max(snapped[0][0],snapped[1][0]),Math.max(snapped[0][1],snapped[1][1])]:null;
      drawChart(tool==='line'?extraction:[],regionBounds?{regionBounds}:{});
    }$('save-profile').disabled=!extraction.length;
  }
  function drawChart(rows, options={}) {
    lastChartRows=rows;lastChartOptions=options;
    const w=Math.max(320,Math.round(chart.clientWidth||960)),h=w<500?240:280,ratio=Math.min(devicePixelRatio||1,2);
    if(chart.width!==Math.round(w*ratio)||chart.height!==Math.round(h*ratio)){chart.width=Math.round(w*ratio);chart.height=Math.round(h*ratio);}
    const ctx=chart.getContext('2d');ctx.setTransform(ratio,0,0,ratio,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);
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
    if(!analysis)return;const rect=overlay.getBoundingClientRect(),point=[(event.clientX-rect.left)/rect.width*overlay.width-.5,(event.clientY-rect.top)/rect.height*overlay.height-.5];
    if(tool==='probe'||points.length>=2)points=[];points.push(point);updateInspection();
  };
  overlay.classList.add('inspecting');
  $('measurement-position').onsubmit=event=>{
    event.preventDefault();stopReplay();if(!analysis)return;
    const positions=[['measure-ax','measure-ay'],...(tool==='probe'?[]:[['measure-bx','measure-by']])].map(ids=>ids.map(id=>Number($(id).value)));
    if(positions.some(([x,y])=>!Number.isFinite(x)||!Number.isFinite(y)||x<0||y<0||x>=analysis.field.width||y>=analysis.field.height)){$('inspection-hint').textContent='Use coordinates inside the reference image.';return;}
    clearHistory();points=positions;updateInspection();
  };
  document.querySelectorAll('[data-inspect]').forEach(button=>button.onclick=()=>{stopReplay();clearHistory();tool=button.dataset.inspect;points=[];document.querySelectorAll('[data-inspect]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));$('inspection-hint').textContent=tool==='probe'?'Click a subset to inspect values and solver status.':tool==='region'?'Click two opposite rectangle corners; valid subset centres are equally weighted.':'Click two endpoints on the field; measurements snap to subset centres.';updateInspection();});
  $('clear-inspection').onclick=()=>{stopReplay();clearHistory();points=[];updateInspection();};
  $('extract-history').onclick=async()=>{
    stopReplay();if(!analysis)return;
    const entry=activeEntry(),candidates=entries.filter(candidate=>sameAnalysis(entry,candidate));
    const controller=new AbortController();historyAbort?.abort();historyAbort=controller;
    const recipe=analysis.recipe,component=$('review-component').value,selection=points.map(point=>[...point]),selectedTool=tool;
    $('extract-history').disabled=true;$('cancel-history').hidden=false;
    try{
      const rows=await extractHistory(candidates,{recipe,component,points:selection,tool:selectedTool,signal:controller.signal,
        progress:(done,total)=>{$('history-state').textContent=`Analysing saved field ${done} / ${total}…`;}});
      if(controller.signal.aborted)return;history={rows,recipe,component,tool:selectedTool};$('spatial-view').hidden=false;
      $('history-state').textContent=`${rows.length} fields from the same analysis epoch · ${entries.length-rows.length} unrelated fields excluded. Missing measurements stay gaps. ${entry.metadata.comparison?.mode==='consecutive'?'Pair increments at fixed image coordinates; not cumulative material tracking.':entry.metadata.comparison?.mode==='fixed'?'Same fixed reference.':'Reference convention not recorded; no accumulated trajectory inferred.'}`;
      updateInspection();
    }catch(e){if(e.name!=='AbortError')$('history-state').textContent=e.message;}
    finally{if(historyAbort===controller){historyAbort=null;$('extract-history').disabled=false;$('cancel-history').hidden=true;}}
  };
  $('cancel-history').onclick=()=>historyAbort?.abort();
  $('spatial-view').onclick=()=>{history=null;$('spatial-view').hidden=true;updateInspection();};
  $('freeze-field').onclick=async()=>{try{const {field,metadata}=await captureCurrent();let mask=currentMask;if(metadata.roi&&!mask)mask=maskFromMetadata(metadata);add(field,metadata,{select:true,mask});}catch(e){error(e.message);}};
  $('return-live').onclick=()=>returnLive(true);
  $('record-fields').onchange=()=>recordChange(recording());
  $('saved-fields').onchange=()=>selectEntry($('saved-fields').value);
  $('result-seek').oninput=()=>{const entry=entries[Number($('result-seek').value)];if(entry)selectEntry(entry.id);};
  $('replay-results').onclick=()=>{
    if(replaying){stopReplay();return;}
    const base=activeEntry()??entries[0];if(!base)return;
    replayIds=entries.filter(entry=>sameAnalysis(base,entry)).map(entry=>entry.id);if(replayIds.length<2)return;
    if(!reviewing||selected===replayIds.at(-1))selectEntry(replayIds[0]);
    replaying=true;$('replay-results').textContent='Pause result replay';
    const next=()=>{
      if(!replaying)return;
      if(workerBusy||pendingAnalysis){replayTimer=setTimeout(next,100);return;}
      if(!analysis){stopReplay();return;}
      const index=replayIds.indexOf(selected),id=replayIds[index+1];
      if(!id){stopReplay();return;}
      selectEntry(id,{replay:true});replayTimer=setTimeout(next,500);
    };replayTimer=setTimeout(next,500);
  };
  $('apply-recipe-analysis').onclick=()=>{
    stopReplay();if(!analysis||computedView)return;
    const candidates=entries.filter(entry=>sameAnalysis(activeEntry(),entry));
    for(const entry of candidates)entry.recipe={...analysis.recipe};
    changed();updateLibrary();$('processing-batch-state').textContent=`Applied recipe to ${candidates.length} matching field(s); ${entries.length-candidates.length} unrelated field(s) unchanged. Computed arrays are preserved.`;
  };
  for(const [id,delta] of [['previous-result',-1],['next-result',1]])$(id).onclick=()=>{const index=entries.findIndex(e=>e.id===selected);if(entries[index+delta])selectEntry(entries[index+delta].id);};
  $('delete-field').onclick=()=>{stopReplay();const index=entries.findIndex(e=>e.id===Number($('saved-fields').value));if(index<0)return;const active=entries[index].id===selected;entries.splice(index,1);changed();if(active){returnLive(true);selected=null;}updateLibrary();};
  $('clear-fields').onclick=()=>{stopReplay();if(!entries.length||confirm('Delete saved fields from this tab? Export NPZ first if needed.')){returnLive(true);entries.length=0;selected=null;updateLibrary();}};
  $('import-result').onclick=()=>$('result-file').click();
  $('save-project').onclick=async()=>{
    stopReplay();try{
      await beforeProjectExport();rememberView();
      const snapshot=[...entries];
      const bytes=projectArchive(snapshot,{selectedId:selected});
      download('live-dic-results.dicproject',bytes,'application/zip');
      downloadedRevision=projectRevision;updateLibrary();
      $('project-state').textContent=`Download requested: ${snapshot.length} computed fields, shared ROI masks and applied recipes. Check the downloaded file; input images are separate.`;
    }catch(e){error(e.message);}
  };
  $('open-project').onclick=()=>$('project-file').click();
  $('project-file').onchange=async()=>{
    const file=$('project-file').files[0];$('project-file').value='';if(!file)return;
    try{
      if(file.size>256*1048576)throw new Error('Project exceeds the 256 MiB import limit');
      const project=await importProjectArchive(await file.arrayBuffer());
      const drafts=project.records.map(record=>({...record,metadataBytes:encoderBytes(normalizeMetadata(record.metadata))}));
      if(memoryBytes(drafts)>LIMIT_BYTES)throw new Error('Project plus saved results exceeds 128 MiB. Export/clear existing results first. Nothing was imported.');
      const added=project.records.map(record=>add(record.field,record.metadata,{mask:record.mask,recipe:record.recipe}));
      const previous=project.records.findIndex(record=>record.originalId===project.settings.selectedId);
      selectEntry(added[Math.max(0,previous)].id);
      $('project-state').textContent=`Opened ${added.length} fields from ${file.name}. Current acquisition settings and source were not replaced.`;
    }catch(e){error(e.message);}
  };
  function addImported(imported, file) {
    const selection=imported.metadata.importGeometry;
    add(imported.field,{...imported.metadata,name:imported.metadata.name??`${file.name}${selection?.frameCount>1?` · field ${selection.frameIndex+1}/${selection.frameCount}`:''}`},
      {mask:imported.roiMask,select:true,recipe:imported.recipe});
  }
  $('result-file').onchange=async()=>{
    const file=$('result-file').files[0];$('result-file').value='';if(!file)return;
    const serial=++importSerial;
    try{
      if(file.size>LIMIT_BYTES*2)throw new Error('NPZ exceeds the 256 MiB import limit');
      const bytes=await file.arrayBuffer();if(serial!==importSerial)return;
      try {const imported=await importFieldArchive(bytes);if(serial!==importSerial)return;addImported(imported,file);}
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
      if(pending.serial!==importSerial)return;addImported(imported,pending.file);$('import-geometry').close();
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
  $('review-component').onchange=()=>{stopReplay();clearHistory();$('review-scale').value='robust';$('manual-limits').hidden=true;render();};
  for(const id of ['review-scale','review-min','review-max','review-palette','review-vectors','review-vector-step','review-vector-gain','review-contour'])$(id).onchange=()=>{stopReplay();$('manual-limits').hidden=$('review-scale').value!=='manual';render();};
  $('save-npz').onclick=()=>{if(!analysis)return;const entry=activeEntry();download(`live-dic-${entry.id}.npz`,fieldArchive(entry.field,{...entry.metadata,workbench:{view:viewSettings()},analysis:{recipe:analysis.recipe,processingWallMs:analysis.processingWallMs}},analysis,entry.mask),'application/octet-stream');};
  $('save-processed-csv').onclick=()=>{if(!analysis)return;const {values,valid,unit}=componentValues(analysis,$('review-component').value),lines=[`x_px,y_px,${$('review-component').value}_${unit},valid,solver_status`];for(let i=0;i<values.length;i++)lines.push([...pointPosition(i),valid[i]?values[i]:'',valid[i],analysis.field.status[i]].join(','));download('live-dic-displayed-field.csv',lines.join('\n'),'text/csv');};
  $('save-profile').onclick=()=>{if(!extraction.length)return;const columns=Object.keys(extraction[0]);download('live-dic-extraction.csv',[columns.join(','),...extraction.map(row=>columns.map(k=>row[k]??'').join(','))].join('\n'),'text/csv');download('live-dic-extraction.json',JSON.stringify({recipe:analysis.recipe,view:viewSettings(),valueUnit:(history?.tool??tool)==='gauge'?'strain':componentValues(analysis,$('review-component').value).unit,coordinateUnit:'px',distanceUnit:analysis.recipe.unit,comparison:activeEntry().metadata.comparison},null,2),'application/json');};
  $('save-figure').onclick=()=>{if(!analysis)return;const figure=document.createElement('canvas');figure.width=Math.max(1600,canvas.width);const imageHeight=Math.round(canvas.height*figure.width/canvas.width);figure.height=imageHeight+160;const ctx=figure.getContext('2d');ctx.imageSmoothingEnabled=false;ctx.drawImage(canvas,0,0,figure.width,imageHeight);ctx.drawImage(overlay,0,0,figure.width,imageHeight);ctx.fillStyle='#fff';ctx.fillRect(0,imageHeight,figure.width,160);ctx.fillStyle='#163540';ctx.font='32px system-ui';ctx.fillText(`${$('review-component').selectedOptions[0].textContent} · ${$('low').textContent} … ${$('high').textContent}`,40,imageHeight+50);ctx.font='22px system-ui';ctx.fillText($('review-component').value.startsWith('computed-p')?`Computed local model coefficient · unaffected by workbench smoothing / VSG · solver ${analysis.field.config.outlierPolicy}`:`${analysis.recipe.strainMeasure} strain · ${analysis.recipe.strainSource} · ${analysis.recipe.smoothing} smoothing · solver ${analysis.field.config.outlierPolicy}`,40,imageHeight+90);ctx.fillText('Reference image coordinates · live-dic · foundational work by Pierre Margerit',40,imageHeight+130);figure.toBlob(blob=>blob&&download('live-dic-field.png',blob,'image/png'),'image/png');};
  updateLibrary();
  return { add, returnLive, setMask(mask){currentMask=mask;}, get reviewing(){return reviewing;}, get recording(){return recording();},
    setReady(ready){$('freeze-field').disabled=!ready;$('record-fields').disabled=!ready;},
    stopRecording(){ $('record-fields').checked=false;recordChange(false); },
    clearInspection(){stopReplay();clearHistory();points=[];updateInspection();},
    report(){const values=analysis?componentValues(analysis,$('review-component').value):null;return{count:entries.length,selected,reviewing,replaying,unsavedProject:hasUnsaved(),memoryBytes:memoryBytes(),provenance:activeEntry()?{analysisEpoch:activeEntry().metadata.analysisEpoch,archiveSha256:activeEntry().metadata.archiveSha256,importGeometry:activeEntry().metadata.importGeometry}:null,recipe:analysis?.recipe??null,statistics:values?statistics(values.values,values.valid):null,extraction};},
    destroy(){stopReplay();historyAbort?.abort();clearTimeout(debounce);chartResize?.disconnect();cancelAnimationFrame(chartResizeFrame);worker?.terminate();window.removeEventListener('beforeunload',beforeUnload);} };
}
