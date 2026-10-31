import { DEFAULT_RECIPE, analyzeField, componentValues, statistics, nearestIndex, extensometer, validateField } from './postprocess.js';
import { fieldArchive, importFieldArchive } from './archive.js';

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
export function installWorkbench({ captureCurrent, pause, error, recordChange, liveChanged, openResults }) {
  const entries = [], canvas = $('review-field'), overlay = $('inspection'), chart = $('profile-chart');
  let nextId = 1, selected = null, analysis = null, currentMask = null, reviewing = false;
  let tool = 'probe', points = [], extraction = [], computedView = false;
  let requestId = 0, workerBusy = false, pendingAnalysis = false, workerFieldId = null, debounce = null;
  const worker = typeof Worker === 'function' ? new Worker(new URL('./analysis-worker.js', import.meta.url), { type: 'module' }) : null;
  const recording = () => $('record-fields').checked;
  const activeEntry = () => entries.find(entry => entry.id === selected);

  function memoryBytes(extra = []) {
    const masks = new Set(); let bytes = 0;
    for (const entry of [...entries, ...extra]) {
      bytes += entry.field.parameters.byteLength + entry.field.status.byteLength + entry.field.iterations.byteLength + entry.metadataBytes;
      if (entry.mask && !masks.has(entry.mask)) { masks.add(entry.mask); bytes += entry.mask.byteLength; }
    }
    return bytes;
  }
  function updateLibrary() {
    const select = $('saved-fields'); select.replaceChildren();
    if (!entries.length) select.add(new Option('No saved fields', ''));
    for (const entry of entries) select.add(new Option(entry.label, entry.id));
    select.value = selected ?? entries.at(-1)?.id ?? '';
    $('library-summary').textContent = `${entries.length} field(s) · ${(memoryBytes()/1048576).toFixed(1)} / 128 MiB in this tab. Save NPZ before closing; images are not recorded.`;
    $('clear-fields').disabled = !entries.length; $('delete-field').disabled = !entries.length;
  }
  function add(field, metadata = {}, { select = false, mask = currentMask, recipe = null } = {}) {
    validateField(field);
    // A single-field archive carries its own timing sample. The live timing log remains in direct export.
    metadata = { ...metadata, samples: metadata.samples?.slice(-1), sampleScope: 'this field',
      roi: metadata.roi ? { ...metadata.roi, data: undefined, encoding: 'roi_mask.npy' } : null };
    const entry = { id: nextId++, field, metadata, mask, recipe, metadataBytes: encoderBytes(metadata) };
    if (mask && mask.length !== field.width*field.height) throw new Error('Saved field mask dimensions differ');
    if (memoryBytes([entry]) > LIMIT_BYTES) {
      $('record-fields').checked = false; recordChange(false); pause();
      throw new Error('Results storage reached 128 MiB. Acquisition is paused. Export or delete fields before continuing; no stored result was discarded.');
    }
    entry.label = fieldLabel(metadata, entry.id); entries.push(entry); updateLibrary();
    if (select) selectEntry(entry.id);
    return entry;
  }
  function returnLive() {
    reviewing = false; requestId++; pendingAnalysis = false; clearTimeout(debounce);
    $('results-workbench').hidden = true; canvas.hidden = true; overlay.hidden = true;
    $('field').hidden = false; $('return-live').hidden = true;
    liveChanged(false);
  }
  function selectEntry(id) {
    const entry = entries.find(value => value.id === Number(id)); if (!entry) return;
    pause(); selected = entry.id; analysis = null; reviewing = true; computedView = false; points = []; extraction = [];
    $('saved-fields').value = entry.id; $('results-workbench').hidden = false;
    $('return-live').hidden = false; canvas.hidden = false; overlay.hidden = false; $('field').hidden = true;
    canvas.width = overlay.width = entry.field.width; canvas.height = overlay.height = entry.field.height;
    $('result-title').textContent = entry.label;
    $('result-provenance').textContent = `${entry.field.width} × ${entry.field.height} · ${entry.field.config.motionModel} · ${entry.field.subsetSize}/${entry.field.subsetStride} · solver spatial policy: ${entry.field.config.outlierPolicy ?? 'unspecified'} · ${entry.metadata.comparison?.mode ?? 'imported'} reference. No image is stored behind this field.`;
    setRecipe(entry.recipe ?? { ...DEFAULT_RECIPE, strainSource: entry.field.config.motionModel === 'translation' ? 'vsg' : 'direct' });
    $('compare-computed').textContent = 'Show computed field';
    liveChanged(true); openResults(); scheduleAnalysis();
  }
  function analysisComplete(id, result, failure) {
    workerBusy = false;
    if (id === requestId && reviewing) {
      if (failure) { analysis = null; $('analysis-state').textContent = failure; error(failure); }
      else {
        analysis = { ...result, field: activeEntry().field }; if (!computedView) activeEntry().recipe = result.recipe;
        $('analysis-state').textContent = `Analysis ${result.processingWallMs.toFixed(1)} ms, outside the live solver. Computed arrays preserved.`;
        $('processing-summary').textContent = result.recipe.smoothing === 'none' && result.recipe.strainSource === 'direct'
          ? 'Computed displacement and local model gradients. Solver smoothing, if enabled during correlation, is already part of these values.'
          : `${result.recipe.smoothing} displacement smoothing · ${result.recipe.strainSource === 'vsg' ? 'VSG strain' : 'direct gradients'} · ${result.recipe.window} × ${result.recipe.window} grid points · nominal support span ${result.supportSpanPx} px. No filling across gaps.`;
        render();
      }
      for (const id of ['save-npz','save-processed-csv','save-figure']) $(id).disabled = !analysis;
    }
    if (pendingAnalysis && reviewing) runAnalysis();
  }
  if (worker) {
    worker.onmessage = ({ data }) => analysisComplete(data.id, data.analysis, data.error);
    worker.onerror = event => { workerBusy = false; error(`Analysis worker: ${event.message}`); };
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
    requestId++; pendingAnalysis = true; clearTimeout(debounce); analysis = null;
    $('analysis-state').textContent = 'Updating analysis…';
    for (const id of ['save-npz','save-processed-csv','save-figure']) $(id).disabled = true;
    debounce = setTimeout(runAnalysis, 60);
  }
  function format(value, unit) { if (value == null) return 'unavailable'; return `${number(STRAIN_COMPONENTS.has($('review-component').value) ? value*1e6 : value)} ${STRAIN_COMPONENTS.has($('review-component').value) ? 'µε' : unit}`; }
  function render() {
    if (!analysis) return;
    const component = $('review-component').value, { values, valid, unit } = componentValues(analysis, component);
    const stats = statistics(values, valid), field = analysis.field, [gw,gh] = field.grid;
    const nonnegative = ['magnitude','vonMises','iterations'].includes(component);
    let limit = Math.max(Math.abs(stats.p02 ?? 0), Math.abs(stats.p98 ?? 0), 1e-12);
    if (nonnegative) limit = Math.max(stats.p98 ?? 0, 1e-12);
    const small = new OffscreenCanvas(gw, gh), context = small.getContext('2d'), image = context.createImageData(gw, gh);
    for (let i=0;i<values.length;i++) {
      let colour = [71,86,96];
      if (component === 'status') colour = { '-3':[160,170,178],'-2':[202,51,64],'-1':[71,86,96],0:[240,193,52],1:[36,151,110],2:[238,135,45] }[Math.round(values[i])] ?? colour;
      else if (valid[i]) {
        const t = Math.max(-1,Math.min(1,values[i]/limit)), endpoint = t<0 ? [39,91,171] : [187,56,54];
        colour = endpoint.map(value => Math.round(247+(value-247)*Math.abs(t)));
      }
      image.data.set([...colour,255],i*4);
    }
    context.putImageData(image,0,0);
    const ctx = canvas.getContext('2d'); ctx.fillStyle = '#263b47'; ctx.fillRect(0,0,canvas.width,canvas.height); ctx.imageSmoothingEnabled = false;
    // Samples occupy a stride-sized cell centred on the actual reference subset centre.
    const offset = (field.subsetSize-field.subsetStride-1)/2;
    ctx.drawImage(small,offset,offset,gw*field.subsetStride,gh*field.subsetStride);
    $('low').textContent = component==='status' ? 'Excluded / failed' : format(nonnegative?0:-limit,unit);
    $('high').textContent = component==='status' ? 'Accepted / flagged' : format(limit,unit);
    document.querySelector('.legend').classList.toggle('status',component==='status');
    document.querySelector('.legend').classList.toggle('magnitude',nonnegative);
    const clipped = Array.from(values).reduce((n,v,i)=>n+ +(valid[i] && Math.abs(v)>limit),0);
    const summary = $('field-summary'); summary.replaceChildren();
    for (const [label,value] of [['Valid values',`${stats.count} / ${values.length}`],['Minimum',format(stats.min,unit)],['Mean',format(stats.mean,unit)],['Maximum',format(stats.max,unit)],['Colour clipping',`${clipped} values · 2–98% scale`]]) {
      const box = document.createElement('div'), strong = document.createElement('strong'); box.textContent = label; strong.textContent = value; box.append(strong); summary.append(box);
    }
    updateInspection();
  }
  function pointPosition(index) {
    const f = analysis.field;return [(index%f.grid[0])*f.subsetStride+(f.subsetSize-1)/2,Math.floor(index/f.grid[0])*f.subsetStride+(f.subsetSize-1)/2];
  }
  function updateInspection() {
    const ctx = overlay.getContext('2d');ctx.clearRect(0,0,overlay.width,overlay.height);extraction=[];
    if (!analysis || !points.length) { $('probe-value').textContent='Click the field above to place a measurement.'; drawChart([]);$('save-profile').disabled=true;return; }
    const component=$('review-component').value,{values,valid,unit}=componentValues(analysis,component),f=analysis.field;
    const indices=points.map(([x,y])=>nearestIndex(f,x,y)),snapped=indices.map(i=>i<0?null:pointPosition(i));
    ctx.strokeStyle='#fff';ctx.fillStyle='#153c48';ctx.lineWidth=Math.max(2,f.width/600);
    for(const point of snapped)if(point){ctx.beginPath();ctx.arc(...point,Math.max(4,f.width/300),0,2*Math.PI);ctx.fill();ctx.stroke();}
    if(tool==='probe') {
      const i=indices[0];
      if(i<0){$('probe-value').textContent='Outside the measured grid.';return;}
      $('probe-value').textContent=`Subset ${i} · x=${snapped[0][0].toFixed(1)}, y=${snapped[0][1].toFixed(1)} px\n${STATUS[Math.round(f.status[i])] ?? 'Unknown status'} · ${f.iterations[i]} iterations\n${component}: ${valid[i]?format(values[i],unit):'unavailable — not a valid measurement'}`;
      extraction=[{distance:0,x:snapped[0][0],y:snapped[0][1],value:valid[i]?values[i]:null,status:f.status[i]}];
    } else if(points.length===2 && snapped.every(Boolean)) {
      const [a,b]=snapped;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();
      if(tool==='gauge') {
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
    drawChart(tool==='line'?extraction:[]);$('save-profile').disabled=!extraction.length;
  }
  function drawChart(rows) {
    const ctx=chart.getContext('2d'),w=chart.width,h=chart.height;ctx.clearRect(0,0,w,h);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);
    const finite=rows.filter(r=>r.value!=null);ctx.fillStyle='#607782';ctx.font='15px system-ui';
    if(!finite.length){ctx.fillText('Select a line profile to plot actual field samples.',24,40);return;}
    let low=Math.min(...finite.map(r=>r.value)),high=Math.max(...finite.map(r=>r.value));const span=Math.max(1e-12,Math.abs(high)*1e-9,high-low),xmax=Math.max(...rows.map(r=>r.distance),1e-12);
    low-=.08*span;high+=.08*span;const x=v=>76+v/xmax*(w-108),y=v=>h-48-(v-low)/(high-low)*(h-82);
    ctx.strokeStyle='#d9e3e7';ctx.lineWidth=1;for(let i=0;i<=4;i++){const v=low+(high-low)*i/4;ctx.beginPath();ctx.moveTo(76,y(v));ctx.lineTo(w-32,y(v));ctx.stroke();ctx.fillText(number(v),4,y(v)+5);}
    ctx.strokeStyle='#117c72';ctx.lineWidth=2.5;ctx.beginPath();let pen=false;for(const row of rows){if(row.value==null){pen=false;continue;}if(pen)ctx.lineTo(x(row.distance),y(row.value));else ctx.moveTo(x(row.distance),y(row.value));pen=true;}ctx.stroke();
    ctx.fillText(`Distance along snapped samples (${analysis.recipe.unit})`,w/2-160,h-10);
    ctx.fillText($('review-component').selectedOptions[0].textContent,76,22);
  }
  overlay.onpointerdown = event => {
    if(!analysis)return;const rect=overlay.getBoundingClientRect(),point=[(event.clientX-rect.left)/rect.width*overlay.width,(event.clientY-rect.top)/rect.height*overlay.height];
    if(tool==='probe'||points.length>=2)points=[];points.push(point);updateInspection();
  };
  overlay.classList.add('inspecting');
  document.querySelectorAll('[data-inspect]').forEach(button=>button.onclick=()=>{tool=button.dataset.inspect;points=[];document.querySelectorAll('[data-inspect]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));$('inspection-hint').textContent=tool==='probe'?'Click a subset to inspect values and solver status.':'Click two endpoints on the field; measurements snap to subset centres.';updateInspection();});
  $('clear-inspection').onclick=()=>{points=[];updateInspection();};
  $('freeze-field').onclick=async()=>{try{const {field,metadata}=await captureCurrent();let mask=currentMask;if(metadata.roi&&!mask)mask=maskFromMetadata(metadata);add(field,metadata,{select:true,mask});}catch(e){error(e.message);}};
  $('return-live').onclick=returnLive;
  $('record-fields').onchange=()=>recordChange(recording());
  $('saved-fields').onchange=()=>selectEntry($('saved-fields').value);
  $('delete-field').onclick=()=>{const index=entries.findIndex(e=>e.id===Number($('saved-fields').value));if(index<0)return;const active=entries[index].id===selected;entries.splice(index,1);if(active){returnLive();selected=null;}updateLibrary();};
  $('clear-fields').onclick=()=>{if(!entries.length||confirm('Delete saved fields from this tab? Export NPZ first if needed.')){returnLive();entries.length=0;selected=null;updateLibrary();}};
  $('import-result').onclick=()=>$('result-file').click();
  $('result-file').onchange=async()=>{const file=$('result-file').files[0];$('result-file').value='';if(!file)return;try{if(file.size>LIMIT_BYTES*2)throw new Error('NPZ exceeds the 256 MiB import limit');const imported=await importFieldArchive(await file.arrayBuffer());add(imported.field,{...imported.metadata,name:imported.metadata.name??file.name},{mask:imported.roiMask,select:true,recipe:imported.recipe});}catch(e){error(e.message);}};
  for(const id of ['post-smoothing','strain-source','post-window','strain-measure','exclude-flagged','length-unit','length-scale'])$(id).onchange=()=>{
    if($('post-smoothing').value!=='none')$('strain-source').value='vsg';
    if(activeEntry()?.field.config.motionModel==='translation')$('strain-source').value='vsg';
    $('length-scale').disabled=$('length-unit').value==='px';if($('length-scale').disabled)$('length-scale').value=1;
    computedView=false;$('compare-computed').textContent='Show computed field';scheduleAnalysis();
  };
  $('reset-processing').onclick=()=>{setRecipe({...DEFAULT_RECIPE,strainSource:activeEntry()?.field.config.motionModel==='translation'?'vsg':'direct'});computedView=false;scheduleAnalysis();};
  $('compare-computed').onclick=()=>{computedView=!computedView;$('compare-computed').textContent=computedView?'Restore processing recipe':'Show computed field';scheduleAnalysis();};
  $('review-component').onchange=render;
  $('save-npz').onclick=()=>{if(!analysis)return;const entry=activeEntry();download(`live-dic-${entry.id}.npz`,fieldArchive(entry.field,{...entry.metadata,analysis:{recipe:analysis.recipe,processingWallMs:analysis.processingWallMs}},analysis,entry.mask),'application/octet-stream');};
  $('save-processed-csv').onclick=()=>{if(!analysis)return;const {values,valid,unit}=componentValues(analysis,$('review-component').value),lines=[`x_px,y_px,${$('review-component').value}_${unit},valid,solver_status`];for(let i=0;i<values.length;i++)lines.push([...pointPosition(i),valid[i]?values[i]:'',valid[i],analysis.field.status[i]].join(','));download('live-dic-displayed-field.csv',lines.join('\n'),'text/csv');};
  $('save-profile').onclick=()=>{if(!extraction.length)return;const columns=Object.keys(extraction[0]);download('live-dic-extraction.csv',[columns.join(','),...extraction.map(row=>columns.map(k=>row[k]??'').join(','))].join('\n'),'text/csv');};
  $('save-figure').onclick=()=>{if(!analysis)return;const figure=document.createElement('canvas');figure.width=canvas.width;figure.height=canvas.height+160;const ctx=figure.getContext('2d');ctx.drawImage(canvas,0,0);ctx.drawImage(overlay,0,0);ctx.fillStyle='#fff';ctx.fillRect(0,canvas.height,figure.width,160);ctx.fillStyle='#163540';ctx.font='32px system-ui';ctx.fillText(`${$('review-component').selectedOptions[0].textContent} · ${$('low').textContent} … ${$('high').textContent}`,40,canvas.height+50);ctx.font='22px system-ui';ctx.fillText(`${analysis.recipe.strainMeasure} strain · ${analysis.recipe.strainSource} · ${analysis.recipe.smoothing} smoothing · solver ${analysis.field.config.outlierPolicy}`,40,canvas.height+90);ctx.fillText('Reference image coordinates · live-dic · foundational work by Pierre Margerit',40,canvas.height+130);figure.toBlob(blob=>blob&&download('live-dic-field.png',blob,'image/png'),'image/png');};
  updateLibrary();
  return { add, returnLive, setMask(mask){currentMask=mask;}, get reviewing(){return reviewing;}, get recording(){return recording();},
    setReady(ready){$('freeze-field').disabled=!ready;$('record-fields').disabled=!ready;},
    stopRecording(){ $('record-fields').checked=false;recordChange(false); },
    clearInspection(){points=[];updateInspection();},
    report(){const values=analysis?componentValues(analysis,$('review-component').value):null;return{count:entries.length,selected,reviewing,memoryBytes:memoryBytes(),recipe:analysis?.recipe??null,statistics:values?statistics(values.values,values.valid):null,extraction};},
    destroy(){worker?.terminate();} };
}
