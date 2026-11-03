import { analyzeField, componentValues, nearestIndex, extensometer, regionSummary } from './postprocess.js';

function snappedPoint(field, point) {
  const index=nearestIndex(field,...point);
  if(index<0)return null;
  const [gw]=field.grid,c=(field.subsetSize-1)/2,s=field.subsetStride;
  return {index,x:index%gw*s+c,y:Math.floor(index/gw)*s+c};
}

/** Evaluate one saved geometry without interpolation or gap filling. */
export function sampleMeasurement(analysis, measurement, component) {
  const {field}=analysis,{id,name,type,points}=measurement;
  component=measurement.component??component;
  if(!['point','probe','line','region','gauge'].includes(type))throw new Error(`Unknown measurement type: ${type}`);
  const expected=type==='point'||type==='probe'?1:2;
  if(!Array.isArray(points)||points.length!==expected||points.some(point=>!Array.isArray(point)||point.length!==2||!point.every(Number.isFinite)))throw new Error(`Invalid geometry for ${name||type}`);
  const base={measurementId:id,measurementName:name,measurementType:type==='probe'?'point':type,component:type==='gauge'?'engineeringStrain':component};
  if(type==='gauge') {
    const gauge=extensometer(analysis,...points);
    return [{...base,sampleIndex:0,...(gauge?{x_px:gauge.points[0][0],y_px:gauge.points[0][1],referenceLength:gauge.referenceLength,deformedLength:gauge.deformedLength,extension:gauge.extension,value:gauge.engineeringStrain,unit:'strain',status:1}:{value:null,unit:'strain',status:null})}];
  }
  if(type==='region') {
    if(component==='status')throw new Error('Choose a scalar quantity for region history; solver status is categorical');
    const summary=regionSummary(analysis,component,...points);
    return [{...base,sampleIndex:0,...(summary?{x0_px:summary.bounds[0],y0_px:summary.bounds[1],x1_px:summary.bounds[2],y1_px:summary.bounds[3],value:summary.mean,mean:summary.mean,standardDeviation:summary.standardDeviation,minimum:summary.min,maximum:summary.max,validSamples:summary.count,selectedGridPositions:summary.selected,totalGridPositions:summary.total,unit:summary.unit}:{value:null,unit:null,validSamples:0,selectedGridPositions:0,totalGridPositions:0})}];
  }
  const values=componentValues(analysis,component),samples=[];
  if(type==='point'||type==='probe') {
    const point=snappedPoint(field,points[0]);
    return [{...base,sampleIndex:0,...(point?{x_px:point.x,y_px:point.y,value:values.valid[point.index]?values.values[point.index]:null,status:field.status[point.index],unit:values.unit}:{value:null,status:null,unit:values.unit})}];
  }
  const [a,b]=points,steps=Math.max(1,Math.ceil(Math.hypot(b[0]-a[0],b[1]-a[1])/field.subsetStride*2)),seen=new Set();
  for(let n=0;n<=steps;n++) {
    const t=n/steps,point=snappedPoint(field,[a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])]);
    if(!point||seen.has(point.index))continue;
    seen.add(point.index);samples.push({...base,sampleIndex:samples.length,distance:Math.hypot(point.x-a[0],point.y-a[1])*analysis.recipe.unitsPerPixel,x_px:point.x,y_px:point.y,value:values.valid[point.index]?values.values[point.index]:null,status:field.status[point.index],unit:values.unit});
  }
  return samples;
}

/** A history never mixes unrelated reference epochs just because their grids happen to match. */
export function sameAnalysis(reference, candidate) {
  const a=reference.field,b=candidate.field,epoch=reference.metadata.analysisEpoch??reference.metadata.archiveSha256;
  return !!epoch && epoch===(candidate.metadata.analysisEpoch??candidate.metadata.archiveSha256)
    && a.width===b.width && a.height===b.height && a.subsetSize===b.subsetSize && a.subsetStride===b.subsetStride
    && a.config.motionModel===b.config.motionModel && a.config.roi?.maskSha256===b.config.roi?.maskSha256
    && reference.metadata.comparison?.mode===candidate.metadata.comparison?.mode;
}

/** Bounded chronological analysis: clone one field at a time; never queue an entire experiment. */
export async function extractHistory(entries, { recipe, component, points, tool, measurements, signal, progress=()=>{}, loadEntry=async entry=>entry }) {
  if(measurements!==undefined) {
    if(!Array.isArray(measurements)||!measurements.length)throw new Error('Add at least one saved measurement before extracting time history');
    if(!entries.length||entries.slice(1).some(entry=>!sameAnalysis(entries[0],entry)))throw new Error('History requires one analysis epoch, ROI and grid');
    const worker=typeof Worker==='function'?new Worker(new URL('./analysis-worker.js',import.meta.url),{type:'module'}):null;
    const rows=[];let pendingReject=null;
    const aborted=()=>{if(signal?.aborted)throw new DOMException('History extraction cancelled','AbortError');};
    const cancel=()=>{worker?.terminate();pendingReject?.(new DOMException('History extraction cancelled','AbortError'));};
    signal?.addEventListener('abort',cancel,{once:true});
    try {
      for(let index=0;index<entries.length;index++) {
        aborted();const entry=await loadEntry(entries[index]);aborted();let samples;
        if(worker)samples=await new Promise((resolve,reject)=>{
          pendingReject=reject;
          worker.onmessage=({data})=>{pendingReject=null;if(data.error)reject(new Error(data.error));else resolve(data.rows);};
          worker.onerror=event=>{pendingReject=null;reject(new Error(event.message));};
          worker.postMessage({id:index,extractMeasurements:true,field:entry.field,...(index===0?{mask:entry.mask}:{}),recipe,component,measurements});
        });
        else {await new Promise(resolve => { setTimeout(resolve,0); });aborted();const analysis=analyzeField(entry.field,recipe,entry.mask);samples=measurements.flatMap(measurement=>sampleMeasurement(analysis,measurement,component));}
        aborted();
        const metadata=entry.metadata,sourceIndex=metadata.comparison?.target?.index??metadata.frameIndex??null,mediaTimeSeconds=metadata.comparison?.target?.timeSeconds??null;
        for(const sample of samples)rows.push({order:index+1,fieldId:entry.id,sourceIndex,mediaTimeSeconds,...sample});
        progress(index+1,entries.length);
      }
      return rows;
    } finally {signal?.removeEventListener('abort',cancel);worker?.terminate();}
  }
  if (!['probe','gauge','region'].includes(tool) || points.length<(tool==='probe'?1:2)) throw new Error('Place a point, two gauge endpoints or two region corners before extracting history');
  if(tool==='region'&&component==='status')throw new Error('Choose a scalar field quantity for a region-mean history; status codes are categories');
  if(!entries.length||entries.slice(1).some(entry=>!sameAnalysis(entries[0],entry)))throw new Error('History requires one analysis epoch, ROI and grid');
  const worker=typeof Worker==='function'?new Worker(new URL('./analysis-worker.js',import.meta.url),{type:'module'}):null;
  const rows=[];let pendingReject=null;
  const aborted=()=>{if(signal?.aborted)throw new DOMException('History extraction cancelled','AbortError');};
  const cancel=()=>{worker?.terminate();pendingReject?.(new DOMException('History extraction cancelled','AbortError'));};
  signal?.addEventListener('abort',cancel,{once:true});
  try {
    for(let index=0;index<entries.length;index++) {
      aborted();const entry=await loadEntry(entries[index]);aborted();let result;
      if(worker) result=await new Promise((resolve,reject)=>{
        pendingReject=reject;
        worker.onmessage=({data})=>{pendingReject=null;if(data.error)reject(new Error(data.error));else resolve(data);};
        worker.onerror=event=>{pendingReject=null;reject(new Error(event.message));};
        worker.postMessage({id:index,extract:true,field:entry.field,...(index===0?{mask:entry.mask}:{}),recipe,component,points,tool});
      });
      else {
        await new Promise(resolve => { setTimeout(resolve,0); });aborted();const analysis=analyzeField(entry.field,recipe,entry.mask);
        const position=nearestIndex(entry.field,...points[0]),values=componentValues(analysis,component);
        const region=tool==='region'?regionSummary(analysis,component,...points):null;
        result={count:region?.count??null,total:region?.total??null,value:tool==='region'?region?.mean??null:tool==='gauge'?extensometer(analysis,...points)?.engineeringStrain??null:position>=0&&values.valid[position]?values.values[position]:null,
          status:position>=0?entry.field.status[position]:null};
      }
      aborted();rows.push({order:index+1,fieldId:entry.id,sourceIndex:entry.metadata.comparison?.target?.index??entry.metadata.frameIndex??null,
        mediaTimeSeconds:entry.metadata.comparison?.target?.timeSeconds??null,value:result.value,status:result.status,validRegionSamples:result.count??null,regionGridPositions:result.total??null});
      progress(index+1,entries.length);
    }
    return rows;
  } finally { signal?.removeEventListener('abort',cancel);worker?.terminate(); }
}
