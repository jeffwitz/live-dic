import { analyzeField, componentValues, nearestIndex, extensometer } from './postprocess.js';

/** A history never mixes unrelated reference epochs just because their grids happen to match. */
export function sameAnalysis(reference, candidate) {
  const a=reference.field,b=candidate.field,epoch=reference.metadata.analysisEpoch??reference.metadata.archiveSha256;
  return !!epoch && epoch===(candidate.metadata.analysisEpoch??candidate.metadata.archiveSha256)
    && a.width===b.width && a.height===b.height && a.subsetSize===b.subsetSize && a.subsetStride===b.subsetStride
    && a.config.motionModel===b.config.motionModel && a.config.roi?.maskSha256===b.config.roi?.maskSha256
    && reference.metadata.comparison?.mode===candidate.metadata.comparison?.mode;
}

/** Bounded chronological analysis: clone one field at a time; never queue an entire experiment. */
export async function extractHistory(entries, { recipe, component, points, tool, signal, progress=()=>{} }) {
  if (!['probe','gauge'].includes(tool) || points.length<(tool==='gauge'?2:1)) throw new Error('Place a point or two gauge endpoints before extracting history');
  if(!entries.length||entries.slice(1).some(entry=>!sameAnalysis(entries[0],entry)))throw new Error('History requires one analysis epoch, ROI and grid');
  const worker=typeof Worker==='function'?new Worker(new URL('./analysis-worker.js',import.meta.url),{type:'module'}):null;
  const rows=[];let pendingReject=null;
  const aborted=()=>{if(signal?.aborted)throw new DOMException('History extraction cancelled','AbortError');};
  const cancel=()=>{worker?.terminate();pendingReject?.(new DOMException('History extraction cancelled','AbortError'));};
  signal?.addEventListener('abort',cancel,{once:true});
  try {
    for(let index=0;index<entries.length;index++) {
      aborted();const entry=entries[index];let result;
      if(worker) result=await new Promise((resolve,reject)=>{
        pendingReject=reject;
        worker.onmessage=({data})=>{pendingReject=null;if(data.error)reject(new Error(data.error));else resolve(data);};
        worker.onerror=event=>{pendingReject=null;reject(new Error(event.message));};
        worker.postMessage({id:index,extract:true,field:entry.field,...(index===0?{mask:entry.mask}:{}),recipe,component,points,tool});
      });
      else {
        await new Promise(resolve=>setTimeout(resolve,0));aborted();const analysis=analyzeField(entry.field,recipe,entry.mask);
        const position=nearestIndex(entry.field,...points[0]),values=componentValues(analysis,component);
        result={value:tool==='gauge'?extensometer(analysis,...points)?.engineeringStrain??null:position>=0&&values.valid[position]?values.values[position]:null,
          status:position>=0?entry.field.status[position]:null};
      }
      aborted();rows.push({order:index+1,fieldId:entry.id,sourceIndex:entry.metadata.comparison?.target?.index??entry.metadata.frameIndex??null,
        mediaTimeSeconds:entry.metadata.comparison?.target?.timeSeconds??null,value:result.value,status:result.status});
      progress(index+1,entries.length);
    }
    return rows;
  } finally { signal?.removeEventListener('abort',cancel);worker?.terminate(); }
}
