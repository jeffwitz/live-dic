import { analyzeField, componentValues, nearestIndex, extensometer, regionSummary } from './postprocess.js';

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
