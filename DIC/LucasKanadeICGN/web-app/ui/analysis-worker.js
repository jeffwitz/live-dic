import { analyzeField, componentValues, nearestIndex, extensometer, regionSummary } from './postprocess.js';
let field, mask, extractionMask;
self.onmessage = ({ data }) => {
  try {
    if (data.clear) { field = mask = null; return; }
    if (data.extract) {
      if (Object.hasOwn(data, 'mask')) extractionMask = data.mask;
      const analysis = analyzeField(data.field, data.recipe, extractionMask);
      let value = null, status = null, region = null;
      if(data.tool==='region'){region=regionSummary(analysis,data.component,...data.points);value=region?.mean??null;}
      else if (data.tool === 'gauge') value = extensometer(analysis, ...data.points)?.engineeringStrain ?? null;
      else {
        const index = nearestIndex(data.field, ...data.points[0]);
        if (index >= 0) { const component = componentValues(analysis, data.component); value = component.valid[index] ? component.values[index] : null; status = data.field.status[index]; }
      }
      self.postMessage({ id: data.id, value, status, count:region?.count??null,total:region?.total??null, processingWallMs: analysis.processingWallMs }); return;
    }
    if (data.field) { field = data.field; mask = data.mask; }
    const { field: source, ...analysis } = analyzeField(field, data.recipe, mask);
    const transfer = [...Object.values(analysis.fields), analysis.valid, analysis.strainValid].map(v => v.buffer);
    self.postMessage({ id: data.id, analysis }, transfer);
  } catch (error) { self.postMessage({ id: data.id, error: error.message }); }
};
