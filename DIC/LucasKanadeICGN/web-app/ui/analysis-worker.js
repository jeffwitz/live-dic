import { analyzeField } from './postprocess.js';
let field, mask;
self.onmessage = ({ data }) => {
  try {
    if (data.field) { field = data.field; mask = data.mask; }
    const { field: source, ...analysis } = analyzeField(field, data.recipe, mask);
    const transfer = [...Object.values(analysis.fields), analysis.valid, analysis.strainValid].map(v => v.buffer);
    self.postMessage({ id: data.id, analysis }, transfer);
  } catch (error) { self.postMessage({ id: data.id, error: error.message }); }
};
