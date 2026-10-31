/** Explicit, reversible field analysis. No GPU/DOM access and no mutation of solver data. */
export const DEFAULT_RECIPE = Object.freeze({ smoothing: 'none', window: 5,
  strainSource: 'direct', strainMeasure: 'small', excludeFlagged: false, unitsPerPixel: 1, unit: 'px' });

export function validateRecipe(options = {}) {
  const recipe = { ...DEFAULT_RECIPE, ...options };
  for (const key of Object.keys(recipe)) if (!(key in DEFAULT_RECIPE)) throw new Error(`Unknown processing option: ${key}`);
  if (!['none', 'median', 'gaussian'].includes(recipe.smoothing)) throw new Error('Unknown displacement smoothing');
  if (![3, 5, 7, 9].includes(recipe.window)) throw new Error('Analysis window must be 3, 5, 7 or 9 grid points');
  if (!['direct', 'vsg'].includes(recipe.strainSource)) throw new Error('Strain source must be direct or vsg');
  if (!['small', 'green'].includes(recipe.strainMeasure)) throw new Error('Unknown strain measure');
  if (recipe.smoothing !== 'none' && recipe.strainSource !== 'vsg') throw new Error('Smoothed displacement requires VSG strain; direct gradients belong to the computed field');
  if (typeof recipe.excludeFlagged !== 'boolean') throw new Error('excludeFlagged must be boolean');
  if (!Number.isFinite(recipe.unitsPerPixel) || recipe.unitsPerPixel <= 0) throw new Error('Length calibration must be positive and finite');
  if (!['px', 'mm', 'µm', 'm'].includes(recipe.unit)) throw new Error('Unsupported length unit');
  if (recipe.unit === 'px' && recipe.unitsPerPixel !== 1) throw new Error('Pixel units require a scale of one');
  return Object.freeze(recipe);
}

export function validateField(field) {
  const { width, height, grid, subsetSize, subsetStride, status, parameters, iterations, config } = field;
  if (![width, height, subsetSize, subsetStride, ...grid ?? []].every(v => Number.isInteger(v) && v > 0)
      || grid?.length !== 2 || subsetSize > width || subsetSize > height
      || grid[0] !== 1 + Math.floor((width-subsetSize)/subsetStride)
      || grid[1] !== 1 + Math.floor((height-subsetSize)/subsetStride)) throw new Error('Inconsistent field image/grid geometry');
  const count = grid[0]*grid[1], components = config?.motionModel === 'quadratic' ? 12 : 6;
  if (!['translation','affine','quadratic'].includes(config?.motionModel)) throw new Error('Unknown field motion model');
  if (status?.length !== count || iterations?.length !== count || parameters?.length !== components*count) throw new Error('Field arrays do not match grid geometry');
  for (const values of [status, iterations, parameters]) if (!ArrayBuffer.isView(values) || !values.every(Number.isFinite)) throw new Error('Field arrays must contain finite numeric values');
  return { count, components };
}

/** Constant-time conservative rectangular visibility: never bridge an excluded pixel. */
function visibility(mask, width, height) {
  if (!mask) return () => true;
  if (mask.length !== width*height) throw new Error('Analysis mask shape does not match the field');
  const pitch = width+1, prefix = new Uint32Array(pitch*(height+1));
  for (let y=0;y<height;y++) { let row=0;for(let x=0;x<width;x++) {
    row += mask[y*width+x] ? 0 : 1;
    prefix[(y+1)*pitch+x+1] = prefix[y*pitch+x+1]+row;
  } }
  return (ax,ay,bx,by) => {
    const x0=Math.max(0,Math.floor(Math.min(ax,bx))), y0=Math.max(0,Math.floor(Math.min(ay,by)));
    const x1=Math.min(width,Math.ceil(Math.max(ax,bx))+1), y1=Math.min(height,Math.ceil(Math.max(ay,by))+1);
    return prefix[y1*pitch+x1]-prefix[y0*pitch+x1]-prefix[y1*pitch+x0]+prefix[y0*pitch+x0] === 0;
  };
}
const median = values => { values.sort((a,b)=>a-b);const n=values.length;return n%2 ? values[n>>1] : .5*(values[n/2-1]+values[n/2]); };

/** Plane slopes from a centred, normalized 3x3 normal system; rank is checked explicitly. */
function plane(points, u, v) {
  let xx=0,xy=0,yy=0,x=0,y=0,ux=0,uy=0,us=0,vx=0,vy=0,vs=0;
  for (const [index,dx,dy] of points) { x+=dx;y+=dy;xx+=dx*dx;xy+=dx*dy;yy+=dy*dy;
    us+=u[index];ux+=dx*u[index];uy+=dy*u[index];vs+=v[index];vx+=dx*v[index];vy+=dy*v[index]; }
  const n=points.length;
  if(n<3)return null;
  xx-=x*x/n;xy-=x*y/n;yy-=y*y/n;ux-=x*us/n;uy-=y*us/n;vx-=x*vs/n;vy-=y*vs/n;
  const det=xx*yy-xy*xy;
  if(det<=1e-12*Math.max(1,(xx+yy)**2))return null;
  return [(yy*ux-xy*uy)/det,(xx*uy-xy*ux)/det,(yy*vx-xy*vy)/det,(xx*vy-xy*vx)/det];
}

export function analyzeField(field, options = {}, roiMask = null) {
  const started=performance.now(), {count,components}=validateField(field),recipe=validateRecipe(options);
  if (recipe.strainSource==='direct' && field.config.motionModel==='translation') throw new Error('Translation-only fields need VSG to estimate strain');
  const [gw,gh]=field.grid,centre=(field.subsetSize-1)/2,stride=field.subsetStride;
  const valid=Uint8Array.from(field.status,s=>+(s>=.5 && (!recipe.excludeFlagged || s<1.5)));
  const u=new Float64Array(count),v=new Float64Array(count);
  for(let i=0;i<count;i++){u[i]=field.parameters[i*components];v[i]=field.parameters[i*components+ (components===12 ? 6:3)];}
  const clear=visibility(roiMask,field.width,field.height),clearGrid=visibility(valid,gw,gh);
  const radius=(recipe.window-1)/2;
  function neighbours(i) {
    const x=i%gw,y=Math.floor(i/gw),cx=x*stride+centre,cy=y*stride+centre,points=[];
    for(let dy=-radius;dy<=radius;dy++)for(let dx=-radius;dx<=radius;dx++) {
      const nx=x+dx,ny=y+dy;if(nx<0||ny<0||nx>=gw||ny>=gh)continue;
      const j=ny*gw+nx;
      if(valid[j] && clearGrid(x,y,nx,ny) && clear(cx,cy,nx*stride+centre,ny*stride+centre))points.push([j,dx,dy]);
    }
    return points;
  }
  if(recipe.smoothing!=='none') {
    const originalU=u.slice(),originalV=v.slice(),sigma=Math.max(.5,radius/2);
    for(let i=0;i<count;i++)if(valid[i]) {
      const points=neighbours(i);if(points.length<3)continue;
      if(recipe.smoothing==='median'){u[i]=median(points.map(([j])=>originalU[j]));v[i]=median(points.map(([j])=>originalV[j]));}
      else {let w=0,a=0,b=0;for(const [j,dx,dy]of points){const weight=Math.exp(-(dx*dx+dy*dy)/(2*sigma*sigma));w+=weight;a+=weight*originalU[j];b+=weight*originalV[j];}u[i]=a/w;v[i]=b/w;}
    }
  }
  const strainValid=valid.slice(),fields=Object.fromEntries(['u','v','magnitude','exx','eyy','exy','principal1','principal2','vonMises','rotation','iterations','status'].map(k=>[k,new Float64Array(count)]));
  for(let i=0;i<count;i++) {
    fields.status[i]=field.status[i];fields.iterations[i]=field.iterations[i];
    if(!valid[i])continue;
    fields.u[i]=u[i]*recipe.unitsPerPixel;fields.v[i]=v[i]*recipe.unitsPerPixel;fields.magnitude[i]=Math.hypot(u[i],v[i])*recipe.unitsPerPixel;
    let slopes;
    if(recipe.strainSource==='vsg'){slopes=plane(neighbours(i),u,v);if(slopes)slopes=slopes.map(value=>value/stride);}
    else {const p=i*components,k=components===12?6:3;slopes=[field.parameters[p+1],field.parameters[p+2],field.parameters[p+k+1],field.parameters[p+k+2]];}
    if(!slopes){strainValid[i]=0;continue;}
    const [ux,uy,vx,vy]=slopes;
    let exx=ux,eyy=vy,exy=.5*(uy+vx);
    if(recipe.strainMeasure==='green'){exx+=.5*(ux*ux+vx*vx);eyy+=.5*(uy*uy+vy*vy);exy+=.5*(ux*uy+vx*vy);}
    const mean=.5*(exx+eyy),diff=Math.hypot(.5*(exx-eyy),exy);
    fields.exx[i]=exx;fields.eyy[i]=eyy;fields.exy[i]=exy;fields.principal1[i]=mean+diff;fields.principal2[i]=mean-diff;
    fields.vonMises[i]=2/3*Math.sqrt(Math.max(0,exx*exx-exx*eyy+eyy*eyy+3*exy*exy));
    fields.rotation[i]=Math.atan2(vx-uy,2+ux+vy); // Polar rotation of the local 2D deformation gradient.
  }
  return {field,recipe,fields,valid,strainValid,processingWallMs:performance.now()-started,
    supportSpanPx:field.subsetSize+(recipe.window-1)*stride};
}

export function componentValues(analysis,component) {
  const values=analysis.fields[component];if(!values)throw new Error('Unknown field component');
  const strain=!['u','v','magnitude','status','iterations'].includes(component);
  return {values,valid:component==='status'?new Uint8Array(values.length).fill(1):strain?analysis.strainValid:analysis.valid,
    unit:strain?component==='rotation'?'rad':'strain':component==='status'?'status':component==='iterations'?'iterations':analysis.recipe.unit};
}
export function statistics(values,valid) {
  const sorted=Array.from(values).filter((v,i)=>valid[i]&&Number.isFinite(v)).sort((a,b)=>a-b),count=sorted.length;
  if(!count)return {count:0,min:null,max:null,mean:null,standardDeviation:null,p02:null,p98:null};
  const mean=sorted.reduce((a,b)=>a+b,0)/count;
  const quantile=q=>{const x=(count-1)*q,i=Math.floor(x);return sorted[i]+(x-i)*(sorted[Math.min(count-1,i+1)]-sorted[i]);};
  return {count,min:sorted[0],max:sorted.at(-1),mean,standardDeviation:Math.sqrt(sorted.reduce((s,v)=>s+(v-mean)**2,0)/count),p02:quantile(.02),p98:quantile(.98)};
}
export function nearestIndex(field,x,y) {
  const c=(field.subsetSize-1)/2,gx=Math.round((x-c)/field.subsetStride),gy=Math.round((y-c)/field.subsetStride);
  return gx<0||gy<0||gx>=field.grid[0]||gy>=field.grid[1] ? -1:gy*field.grid[0]+gx;
}
export function extensometer(analysis,a,b) {
  const {field,recipe}=analysis,indices=[a,b].map(([x,y])=>nearestIndex(field,x,y));
  if(indices.some(i=>i<0||!analysis.valid[i]))return null;
  const point=i=>[(i%field.grid[0])*field.subsetStride+(field.subsetSize-1)/2,Math.floor(i/field.grid[0])*field.subsetStride+(field.subsetSize-1)/2];
  const [p,q]=indices.map(point),length=Math.hypot(q[0]-p[0],q[1]-p[1])*recipe.unitsPerPixel;if(length===0)return null;
  const deformed=Math.hypot((q[0]-p[0])*recipe.unitsPerPixel+analysis.fields.u[indices[1]]-analysis.fields.u[indices[0]],
    (q[1]-p[1])*recipe.unitsPerPixel+analysis.fields.v[indices[1]]-analysis.fields.v[indices[0]]);
  return {indices,points:[p,q],referenceLength:length,deformedLength:deformed,extension:deformed-length,engineeringStrain:(deformed-length)/length,unit:recipe.unit};
}
