/** Mask images are cold reference geometry, never target-image preprocessing. */
export function maskFromPixels(rgba, width, height, { threshold = null } = {}) {
  if (!(rgba instanceof Uint8Array || rgba instanceof Uint8ClampedArray) || rgba.length !== width*height*4) throw new Error('Expected full-size 8-bit RGBA mask pixels');
  if (threshold !== null && (!Number.isInteger(threshold) || threshold < 0 || threshold > 255)) throw new Error('Mask threshold must be an integer from 0 to 255');
  const mask = new Uint8Array(width*height);
  for (let i=0;i<mask.length;i++) {
    const k=i*4,r=rgba[k],g=rgba[k+1],b=rgba[k+2],a=rgba[k+3];
    if (threshold === null) {
      if (a!==255 || r!==g || r!==b || (r!==0 && r!==255)) throw new Error(`Mask pixel (${i%width}, ${Math.floor(i/width)}) is not opaque black/white. Use a binary PNG or choose explicit threshold conversion.`);
      mask[i]=+(r===255);
    } else mask[i]=+(a>=128 && Math.round(.2126*r+.7152*g+.0722*b)>=threshold);
  }
  return mask;
}

export function intersectMasks(base, shapes) {
  if (!base) return shapes?.slice() ?? null;
  if (!shapes) return base.slice();
  if (base.length !== shapes.length) throw new Error('Image and vector masks have different dimensions');
  const mask=new Uint8Array(base.length);for(let i=0;i<mask.length;i++)mask[i]=+(base[i]&&shapes[i]);return mask;
}

export async function decodeMaskImage(file, width, height, options = {}) {
  if(options.signal?.aborted)throw new DOMException('Mask import cancelled','AbortError');
  const bitmap=await createImageBitmap(file);
  try {
    if(options.signal?.aborted)throw new DOMException('Mask import cancelled','AbortError');
    if(bitmap.width!==width||bitmap.height!==height)throw new Error(`Mask is ${bitmap.width}×${bitmap.height}; reference is ${width}×${height}. Masks are never resized.`);
    const canvas=typeof OffscreenCanvas==='function'?new OffscreenCanvas(width,height):Object.assign(document.createElement('canvas'),{width,height});
    const context=canvas.getContext('2d',{willReadFrequently:true});
    if(!context)throw new Error('The browser cannot allocate the mask image');
    context.drawImage(bitmap,0,0);
    return maskFromPixels(context.getImageData(0,0,width,height).data,width,height,options);
  } finally {bitmap.close();}
}

export async function maskPng(mask, width, height) {
  if(mask&&mask.length!==width*height)throw new Error('Mask dimensions differ');
  const canvas=typeof OffscreenCanvas==='function'?new OffscreenCanvas(width,height):Object.assign(document.createElement('canvas'),{width,height});
  const context=canvas.getContext('2d'),image=context.createImageData(width,height);
  for(let i=0;i<width*height;i++){const value=mask&&!mask[i]?0:255;const k=i*4;image.data[k]=image.data[k+1]=image.data[k+2]=value;image.data[k+3]=255;}
  context.putImageData(image,0,0);
  return canvas.convertToBlob?canvas.convertToBlob({type:'image/png'}):new Promise((resolve,reject)=>canvas.toBlob(blob=>blob?resolve(blob):reject(new Error('Mask PNG encoding failed')),'image/png'));
}
