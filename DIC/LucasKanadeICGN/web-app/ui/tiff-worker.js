/* Classic Worker keeps the upstream browser bundles byte-for-byte unchanged. */
importScripts('../vendor/pako-inflate-1.0.11.min.js','../vendor/UTIF-3.1.0.js');
onmessage = async ({data}) => {
  try {
    const {decodeTiff,MAX_TIFF_BYTES}=await import('./tiff-codec.js');
    if(data.file.size>MAX_TIFF_BYTES)throw new Error('TIFF exceeds the 256 MiB per-file limit');
    const decoded=decodeTiff(await data.file.arrayBuffer(),self.UTIF,data.expected);
    const pixels=new ImageData(new Uint8ClampedArray(decoded.rgba.buffer),decoded.width,decoded.height);
    const bitmap=await createImageBitmap(pixels,{premultiplyAlpha:'none',colorSpaceConversion:'none'});
    postMessage({bitmap,metadata:decoded.metadata},[bitmap]);
  } catch(error) {postMessage({error:String(error?.message??error)});}
};
