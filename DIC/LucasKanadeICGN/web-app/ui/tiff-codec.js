/** Conservative TIFF admission around the vendored UTIF pixel decoder.
 * Read only the image tags we use: unrelated EXIF/SubIFDs never trigger
 * recursive parsing. Reject stacks rather than silently choosing a page.
 */
export const MAX_TIFF_BYTES = 256 * 1024 * 1024;
const MAX_PIXELS = 64 * 1024 * 1024;
const TAGS = new Set([256,257,258,259,262,266,273,274,277,278,279,284,317,320,322,323,324,325,338,339]);

function directory(buffer) {
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength < 8) throw new Error('Truncated TIFF header');
  const view = new DataView(buffer), marker = view.getUint16(0);
  if (marker !== 0x4949 && marker !== 0x4d4d) throw new Error('Invalid TIFF byte order');
  const little = marker === 0x4949, short = offset => view.getUint16(offset,little), long = offset => view.getUint32(offset,little);
  if (short(2) === 43) throw new Error('BigTIFF is not supported; export separate classic TIFF images');
  if (short(2) !== 42) throw new Error('Invalid TIFF signature');
  const start = long(4);
  if (start < 8 || start+2 > buffer.byteLength) throw new Error('Invalid TIFF directory offset');
  const count = short(start), end = start+2+12*count;
  if (end+4 > buffer.byteLength) throw new Error('Truncated TIFF directory');
  if (long(end) !== 0) throw new Error('Multipage TIFF is not supported; export one image per file for the sequence');
  const image = {};
  for (let i=0;i<count;i++) {
    const entry=start+2+12*i, tag=short(entry);
    if (!TAGS.has(tag)) continue;
    if (image[`t${tag}`]) throw new Error(`Duplicate TIFF tag ${tag}`);
    const type=short(entry+2), length=long(entry+4), size=type===3?2:type===4?4:0;
    if (!size || !length || length > 65536) throw new Error(`Unsupported TIFF tag ${tag} representation`);
    const offset=length*size<=4?entry+8:long(entry+8);
    if (offset+length*size > buffer.byteLength) throw new Error(`Truncated TIFF tag ${tag}`);
    image[`t${tag}`]=Array.from({length},(_,k)=>size===2?short(offset+k*size):long(offset+k*size));
  }
  return image;
}

export function decodeTiff(buffer, decoder, expected = null) {
  if (buffer.byteLength > MAX_TIFF_BYTES) throw new Error('TIFF exceeds the 256 MiB per-file limit');
  const image=directory(buffer), value=(tag,fallback)=>image[`t${tag}`]?.[0]??fallback;
  const width=value(256),height=value(257),bits=value(258,1),samples=value(277,1),photo=value(262,1),orientation=value(274,1);
  if (!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||width>16384||height>16384||width*height>MAX_PIXELS) throw new Error('TIFF dimensions exceed the supported image limits');
  if (!Number.isInteger(orientation)||orientation<1||orientation>8) throw new Error('Unsupported TIFF orientation');
  const outputSize=orientation>=5?[height,width]:[width,height];
  if (expected && (outputSize[0]!==expected[0]||outputSize[1]!==expected[1])) throw new Error(`TIFF is ${outputSize.join('×')}; the reference is ${expected.join('×')}. Images are never resized implicitly.`);
  if ((image.t339??[1]).some(format=>format!==1)) throw new Error('Only unsigned integer TIFF samples are supported; floating-point and signed TIFF require explicit conversion');
  if (value(284,1)!==1) throw new Error('Planar TIFF is not supported; export interleaved samples');
  const gray=(photo===0||photo===1)&&samples===1&&[1,8,16].includes(bits);
  const rgb=photo===2&&[3,4].includes(samples)&&[8,16].includes(bits);
  const palette=photo===3&&samples===1&&bits===8&&image.t320?.length===768;
  if (!gray&&!rgb&&!palette) throw new Error('Supported TIFF pixels: 1/8/16-bit grayscale, 8/16-bit RGB(A), or 8-bit palette');
  if ((image.t258??[bits]).some(depth=>depth!==bits) || (rgb&&image.t258?.length!==samples)) throw new Error('Mixed TIFF channel depths are not supported');
  const compression=value(259,1);
  if (![1,5,8,32946,32773].includes(compression)) throw new Error(`Unsupported TIFF compression ${compression}; use uncompressed, LZW, Deflate or PackBits`);
  if (![1,2].includes(value(317,1)) || (bits===1&&value(317,1)!==1)) throw new Error('Unsupported TIFF predictor');
  if (value(266,1)!==1) throw new Error('Reversed TIFF bit fill order is not supported');
  const offsets=image.t273??image.t324,counts=image.t279??image.t325;
  if (!offsets?.length||offsets.length!==counts?.length||offsets.some((offset,i)=>!counts[i]||offset+counts[i]>buffer.byteLength)) throw new Error('Truncated or invalid TIFF pixel blocks');
  if (image.t322 || image.t323) {
    if (!value(322)||!value(323)||Math.ceil(width/value(322))*Math.ceil(height/value(323))!==offsets.length) throw new Error('Invalid TIFF tile layout');
    // UTIF applies horizontal prediction using full-image row widths. Keep
    // tiled prediction explicit until it has an independent qualification.
    if (value(317,1)!==1) throw new Error('Predicted tiled TIFF is not supported; export strips or disable prediction');
  } else {
    const rows=value(278,height);
    if (rows<1||Math.ceil(height/rows)!==offsets.length) throw new Error('Invalid TIFF strip layout');
    if (compression===1&&counts.some((bytes,i)=>bytes<Math.min(rows,height-i*rows)*Math.ceil(width*bits*samples/8))) throw new Error('Truncated uncompressed TIFF pixels');
  }
  if (compression===32946) image.t259=[8];
  decoder.decodeImage(buffer,image);
  const stride=Math.ceil(width*bits*samples/8);
  if (image.data?.length!==stride*height) throw new Error('Incomplete TIFF pixel decoding');
  // UTIF normalizes 16-bit decompressed samples to little endian. Retaining
  // their high byte is a fixed floor(value/256), never image-wise scaling.
  let rgba;
  if (gray) {
    rgba=new Uint8Array(width*height*4);
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const source=y*stride, target=(y*width+x)*4;
      let sample=bits===1?((image.data[source+(x>>3)]>>(7-(x&7)))&1)*255:image.data[source+x*(bits/8)+(bits===16?1:0)];
      if(photo===0)sample=255-sample;
      rgba[target]=rgba[target+1]=rgba[target+2]=sample;rgba[target+3]=255;
    }
  } else rgba=decoder.toRGBA8(image);
  if (rgb&&samples===4&&value(338,0)===1) {
    for(let i=0;i<rgba.length;i+=4)for(let k=0;k<3;k++)rgba[i+k]=rgba[i+3]?Math.min(255,Math.round(rgba[i+k]*255/rgba[i+3])):0;
  }
  let oriented=rgba;
  if(orientation!==1){
    oriented=new Uint8Array(rgba.length);
    for(let y=0;y<height;y++)for(let x=0;x<width;x++){
      const [a,b]=orientation===2?[width-1-x,y]:orientation===3?[width-1-x,height-1-y]:orientation===4?[x,height-1-y]:orientation===5?[y,x]:orientation===6?[height-1-y,x]:orientation===7?[height-1-y,width-1-x]:[y,width-1-x];
      const source=(y*width+x)*4,target=(b*outputSize[0]+a)*4;
      oriented.set(rgba.subarray(source,source+4),target);
    }
  }
  return { width:outputSize[0],height:outputSize[1],rgba:oriented,
    metadata:{format:'TIFF',bitsPerSample:bits,samplesPerPixel:samples,compression,orientation,
      conversion:bits===16?'unsigned 16-bit → uint8: floor(value / 256), fixed for every image':'unsigned samples → 8-bit RGBA; no automatic contrast scaling',
      decoder:'UTIF 3.1.0',colourManagement:'TIFF ICC profiles are not applied'} };
}
