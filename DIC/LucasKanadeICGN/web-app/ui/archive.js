/** Dependency-free NumPy archive interchange. No pickle, eval or filesystem extraction. */
import { validateField } from './postprocess.js';
const encoder=new TextEncoder(),decoder=new TextDecoder();
const table=Uint32Array.from({length:256},(_,n)=>{for(let k=0;k<8;k++)n=n&1?0xedb88320^(n>>>1):n>>>1;return n>>>0;});
function crc32(bytes){let c=0xffffffff;for(const b of bytes)c=table[(c^b)&255]^(c>>>8);return(c^0xffffffff)>>>0;}
function join(parts){const out=new Uint8Array(parts.reduce((n,p)=>n+p.length,0));let offset=0;for(const p of parts){out.set(p,offset);offset+=p.length;}return out;}
function header(size){const bytes=new Uint8Array(size);return {bytes,view:new DataView(bytes.buffer)};}
export function npy(values,shape,descriptor) {
  const units=descriptor.startsWith('<U')?Number(descriptor.slice(2)):1;
  if(shape.reduce((n,v)=>n*v,1)*units!==values.length)throw new Error('NPY shape mismatch');
  const h=`{'descr': '${descriptor}', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length===1?',':''}), }`;
  const text=h+' '.repeat((64-((10+h.length+1)%64))%64)+'\n',prefix=header(10);
  prefix.bytes.set([147,78,85,77,80,89,1,0]);prefix.view.setUint16(8,text.length,true);
  return join([prefix.bytes,encoder.encode(text),new Uint8Array(values.buffer,values.byteOffset,values.byteLength)]);
}
function textNpy(value){const chars=Array.from(value),data=new Uint32Array(Math.max(1,chars.length));chars.forEach((c,i)=>data[i]=c.codePointAt(0));return npy(data,[],`<U${Math.max(1,chars.length)}`);}
function scalar(value){return npy(new Uint32Array([value]),[],'<u4');}
export function zipStore(entries) {
  const parts=[],central=[];let offset=0;
  for(const [name,data] of Object.entries(entries)){
    const filename=encoder.encode(name),crc=crc32(data),local=header(30),directory=header(46);
    local.view.setUint32(0,0x04034b50,true);local.view.setUint16(4,20,true);local.view.setUint32(14,crc,true);
    local.view.setUint32(18,data.length,true);local.view.setUint32(22,data.length,true);local.view.setUint16(26,filename.length,true);
    directory.view.setUint32(0,0x02014b50,true);directory.view.setUint16(4,20,true);directory.view.setUint16(6,20,true);
    directory.view.setUint32(16,crc,true);directory.view.setUint32(20,data.length,true);directory.view.setUint32(24,data.length,true);
    directory.view.setUint16(28,filename.length,true);directory.view.setUint32(42,offset,true);
    parts.push(local.bytes,filename,data);central.push(directory.bytes,filename);offset+=30+filename.length+data.length;
  }
  const directory=join(central),end=header(22),count=Object.keys(entries).length;
  end.view.setUint32(0,0x06054b50,true);end.view.setUint16(8,count,true);end.view.setUint16(10,count,true);
  end.view.setUint32(12,directory.length,true);end.view.setUint32(16,offset,true);
  return join([...parts,directory,end.bytes]);
}
export function fieldArchive(field,metadata={},analysis=null,roiMask=null) {
  const {components}=validateField(field),shape=[field.grid[1],field.grid[0]];
  const entries={
    'parameters.npy':npy(field.parameters instanceof Float64Array?field.parameters:Float32Array.from(field.parameters),[...shape,components],field.parameters instanceof Float64Array?'<f8':'<f4'),
    'status.npy':npy(Float32Array.from(field.status),shape,'<f4'),
    'iterations.npy':npy(Float32Array.from(field.iterations),shape,'<f4'),
    'width.npy':scalar(field.width),'height.npy':scalar(field.height),
    'subset_size.npy':scalar(field.subsetSize),'subset_stride.npy':scalar(field.subsetStride),
    'config_json.npy':textNpy(JSON.stringify(field.config)),
    'metadata_json.npy':textNpy(JSON.stringify(metadata)),
  };
  if(roiMask){if(roiMask.length!==field.width*field.height)throw new Error('Mask shape mismatch');entries['roi_mask.npy']=npy(roiMask,[field.height,field.width],'|u1');}
  if(analysis){entries['recipe_json.npy']=textNpy(JSON.stringify(analysis.recipe));
    entries['analysis_valid.npy']=npy(analysis.valid,shape,'|u1');entries['strain_valid.npy']=npy(analysis.strainValid,shape,'|u1');
    for(const [key,values]of Object.entries(analysis.fields))entries[`processed_${key}.npy`]=npy(values,shape,'<f8');}
  return zipStore(entries);
}

const MAX_BYTES=256*1024*1024;
export function readNpy(bytes) {
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  if(bytes.length<10||!bytes.slice(0,6).every((v,i)=>v===[147,78,85,77,80,89][i]))throw new Error('Invalid NumPy header');
  const version=bytes[6],base=version===1?10:12;
  if(![1,2,3].includes(version))throw new Error('Unsupported NumPy version');
  const length=version===1?view.getUint16(8,true):view.getUint32(8,true);
  if(length>65536||base+length>bytes.length)throw new Error('Invalid NumPy header length');
  const h=decoder.decode(bytes.subarray(base,base+length));
  const descriptor=h.match(/['"]descr['"]\s*:\s*['"]([^'"]+)['"]/)?.[1],shapeText=h.match(/['"]shape['"]\s*:\s*\(([^)]*)\)/)?.[1];
  if(!/['"]fortran_order['"]\s*:\s*False/.test(h)||shapeText===undefined)throw new Error('Only C-order numeric NumPy arrays are supported');
  const shape=shapeText.split(',').map(s=>s.trim()).filter(Boolean).map(s=>/^\d+$/.test(s)?Number(s):NaN);
  if(!shape.every(v=>Number.isSafeInteger(v)&&v>=0))throw new Error('Invalid NumPy shape');
  const count=shape.reduce((a,b)=>a*b,1),match=descriptor?.match(/^([<|=])([fiuUb])(\d+)$/);
  if(!match||!Number.isSafeInteger(count))throw new Error('Only little-endian numeric/Unicode arrays are supported; object/pickle arrays are forbidden');
  const [,endian,kind,sizeText]=match,size=Number(sizeText),itemBytes=kind==='U'?size*4:size;
  if(itemBytes*count!==bytes.length-base-length||itemBytes*count>MAX_BYTES)throw new Error('NumPy payload size mismatch');
  if(endian==='|'&&itemBytes>1)throw new Error('Invalid endian-independent numeric array');
  const data=bytes.slice(base+length),dataView=new DataView(data.buffer);
  if(kind==='U'){
    const values=[];for(let i=0;i<count;i++){let text='';for(let j=0;j<size;j++){const c=dataView.getUint32((i*size+j)*4,true);if(c)text+=String.fromCodePoint(c);}values.push(text);}return{shape,values,descriptor};
  }
  const types={f4:Float32Array,f8:Float64Array,i4:Int32Array,u4:Uint32Array,i2:Int16Array,u2:Uint16Array,i1:Int8Array,u1:Uint8Array,b1:Uint8Array};
  if((kind==='i'||kind==='u')&&size===8){const values=new Float64Array(count);for(let i=0;i<count;i++){
    values[i]=Number(kind==='i'?dataView.getBigInt64(i*8,true):dataView.getBigUint64(i*8,true));
    if(!Number.isSafeInteger(values[i]))throw new Error('NumPy integer exceeds JavaScript exact integer range');
  }return{shape,values,descriptor};}
  const Type=types[kind+size];if(!Type)throw new Error(`Unsupported NumPy scalar type ${descriptor}`);
  return {shape,values:new Type(data.buffer),descriptor};
}
export async function readZip(buffer) {
  const bytes=buffer instanceof Uint8Array?buffer:new Uint8Array(buffer);
  if(bytes.length>MAX_BYTES||bytes.length<22)throw new Error('Archive is empty or exceeds the 256 MiB import limit');
  const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);let end=-1;
  for(let i=bytes.length-22;i>=Math.max(0,bytes.length-65557);i--)if(view.getUint32(i,true)===0x06054b50){end=i;break;}
  if(end<0||view.getUint16(end+4,true)||view.getUint16(end+6,true))throw new Error('Unsupported ZIP directory');
  const count=view.getUint16(end+10,true);if(count>128)throw new Error('Too many archive entries');
  let offset=view.getUint32(end+16,true),total=0;const entries=Object.create(null);
  for(let i=0;i<count;i++){
    if(offset+46>end||view.getUint32(offset,true)!==0x02014b50)throw new Error('Invalid ZIP central directory');
    const flags=view.getUint16(offset+8,true),method=view.getUint16(offset+10,true),crc=view.getUint32(offset+16,true),compressed=view.getUint32(offset+20,true),size=view.getUint32(offset+24,true);
    const nameLength=view.getUint16(offset+28,true),extra=view.getUint16(offset+30,true),comment=view.getUint16(offset+32,true),local=view.getUint32(offset+42,true);
    const name=decoder.decode(bytes.subarray(offset+46,offset+46+nameLength));offset+=46+nameLength+extra+comment;
    total+=size;if(total>MAX_BYTES||flags&1||![0,8].includes(method)||entries[name])throw new Error('Unsupported or oversized archive entry');
    if(local+30>bytes.length||view.getUint32(local,true)!==0x04034b50)throw new Error('Invalid ZIP entry');
    const start=local+30+view.getUint16(local+26,true)+view.getUint16(local+28,true);
    if(start+compressed>bytes.length)throw new Error('Truncated ZIP payload');
    let payload=bytes.slice(start,start+compressed);
    if(method===8){if(typeof DecompressionStream==='undefined')throw new Error('This browser cannot decompress NPZ files');
      const stream=new Blob([payload]).stream().pipeThrough(new DecompressionStream('deflate-raw')),reader=stream.getReader(),chunks=[];let read=0;
      try{while(true){const {done,value}=await reader.read();if(done)break;read+=value.length;if(read>size||read>MAX_BYTES)throw new Error('Decompressed archive exceeds declared size');chunks.push(value);}}
      finally{await reader.cancel();}payload=join(chunks);}
    if(payload.length!==size||crc32(payload)!==crc)throw new Error('Archive integrity check failed');entries[name]=payload;
  }
  return entries;
}
export async function importFieldArchive(buffer, suppliedGeometry = {}) {
  const zip=await readZip(buffer),read=name=>{if(!zip[name+'.npy'])throw new Error(`Archive is missing ${name}`);return readNpy(zip[name+'.npy']);};
  const parameters=read('parameters'),status=read('status'),iterations=read('iterations');
  const sequence=status.shape.length===3;
  if(![2,3].includes(status.shape.length)||parameters.shape.length!==status.shape.length+1)throw new Error('Unsupported field/sequence array dimensions');
  if(iterations.shape.length!==status.shape.length||iterations.shape.some((v,i)=>v!==status.shape[i])||status.shape.some((v,i)=>parameters.shape[i]!==v))throw new Error('Field array shapes differ');
  const config=JSON.parse(read('config_json').values[0]),roi=zip['roi_mask.npy']?read('roi_mask'):null;
  if(config.roi&&!roi)throw new Error('Archive declares an ROI but contains no roi_mask; the missing mask must not be ignored');
  const scalar=name=>{const item=read(name);if(item.values.length!==1)throw new Error(`${name} must be scalar`);return item.values[0];};
  const width=zip['width.npy']?scalar('width'):roi?.shape[1]??suppliedGeometry.width;
  const height=zip['height.npy']?scalar('height'):roi?.shape[0]??suppliedGeometry.height;
  const subsetSize=zip['subset_size.npy']?scalar('subset_size'):config.subsetSize??suppliedGeometry.subsetSize;
  const subsetStride=zip['subset_stride.npy']?scalar('subset_stride'):config.subsetStride??suppliedGeometry.subsetStride;
  const frameCount=sequence?status.shape[0]:1,frameIndex=suppliedGeometry.frameIndex??0;
  if([width,height,subsetSize,subsetStride].some(v=>v===undefined)||(frameCount>1&&suppliedGeometry.frameIndex===undefined)){
    const error=new Error('Confirm image geometry and the field to import; image extent is never guessed from the grid');
    error.code='FIELD_GEOMETRY_REQUIRED';error.geometry={width,height,subsetSize,subsetStride,frameCount};throw error;
  }
  if(!Number.isInteger(frameIndex)||frameIndex<0||frameIndex>=frameCount)throw new Error('Requested archive frame does not exist');
  const shape=sequence?status.shape.slice(1):status.shape,count=shape[0]*shape[1],components=parameters.shape.at(-1);
  const field={width,height,subsetSize,subsetStride,grid:[shape[1],shape[0]],config,
    parameters:parameters.values.slice(frameIndex*count*components,(frameIndex+1)*count*components),
    status:status.values.slice(frameIndex*count,(frameIndex+1)*count),iterations:iterations.values.slice(frameIndex*count,(frameIndex+1)*count)};
  const expected=validateField(field);
  if(components!==expected.components)throw new Error('Parameter shape mismatch');
  if(roi&&(roi.shape[0]!==field.height||roi.shape[1]!==field.width||roi.shape.length!==2||roi.values.some(v=>v!==0&&v!==1)))throw new Error('Invalid archive ROI');
  const metadata=zip['metadata_json.npy']?JSON.parse(read('metadata_json').values[0]):{};
  metadata.archiveSha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',buffer)),b=>b.toString(16).padStart(2,'0')).join('');
  metadata.importGeometry={width,height,subsetSize,subsetStride,frameIndex,frameCount,
    imageExtentSource:zip['width.npy']&&zip['height.npy']?'archive scalars':roi?'archived ROI shape':'explicit user input'};
  return {field,metadata,roiMask:roi?.values??null,
    recipe:zip['recipe_json.npy']?JSON.parse(read('recipe_json').values[0]):null};
}
