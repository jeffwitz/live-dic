export function instrument(device) {
  const capacity=8192, queries=device.createQuerySet({type:"timestamp",count:capacity});
  const pipelines=new WeakMap(), rows=[], maps=[];let active=false,frame=null,cursor=0;
  const asyncPipeline=device.createComputePipelineAsync.bind(device);
  device.createComputePipelineAsync=async descriptor=>{const pipeline=await asyncPipeline(descriptor);pipelines.set(pipeline,{entry:descriptor.compute.entryPoint,label:descriptor.label??null,constants:descriptor.compute.constants??null});return pipeline;};
  const create=device.createCommandEncoder.bind(device);
  device.createCommandEncoder=descriptor=>{
    const encoder=create(descriptor), begin=encoder.beginComputePass.bind(encoder);
    encoder.beginComputePass=options=>{
      if(!active)return begin(options);
      if(cursor+2>capacity)throw new Error("GPU profile query capacity exhausted");
      const row={frame,index:cursor,encoder:descriptor?.label??null,label:options?.label??null,pipelines:[],dispatches:[]};cursor+=2;rows.push(row);
      const pass=begin({...options,timestampWrites:{querySet:queries,beginningOfPassWriteIndex:row.index,endOfPassWriteIndex:row.index+1}});
      const set=pass.setPipeline.bind(pass), dispatch=pass.dispatchWorkgroups.bind(pass);
      pass.setPipeline=pipeline=>{row.pipelines.push(pipelines.get(pipeline)??{label:pipeline.label});return set(pipeline);};
      pass.dispatchWorkgroups=(...dimensions)=>{row.dispatches.push(dimensions);return dispatch(...dimensions);};
      return pass;
    };return encoder;
  };
  const map=GPUBuffer.prototype.mapAsync;
  GPUBuffer.prototype.mapAsync=function(...args){
    if(!active)return map.apply(this,args);
    const row={frame,bytes:this.size,label:this.label,start:performance.now()};
    return map.apply(this,args).then(value=>{row.wallMs=performance.now()-row.start;maps.push(row);return value;});
  };
  return {
    begin(id){frame=id;active=true;}, end(){active=false;frame=null;},
    async read(){
      active=false;await device.queue.onSubmittedWorkDone();
      const bytes=cursor*8, resolved=device.createBuffer({size:bytes,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
      const readback=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      const encoder=create();encoder.resolveQuerySet(queries,0,cursor,resolved,0);encoder.copyBufferToBuffer(resolved,0,readback,0,bytes);device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);const timestamps=new BigUint64Array(readback.getMappedRange());const base=timestamps[0];
      const result=rows.map(row=>({...row,gpuMs:Number(timestamps[row.index+1]-timestamps[row.index])/1e6,gpuStartMs:Number(timestamps[row.index]-base)/1e6}));
      readback.unmap();readback.destroy();resolved.destroy();queries.destroy();return {passes:result,mapWaits:maps};
    }
  };
}
