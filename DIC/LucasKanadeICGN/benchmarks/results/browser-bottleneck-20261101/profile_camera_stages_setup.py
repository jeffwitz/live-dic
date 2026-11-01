from pathlib import Path
import shutil
site=Path('/tmp/live-dic-camera-stage-site')
shutil.copytree('/tmp/live-dic-cadence-site',site,dirs_exist_ok=True)
hooks='''
globalThis.__stageEvents=[];globalThis.__stageFrame=null;
globalThis.__stageEvent=(stage,start,extra={})=>{
 if(!globalThis.__stageFrame)return;
 __stageEvents.push({stage,frame:__stageFrame,startEpochMs:performance.timeOrigin+start,wallMs:performance.now()-start,...extra});
 if(__stageEvents.length>8192)__stageEvents.shift();
};
const map=GPUBuffer.prototype.mapAsync;
GPUBuffer.prototype.mapAsync=function(...args){const start=performance.now(),frame=__stageFrame;
 return map.apply(this,args).then(result=>{if(frame){__stageEvents.push({stage:'buffer map',frame,startEpochMs:performance.timeOrigin+start,wallMs:performance.now()-start,bytes:this.size,label:this.label});}return result;});};
'''
(site/'stage-hooks.js').write_text(hooks)
p=site/'worker.js';p.write_text("import './stage-hooks.js';\n"+p.read_text())
p=site/'ui/session.js';s=p.read_text();needle='        const available = performance.now();';assert s.count(needle)==1
s=s.replace(needle,needle+"\n        globalThis.__stageFrame={id,ordinal:message.comparison?.target?.ordinal};")
needle='samples: sampleLog, capabilities,';assert s.count(needle)==1
s=s.replace(needle,'samples: sampleLog, capabilities, diagnostic:globalThis.__stageEvents,')
p.write_text(s)
p=site/'web/core/pyramid.js';s=p.read_text()
needle='    if(pair){slot.temporal?.clearForNewReference(encoder);gpuReference.encode(encoder);}'
assert s.count(needle)==1
s=s.replace(needle,'''    if(pair){
      const refStart=performance.now(),referenceEncoder=device.createCommandEncoder({label:'Diagnostic reference preparation'});
      slot.temporal?.clearForNewReference(referenceEncoder);gpuReference.encode(referenceEncoder);
      device.queue.submit([referenceEncoder.finish()]);await device.queue.onSubmittedWorkDone();
      globalThis.__stageEvent('reference preparation completion',refStart);
    }
    const solveStarted=performance.now();''')
needle='''      await device.queue.onSubmittedWorkDone();
      return null;''';assert s.count(needle)==1
s=s.replace(needle,"      await device.queue.onSubmittedWorkDone();\n      globalThis.__stageEvent('target preprocessing + solve + summaries completion',solveStarted);\n      return null;")
needle='''    slot.external.converter.copy(frame);
    const gpuMs = await executePipeline({ slot });''';assert s.count(needle)==1
s=s.replace(needle,"    const uploadStart=performance.now();slot.external.converter.copy(frame);const apiMs=performance.now()-uploadStart;await device.queue.onSubmittedWorkDone();\n    globalThis.__stageEvent('current input upload completion',uploadStart,{apiMs});\n    const gpuMs = await executePipeline({ slot });")
needle='    if(roiPlans[0]?.count!==0)gpuReference.copy(reference);';assert s.count(needle)==1
s=s.replace(needle,"    const uploadStart=performance.now();\n"+needle)
needle='    await executePipeline({slot,pair:true});';assert s.count(needle)==1
s=s.replace(needle,"    const apiMs=performance.now()-uploadStart;await device.queue.onSubmittedWorkDone();\n    globalThis.__stageEvent('both pair inputs upload completion',uploadStart,{apiMs});\n"+needle)
p.write_text(s)

source=Path('tools/diagnose_camera_pipeline.py').read_text().replace('ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path.cwd()')
needle='                for mode, recording in [';assert source.count(needle)==1
inject='''                browser._evaluate("""(async()=>{
                  globalThis.__mainStageEvents=[];
                  const epoch=()=>performance.timeOrigin+performance.now();
                  const {RecordingQueue}=await import('./ui/recording-queue.js'),{IndexedFieldDatabase,FieldStorage}=await import('./ui/field-storage.js');
                  const submit=RecordingQueue.prototype.submit;
                  RecordingQueue.prototype.submit=function(ticket,record){__mainStageEvents.push({stage:'received field ownership',ticket,ordinal:record.metadata.comparison?.target?.ordinal,startEpochMs:epoch(),workerArrivalEpochMs:record.workerArrivalEpochMs,recordingReadbackMs:record.metadata.recordingReadbackWallMs,workerToRecordedFieldMs:record.metadata.workerArrivalToRecordedFieldMs});return submit.call(this,ticket,record);};
                  const write=IndexedFieldDatabase.prototype.write;
                  IndexedFieldDatabase.prototype.write=async function(session,records,masks){const started=epoch();try{return await write.call(this,session,records,masks);}finally{__mainStageEvents.push({stage:'IndexedDB transaction',startEpochMs:started,wallMs:epoch()-started,ordinal:records[0]?.metadata.comparison?.target?.ordinal});}};
                  const add=FieldStorage.prototype.addMany;
                  FieldStorage.prototype.addMany=async function(entries,...args){const started=epoch();try{return await add.call(this,entries,...args);}finally{__mainStageEvents.push({stage:'storage full admission',startEpochMs:started,wallMs:epoch()-started,ordinal:entries[0]?.metadata.comparison?.target?.ordinal});}};
                  return true;
                })()""")
'''
source=source.replace(needle,inject+needle)
needle='                    report["cases"].append(case)';assert source.count(needle)==1
source=source.replace(needle,'''                    case["worker_stages"]=[event for event in worker.get("diagnostic",[]) if start["epochMs"] <= event["startEpochMs"] <= stop["epochMs"]]
                    case["main_stages"]=browser._evaluate("__mainStageEvents");case["main_stages"]=[event for event in case["main_stages"] if start["epochMs"] <= event["startEpochMs"] <= stop["epochMs"]]
'''+needle)
source=source.replace('"kind": "Static simulated camera control; no physical camera or general real-time qualification",','"kind": "Intrusive stage diagnosis: upload, GPU reference and solving are separated by extra completion fences. Not an unmodified live-FPS benchmark.",')
Path('/tmp/profile_camera_stages.py').write_text(source)
print(site)
