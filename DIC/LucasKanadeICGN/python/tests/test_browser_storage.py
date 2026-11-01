"""Cache/failure contracts; the browser qualification separately executes real IndexedDB."""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

UI = Path(__file__).resolve().parents[2] / "web-app/ui"
PREAMBLE = f"import {{FieldStorage,defaultCacheBytes,MiB}} from {json.dumps((UI / 'field-storage.js').as_uri())};\n"
FIXTURE = """
class Disk {
  constructor(){this.rows=new Map();this.indices=new Map();this.staticMasks=new Map();this.summaries=new Map();this.savedSettings=new Map();this.fail=false;}
  async write(session,records,masks){if(this.fail)throw new DOMException('Injected disk quota','QuotaExceededError');
    for(const r of records){this.rows.set(r.id,structuredClone(r));this.indices.set(r.id,structuredClone(r.descriptor));}
    for(const mask of masks)this.staticMasks.set(mask.id,structuredClone(mask.data));this.summaries.set(session.id,structuredClone(session));}
  async read(session,id){return structuredClone(this.rows.get(id));}
  async mask(session,id){return structuredClone(this.staticMasks.get(id));}
  async sessions(){return [...this.summaries.values()].map(v=>structuredClone(v));}
  async catalog(session,map=v=>v){return [...this.indices.values()].map(v=>map(structuredClone(v)));}
  async settings(id){return this.savedSettings.get(id)??{value:{},revision:0};}
  async saveSettings(id,value,revision){if((this.savedSettings.get(id)?.revision??0)!==revision)throw new Error('Concurrent settings');this.savedSettings.set(id,{value:structuredClone(value),revision:revision+1});}
}
function entry(id,width=64,height=64,mask=null){const n=width/16*height/16;
 const field={width,height,subsetSize:16,subsetStride:16,grid:[width/16,height/16],config:{motionModel:'affine'},parameters:new Float64Array(n*6),status:new Float32Array(n).fill(1),iterations:new Float32Array(n).fill(4)};
 field.parameters[0]=id+.123456789012345;
 return{id,label:'Field '+id,field,metadata:{analysisEpoch:'test',comparison:{mode:'fixed',target:{index:id}}},metadataBytes:120,mask,recipe:null};}
function setup(memoryLimit=4096){const disk=new Disk(),store=new FieldStorage({memoryLimit,openDatabase:async()=>disk});return{disk,store};}
"""


def javascript(script):
    return json.loads(
        subprocess.check_output(
            ["node", "--input-type=module"],
            input=PREAMBLE + FIXTURE + script,
            text=True,
            encoding="utf-8",
        )
    )


@unittest.skipUnless(shutil.which("node"), "Node is needed for browser storage contracts")
class BrowserStorageTests(unittest.TestCase):
    def test_default_budget_is_bounded_and_unknown_memory_stays_conservative(self):
        result = javascript("""
          console.log(JSON.stringify([defaultCacheBytes(32),defaultCacheBytes(8),defaultCacheBytes(4),defaultCacheBytes(2),defaultCacheBytes(0),defaultCacheBytes(-1)]));
        """)
        self.assertEqual(
            [value // (1024 * 1024) for value in result], [1024, 1024, 512, 256, 128, 128]
        )

    def test_uhd_fields_spill_and_reload_without_precision_or_mask_duplication(self):
        result = javascript("""
          const {disk,store}=setup(16*MiB),mask=new Uint8Array(3840*2160).fill(1),entries=[];
          for(let id=1;id<=32;id++){const e=entry(id,3840,2160,mask);entries.push(e);await store.addMany([e]);}
          const evicted=entries.filter(e=>!e.field.parameters).length,initial=store.report();let exact=true,peak=0;
          for(const e of entries){await store.load(e);exact&&=e.field.parameters instanceof Float64Array&&e.field.parameters[0]===e.id+.123456789012345&&e.mask.length===3840*2160;peak=Math.max(peak,store.report().residentBytes);}
          console.log(JSON.stringify({exact,evicted,initial,peak,masks:disk.staticMasks.size,fields:disk.rows.size}));
        """)
        self.assertTrue(result["exact"])
        self.assertGreater(result["evicted"], 20)
        self.assertEqual(result["fields"], 32)
        self.assertEqual(result["masks"], 1)
        self.assertLessEqual(result["peak"], 16 * 1024 * 1024)

    def test_quota_failure_keeps_completed_field_and_retry_does_not_duplicate_it(self):
        result = javascript("""
          const {disk,store}=setup(2048),a=entry(1),b=entry(2),c=entry(3);
          await store.addMany([a]);await store.addMany([b]);disk.fail=true;
          const failure=await store.addMany([c]),held=c.field.parameters[0],before=store.report();disk.fail=false;await store.retry();
          await store.load(c);console.log(JSON.stringify({failure,held,before,after:store.report(),count:disk.rows.size,value:c.field.parameters[0]}));
        """)
        self.assertIn("QuotaExceeded", result["failure"]["failure"])
        self.assertEqual(result["before"]["unsavedFields"], 1)
        self.assertEqual(result["before"]["diskFields"], 2)
        self.assertEqual(result["after"]["unsavedFields"], 0)
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["held"], result["value"])

    def test_no_disk_never_evicts_unsaved_data_and_stops_at_budget(self):
        result = javascript("""
          const store=new FieldStorage({memoryLimit:2048,openDatabase:async()=>{throw new Error('Storage disabled');}}),entries=[entry(1),entry(2),entry(3)];let result;
          for(const e of entries)result=await store.addMany([e]);
          console.log(JSON.stringify({failure:result.failure,report:store.report(),values:entries.map(e=>e.field.parameters[0])}));
        """)
        self.assertIn("Storage disabled", result["failure"])
        self.assertEqual(result["report"]["unsavedFields"], 3)
        self.assertEqual(len(result["values"]), 3)

    def test_pending_shared_mask_is_accounted_once_before_disk_recovery(self):
        result = javascript("""
          const {disk,store}=setup(16*MiB),mask=new Uint8Array(64*64).fill(1);disk.fail=true;
          const entries=[entry(1,64,64,mask),entry(2,64,64,mask),entry(3,64,64,mask)];
          for(const e of entries)await store.addMany([e]);
          const bytes=store.session.bytes;disk.fail=false;await store.retry();
          console.log(JSON.stringify({bytes,after:store.session.bytes,masks:disk.staticMasks.size}));
        """)
        self.assertEqual(result["bytes"], 4096 + 3 * 16 * (6 * 8 + 4 + 4))
        self.assertEqual(result["after"], result["bytes"])
        self.assertEqual(result["masks"], 1)

    def test_identical_masks_from_different_archive_parts_share_resident_memory(self):
        result = javascript("""
          const {disk,store}=setup(16*MiB),a=entry(1,64,64,new Uint8Array(4096).fill(1)),b=entry(2,64,64,new Uint8Array(4096).fill(1));
          await store.addMany([a]);await store.addMany([b]);
          console.log(JSON.stringify({shared:a.mask===b.mask,masks:disk.staticMasks.size,bytes:store.residentBytes()}));
        """)
        self.assertTrue(result["shared"])
        self.assertEqual(result["masks"], 1)
        self.assertEqual(result["bytes"], 4096 + 2 * (16 * (6 * 8 + 4 + 4) + 120))

    def test_auxiliary_field_arrays_are_included_in_the_cache_budget(self):
        result = javascript("""
          const {store}=setup(4096),e=entry(1);e.field.roiSelected=new Uint8Array(16).fill(1);
          e.field.selectionAlias=e.field.roiSelected;
          await store.addMany([e]);console.log(JSON.stringify({resident:store.residentBytes(),payload:e.storage.descriptor.bytes}));
        """)
        self.assertEqual(result["payload"], 16 * (6 * 8 + 4 + 4) + 16)
        self.assertEqual(result["resident"], result["payload"] + 120)

    def test_large_recovery_uses_bounded_batches_and_resumes_after_partial_commit(self):
        result = javascript("""
          const {disk,store}=setup(1024),mask=new Uint8Array(512*512).fill(1);disk.fail=true;
          for(let i=1;i<=20;i++)await store.addMany([entry(i,512,512,mask)]);
          const expected=store.session.bytes,original=disk.write.bind(disk),batches=[];disk.fail=false;let fail=true;
          disk.write=async(s,records,masks)=>{if(fail&&batches.length===1)throw Error('Second recovery batch failed');batches.push(records.length);await original(s,records,masks);};
          let message='';try{await store.retry();}catch(e){message=e.message;}
          const partial=store.report();fail=false;await store.retry();
          console.log(JSON.stringify({message,partial,after:store.report(),count:disk.rows.size,masks:disk.staticMasks.size,batches,bytes:disk.summaries.get(store.session.id).bytes,expected}));
        """)
        self.assertIn("Second recovery batch", result["message"])
        self.assertEqual(result["partial"]["diskFields"], 1)
        self.assertEqual(result["partial"]["unsavedFields"], 19)
        self.assertEqual(result["after"]["unsavedFields"], 0)
        self.assertEqual(result["count"], 20)
        self.assertEqual(result["masks"], 1)
        self.assertLessEqual(max(result["batches"]), 4)
        self.assertEqual(result["bytes"], result["expected"])

    def test_reopen_is_lazy_and_restores_analysis_without_loading_all_fields(self):
        result = javascript("""
          const {disk,store}=setup(),entries=[entry(1),entry(2),entry(3)];for(const e of entries)await store.addMany([e]);
          entries[0].recipe={smoothing:'gaussian'};entries[0].metadata.workbench={view:{component:'u'}};await store.saveSettings(entries,1);
          const reopened=new FieldStorage({memoryLimit:4096,openDatabase:async()=>disk}),saved=await reopened.openSession(store.session.id);
          const lazy=saved.entries.every(e=>!e.field.parameters),settings=saved.entries[0].recipe;
          await reopened.load(saved.entries[0]);let refused=false;try{await reopened.addMany([entry(4)]);}catch{refused=true;}
          console.log(JSON.stringify({lazy,settings,selected:saved.selectedId,view:saved.entries[0].metadata.workbench,refused,value:saved.entries[0].field.parameters[0]}));
        """)
        self.assertTrue(result["lazy"])
        self.assertTrue(result["refused"])
        self.assertEqual(result["selected"], 1)
        self.assertEqual(result["settings"]["smoothing"], "gaussian")
        self.assertEqual(result["view"]["view"]["component"], "u")

    def test_missing_mask_fails_atomically_instead_of_returning_an_unmasked_field(self):
        result = javascript("""
          const {disk,store}=setup(1024),mask=new Uint8Array(64*64).fill(1),e=entry(1,64,64,mask);await store.addMany([e]);
          disk.staticMasks.clear();let message='';try{await store.load(e);}catch(error){message=error.message;}
          console.log(JSON.stringify({message,notLoaded:!e.field.parameters}));
        """)
        self.assertIn("ROI", result["message"])
        self.assertTrue(result["notLoaded"])

    def test_small_views_do_not_keep_large_backing_buffers_alive(self):
        result = javascript("""
          const {store}=setup(),e=entry(1),large=new Float64Array(1000000);large[3]=.123456789012345;e.field.parameters=large.subarray(3,3+e.field.parameters.length);
          await store.addMany([e]);console.log(JSON.stringify({bytes:e.field.parameters.buffer.byteLength,expected:e.field.parameters.byteLength,value:e.field.parameters[0]}));
        """)
        self.assertEqual(result["bytes"], result["expected"])
        self.assertEqual(result["value"], 0.123456789012345)

    def test_conflicting_settings_are_reported_instead_of_overwritten(self):
        result = javascript("""
          const {store,disk}=setup(),e=entry(1);await store.addMany([e]);const other=new FieldStorage({openDatabase:async()=>disk});const saved=await other.openSession(store.session.id);
          await store.saveSettings([e],1);let conflict='';try{await other.saveSettings(saved.entries,1);}catch(error){conflict=error.message;}
          console.log(JSON.stringify({conflict}));
        """)
        self.assertIn("Concurrent", result["conflict"])

    def test_history_reads_evicted_fields_lazily_in_saved_order(self):
        result = javascript(
            f"import {{extractHistory}} from {json.dumps((UI / 'history.js').as_uri())};"
            + """
          const {store}=setup(1024),entries=[entry(1),entry(2),entry(3),entry(4)];for(const e of entries)await store.addMany([e]);
          const order=[],rows=await extractHistory(entries,{recipe:{},component:'u',points:[[7.5,7.5]],tool:'probe',loadEntry:async e=>{order.push(e.id);return store.load(e);}});
          console.log(JSON.stringify({order,values:rows.map(row=>row.value),ids:rows.map(row=>row.fieldId)}));
        """
        )
        self.assertEqual(result["order"], [1, 2, 3, 4])
        self.assertEqual(result["ids"], [1, 2, 3, 4])
        np_values = [value + 0.123456789012345 for value in range(1, 5)]
        self.assertEqual(result["values"], np_values)

    def test_gpu_api_resource_accounting_deduplicates_handles_and_keeps_vram_unknown(self):
        module = UI.parents[1] / "python/src/lucas_kanade_icgn/web/core/gpu-memory.js"
        result = javascript(
            f"import {{gpuMemoryUsage}} from {json.dumps(module.as_uri())};"
            + """
          const buffer={[Symbol.toStringTag]:'GPUBuffer',size:4096,usage:1};
          const texture={[Symbol.toStringTag]:'GPUTexture',width:3840,height:2160,depthOrArrayLayers:1,mipLevelCount:1,sampleCount:1,format:'rgba8unorm'};
          const graph={a:buffer,slots:[buffer,texture],map:new Map([['same',texture]]),hostData:new Float64Array(100)};graph.cycle=graph;
          console.log(JSON.stringify({filled:gpuMemoryUsage(graph),empty:gpuMemoryUsage(null)}));
        """
        )
        self.assertEqual(result["filled"]["bufferCount"], 1)
        self.assertEqual(result["filled"]["mappedBufferBytes"], 4096)
        self.assertEqual(result["filled"]["textureCount"], 1)
        self.assertEqual(result["filled"]["knownResourceBytes"], 4096 + 3840 * 2160 * 4)
        self.assertIsNone(result["filled"]["freeVramBytes"])
        self.assertIsNone(result["filled"]["physicalVramBytes"])
        self.assertEqual(result["empty"]["knownResourceBytes"], 0)
