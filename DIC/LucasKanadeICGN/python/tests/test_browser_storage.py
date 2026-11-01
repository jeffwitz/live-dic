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
