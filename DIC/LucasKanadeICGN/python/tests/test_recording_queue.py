"""Bounded recording ownership, backpressure, order and failure contracts."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

UI = Path(__file__).resolve().parents[2] / "web-app/ui"


def run(script):
    return json.loads(
        subprocess.check_output(
            ["node", "--input-type=module"],
            input=f"import {{RecordingQueue}} from {json.dumps((UI / 'recording-queue.js').as_uri())};\n"
            + script,
            text=True,
        )
    )


@unittest.skipUnless(shutil.which("node"), "Node is needed for queue ownership tests")
class RecordingQueueTests(unittest.TestCase):
    def test_two_reservations_bound_the_gpu_and_pending_disk_field(self):
        actual = run("""
          let release;const writes=[],q=new RecordingQueue({write:async r=>{writes.push(r.id);if(r.id===1)await new Promise(resolve=>release=resolve);},retain:async()=>{throw Error('unexpected retention');}});
          const first=q.reserve(),second=q.reserve(),full=q.reserve();
          const record=id=>({id,field:{parameters:new Float64Array([id+.125])}});
          const a=q.submit(first,record(1));await Promise.resolve();
          const b=q.submit(second,record(2));let drained=false;const drain=q.drain().then(()=>drained=true);
          await Promise.resolve();const before={pending:q.pending,full,drained,writes:[...writes]};
          q.release(first);const stillHeld=q.pending;
          release();await Promise.all([a,b,drain]);console.log(JSON.stringify({before,stillHeld,writes,report:q.report()}));
        """)
        self.assertEqual(actual["before"], dict(pending=2, full=None, drained=False, writes=[1]))
        self.assertEqual(actual["stillHeld"], 2)
        self.assertEqual(actual["writes"], [1, 2])
        self.assertEqual(actual["report"]["pending"], 0)
        self.assertEqual(actual["report"]["peakPending"], 2)

    def test_disk_failure_retains_both_completed_fields_and_blocks_new_admission(self):
        actual = run("""
          const written=[],retained=[],failures=[];
          const q=new RecordingQueue({write:async r=>{written.push(r.id);throw Error('quota');},retain:async r=>retained.push(r.id),onFailure:e=>failures.push(e.message)});
          const a=q.reserve(),b=q.reserve(),record=id=>({id,field:{parameters:new Float64Array([id])}});
          await Promise.all([q.submit(a,record(1)),q.submit(b,record(2))]);await q.drain();
          console.log(JSON.stringify({written,retained,failures,next:q.reserve(),report:q.report()}));
        """)
        self.assertEqual(actual["written"], [1])
        self.assertEqual(actual["retained"], [1, 2])
        self.assertEqual(actual["failures"], ["quota"])
        self.assertIsNone(actual["next"])
        self.assertEqual(actual["report"]["retainedAfterFailure"], 2)

    def test_storage_layer_ram_retention_is_not_duplicated(self):
        actual = run("""
          const writes=[],retained=[];const q=new RecordingQueue({write:async r=>{writes.push(r.id);return{failure:'quota; this field is already in RAM'};},retain:async r=>retained.push(r.id)});
          const a=q.reserve(),b=q.reserve();await Promise.all([q.submit(a,{id:1,field:{}}),q.submit(b,{id:2,field:{}})]);await q.drain();console.log(JSON.stringify({writes,retained,count:q.report().retainedAfterFailure}));
        """)
        self.assertEqual(actual, dict(writes=[1], retained=[2], count=2))

    def test_failed_ram_retention_keeps_the_original_object_and_rejects_drain(self):
        actual = run("""
          const record={field:{parameters:new Float64Array([.123456789012345])}};
          const q=new RecordingQueue({write:async()=>{throw Error('disk');},retain:async()=>{throw Error('RAM admission');}}),ticket=q.reserve();
          await q.submit(ticket,record).catch(()=>{});let rejected=false;try{await q.drain();}catch{rejected=true;}
          console.log(JSON.stringify({rejected,same:q.entries.get(ticket).record===record,report:q.report()}));
        """)
        self.assertTrue(actual["rejected"])
        self.assertTrue(actual["same"])
        self.assertEqual(actual["report"]["pendingArrayBytes"], 8)

    def test_waiting_file_admission_and_failed_gpu_reservations_release_capacity(self):
        actual = run("""
          const q=new RecordingQueue({write:async()=>{},retain:async()=>{}}),a=q.reserve(),b=q.reserve();let ticket;
          const waiting=q.waitReserve().then(value=>ticket=value);await Promise.resolve();const blocked=ticket===undefined;
          q.release(a);await waiting;q.release(b);q.release(ticket);await q.drain();
          console.log(JSON.stringify({blocked,valid:ticket!==null,pending:q.pending}));
        """)
        self.assertEqual(actual, dict(blocked=True, valid=True, pending=0))


@unittest.skipUnless(shutil.which("node"), "Node is needed for supervisor contracts")
class SupervisionOptionsTests(unittest.TestCase):
    def test_defaults_are_synchronous_and_preview_geometry_does_not_change_source(self):
        script = f"""
          import {{supervisionOptions,presentationSize}} from {json.dumps((UI / "supervision.js").as_uri())};
          const invalid=[null,{{overlapRecording:1}},{{previewEnabled:0}},{{displayRate:-1}},{{displayRate:NaN}},{{previewMaxWidth:1.5}},{{unknown:1}}];
          const rejected=invalid.map(value=>{{try{{supervisionOptions(value);return false;}}catch{{return true;}}}});
          console.log(JSON.stringify({{defaults:supervisionOptions(),native:presentationSize(3840,2160),preview:presentationSize(3840,2160,1920),odd:presentationSize(99,97,64),rejected}}));
        """
        actual = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertEqual(
            actual["defaults"],
            dict(overlapRecording=False, displayRate=0, previewMaxWidth=0, previewEnabled=True),
        )
        self.assertEqual(actual["native"], [3840, 2160])
        self.assertEqual(actual["preview"], [1920, 1080])
        self.assertEqual(actual["odd"], [64, 63])
        self.assertTrue(all(actual["rejected"]))


if __name__ == "__main__":
    unittest.main()
