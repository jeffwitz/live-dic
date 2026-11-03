"""Check that offered UI combinations satisfy the actual core option contract."""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("node"), "Node is needed for solver-control contracts")
class SolverControlTests(unittest.TestCase):
    def test_initialization_controls_are_one_radio_group(self):
        class InputParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.inputs = []

            def handle_starttag(self, tag, attrs):
                if tag == "input":
                    self.inputs.append(dict(attrs))

        parser = InputParser()
        parser.feed((ROOT / "web-app/index.html").read_text(encoding="utf-8"))
        initialization = [item for item in parser.inputs if item.get("name") == "initialization"]
        self.assertEqual(
            {item.get("id") for item in initialization},
            {"fourier-start", "global-affine-start", "temporal-start"},
        )
        self.assertTrue(all(item.get("type") == "radio" for item in initialization))
        self.assertEqual(
            [item["id"] for item in initialization if "checked" in item],
            ["temporal-start"],
        )

    def test_analysis_only_initializers_and_solver_controls_are_hidden_from_live(self):
        html = (ROOT / "web-app/index.html").read_text(encoding="utf-8")
        self.assertIn('<fieldset id="initialization-options" data-workspace-only="analysis">', html)
        self.assertIn('<button id="large-shift-preset" data-workspace-only="analysis">', html)
        self.assertIn(
            '<label class="check" data-workspace-only="analysis"><input id="custom-solver"', html
        )
        self.assertIn(
            '<label class="check" data-workspace-only="analysis"><input id="gpu-reference-preparation"',
            html,
        )
        self.assertIn('data-workspace-only="live" hidden>Live uses the fixed', html)
        self.assertNotIn('id="profile"', html)

    def test_preview_controls_are_independent_of_recording_overlap(self):
        script = f"""
          import {{solverOptions}} from {json.dumps((ROOT / "web-app/ui/workspace.js").as_uri())};
          const values={{'custom-solver':{{checked:false}},
            'display-rate':{{value:'10'}},'preview-width':{{value:'1920'}},'overlap-pipeline':{{checked:false}}}};
          globalThis.document={{getElementById:id=>values[id]}};
          const normal=solverOptions();values['display-rate'].value='off';
          const suspended=solverOptions();values['overlap-pipeline'].checked=true;
          console.log(JSON.stringify({{normal,suspended,overlapped:solverOptions()}}));
        """
        result = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertEqual(result["normal"]["profile"], "portable")
        self.assertEqual(
            result["normal"]["supervision"],
            dict(overlapRecording=False, displayRate=10, previewMaxWidth=1920, previewEnabled=True),
        )
        self.assertFalse(result["suspended"]["supervision"]["previewEnabled"])
        self.assertFalse(result["suspended"]["supervision"]["overlapRecording"])
        self.assertTrue(result["overlapped"]["supervision"]["overlapRecording"])

    def test_live_mode_uses_temporal_preset_and_ignores_analysis_initializers(self):
        ui = ROOT / "web-app/ui/workspace.js"
        script = f"""
          import {{solverOptions}} from {json.dumps(ui.as_uri())};
          const values={{'custom-solver':{{checked:true}},'fourier-start':{{checked:true}},
            'global-affine-start':{{checked:true}},'global-affine-levels':{{value:'8'}},
            'temporal-start':{{checked:false}},'pyramid-levels':{{value:'6'}},
            'gpu-reference-preparation':{{checked:true}},'display-rate':{{value:'10'}},
            'preview-width':{{value:'1920'}},'overlap-pipeline':{{checked:false}}}};
          globalThis.document={{body:{{dataset:{{workspaceMode:'live'}}}},getElementById:id=>values[id]??null}};
          console.log(JSON.stringify(solverOptions()));
        """
        actual = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertEqual(actual["profile"], "portable")
        self.assertEqual(actual["initialization"], "temporal")
        self.assertEqual(actual["temporalPrediction"], "velocity")
        self.assertTrue(actual["temporalFineCorrection"])
        self.assertEqual(actual["divergencePatience"], 3)
        self.assertNotIn("globalAffineLevels", actual)
        self.assertNotIn("pyramidLevels", actual)
        self.assertNotIn("referencePreparation", actual)
        self.assertEqual(actual["supervision"]["displayRate"], 10)

    def test_only_cold_configuration_changes_require_preparation(self):
        module = ROOT / "web-app/ui/workspace.js"
        script = f"""
          import {{solverPreparationKey}} from {json.dumps(module.as_uri())};
          const original={{profile:'portable',initialization:'zero',supervision:{{displayRate:10,previewMaxWidth:1920,previewEnabled:true,overlapRecording:false}}}};
          const display={{...original,supervision:{{...original.supervision,displayRate:1,previewMaxWidth:1280,previewEnabled:false}}}};
          const changed={{...original,initialization:'fourier'}};
          const overlap={{...original,supervision:{{...original.supervision,overlapRecording:true}}}};
          console.log(JSON.stringify([solverPreparationKey(original)===solverPreparationKey(display),solverPreparationKey(original)!==solverPreparationKey(changed),solverPreparationKey(original)!==solverPreparationKey(overlap)]));
        """
        actual = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertEqual(actual, [True, True, True])

    def test_global_affine_scales_are_a_cold_solver_setting(self):
        ui = ROOT / "web-app/ui/workspace.js"
        script = f"""
          import {{solverOptions,solverPreparationKey}} from {json.dumps(ui.as_uri())};
          const values={{'custom-solver':{{checked:false}},
            'global-affine-start':{{checked:true}},'global-affine-levels':{{value:'6'}}}};
          globalThis.document={{getElementById:id=>values[id]}};
          const six=solverOptions();values['global-affine-levels'].value='7';const seven=solverOptions();
          console.log(JSON.stringify({{six,seven,different:solverPreparationKey(six)!==solverPreparationKey(seven)}}));
        """
        actual = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertEqual(actual["six"]["globalAffineLevels"], 6)
        self.assertEqual(actual["seven"]["globalAffineLevels"], 7)
        self.assertTrue(actual["different"])

    def test_global_affine_initializer_keeps_translation_local_model(self):
        ui = ROOT / "web-app/ui/workspace.js"
        script = f"""
          import {{solverOptions}} from {json.dumps(ui.as_uri())};
          const values={{'custom-solver':{{checked:true}},
            'global-affine-start':{{checked:true}},'global-affine-levels':{{value:'6'}},
            'fourier-start':{{checked:false}},'subset-size':{{value:'16'}},'subset-stride':{{value:'16'}},
            'pyramid-levels':{{value:'2'}},'coarse-iterations':{{value:'8'}},'fine-iterations':{{value:'16'}},
            'temporal-start':{{checked:false}},'motion-model':{{value:'translation'}},
            interpolation:{{value:'bspline'}},'outlier-policy':{{value:'disabled'}},'median-filter-size':{{value:'3'}}}};
          globalThis.document={{getElementById:id=>values[id]??null}};
          console.log(JSON.stringify({{options:solverOptions(),selectedModel:values['motion-model'].value}}));
        """
        actual = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertEqual(actual["options"]["initialization"], "global-affine")
        self.assertEqual(actual["options"]["motionModel"], "translation")
        self.assertEqual(actual["options"]["globalAffineLevels"], 6)
        self.assertEqual(actual["options"]["temporalPrediction"], "none")
        self.assertEqual(actual["options"]["profile"], "custom")
        self.assertEqual(actual["selectedModel"], "translation")

    def test_preset_is_unchanged_and_custom_combinations_reach_the_core(self):
        ui = ROOT / "web-app/ui/workspace.js"
        core = ROOT / "python/src/lucas_kanade_icgn/web/core/engine.js"
        script = f"""
          import {{solverOptions}} from {json.dumps(ui.as_uri())};
          import {{DICEngine}} from {json.dumps(core.as_uri())};
          const values={{'custom-solver':{{checked:false}},'subset-size':{{value:'16'}},'subset-stride':{{value:'16'}},'pyramid-levels':{{value:'2'}},'coarse-iterations':{{value:'8'}},'fine-iterations':{{value:'16'}},'temporal-start':{{checked:true}},'motion-model':{{value:'affine'}},interpolation:{{value:'bspline'}},'outlier-policy':{{value:'median'}}}};
          globalThis.document={{getElementById:id=>values[id]}};const preset=solverOptions();values['custom-solver'].checked=true;
          let count=0;const errors=[];
          for(const model of ['translation','affine'])for(const level of [1,2,3,4])for(const mode of ['bspline','linear','cubic'])for(const policy of ['median','replace','flag','disabled'])for(const temporal of [false,true]){{
            values['motion-model'].value=model;values['pyramid-levels'].value=String(level);values.interpolation.value=mode;values['outlier-policy'].value=policy;values['temporal-start'].checked=temporal;
            try{{const {{profile,...options}}=solverOptions();const engine=new DICEngine({{...options,width:3840,height:2160}});engine.destroy();count++;}}catch(e){{errors.push(e.message);}}
          }}console.log(JSON.stringify({{preset,count,errors}}));
        """
        result = json.loads(
            subprocess.check_output(
                ["node", "--input-type=module"], input=script, text=True, encoding="utf-8"
            )
        )
        self.assertEqual(result["preset"], {"profile": "portable"})
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["count"], 192)

    def test_median_neighbourhood_is_a_cold_setting_and_reaches_the_core(self):
        ui = ROOT / "web-app/ui/workspace.js"
        core = ROOT / "python/src/lucas_kanade_icgn/web/core/engine.js"
        script = f"""
          import {{solverOptions,solverPreparationKey}} from {json.dumps(ui.as_uri())};
          import {{DICEngine}} from {json.dumps(core.as_uri())};
          const values={{'custom-solver':{{checked:true}},'subset-size':{{value:'16'}},'subset-stride':{{value:'16'}},'median-filter-size':{{value:'5'}},'pyramid-levels':{{value:'2'}},'coarse-iterations':{{value:'8'}},'fine-iterations':{{value:'16'}},'temporal-start':{{checked:false}},'motion-model':{{value:'affine'}},interpolation:{{value:'bspline'}},'outlier-policy':{{value:'median'}}}};
          globalThis.document={{getElementById:id=>values[id]}};
          const options=solverOptions();const {{profile,...coreOptions}}=options;const engine=new DICEngine({{...coreOptions,width:128,height:128}});engine.destroy();
          values['median-filter-size'].value='9';const larger=solverOptions();
          values['outlier-policy'].value='replace';const unsmoothed=solverOptions();
          console.log(JSON.stringify({{options,larger,unsmoothed,cold:solverPreparationKey(options)!==solverPreparationKey(larger)}}));
        """
        actual = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertEqual(actual["options"]["medianFilterSize"], 5)
        self.assertEqual(actual["larger"]["medianFilterSize"], 9)
        self.assertEqual(actual["unsmoothed"]["medianFilterSize"], 3)
        self.assertTrue(actual["cold"])

    def test_workspace_modes_preserve_the_current_pane_and_toggle_mode_content(self):
        ui = ROOT / "web-app/ui/workspace.js"
        script = f"""
          import {{installWorkspace}} from {json.dumps(ui.as_uri())};
          const element=(dataset={{}})=>({{dataset,hidden:false,attributes:{{}},listeners:{{}},
            setAttribute(name,value){{this.attributes[name]=value;}},
            toggleAttribute(name,force){{if(force)this.attributes[name]=true;else delete this.attributes[name];}},
            addEventListener(name,callback){{this.listeners[name]=callback;}},
            click(){{this.onclick?.();}},focus(){{}},showModal(){{}},open:false}});
          const panes=['source','mask','solve','view'].map(name=>element({{pane:name}}));
          const modes=['live','analysis'].map(name=>element({{workspaceMode:name}}));
          const copies=['live','analysis'].map(name=>element({{modeCopy:name}}));
          const liveOnly=element({{workspaceOnly:'live'}}),analysisOnly=element({{workspaceOnly:'analysis'}});
          const panels=new Map(['source','mask','solve','view'].map(name=>[`pane-${{name}}`,element()]));
          const help=element(),policy=element();policy.value='median';
          const medianSize=element();
          globalThis.document={{body:{{dataset:{{}}}},
            querySelectorAll(selector){{return {{'[data-pane]':panes,'button[data-workspace-mode]':modes,
              '[data-mode-copy]':copies,'[data-workspace-only]':[liveOnly,analysisOnly]}}[selector]??[];}},
            getElementById(id){{return panels.get(id)??({{help,'quick-guide':help,'outlier-policy':policy,'median-filter-size':medianSize}}[id]??element());}},
            querySelector(){{return null;}}}};
          globalThis.window={{addEventListener(){{}}}};
          globalThis.localStorage={{writes:[],getItem(){{return null;}},setItem(key,value){{this.writes.push([key,value]);}}}};
          const paneCalls=[],modeCalls=[];const workspace=installWorkspace({{shortcuts:{{play(){{}},reference(){{}},clear(){{}}}},
            onPane:name=>paneCalls.push(name),onModeChange:name=>modeCalls.push(name)}});
          workspace.showPane('view');modes[0].click();
          const live={{mode:workspace.mode,pane:workspace.pane,cameraHidden:liveOnly.hidden,
            analysisHidden:analysisOnly.hidden,liveCopyHidden:copies[0].hidden,analysisCopyHidden:copies[1].hidden}};
          modes[1].click();
          const analysis={{mode:workspace.mode,pane:workspace.pane,cameraHidden:liveOnly.hidden,
            analysisHidden:analysisOnly.hidden,liveCopyHidden:copies[0].hidden,analysisCopyHidden:copies[1].hidden}};
          let invalidRejected=false;try{{workspace.showMode('unknown');}}catch{{invalidRejected=true;}}
          console.log(JSON.stringify({{live,analysis,paneCalls,modeCalls,modeWrites:localStorage.writes,invalidRejected}}));
        """
        actual = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )
        self.assertEqual(
            actual["live"],
            dict(
                mode="live",
                pane="view",
                cameraHidden=False,
                analysisHidden=True,
                liveCopyHidden=False,
                analysisCopyHidden=True,
            ),
        )
        self.assertEqual(
            actual["analysis"],
            dict(
                mode="analysis",
                pane="view",
                cameraHidden=True,
                analysisHidden=False,
                liveCopyHidden=True,
                analysisCopyHidden=False,
            ),
        )
        self.assertEqual(actual["paneCalls"], ["view"])
        self.assertEqual(actual["modeCalls"], ["analysis", "live", "analysis"])
        self.assertEqual(
            actual["modeWrites"],
            [["live-dic-workspace-mode", "live"], ["live-dic-workspace-mode", "analysis"]],
        )
        self.assertTrue(actual["invalidRejected"])
