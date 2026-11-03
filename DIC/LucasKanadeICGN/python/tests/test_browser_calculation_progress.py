"""Contracts for the Solve pane's calculation progress indicator."""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "web-app/ui/calculation-progress.js"


@unittest.skipUnless(shutil.which("node"), "Node is needed for browser progress contracts")
class BrowserCalculationProgressTests(unittest.TestCase):
    def run_progress(self, actions: str) -> dict:
        script = f"""
          import {{createCalculationProgress, imageSequenceTargetCount}} from {json.dumps(MODULE.as_uri())};
          const elements = new Map();
          const getElement = id => {{
            if (!elements.has(id)) elements.set(id, {{
              hidden: false, textContent: '', _value: 0, max: 1,
              attributes: new Map(),
              get value() {{ return this._value; }},
              set value(value) {{ this._value = value; this.attributes.set('value', String(value)); }},
              removeAttribute(name) {{ this.attributes.delete(name); }},
              setAttribute(name, value) {{ this.attributes.set(name, value); }},
            }});
            return elements.get(id);
          }};
          const progress = createCalculationProgress({{getElement}});
          {actions}
          console.log(JSON.stringify({{
            state: progress.snapshot(),
            containerHidden: getElement('solve-progress-ui').hidden,
            barHidden: getElement('solve-progress-bar').hidden,
            value: getElement('solve-progress-bar').value,
            max: getElement('solve-progress-bar').max,
            hasValue: getElement('solve-progress-bar').attributes.has('value'),
            detail: getElement('solve-progress-detail').textContent,
          }}));
        """
        output = subprocess.check_output(
            ["node", "--input-type=module"], input=script, text=True, encoding="utf-8"
        )
        return json.loads(output)

    def test_known_sequence_reports_partial_progress_when_stopped(self):
        result = self.run_progress(
            "progress.begin('Preparing reference…');"
            "progress.setTargets(4); progress.advance(); progress.advance(); progress.finish();"
        )
        self.assertFalse(result["containerHidden"])
        self.assertFalse(result["barHidden"])
        self.assertEqual((result["value"], result["max"]), (2, 4))
        self.assertTrue(result["hasValue"])
        self.assertEqual(result["detail"], "Calculation stopped · 2 / 4 frame(s)")

    def test_known_sequence_reports_completion_at_its_target(self):
        result = self.run_progress(
            "progress.begin('Preparing reference…'); progress.setTargets(2);"
            "progress.advance(); progress.advance(); progress.finish();"
        )
        self.assertEqual((result["value"], result["max"]), (2, 2))
        self.assertEqual(result["detail"], "Calculation complete · 2 / 2 frame(s)")

    def test_indefinite_stream_shows_count_then_stops_cleanly(self):
        result = self.run_progress(
            "progress.begin('Preparing reference…'); progress.setTargets(null);"
            "progress.advance(); progress.advance(); progress.finish();"
        )
        self.assertFalse(result["containerHidden"])
        self.assertTrue(result["barHidden"])
        self.assertFalse(result["hasValue"])
        self.assertEqual(result["detail"], "Calculation ended after 2 frame(s).")

    def test_reset_hides_progress_for_a_new_source(self):
        result = self.run_progress("progress.begin('Preparing…'); progress.reset();")
        self.assertTrue(result["containerHidden"])
        self.assertTrue(result["barHidden"])
        self.assertEqual(result["state"]["completed"], 0)

    def test_sequence_target_count_matches_current_and_consecutive_ranges(self):
        script = f"""
          import {{imageSequenceTargetCount}} from {json.dumps(MODULE.as_uri())};
          const count = options => imageSequenceTargetCount({{imageCount:5,...options}});
          console.log(JSON.stringify([
            count({{currentIndex:1,includeCurrent:true,range:'remaining',comparison:'fixed'}}),
            count({{currentIndex:1,includeCurrent:false,range:'remaining',comparison:'fixed'}}),
            count({{currentIndex:0,includeCurrent:true,range:'remaining',comparison:'consecutive'}}),
            count({{currentIndex:0,includeCurrent:false,range:'single',comparison:'consecutive'}}),
          ]));
        """
        output = subprocess.check_output(
            ["node", "--input-type=module"], input=script, text=True, encoding="utf-8"
        )
        self.assertEqual(json.loads(output), [4, 3, 4, 0])


if __name__ == "__main__":
    unittest.main()
