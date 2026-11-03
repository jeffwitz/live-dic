"""Tests for the local browser compatibility report."""

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class BrowserDeviceReportTests(unittest.TestCase):
    def test_report_contains_runtime_capabilities_without_experiment_data(self):
        module = (ROOT / "web-app/ui/device-report.js").as_uri()
        script = f"""
          import {{createDeviceReport}} from {json.dumps(module)};
          const report = createDeviceReport({{
            capabilities: {{supported: true, adapter: {{vendor: 'test'}}, limits: {{maxBufferSize: 4096}}}},
            build: {{commit: 'abc123', dirty: false}},
            navigatorObject: {{userAgent: 'Example Browser', platform: 'TestOS',
              userAgentData: {{brands: [{{brand: 'Example', version: '1'}}], mobile: true, platform: 'TestOS'}},
              language: 'en', onLine: true, hardwareConcurrency: 8, deviceMemory: 4}},
            secureContext: true,
            generatedAt: '2026-11-03T12:00:00.000Z',
          }});
          console.log(JSON.stringify(report));
        """
        report = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )

        self.assertEqual(report["format"], "live-dic-device-report")
        self.assertEqual(report["schemaVersion"], 1)
        self.assertEqual(report["generatedAt"], "2026-11-03T12:00:00.000Z")
        self.assertEqual(report["app"]["build"]["commit"], "abc123")
        self.assertEqual(report["browser"]["hardwareConcurrency"], 8)
        self.assertTrue(report["browser"]["userAgentData"]["mobile"])
        self.assertEqual(report["webgpu"]["limits"]["maxBufferSize"], 4096)
        self.assertFalse({"acquisition", "images", "mask", "fields"} & report.keys())

    def test_report_is_serializable_when_browser_capabilities_are_unavailable(self):
        module = (ROOT / "web-app/ui/device-report.js").as_uri()
        script = f"""
          import {{createDeviceReport}} from {json.dumps(module)};
          console.log(JSON.stringify(createDeviceReport({{
            capabilities: null,
            build: null,
            navigatorObject: {{userAgent: 'No WebGPU browser'}},
            secureContext: false,
            generatedAt: '2026-11-03T12:00:00.000Z',
          }})));
        """
        report = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )

        self.assertIsNone(report["webgpu"])
        self.assertIsNone(report["browser"]["platform"])
        self.assertFalse(report["browser"]["secureContext"])

    def test_device_report_download_is_explained_in_the_ui(self):
        html = (ROOT / "web-app/index.html").read_text(encoding="utf-8")
        self.assertIn('id="download-device-report"', html)
        self.assertIn("it is never uploaded", html)


if __name__ == "__main__":
    unittest.main()
