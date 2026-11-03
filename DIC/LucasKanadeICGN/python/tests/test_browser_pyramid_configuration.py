"""The WebGPU host keeps solver policy validation outside GPU orchestration."""

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class BrowserPyramidConfigurationTests(unittest.TestCase):
    def test_configuration_defaults_and_level_schedules_are_unchanged(self):
        module = (ROOT / "python/src/lucas_kanade_icgn/web/core/pyramid.js").as_uri()
        script = f"""
          import {{createPyramidHost}} from {json.dumps(module)};
          const host = createPyramidHost();
          const initial = host.configure();
          const changed = host.configure({{pyramidLevels: 3, iterations: 12}});
          console.log(JSON.stringify({{initial, changed}}));
        """
        result = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )

        self.assertEqual(result["initial"]["pyramidLevels"], 2)
        self.assertEqual(result["initial"]["iterationsPerLevel"], [8, 16])
        self.assertEqual(result["initial"]["tolerancePerLevel"], [0.005, 0.005])
        self.assertEqual(result["changed"]["pyramidLevels"], 3)
        self.assertEqual(result["changed"]["iterationsPerLevel"], [6, 8, 12])
        self.assertEqual(result["changed"]["tolerancePerLevel"], [0.01, 0.005, 0.005])

    def test_invalid_configuration_does_not_mutate_the_current_settings(self):
        module = (ROOT / "python/src/lucas_kanade_icgn/web/core/pyramid.js").as_uri()
        script = f"""
          import {{createPyramidHost}} from {json.dumps(module)};
          const host = createPyramidHost();
          const before = host.configure();
          let error = null;
          try {{ host.configure({{motionModel: 'rigid-plus'}}); }}
          catch (caught) {{ error = caught.message; }}
          console.log(JSON.stringify({{before, after: host.configure(), error}}));
        """
        result = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )

        self.assertEqual(result["before"], result["after"])
        self.assertEqual(
            result["error"], "motionModel must be 'affine', 'translation', or 'quadratic'"
        )


if __name__ == "__main__":
    unittest.main()
