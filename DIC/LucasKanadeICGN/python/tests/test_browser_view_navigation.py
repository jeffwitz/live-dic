"""Tests for the isolated image viewport controls."""

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class BrowserViewNavigationTests(unittest.TestCase):
    def test_zoom_pan_and_fit_are_viewport_only_operations(self):
        module = (ROOT / "web-app/ui/view-navigation.js").as_uri()
        script = f"""
          import {{installViewNavigation}} from {json.dumps(module)};
          const elements = new Map();
          const element = id => {{
            if (!elements.has(id)) elements.set(id, {{
              style: {{}}, textContent: '', clientWidth: 400, clientHeight: 300,
              scrollWidth: 700, scrollHeight: 550, scrollLeft: 100, scrollTop: 80,
              scrollBy(value) {{ this.lastScrollBy = value; }},
              scrollTo(value) {{ this.lastScrollTo = value; }},
            }});
            return elements.get(id);
          }};
          globalThis.requestAnimationFrame = callback => {{ callback(); return 1; }};
          globalThis.ResizeObserver = undefined;
          installViewNavigation({{getElement: element, updateVectorLengthLegend() {{}}}});
          element('view-zoom-in').onclick();
          element('view-pan-right').onclick();
          const zoomed = {{width: element('scene').style.width, label: element('view-zoom-label').textContent}};
          const panned = element('image-viewport').lastScrollBy;
          element('view-reset').onclick();
          console.log(JSON.stringify({{
            zoomed, panned,
            fitted: {{width: element('scene').style.width, label: element('view-zoom-label').textContent,
              scroll: element('image-viewport').lastScrollTo}},
          }}));
        """
        result = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )

        self.assertEqual(result["zoomed"], {"width": "150%", "label": "150%"})
        self.assertEqual(result["panned"]["left"], 195)
        self.assertEqual(result["fitted"]["width"], "100%")
        self.assertEqual(result["fitted"]["label"], "100%")
        self.assertEqual(result["fitted"]["scroll"]["left"], 150)
        self.assertEqual(result["fitted"]["scroll"]["top"], 125)


if __name__ == "__main__":
    unittest.main()
