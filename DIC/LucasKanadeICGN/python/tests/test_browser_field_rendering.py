"""Tests for the isolated browser results raster renderer."""

import json
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class BrowserFieldRenderingTests(unittest.TestCase):
    def test_percentiles_palette_and_raster_rendering(self):
        module = (ROOT / "web-app/ui/field-rendering.js").as_uri()
        script = f"""
          class Context {{
            constructor() {{ this.strokes = []; }}
            createImageData(width, height) {{
              return {{width, height, data: new Uint8ClampedArray(width * height * 4)}};
            }}
            putImageData(image) {{ this.image = image; }}
            fillRect(...args) {{ this.fillRectArgs = args; }}
            drawImage(...args) {{ this.drawImageArgs = args; }}
            stroke(path) {{ this.strokes.push(path); }}
          }}
          class MockCanvas {{
            constructor(width, height) {{ this.width = width; this.height = height; this.context = new Context(); }}
            getContext() {{ return this.context; }}
          }}
          class MockPath {{
            constructor() {{ this.operations = []; }}
            moveTo(...point) {{ this.operations.push(['move', ...point]); }}
            lineTo(...point) {{ this.operations.push(['line', ...point]); }}
          }}
          globalThis.OffscreenCanvas = MockCanvas;
          globalThis.Path2D = MockPath;
          const {{fieldColour, percentileBounds, renderFieldRaster}} = await import({json.dumps(module)});
          const analysis = {{
            field: {{width: 4, height: 2, grid: [2, 1], subsetSize: 2, subsetStride: 2}},
            valid: new Uint8Array([1, 0]),
            fields: {{u: new Float32Array([0.5, 0]), v: new Float32Array([0, 0])}},
            recipe: {{unitsPerPixel: 1}},
          }};
          const values = new Float32Array([0.25, 0.75]);
          const valid = new Uint8Array([1, 0]);
          const diagnostic = {{values: new Float32Array([0, 0.5]), valid: new Uint8Array([0, 1])}};
          const canvas = new MockCanvas(0, 0), overlay = new MockCanvas(0, 0);
          const mask = new Uint8Array([1, 0, 0, 0, 0, 0, 0, 0]);
          renderFieldRaster({{
            canvas, overlay, analysis, values, valid, diagnostic,
            component: 'magnitude', low: 0, high: 1, palette: 'gray',
            statusColours: {{}}, showContour: true, mask,
            showVectors: true, vectorStep: 1, vectorGain: 1,
          }});
          const percentileAnalysis = {{}};
          const bounds = percentileBounds(percentileAnalysis, 'u',
            new Float32Array([1, 2, 3, 4, Number.NaN]), new Uint8Array([1, 0, 1, 1, 1]), 25, 75);
          console.log(JSON.stringify({{
            bounds,
            palette: fieldColour(0.25, 0, 1, 'gray'),
            dimensions: [canvas.width, canvas.height, overlay.width, overlay.height],
            pixels: Array.from(canvas.context.drawImageArgs[0].context.image.data),
            draw: canvas.context.drawImageArgs.slice(1),
            fillStyle: canvas.context.fillStyle,
            strokeCount: canvas.context.strokes.length,
          }}));
        """
        result = json.loads(
            subprocess.check_output(["node", "--input-type=module"], input=script, text=True)
        )

        self.assertEqual(result["bounds"], [2, 3.5])
        self.assertEqual(result["palette"], [79, 79, 79])
        self.assertEqual(result["dimensions"], [4, 2, 4, 2])
        self.assertEqual(result["pixels"], [79, 79, 79, 255, 134, 134, 134, 255])
        self.assertEqual(result["draw"], [0, 0, 4, 2])
        self.assertEqual(result["fillStyle"], "#263b47")
        self.assertEqual(result["strokeCount"], 3)


if __name__ == "__main__":
    unittest.main()
