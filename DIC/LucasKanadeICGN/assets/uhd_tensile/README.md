# Original UHD virtual tensile-test fixtures

These are four **native 3840 × 2160 grayscale images**, copied byte for byte
from `DIC_FROM_H264/experiments/tensile_extensometer_4k/inputs`.
They are not tiled or resized versions of the 512 × 512 fixtures.

| Local image | Original image | Frame | Axial stretch |
|---|---|---:|---:|
| `reference.png` | `anchor_preview.png` | 0 | 1.0 |
| `frame_040.png` | `frame_040_preview.png` | 40 | 1.0011196739864576 |
| `frame_100.png` | `frame_100_preview.png` | 100 | 1.0384330045576058 |
| `frame_199.png` | `frame_199_preview.png` | 199 | 1.1 |

The approximately 20 MiB PNG set is included directly in the repository.
Tests do not download data or read the original checkout. `manifest.json`
records SHA256 hashes, source revision at import, selected frames, coordinate
conventions, rendering, and numerical bounds. `machine_truth.csv` retains all
200 states. `provenance/source_manifest.json` and `provenance/generate.py`
are unchanged copies of the original experiment, not executed by tests.
The local `.gitattributes` disables line-ending conversion for the archived
inputs, including the original CRLF CSV, so clones preserve their byte hashes.

## Analytic ground truth

Pixel coordinates are zero based, with centres at integer coordinates. The
generator stretches about **(1920, 1080)**, not `(1919.5, 1079.5)` or the
different centre used by later experiments in the source repository.
For the frame's stretches `lx`, `ly`, the forward map is

```text
x_target = 1920 + lx * (x_reference - 1920)
y_target = 1080 + ly * (y_reference - 1080)
u = (lx - 1) * (x_reference - 1920)
v = (ly - 1) * (y_reference - 1080)
du/dx = lx - 1; dv/dy = ly - 1; du/dy = dv/dx = 0
```

These PNGs were written **before video encoding** by the original generator.
Targets were rendered from the reference with OpenCV `INTER_LINEAR` inverse
remapping and reflected borders. That interpolation and uint8 quantization
introduce discrete image-formation error: these are not exact continuous
texture samples. No independent sensor noise was added.
Reading PNGs needs only Pillow and NumPy. Tests do not need OpenCV, SciPy,
FFmpeg, or a video decoder. Those historical generator dependencies are not
installation requirements; regeneration is unnecessary.

## Regression scope

Portable tests check all image dimensions, grayscale mode, file and provenance
hashes, and the archived strain/stretch relations. The opt-in GPU tests:

- register all four images against themselves at full UHD, checking shape,
  finite values, interior coverage, and zero displacement;
- register frame 40 against the initial reference with affine 16/16,
  B-spline and two levels, using both raw and median outputs;
- require displacement RMSE < 0.05 px and p95 < 0.10 px, with at least 90%
  of interior subsets accepted;
- additionally require median-output gradient RMSE per component < 0.002.

The raw 16/16 gradient RMSE measured about 0.0030 on the first hardware check;
the raw derivative estimate is **not** qualified to the median-output bound.
This is explicit smoothing qualification, not a claim of improved spatial
resolution or general robustness. Interior scores exclude a 32-pixel border;
coverage is checked in addition to errors on accepted subsets.

Frames 100 and 199 are preserved as later-stage inputs and identity
regressions. Their direct initial-reference motions exceed the ordinary
two-level capture range over much of the image. The tests do not claim
complete zero-seed recovery of the 3.8% or 10% strain fields.

```bash
ICGN_RUN_GPU_TESTS=1 PYTHONPATH=DIC/LucasKanadeICGN/python/src \
  python3 -m unittest discover -s DIC/LucasKanadeICGN/python/tests \
  -p test_uhd_fixtures.py -v
```

Without the environment flag, integrity/truth checks run and numerical checks
are skipped explicitly. The Linux lavapipe CI job runs both. These are
correctness tests, with no machine-specific timing limit.
