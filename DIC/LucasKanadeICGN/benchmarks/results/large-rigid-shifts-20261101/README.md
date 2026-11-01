# Large rigid translations: local-pyramid controls and Fourier bootstrap

Code revision: `f514eb0` (clean during these four captures). Canonical WGSL is
unchanged. Source hashes, input hashes, exact options, versions, selected
adapter and native samples are in each JSON. Hardware: Intel Arc 130V,
Linux/Vulkan for native and Chromium WebGPU for the standalone workbench.
This does not qualify another GPU, OS, rotation or general nonrigid motion.

## Inputs and populations

The archived native 3840 × 2160 `assets/uhd_tensile/reference.png` supplies the
texture. The tool creates exact integer translations by copying overlapping
pixels and filling newly exposed content with zero. It never tiles, wraps,
regenerates or changes the fixture. Input generation/PNG encoding are outside
the native resident-image timing boundary.

Local-only controls use the fixed 23,976 reference centres more than 192 pixels
from every reference edge. Their shifts are at most 64 pixels, so this population
has genuine translated source pixels. Fourier cases use the stricter common
interior: both the reference centre and its translated target centre must be
192 pixels from image edges. The population varies with motion and is identical
between Fourier models/policies at that motion. Do not compare exact coverage
percentages across these different populations as a paired statistical test.
A global seed does not turn newly exposed pixels into measurements.

Predefined gates remain vector displacement RMSE <0.05 px, p95 <0.10 px and
accepted coverage >=90%. All returned parameters must be finite. Integer
registration errors here do not qualify subpixel metrology: the separate native
and browser UHD regressions refine `(128.35, -64.2)` with the same gates, for
translation and affine, raw outputs. The four-level subpixel regression uses
`(16.35, -8.2)` and the same gates.

## Observations

- The four-level local-only trial, with `(20,8,8,16)` iterations, qualifies
  `(16,-8)` for both native models/policies and the browser affine/raw case.
  It fails the larger `(32,-16)` and `(64,-32)` controls. More levels alone
  do not solve the owner's sparse-pair problem on this texture.
- Fourier plus the ordinary two-level `(8,16)` pyramid qualifies every tested
  `(64,-32)`, `(256,-128)`, `(512,-128)`, `(700,-200)` case. Translation/affine,
  raw/median native outputs each have 100% acceptance and accurate coverage on
  their common interior. The workbench affine/raw controls do too.
- The actual workbench file picker, reference preparation, Fourier checkbox,
  Solve and field export are used; `errors` is empty. Exports retain the per-pair
  seed, crop, scores and initialization time. They are not only module tests.
- Confidence checks are heuristic. In the last case the crop overlap is only
  25.46%; this is not a promise of capture beyond 700 pixels or on arbitrary data.

## Timing boundaries and limitations

Native: resident NumPy target to returned/materialized NumPy field and
statistics, one synchronous slot, preparation excluded, Fourier included.
Five warmups and thirty complete calls per case. Raw samples, median and p95
are retained. These are descriptive latency measurements, not a long-stream
4K60 qualification or an old-versus-new default-performance nonregression test.
The modes were measured sequentially, without tests/builds during timing.

Browser: four distinct single solves in the real PWA. Sample `ageMs` is Worker
arrival to observable GPU field plus small summary, not materialized Python
output and not exposure age. Full field export has its own readback time.
`globalTranslation.wallMs` includes external-window conversion/readback plus
JavaScript FFT and validation. These samples have no credible p95/FPS claim.
Native and browser timing boundaries are deliberately not merged.

This first initializer is CPU-based and opt-in. It resolves much larger motion
but adds substantial latency: see the table in `docs/reference/performance.rst`.
No giant-window LK implementation was benchmarked, so no speedup relative to
such an implementation is claimed. Normal zero/temporal paths do not run FFT or
read targets back to CPU.

## Reproduction

From the repository root with the native environment and Chromium installed:

```bash
python3 tools/build_pages.py --output /tmp/live-dic-fourier-site
python3 tools/qualify_large_translations.py --warmups 5 --repeats 30 \
  --output /tmp/local-controls.json
python3 tools/qualify_large_translations.py --fourier --warmups 5 --repeats 30 \
  --output /tmp/native-fourier.json
python3 tools/qualify_large_translations.py --browser-only \
  --site /tmp/live-dic-fourier-site --output /tmp/browser-controls.json
python3 tools/qualify_large_translations.py --fourier --browser-only \
  --site /tmp/live-dic-fourier-site --output /tmp/browser-fourier.json
ICGN_RUN_GPU_TESTS=1 ICGN_RUN_BROWSER_TESTS=1 python3 -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests -p 'test_fourier.py' -v
ICGN_RUN_GPU_TESTS=1 ICGN_RUN_BROWSER_TESTS=1 python3 -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests -p 'test_uhd_fixtures.py' -v
```

Keep native timed campaigns isolated from the functional tests. `--software`
selects SwiftShader for browser correctness only. The separate 512² browser
regression also executes byte/external inputs, CPU/GPU reference preparation,
GPU pairs, failure/recovery, exports and a two-slot ring on hardware and locally
on SwiftShader. A fresh installed wheel executed native Fourier and NPZ export
outside the checkout; it contains both new Python and JavaScript modules and
remains `py3-none-any`.

## Final local validation

The complete enabled native/browser suite passed **300 tests without skips**
(`native-browser-tests.log`). The new browser execution test also passed on
SwiftShader. Ruff, formatting, 19 Node syntax checks, strict Sphinx, repository
link checks and installed-wheel checks passed locally. No new remote CI run is
claimed. The source revision is the same for the functional and timed captures.
