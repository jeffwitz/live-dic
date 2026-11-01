# Opt-in browser GPU reference preparation

The original 1080p diagnosis identified per-pair CPU reference construction as
the consecutive-camera bottleneck. This campaign qualifies the new explicit
`referencePreparation: 'gpu'` / `registerPair` path. CPU/default, native Python,
fixed-reference IC-GN and its mathematical shaders are unchanged.

## Native UHD numerical and image-to-field controls

- Linux x86_64, Intel Lunar Lake integrated GPU, Chromium 155 (exact user agent in reports).
- Original 3840 × 2160 tensile PNGs; no resize, tiling or video codec. Input
  hashes/provenance are in each report's fixture metadata. Translation uses
  the original reference rolled by (+2,-1) pixels, evaluated on the interior.
- Uniform 16/16 subsets, B-spline, two fixed levels, [8,16] iterations,
  [0.005,0.005] tolerances, explicit median policy, full JS typed-field readback.
- Boundary: resident Uint8Array reference/target -> materialized JS field.
  Every CPU call includes `setReference` and `register`; GPU calls reuse cold
  geometry/resources and include preparation plus solving in `registerPair`.
  PNG decode, Python transport, camera capture, display and recording are excluded.
- Five warmups; alternating CPU/GPU order per repetition. Thirty affine pairs,
  twenty translation pairs; these short distributions are descriptive controls,
  not universal real-time/thermal qualification. No test/build ran during them.
- Preparation arithmetic is float32 on GPU versus float64 on CPU. Existing
  scientific limits were retained. Cold costs, all raw samples, source revision,
  dirty state, source-file hashes, browser, adapter and tracked memory are saved.

| UHD model | CPU median / p95 | GPU median / p95 | GPU sustained call rate |
|---|---:|---:|---:|
| Affine, original frame 40 | 1272.10 / 1375.43 ms | 75.50 / 82.57 ms | 13.06 fields/s |
| Translation (+2,-1) | 946.60 / 960.07 ms | 64.60 / 73.73 ms | 15.18 fields/s |

Affine has 31,648 accepted subsets and zero status mismatches between preparation
paths. On their 30,910 accepted interior subsets, displacement RMSE versus the
analytical tensile truth is 0.024202 px, p95 0.045534 px, gradient RMSE per
component 0.0010642; interior coverage is 99.981%. CPU/GPU displacement-field
difference RMSE is 0.0000559 px and gradient difference RMSE 0.00000317.

Translation has 31,654 accepted subsets, zero status mismatches and all 30,916
interior subsets accepted. Analytical displacement RMSE is 0.00004235 px; CPU/GPU
field difference RMSE is 0.00000415 px. Its inactive derivatives remain zero.

The improvement does **not** reach 4K60 for newly prepared consecutive pairs on
this GPU. It must not be compared to the much cheaper fixed-reference workload
as though both include Hessian reconstruction.

## Reproduce

`affine-uhd.json` records clean source `eae271b`; `translation-uhd.json` records
clean source `7a3ab9e`. Their numerical core/shader hashes are included. The
current tool additionally verifies original PNG hashes and records its command.
From a checkout with the Python/browser dependencies installed:

```bash
python tools/qualify_gpu_reference.py --output /tmp/affine-uhd.json \
  --motion-model affine --warmup 5 --repetitions 30
python tools/qualify_gpu_reference.py --output /tmp/translation-uhd.json \
  --motion-model translation --warmup 5 --repetitions 20
ICGN_RUN_BROWSER_TESTS=1 python -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests -p test_gpu_reference.py -v
```

`--software`/`ICGN_BROWSER_SOFTWARE=1` are numerical controls, never hardware FPS.
The original static fake-camera baseline is archived separately in
`../browser-camera-diagnosis-20261101/`. It is not a physical-camera result.

## Application camera controls

Three additional reports exercise the actual application with a **static simulated
60 Hz camera**, no physical device. Negotiated/output sizes are verified in the
reports. The 1080p source is an unresized crop of the original UHD reference;
the UHD source uses the entire original image. This is a scheduling/storage
control, not camera or moving-sequence metrology. Ten seconds per case after
three seconds warmup; clean site build `1c28373` with canonical shader hashes.
Tests/builds were stopped before these observations.

| Application / mode | CPU preparation completed FPS | GPU preparation completed FPS |
|---|---:|---:|
| 1080p fixed, recording off | 60.06 | 57.94 |
| 1080p fixed, recording on | 31.68 | 32.95 |
| 1080p consecutive, recording off | 2.50 | 26.39 |
| 1080p consecutive, recording on | 2.20 | 19.49 |
| UHD fixed, recording off | not measured | 18.28 |
| UHD fixed, recording on | not measured | 10.19 |
| UHD consecutive, recording off | not measured | 7.89 |
| UHD consecutive, recording on | not measured | 5.27 |

Completed FPS includes delivery, solving, display and, when enabled, recorded-field
readback plus main-thread durable-storage admission. Observable-worker ages
exclude later recording admission and are not exposure-to-display latency.
GPU pair-call medians are 25.6/25.2 ms in 1080p without/with recording; UHD medians
are 100.3/91.0 ms. CPU consecutive reference-preparation medians are 388.1/378.0 ms
at 1080p. GPU mode reports the combined preparation/solve time, not a fictitious
separate zero-duration preparation.

These application observations differ from the resident-array controls above:
external image conversion, rendering, browser scheduling and sustained competing
work are included. No claim attributes the entire difference to one stage.
The final 4K application observation is **not 60 FPS**. None of these results
renews a claim about physical cameras, other vendors or other operating systems.

Camera JSON files are losslessly gzip-compressed. `environment.json` records
their uncompressed SHA256 hashes; it also distinguishes the native adapter
inventory from the selected browser adapter in each report. Reproduce after
assembling a separate site:

```bash
python3 tools/build_pages.py --output /tmp/live-dic-reference-site
python tools/diagnose_camera_pipeline.py --site /tmp/live-dic-reference-site \
  --output /tmp/camera-cpu-1080.json --duration 10 --warmup 3
python tools/diagnose_camera_pipeline.py --site /tmp/live-dic-reference-site \
  --output /tmp/camera-gpu-1080.json --gpu-reference --duration 10 --warmup 3
python tools/diagnose_camera_pipeline.py --site /tmp/live-dic-reference-site \
  --output /tmp/camera-gpu-uhd.json --gpu-reference --resolution 3840x2160 \
  --duration 10 --warmup 3
```

## Validation

- Complete local native-GPU/browser-enabled suite: 270 tests, no skips, passed.
- Seven new tests: 42 CPU/GPU preparation combinations covering all supported
  subset sizes, plus flat rejection, analytic fractional translation/affine,
  ROI, input ownership, all readback modes, history reset, resource reuse and
  exactly one hot submission. All seven pass on Chromium SwiftShader as well.
- Pure-Python wheel installed in fresh Python 3.12 outside the checkout: all
  seven new browser tests pass against the installed modules/shaders; native ROI
  registration and strict no-pickle NPZ export also pass. Seven WGSL sources are
  shipped. `wheel-check.txt` identifies the tested artifact/hash.
- Native wgpu-py/Naga also compiles the new shader and both model pipelines.
  This is compilation evidence only: the native Python pair API does not use
  this new preparation graph yet.
- Ruff, Node syntax, strict Sphinx and repository/wheel resource checks pass.
  This is local evidence; no new remote CI run or Pages deployment is claimed.
