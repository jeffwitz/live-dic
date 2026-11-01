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
