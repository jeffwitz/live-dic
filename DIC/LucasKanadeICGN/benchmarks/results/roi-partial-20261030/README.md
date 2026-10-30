# Partial-support ROI qualification

The full-only baseline is `31e7e25`; native implementation `0e1f471` adds
explicit `roi_policy="partial"` and `roi_truncation="none"|"visibility"`.
`full` remains the default. Source SHA256 maps identify the actual Python/WGSL
in every report, including reports marked dirty while documentation changed.
The implementation does not add a dependency or a native project binary.

## Fixed geometry and numerical meaning

Complete windows retain their ordinary reference preparation and fast solver.
Partial windows use the fixed 4-connected component containing the central
anchor; even windows require all four central pixels. Their actual support
multiplies the selected spatial weights in every mean, norm, Hessian and
correlation sum. Preparation in double precision checks contrast, normalized
Hessian rank/condition and computes the inverse once. Local bit masks and a
compact work list share existing read-only storage between stream slots.

Visibility truncation is a separate line-of-sight heuristic. It checks every
closed pixel cell touched by a centre-to-pixel segment and removes hidden pixels.
It addresses wrap-around near tips but is **not an exact port of Ncorr's
contour-based circular-support truncation**. The mask is in reference
coordinates and does not discover new cracks. See the
[advanced guide](../../../docs/how-to/use-partial-roi.rst), including the
Ncorr references, diagram, support thresholds and archive layout.

## Inputs, hardware and measured boundary

Intel Core Ultra 5 228V, Intel Lunar Lake integrated GPU, Vulkan, Mesa
25.2.8-0ubuntu0.24.04.4, wgpu-py 0.32.0. Every native timing explicitly requests
`device_type="gpu"`. A software adapter only validates correction tests.

Both models use the original 3840 x 2160 texture from `assets/uhd_tensile`:

* Translation: identity and pure `(2,-1)` / `(4,-2)` integer shifts of the
  original reference, made in memory with `np.roll`. The ROI excludes the roll
  seam. This is an execution/large-texture test; its small errors do not prove
  subpixel interpolation accuracy. Fractional translation is tested separately.
* Affine: original reference and archived tensile frame 40, with its analytic
  deformation from `machine_truth.csv`. No image is resized, downloaded or
  overwritten. Frames 100/199 are not required to converge from zero seeds.

The deterministic ROI is an ellipse centred at `(1920,1080)`, radii `(1600,900)`,
with a two-pixel slit at `mask[400:1205,1930:1932] = False`. Its canonical mask
SHA256 is `62c6521974ed5cc6ca1688b85a90ce72a97cbf0e6ba6f2dacc026f18bd825e1d`.
It retains 17,326 complete windows; partial mode adds 334 boundary windows.

Settings are 16/16 subsets, B-spline interpolation, uniform weighting, pure
median, two pyramid levels, normal `(8,16)` iteration schedule and existing
tolerances. Each report contains three campaigns, each warming the actual
`register_stream` for 12 images at depth 2, then measuring 160 changing inputs.
The boundary is the resident NumPy image to the materialized NumPy field.
Acquisition, decoding and setup are excluded. `preparation_ms` measures
`set_reference`, including its partial preparation/pipelines; engine creation
and initial common pipeline compilation precede that counter.

## Measured streams

Times are medians across the three campaign summaries, including median p95/p99.
Preparation is one setup measurement per report, not a per-frame cost.

| Model | Policy | Selected | Minimum accepted | FPS | Result age median ms | p95 ms | p99 ms | Preparation ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Translation | full | 17,326 | 17,326 | 107.82 | 18.46 | 20.52 | 21.40 | 1,596 |
| Translation | partial | 17,660 | 17,660 | 108.80 | 18.62 | 20.22 | 21.24 | 1,906 |
| Translation | partial + visibility | 17,660 | 17,660 | 113.75 | 17.30 | 21.35 | 22.14 | 1,876 |
| Affine | full | 17,326 | 17,325 | 103.89 | 18.97 | 22.74 | 26.41 | 2,011 |
| Affine | partial | 17,660 | 17,659 | 105.87 | 18.96 | 21.74 | 22.59 | 2,286 |
| Affine | partial + visibility | 17,660 | 17,659 | 107.02 | 18.93 | 20.29 | 21.80 | 2,328 |

These are observations, not evidence that partial solving or truncation makes
the solver faster. The environment varies, populations and seeds change, and
partial processing adds a solve pass. An image period below 16.67 ms at depth 2
does not imply a result age below 16.67 ms.

### Separate boundary metrology

The maximum over the changing inputs is reported on each accepted population;
excluded windows never contribute. All 334 partial windows are accepted for
both models in these cases. Complete affine coverage is 17,325 / 17,326.

| Case | Population | Displacement RMSE px | Displacement p95 px | Gradient component RMSE |
|---|---|---:|---:|---:|
| Translation, partial | complete | 0.0000423 | 0.0000792 | not applicable |
| Translation, partial | partial | 0.0000595 | 0.0001102 | not applicable |
| Affine, partial | complete | 0.02323 | 0.04305 | 0.001029 |
| Affine, partial | partial | 0.03261 | 0.06013 | 0.001618 |
| Affine, visibility | partial | 0.03268 | 0.06002 | 0.001616 |

The archived tensile test checks complete and partial populations independently:
coverage >=90%, displacement RMSE <0.05 px and p95 <0.10 px. It retains the
existing median-gradient bound for complete windows. Partial gradient error is
reported separately rather than inheriting that bound without qualification.
Raw outputs do not inherit the median-gradient criterion.

## Temporal non-regression remains open

The runner writes each `*-predeclared-margins.json` from baseline/baseline
campaign spread **before** executing the candidate. It then rechecks the old
revision after the unmasked candidate. Controls are retained:

| Model | Before, no ROI FPS | Candidate, no ROI FPS | Before recheck FPS | Candidate, all-true FPS |
|---|---:|---:|---:|---:|
| Translation | 107.35 | 106.83 | 101.97 | 104.95 |
| Affine | 88.44 | 83.49 | 91.72 | 90.80 |

Translation's absent-mask comparison fits its predeclared spread, but control
drift and tail uncertainty prevent a general non-regression claim. Affine's
p99 difference exceeds its initially very narrow p99 spread. These data do not
close temporal non-regression reception. In particular the higher all-true
affine rate is not an attributed ROI optimization: it uses the same unmasked
pipelines, and numerical outputs/iteration histograms agree. The physical cause
of timing drift was not established. No tolerance, iteration budget or level
count was lowered to improve a number.

## Correctness, installation and reproducibility

Tests cover exact mask/bit geometry, odd/even centres, all supported sizes,
overlap, uniform/Gaussian integration, rank deficiency, retry, reconfiguration,
all-true/empty masks, mixed-image stream slots, installed exports and original
UHD data. A separate matrix-based double-precision CPU IC-GN oracle compares
partial translation and actual affine iterates with bilinear interpolation.
GPU tests exercise cubic/B-spline paths and an admissible subgroup path as well.
The additional 8/8 cubic translation case is checked against an independent
double-precision Keys-kernel oracle using the same fixed 30-iteration budget
in both solvers. This avoids treating a different last iteration at the stop
threshold as an error in the equations. On this small smooth analytic texture its
accepted partial-window displacement RMSE is **0.1471 px**, versus **0.0738 px**
for the unmasked 8/8 solve. It does not meet the 0.05 px bound qualified for
16/16 UHD subsets. Supports passing rank/contrast checks do not guarantee that
metrological accuracy. This limitation is retained, not removed by selecting
only favourable windows or weakening the established 16/16 criteria.
The visibility ray oracle uses crossing events rather than the production slab
algorithm; a displacement-jump case checks reduced cross-side mixing. See
`validation.json` for the actual final test counts and installation evidence.

The wheel is `py3-none-any`. A fresh Python 3.12 environment installs it with
declared dependencies and runs both native engines outside the checkout, with
partial + visibility, both models, and NPZ reading with `allow_pickle=False`.
No compiler or new dependency is needed. The shared raw-WGSL browser also runs
after the binding-8 storage declaration change; browser ROI remains explicitly
unsupported. Windows/macOS GPU execution was not measured here; local tests do
not assert a remote CI result.

Reproduce from the repository root with the project interpreter:

```bash
git worktree add --detach /tmp/live-dic-full-baseline 31e7e25
export PYTHONPATH="$PWD/DIC/LucasKanadeICGN/python/src"
python DIC/LucasKanadeICGN/benchmarks/results/roi-partial-20261030/run_campaigns.py \
  /tmp/partial-masks /tmp/live-dic-full-baseline/DIC/LucasKanadeICGN/python/src
ICGN_RUN_GPU_TESTS=1 python -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests -v
```

The runner recreates masks and reports. Add `--model translation` or `--model
affine` to repeat one family. For a direct invocation use
`benchmark_image_to_field.py --fixture-set uhd-translation` or `uhd-tensile`,
with `--roi-mask`, `--roi-policy partial` and optional
`--roi-truncation visibility`. Every JSON includes the exact command, versions,
adapter, source/image hashes, mask configuration, iteration histogram, latency
samples, completion intervals and per-population accuracy. Quality evaluation
runs after the timing window. Geometry/support preparation and exports do not
run in the live loop.
