# Temporal initialization at native UHD

Every target runs the entire selected pyramid. The previous accepted fine
field is projected onto the next target's coarsest grid; no conditional retry
or reference update is performed. Initialization remains opt-in.

The 16.62 ms tensile result is not a replacement for earlier small-motion
timings. The subsequent [same-workload diagnostic](performance-diagnostic.md)
remeasures the pre-temporal code, current zero mode and current temporal mode
on the same native-UHD frame: all remain around 12.4–13.1 ms. It also records
long-sequence variability and observed GPU power limits.

The complete [performance-profile repeat](performance-retest/README.md)
reapplies and logs the system profile, which was already reported as performance
before the command. The same three-campaign affine/two-level/one-slot test
measures 16.90 ms median / 20.36 ms p95 and 59.23 FPS, with unchanged numerical
results. No speedup or balanced-versus-performance causal claim is established.

## Reproduce

From the repository root, with the package installed:

```bash
python DIC/LucasKanadeICGN/benchmarks/benchmark_temporal_init.py --campaigns 3
```

The computational revision is `150ca9d` (host/kernel introduced in `0e9d088`),
starting from `67e264f`. `report.json` records dirty=true because documentation
and the CI workflow were edited during the run; the computation sources were
unchanged throughout. No tests, rendering, wheel builds or other qualification
campaigns ran concurrently with hardware timing. CPU scoring is outside each
timed region. No baseline performance non-regression claim is made.

Machine: Intel Graphics LNL / integrated Lunar Lake, Vulkan, Mesa
25.2.8-0ubuntu0.24.04.4, Python 3.11, NumPy 2.4.6, wgpu-py 0.32.0,
Pillow 13.0.0. Full adapter information, command, input hashes, all age and
completion-period samples, counts and iteration histograms are in the JSON.

## Image formation and populations

The 200 native 3840 × 2160 targets use the archived tensile reference and
all 200 stretches from `assets/uhd_tensile/machine_truth.csv`. No fixture is
tiled, resized or downloaded. An inverse float64 NumPy bilinear warper reflects
coordinates at the image boundary and rounds to the nearest byte. Byte
rounding error is at most 0.5 gray level relative to this interpolant; this
does not bound displacement error. The historical OpenCV generator quantized
fractions to 1/32, so these targets are not claimed identical to archived PNGs.
The physical stretch centre is (1920,1080); subset centres are at 7.5+16k.

All runs use B-spline interpolation, uniform weighting, size/stride 16/16,
explicit `outlier_policy="median"`, default tolerances and iteration ceilings:
one level `(30,)`, two levels `(8,16)`. Raw states seed history, so median
smoothing is not accumulated. The fixed common support contains **26,924**
positions: centres x in [224,3616), y in [64,2096). Its denominator stays fixed
for every method and frame. Missing estimates remain missing: coverage is
reported alongside errors conditional on accepted points. An additional own
population includes the full 32,400-position grid, with its own coverage and
error. This prevents a small surviving population from appearing successful.
`metrics.csv` reports every tensile frame for all three configurations.

Final frame, stretch x=1.1, y=0.9539017454:

| Initialization / levels | Fixed coverage | Displacement RMSE | Strain component RMSE |
|---|---:|---:|---:|
| zero / 2 | 1.645% | 63.369 px | 0.211505 |
| temporal / 1 | 99.974% | 0.028661 px | 0.001447 |
| temporal / 2 | 99.967% | 0.028376 px | 0.001431 |

Strain uses exx=du/dx, eyy=dv/dy, exy=(du/dy+dv/dx)/2. These are explicit
median-smoothed outputs; their error does not qualify raw gradients or
resolution of a localized strain feature. The zero mode demonstrably fails
the large-motion case, including accepted wrong-basin estimates.

## Image-to-field timing

Boundary: **resident uint8 NumPy target → materialized NumPy field and result
statistics**. Acquisition, rendering, truth scoring and serialization are
excluded. Actual stream slots are warmed with 20 images, history is reset,
frames 0–19 establish ordered history, and frames 20–199 are timed. Three
campaigns alternate configuration order. Numbers below are medians of the
three campaign statistics, not pooled quantiles. Repeated wall-time variation
is substantial; all samples are retained. Preparation is reported separately.

| Motion | Initialization / levels | Slots | FPS | Age median | Age p95 | Age p99 |
|---|---|---:|---:|---:|---:|---:|
| affine tensile | zero / 2 | 1 | 16.55 | 69.36 ms | 81.43 ms | 91.85 ms |
| affine tensile | zero / 2 | 2 | 16.75 | 136.97 ms | 167.74 ms | 180.43 ms |
| affine tensile | temporal / 1 | 1 | 66.99 | 15.13 ms | 19.70 ms | 22.12 ms |
| affine tensile | temporal / 1 | 2 | 63.66 | 32.13 ms | 37.21 ms | 41.30 ms |
| affine tensile | temporal / 2 | 1 | 59.92 | 16.62 ms | 20.15 ms | 21.65 ms |
| affine tensile | temporal / 2 | 2 | 57.64 | 35.30 ms | 41.94 ms | 50.22 ms |
| pure translation | zero / 2 | 1 | 29.34 | 37.30 ms | 42.48 ms | 44.65 ms |
| pure translation | zero / 2 | 2 | 30.04 | 73.51 ms | 79.59 ms | 80.09 ms |
| pure translation | temporal / 1 | 1 | 86.67 | 11.37 ms | 14.14 ms | 14.60 ms |
| pure translation | temporal / 1 | 2 | 92.98 | 22.05 ms | 26.23 ms | 27.37 ms |
| pure translation | temporal / 2 | 1 | 86.76 | 11.37 ms | 13.77 ms | 14.13 ms |
| pure translation | temporal / 2 | 2 | 81.58 | 24.11 ms | 29.86 ms | 30.49 ms |

The zero-mode times describe mostly failed large-motion solves and must not
be compared to earlier small-motion fixture timings as a performance
regression. Translation uses 200 pure shifts up to (+39.8,-19.9) pixels,
not the tensile deformation. Per-frame quality accompanies every timing run.
For temporal/two levels the last translation frame has 100% fixed coverage
and 0.00921 px displacement RMSE. Its own-population RMSE is 0.27209 px:
edge estimates are less reliable and must not be hidden by interior scoring.

**Latency conclusion:** affine does not reach p95 <16.67 ms. Translation at
one level and one slot does in all three campaigns (11.01–14.81 ms).
Translation/two levels/one slot does not consistently do so (12.48–17.12 ms),
despite its median campaign p95 being below the threshold. No two-slot
configuration meets that age criterion. More slots do not guarantee more FPS.

Iterations are over accepted estimates only; failures and their population
are retained separately. Affine temporal/one level averages 4.39 iterations,
with p95=7, p99=9 but maximum=30. Two levels average 2.65 fine iterations,
with maximum=16. Translation averages 3.01 and 1.73, with maxima 30 and 16.
Ceilings were not reduced: the accepted tails reach the existing limits,
and early exit already saves work for the majority. Cutting the tails merely
to improve timing would alter the qualified measurement population.

## Corruption and correctness

A black image replaces tensile frame 100. It accepts zero subsets and updates
zero temporal states. At frame 101 the normal two-level solve recovers: fixed
coverage is within one percentage point of control and displacement RMSE is
below 0.05 px. Thus recovery takes **one subsequent frame**, with no retry.
This does not qualify arbitrary noise, occlusions or accepted wrong solutions.

The test suite checks first-frame/default equality, shared synchronous/stream
history at depths 1/2/3, reset, expiry, full/partial ROI exclusions, and a
101-frame cropped tensile sequence. Native UHD remains the qualification
above. A separate pre-edit comparison matched all 30 arrays (parameters,
statuses, iterations) exactly across three models, three level counts and
the archived native UHD frame 40. See `default-parity.json` for fingerprints.

Local final validation: 171/171 opt-in tests on Intel, 118 portable passes
plus 53 explicitly disabled GPU tests, 7/7 temporal tests on lavapipe, Ruff
checks and warning-free Sphinx. Raw browser WGSL still passes native UHD
frame 40 (99.981% interior coverage, 0.02420 px RMSE); browser temporal mode
is not implemented. The platform-independent wheel was installed into a
fresh Python 3.12 environment and tested from `/tmp`, using native UHD
full/partial ROI streams at depth 3 and NPZ exports without pickle. See
`validation.json`. The CI workflow now includes temporal lavapipe regressions,
but no remote CI execution is claimed or triggered here.

The implementation does not project a new fine-only seed when no admissible
coarse ROI route exists, does not predict velocity, and does not automatically
retry. Browser temporal initialization is not implemented. Temporal mode is
not made the default by this qualification.
