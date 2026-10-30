# Native-UHD fractional-motion and crack qualification

**All images in this qualification are native 3840 x 2160.** Small-image
geometry and double-precision oracle checks remain unit tests; they are not a
substitute for these accuracy and image-to-field measurements. No input is
tiled or resized. The shipped Python/WGSL is unchanged from `0e1cbbf`.

## Data and controlled comparison

Two distinct image-formation models use the same known fractional displacement
`(0.35, -0.2)` pixels:

* `archived-speckle`: the original UHD `assets/uhd_tensile/reference.png`.
  A target is rendered in NumPy with inverse bilinear sampling, reflected
  borders and uint8 rounding. This introduces discrete image-formation error,
  as does the archived tensile generator. No sensor noise is added.
* `analytic-wave`: the existing continuous analytic texture evaluated directly
  at all UHD pixel coordinates and translated coordinates, followed by uint8
  rounding. This separates continuous sampling from discrete source remapping.

The ellipse/slit ROI is unchanged from the previous report. Size and stride
are equal. At 8/8 it selects 70,497 windows, including 457 partial supports;
at 16/16 it selects 17,660, including 334 partial supports. The different grid
densities are part of these configurations; the tables do not compare equal
numbers of measurements. Complete and partial populations are evaluated
separately in the JSON.

Within each pair of sizes the interpolator, model, integration weights,
iteration schedule and postprocessing are identical. Cubic rows use the native
single-level raw solver, 30 iterations and tolerance 0.005. B-spline rows use
the full two-level `(8,16)` schedule with its normal tolerances; raw and explicit
median results are separate rows. Do not attribute differences between those
families solely to the subset size.

## Original UHD speckle: fractional translation

Intel Lunar Lake integrated GPU, Vulkan, Mesa 25.2.8, wgpu-py 0.32.0.
Twelve actual-stream warmups precede each of three campaigns of 160 changing
images, alternating identity and fractional translation. Two images are in
flight. The measured boundary is resident NumPy image to complete NumPy field;
acquisition, decoding, preparation and post-timing quality evaluation are excluded.

| Model and subset/stride | Interpolation / path / output | FPS | Result age median ms | p95 ms | Partial accepted | Partial displacement RMSE px | Partial error p95 px |
|---|---|---:|---:|---:|---:|---:|---:|
| Translation 8/8 | Cubic / single level / raw | 138.66 | 14.05 | 15.99 | 432 / 457 | 0.12265 | 0.27739 |
| Translation 16/16 | Cubic / single level / raw | 262.28 | 7.55 | 8.83 | 334 / 334 | 0.03252 | 0.04736 |
| Translation 8/8 | B-spline / two levels / raw | 84.11 | 23.73 | 25.39 | 454 / 457 | 0.05334 | 0.10811 |
| Translation 16/16 | B-spline / two levels / raw | 93.77 | 20.72 | 25.15 | 334 / 334 | 0.01826 | 0.03124 |
| Translation 8/8 | B-spline / two levels / median | 72.50 | 28.10 | 31.18 | 454 / 457 | 0.02131 | 0.03685 |
| Translation 16/16 | B-spline / two levels / median | 85.61 | 22.59 | 27.70 | 334 / 334 | 0.01037 | 0.01551 |
| Affine 16/16 | B-spline / two levels / median | 91.27 | 21.06 | 27.20 | 332 / 334 | 0.03450 | 0.06364 |

Times are medians of the campaign summaries. Error/acceptance values are the
worst over the changing inputs; the raw JSON retains each campaign, all latency
samples, completion intervals, iterations and both support populations.
This affine row measures an affine model on a pure translation; the archived
frame-40 tensile deformation is qualified separately in [the main report](README.md).

The 8/8 B-spline/median path meets the tested displacement RMSE <0.05 px,
error p95 <0.10 px and coverage >=90% criteria on both populations. Its raw
boundary estimates do not meet those error bounds on this scene. Median is an
explicit smoothing operation; these results do not prove preserved spatial
resolution or general outlier robustness. The single-level cubic 8/8 path also
misses these error limits. Its behaviour does not disqualify the separate
B-spline pyramidal/median path.

All listed stream rates exceed 60 FPS on these measured configurations and
this ROI. The age of a result is a different criterion: the B-spline rows exceed
16.67 ms at depth 2. Variable clocks/system load remain uncontrolled, so these
campaigns do not establish a non-regression result or universal deadline.

## Continuous analytic UHD texture

The same protocols give:

| Translation subset/stride | Path / output | FPS | Result age median ms | Partial RMSE px | Partial error p95 px |
|---|---|---:|---:|---:|---:|
| 8/8 | Cubic single level / raw | 85.66 | 23.11 | 0.10950 | 0.21048 |
| 16/16 | Cubic single level / raw | 212.23 | 9.32 | 0.01330 | 0.02525 |
| 8/8 | B-spline two levels / raw | 91.49 | 21.29 | 0.02321 | 0.04092 |
| 16/16 | B-spline two levels / raw | 121.95 | 16.45 | 0.00732 | 0.01381 |
| 8/8 | B-spline two levels / median | 84.59 | 23.48 | 0.00836 | 0.01482 |
| 16/16 | B-spline two levels / median | 115.99 | 17.18 | 0.00314 | 0.00542 |

Both B-spline/median sizes again meet the precision/coverage bounds on the
complete and partial populations. The numerical results depend on texture and
image formation, so a small-image phase-specific RMSE must not be extrapolated
to this complete UHD field.

## Crack-tip case computed as a full UHD field

The continuous texture has opposite translations on the two sides of a slit.
The slit ends inside a central 17/17 subset; all field positions are dispatched
at UHD. This is a controlled displacement-jump case, not a physical fracture
model. The table concerns the specific subset containing the tip, centred at
`(1929,1079)`; it is not a global crack-field accuracy score.

Uniform integration, bilinear interpolation, single-level raw solve,
30 iterations, tolerance 0.005, same reference/target and requested model:

| Model | Tip error without truncation px | With visibility px | Retained local support |
|---|---:|---:|---|
| Translation | 0.44920 | 0.01126 | 275 -> 189 pixels |
| Affine | 0.32621 | 0.01588 | 275 -> 189 pixels |

Both truncated results are accepted and the complete output fields are finite.
The `visibility` heuristic limits mixing around this tip; it remains distinct
from Ncorr's circular-contour truncation. See `uhd-crack.json`. No timing result
is inferred from this correctness run.

## Regression and reproduction

`test_uhd_fixtures.py` now executes the fractional 8/8 and 16/16 B-spline/median
cases on both UHD textures, checking complete and partial populations
independently, with the existing error/coverage limits. It also runs the UHD
crack-tip case for both models and checks other supported sizes, overlapping
strides and Gaussian support preparation at UHD. There are no CI timing limits.
The lavapipe job already executes this file; hardware timing is measured separately.

From the repository root:

```bash
export PYTHONPATH=DIC/LucasKanadeICGN/python/src
python DIC/LucasKanadeICGN/benchmarks/qualify_uhd_partial_roi.py \
  --scene archived-speckle --output /tmp/uhd-speckle.json
python DIC/LucasKanadeICGN/benchmarks/qualify_uhd_partial_roi.py \
  --scene analytic-wave --output /tmp/uhd-analytic.json
python DIC/LucasKanadeICGN/benchmarks/results/roi-partial-20261030/run_uhd_crack.py
ICGN_RUN_GPU_TESTS=1 python -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests -p test_uhd_fixtures.py -v
```

`uhd-fractional-speckle.json`, `uhd-fractional-analytic.json` and `uhd-crack.json`
contain input/source/helper hashes, exact commands, Python/wgpu/adapter versions
and configurations. `uhd-validation.json` records the final local checks.
Archived input hashes are verified by the existing fixture-integrity tests.
