# Native-UHD plate: ROI masks and equivalent strain

Full 3840 x 2160 registration on the original analytical plate from SYSCL,
with affine 16/16, uniform integration, B-spline and two pyramid levels using
the normal `(8,16)` iteration schedule. Raw and pure-component-median outputs
are separate runs. No solver/default change is made for this study.

The exact image mask is outside the circle `(1920,1080), radius 112`. Four
variants are compared: no ROI, full squares, partial connected supports, and
partial supports with the explicit visibility heuristic. All images are
included in [the offline fixture](../../../assets/uhd_plate_hole/README.md).

## Matplotlib figures

All fields are computed at native UHD. Only the **view** is zoomed to +/-320 px
around the hole. Columns show analytical truth and the four mask variants;
rows show raw affine gradients and the explicit median output.

| Load | Equivalent-strain comparison | Signed errors | Boundary precision / coverage |
|---|---|---|---|
| Frame 95: primary load | [PNG](von_mises_095.png), [PDF](von_mises_095.pdf) | [PNG](von_mises_errors_095.png), [PDF](von_mises_errors_095.pdf) | [PNG](boundary_bands_095.png) |
| Frame 225: two-mode load | [PNG](von_mises_225.png), [PDF](von_mises_225.pdf) | [PNG](von_mises_errors_225.png), [PDF](von_mises_errors_225.pdf) | [PNG](boundary_bands_225.png) |

The same value scale is used in every field panel and both frames. Signed-error
panels also share fixed limits. Colour saturation is reported on each panel;
unclipped fields and errors are retained in NPZ/CSV. Grey is unavailable, white
is the hole. Cells show point samples, without smooth display interpolation.
Geometry is plotted in reference coordinates, with image y increasing downward.

These are **equivalent strains**, with `ezz=0`, not Von Mises stresses. Strain
comes from direct affine derivatives, `exy=(du/dy+dv/dx)/2`. The original SYSCL
5x5 displacement-regression operator is a different measurement operator.

## Result: coverage improves; new boundary strains are not qualified

The physical 0--16 px band outside the hole contains 42 grid centres. Full
squares retain 18; partial retains all 42, adding 24 points. On the tested
frame-95 and frame-225 images, all 24 partial solves are accepted.

That acceptance does **not** establish their strain precision. Raw gradients
at these added points are particularly inaccurate. Their VM RMSE is retained:

| Output / mask | Frame 95 VM RMSE on added points, microstrain | Frame 225 VM RMSE on added points, microstrain |
|---|---:|---:|
| Raw, partial | 13,160 | 14,231 |
| Raw, partial + visibility | 12,574 | 13,560 |
| Median, partial | 497 | 1,402 |
| Median, partial + visibility | 338 | 1,340 |
| Full | No measurement | No measurement |

Median partial displacement RMSE on those 24 points is 0.0135 / 0.0132 px;
visibility gives 0.0133 / 0.0123 px. Accurate displacement is therefore
insufficient to certify these local gradients. The mechanisms causing the
boundary gradient error have not been isolated in this comparison; the nominal
mask does not isolate reference/target preprocessing from the dark hole.
Visibility is not a general cure for this free-boundary metrology problem.

The **same 835 accepted near-hole centres** are also scored for every method,
including raw and median. On that fixed support the median VM RMSE is:

| Mask | Frame 95, microstrain | Frame 225, microstrain |
|---|---:|---:|
| None | 169.05 | 136.99 |
| Full | 159.79 | 139.96 |
| Partial | 159.04 | 140.30 |
| Partial + visibility | 158.81 | 139.92 |

The mask variants give similar common-support errors. Median reduces noise
relative to the raw 539--540 / 446 microstrain levels, while explicitly changing
spatial resolution. These results are not evidence of better outlier detection.
Comparing full's smaller error on its own retained boundary points with partial's
larger point population would mix support selection with accuracy. The radial
figures therefore show error and coverage together. `metrics.csv` keeps both
fixed common support and all method-specific populations, without deleting
difficult points from the comparison.

## Timing boundary and observations

Intel Lunar Lake integrated GPU, Vulkan, Mesa 25.2.8, wgpu-py 0.32.0.
Each variant runs three campaigns of 160 changing inputs, cycling the original
reference, frame 95 and frame 225. Each campaign warms the actual stream with
12 images, at depth 2. The measurement starts with resident NumPy pixels and
ends with the full NumPy parameter field. Mechanics/VM scoring, figure rendering,
acquisition, decoding and setup are excluded; this is not acquisition-to-VM time.

| Median output | FPS | Result age median ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|
| No ROI | 83.48 | 23.60 | 25.90 | 26.66 |
| Full | 85.97 | 22.95 | 24.39 | 24.70 |
| Partial | 83.62 | 22.65 | 28.95 | 29.45 |
| Partial + visibility | 84.98 | 23.24 | 25.18 | 26.32 |

These are measured observations, not a controlled causal speedup study or a
universal deadline guarantee. The depth-2 throughput must not be confused with
a result age below 16.67 ms. Raw-output timing and all latency/completion samples
are retained in `report.json`. No figure rendering or correctness job ran during
these timing campaigns.

## Truth, provenance and reproduction

The NumPy truth matches the original C++ at the archived boundary/interior
samples to <6e-16 px / <5e-19 strain. A derivative check independently links
complex displacement and stress-derived strain. Truth is evaluated at live-dic's
actual `7.5+16*i` centres, while retaining the original generator's section
calibration. No load or material constant enters the image registration.

```bash
export PYTHONPATH=DIC/LucasKanadeICGN/python/src
python DIC/LucasKanadeICGN/benchmarks/benchmark_plate_hole_roi.py
python -m pip install matplotlib
python DIC/LucasKanadeICGN/benchmarks/plot_plate_hole_roi.py \
  DIC/LucasKanadeICGN/benchmarks/results/plate-hole-roi-20261030
ICGN_RUN_GPU_TESTS=1 python -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests -p test_plate_hole.py -v
```

Matplotlib is an optional figure-generation dependency, outside the native
runtime. The shipped package and its dependencies are unchanged. Plotting can
run entirely from the saved `fields/*.npz`, `truth_*.npz` and `report.json` with
no GPU execution. Archives load with `allow_pickle=False`. Static ROI masks,
configuration and statuses are included in each field export.
Truth archives carry a ``physical`` boolean mask. Their zero placeholders
inside the hole do not represent zero-strain material and are never scored.

`validation.json` records the actual local checks; remote CI is not asserted
for these local changes. The workflow adds this native-UHD plate regression
to the lavapipe job. Its accuracy assertion is explicitly far-field; near-hole
limitations are shown here in full, rather than hidden behind a passing test.
