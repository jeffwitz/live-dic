# GPU seed selection, median controls and OpenCV DIS

Research implementation, not a public native/browser option. Defaults remain
unchanged. Source revision for the final campaign:
`54b0901` (clean tree); `report.json` records the full revision and source hashes.
The earlier implementation at `7352fd1` is retained in `initial.json`.

## Implementation and boundaries

Translation 32/16, B-spline, two levels, iteration caps 8/16, tolerance
0.005/0.005, ZNCC 0.8, uniform integration, raw finest field, one synchronous
slot. The original 3840 x 2160 tensile reference and frame 40 are unchanged.
Reference preparation happens once. Timings include resident NumPy target
upload, GPU reduction, prefilters, candidate scoring, atomic compaction,
indirect re-solves, final NumPy field and statistics. They exclude decoding,
display and disk I/O. No intermediate host readback or per-frame compilation
is used. Diagnostic registrations/counter readbacks are outside timing.

Intel Graphics LNL / Vulkan, Mesa 25.2.8, wgpu 0.32.0, NumPy 2.4.6, Linux;
CPU governor `powersave`. Three warm-ups per case, then three alternating-order
rounds with twenty samples per case. All sixty raw timings per row are retained.
Fields/s is completed calls divided by their summed measured durations.

The scorer tests hypotheses on the destination window. Only an improved seed
queues another solve; it never becomes a copied measurement. Static ROI and
reference-validity permissions remain enforced. An atomic work list suffices;
no global sorting is needed. Its finalization kernel emits a 2-D indirect
dispatch when necessary. Image-support guards match the existing IC-GN guard.

In failed-only mode accepted windows skip candidate search. The input state is
immutable during that dispatch, so a snapshot copy and accepted-point restore
are unnecessary. In all-window mode a snapshot protects Jacobi propagation,
and an accepted old measurement is restored if a trial fails. Count-copy
instrumentation is explicitly disabled during timed registrations.

## UHD results

| Search policy | Extra passes/level | Median ms | p95 ms | Fields/s |
|---|---:|---:|---:|---:|
| Baseline in all-window group | 0 | 15.095 | 15.902 | 66.41 |
| Accepted-measurement median before transfer | 0 | 14.644 | 15.933 | 67.78 |
| Initialization-only median before transfer | 0 | 15.097 | 16.104 | 65.93 |
| All windows, full-pixel scoring | 1 | 41.969 | 43.203 | 23.82 |
| All windows, full-pixel scoring | 2 | 67.093 | 68.922 | 14.87 |
| Baseline in failed-only group | 0 | 14.377 | 15.542 | 69.25 |
| Failed only, full-pixel scoring | 1 | 15.010 | 16.404 | 66.75 |
| Failed only, full-pixel scoring | 2 | 16.123 | 16.872 | 62.19 |
| All windows, sampled every 4 pixels | 2 | 38.961 | 40.824 | 25.51 |

The failed-only two-pass median overhead relative to its own baseline is
1.746 ms (12.1%); measured fields/s falls by 10.2%. The one-pass overhead is
0.633 ms. The two-pass p95 exceeds 16.67 ms. These short repeated-pair results
do not qualify all-frame latency, thermally sustained recovery, camera input
or recording at 4K60. Baselines vary between groups/campaigns: the difference
between initial and final failed-only timings does not isolate the effect of
individual orchestration optimizations. Small median-filter timing differences
must not be interpreted as a speedup.

Every case retains 31,284 accepted positions, all 30,916 archived interior
positions, finite parameters and exactly zero local gradients. Vector RMSE is
about 0.0102 px. The baseline parameters, statuses and iterations are identical
to the ordinary production full pyramid. Previously fixed relative error gates
are retained in the report. No thresholds or iteration limits were relaxed.

`plate-control.json` additionally checks the original UHD plate-hole reference
and frame 95, exact nominal material mask, full subsets, same 32/16 translation
configuration. All cases cover 100% of the 30,706 selected interior positions,
with approximately 0.00802 px vector RMSE against the independent Kirsch truth.
No additional seeds are needed on that case. This is a correctness check,
not a timing or universal translation/ROI qualification.

## What differs from OpenCV DIS

Inspection of OpenCV 5.0.0 `modules/video/src/dis_flow.cpp` confirms that each
directional sweep tests the current vector and two already-updated neighbours,
plus an optional temporal candidate. The configured gradient-descent budget
is divided between forward/backward sweeps, with an early exit when SSD ceases
decreasing. Mean-centred SSD, bilinear interpolation and an 8 x 8 SIMD path
are used. Dense photometric aggregation and optional variational refinement
follow at each level. References: [source](https://github.com/opencv/opencv/blob/5.0.0/modules/video/src/dis_flow.cpp),
[DIS paper](https://arxiv.org/html/1603.03590v1).

Our all-window prototype instead tests eight neighbours, prior and current
seeds using NCC reductions, then adds full native solves. Jacobi passes also
propagate only a limited distance per pass. It is not an equal-work port of
DIS. The targeted policy reduces work rather than demonstrating that GPU
hardware must accelerate any arbitrary host/kernel organization. Further
work could test fewer directional candidates and share an initialization
iteration budget; it must preserve final scientific acceptance and be qualified.

## CPU DIS application controls

`opencv-uhd.json` and the exact measured `opencv_uhd.py` retain all six cases.
OpenCV 5.0.0, eight threads, OpenCL disabled, NumPy inputs, one warm-up and five
timed calls each. `calc` prepares both images on every call and returns a dense
full-size flow. The native engine retains its prepared reference and returns
the subset grid. These are application boundary observations, not an isolated
GPU/CPU kernel speedup or identical algorithms.

| CPU case | Median ms | Fields/s | Interior vector RMSE px |
|---|---:|---:|---:|
| DIS ultrafast, final solve at quarter width/height | 21.304 | 46.73 | 0.07809 |
| DIS fast, final solve at quarter width/height | 59.860 | 16.57 | 0.10400 |
| DIS medium, final solve at half width/height | 214.340 | 4.69 | 0.02599 |
| Requested 4/2, finest scale 0, refinement 10 | 1366.360 | 0.73 | 0.06035 |
| Geometry-matched 32/16, two levels, refinement 0 | 235.502 | 4.18 | 0.02673 |
| Geometry-matched 32/16, two levels, refinement 10 | 1068.455 | 0.94 | 0.03424 |

The dense returned flow is sampled at the same 30,916 subset centres for the
truth error. OpenCV has no corresponding solver acceptance/status mask here;
coverage must not be equated with live-dic acceptance. Geometry-matched means
32/16 and two levels, not equal equations, iteration budgets or preparation.
The CPU comparison is sequential, not thermally interleaved with the GPU run.

## Median verification and remaining scope

The previous CPU seed experiment disabled medians and did not establish their
ineffectiveness. Independent GPU tests verify the existing 3 x 3 median over
accepted measurements and actual transfer of its filtered buffer. A second
initialization-only median includes unfinished status-0 estimates and may
initialize an unavailable seed from four permitted neighbours. Raw coarse
statuses stay untouched and fine LK must solve the point normally.

The associated private large-image checks remain local under
`/tmp/live-dic-gpu-seed-20261104/`. They have no ground truth and are not
published as metrology qualification. The final accepted masks/parameters
are unchanged by the failed-only orchestration optimization; rejected states
may differ after pruning impossible image supports. The public repository
contains no private images or derived fields.

Validation: the complete suite with native GPU and browser flags executed
348 tests with no skips. Six targeted GPU controls also pass after the final
support guard change. Ruff, ESLint, strict Sphinx, resource checks, a pure-Python
wheel and outside-checkout installed native execution/NPZ export were checked.
No new remote CI or browser propagation execution is claimed. The shared
default browser shader paths were exercised; this research supervisor remains
native-only and synchronous, with no GUI setting or streaming support.

## Reproduction

From the repository root, with the intended native Python environment:

```bash
python3 DIC/LucasKanadeICGN/benchmarks/benchmark_gpu_spatial_seed.py \
  --output /tmp/gpu-spatial-seeds
```

The output directory must be new. For the CPU comparison, install the optional
`opencv-python-headless` separately, create
`/tmp/live-dic-gpu-seed-20261104`, and run the archived `opencv_uhd.py` from the
repository root. It writes only under that temporary directory. The GPU
prototype does not depend on OpenCV. For the plate control, use
`GpuSpatialPyramid.prepare_research(reference, roi_mask=mask)` with the exact
mask formula in `assets/uhd_plate_hole/README.md`, and compare its raw translation
at the existing grid centres with
`python/tests/plate_hole_reference.py:displacement(x, y, 250)`.
