# UHD spatial-seed CPU prototype performance

Measured on the original 3840 x 2160 tensile reference and frame 40 at source
commit `d73df899ec491abc6bfb46f1eefff58c27b23bc9`, with a clean working tree.
`report.json` retains commands, input/script hashes, adapter/driver/software,
all 84 timing samples, stage breakdowns and analytic quality controls.

## Boundary and controls

Resident uint8 NumPy target to materialized NumPy field/statistics, fixed
reference prepared once. Excludes image decoding, display, disk storage and
any global-affine bootstrap. Includes target reduction, CPU scoring and seed
transfer where applicable, all uploads, re-solves and intermediate readbacks.

Translation, size 32, stride 16, B-spline, uniform weights, no median/UOD,
zero initial seeds on each call, full image/no ROI, two levels, iteration caps
8/16, tolerance 0.005 at both levels, ZNCC 0.8, one in-flight frame. No adaptive
workload reduction. The extra passes run on all selected windows.

Hardware: Intel Graphics LNL, Vulkan, Mesa 25.2.8, Linux x86_64;
wgpu 0.32.0, NumPy 2.4.6, OpenCV 5.0.0 (eight threads, OpenCL disabled).
CPU governor was `powersave`; no performance-mode claim. Orphaned headless
test Chromium instances were terminated before timing; user Firefox stayed open.

One warm-up per case, then three rounds in forward/reverse/forward case order;
20 production calls and two calls of each research case per round. No tests or
builds ran during timing. Preparation including engine creation is separate:
1.654 s production / 1.327 s independent levels.

## Results

| Case | Median ms | p95 ms | Completed fields/s | Samples |
|---|---:|---:|---:|---:|
| Production full GPU pyramid | 14.397 | 16.110 | 68.7141 | 60 |
| Independent levels, no extra solve | 63.812 | 76.103 | 15.1289 | 6 |
| Independent levels, two extra solves per level | 90.271 | 97.650 | 11.0048 | 6 |
| CPU spatial selection, one extra pass per level | 3351.324 | 3402.432 | 0.2983 | 6 |
| CPU spatial selection, two extra passes per level | 6535.071 | 6743.773 | 0.1518 | 6 |

Fields/s is sample count divided by the sum of measured wall durations; it is
not the reciprocal of median latency. These are sequential repeated-pair calls,
not a camera stream. Six samples are insufficient for a reliable tail-latency
qualification: p95 is descriptive. This does not certify 4K60 recording or age.

The two-pass prototype is 453.9 times slower in median
than the production pyramid, a 99.78% reduction in
measured fields/s. Median CPU candidate-scoring stages total about
6403 ms;
coarse/fine solve stages total about
92 ms.
Sums of stage medians need not equal the median whole call.

The independent-level no-propagation control is slower than production because
it reduces images and transfers coarse seeds on CPU. Additional re-solves alone
cost far less than CPU candidate scoring. This attributes the bottleneck to the
current oracle; it does not predict a future GPU propagation kernel's FPS.

All cases retain 31,284 accepted grid points, 100% of the 30,916-point archived
interior population, displacement vector RMSE below 0.01020 px and p95 error
below 0.01724 px. No private input images or derived fields are included.

## Reproduction

Install the optional `opencv-python-headless` in the intended environment, then
from the repository root (output directory must not exist):

```bash
DIC/LucasKanadeICGN/python/.venv/bin/python \
  DIC/LucasKanadeICGN/benchmarks/benchmark_spatial_seed_performance.py \
  --output /tmp/live-dic-spatial-performance
```

No engine or browser option is added. A GPU-resident implementation and separate
numerical/performance qualification are required before live use.
