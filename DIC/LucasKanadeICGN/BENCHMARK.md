# live-dic benchmark protocol

The [performance reference](docs/reference/performance.rst) is the canonical
location for timing tables and the GPU hardware coverage matrix. The
[campaign index](benchmarks/results/README.md) points to raw reports, input
hashes, source revisions, diagnostic logs and final fields. This document
explains how to measure; it deliberately does not duplicate numerical tables.

## Define the measured workload

For the Python product boundary, start with a resident, contiguous uint8 NumPy
image and stop with the complete NumPy field and statistics. Preparation,
compilation, acquisition, decoding and downstream consumption are separate
measurements. GPU timestamps and completion waits are useful diagnostics,
not substitutes for the complete call.

Keep the reference, targets, model, interpolation, subset size/stride,
weighting, ROI, spatial policy, pyramid levels, iteration/tolerance schedules,
initialization and slot depth identical between A/B cases. Use a hardware GPU
explicitly for live performance; software adapters qualify correctness.
Report the actual adapter and selected solver rather than assuming the request
was honored.

The checked-in [UHD tensile](assets/uhd_tensile/README.md) and
[plate-hole](assets/uhd_plate_hole/README.md) fixtures have analytic truth and
provenance. Tiling the small integer-translation fixtures is useful for timing,
but must not be presented as a native-UHD subpixel metrology qualification.
Do not require large-strain late frames to converge from an unqualified zero
initialization. Preserve each dataset's documented capture-range and error gates.

## Changing-frame image-to-field benchmark

From the repository root, after installing the Python package:

```bash
python3 DIC/LucasKanadeICGN/benchmarks/benchmark_image_to_field.py \
  --backend native-pyramid --fixture-set uhd-tensile --device-type gpu \
  --width 3840 --height 2160 --frames 50 --warmups 8 --campaigns 3 \
  --pipeline-depth 1 --motion-model affine --outlier-policy median
```

Run `--help` for ROI NPY inputs, subset geometry, interpolation and output
options. Use `--backend three-tsl` or `--backend webgpu-wgsl` for Python/browser
comparisons without ROI. Their algorithmic coverage differs; Three.js is
single-scale, so label the comparison accordingly.

For the 200-frame temporal trajectory, follow the exact cache preparation,
settings and commands in the [latency campaign](benchmarks/results/latency-reduction-20261030/README.md).
The width [16/32](benchmarks/results/workgroup-width-20261030/README.md),
[AMD](benchmarks/results/workgroup-width-amd-20261030/README.md) and
[16/128](benchmarks/results/workgroup-128-20261030/README.md) controls archive
three alternating repetitions at depths one and two. Do not overwrite these
archives when running a new campaign; choose a new output directory.

## Required reporting

- Input hashes, mask hash/data, code revision and dirty state, relevant source
  hashes, exact command, software versions, GPU/driver and power profile.
- Preparation time separately from warmed steady-state measurements.
- Sustained FPS and stream period separately from result age p50/p95/p99.
  State whether quantiles pool samples or summarize per-run quantiles.
- Raw samples, warmup and repetition counts, slot depth and actual levels used.
- Reference population, ROI selection, numerical validity, accepted/rejected
  counts, displacement and derivative errors on explicitly defined populations.
- Whole-grid failures as well as any fixed-interior qualification.
- Iteration distributions over all windows, including nonconverged windows.
- Numerical agreement of optimized variants before choosing them by speed.

Alternate matched runs and keep tests, builds, rendering and scoring outside
timed streams. Derive a non-regression margin from repeated reference runs;
a noisy comparison is indeterminate. Never improve timings by silently
lowering accuracy requirements, changing the requested model, reducing levels,
or omitting eligible windows.

## Qualification and limitations

Run the [development checks](../../CONTRIBUTING.md) before publishing evidence.
Portable CI, lavapipe execution, hardware-GPU correctness and hardware-GPU
performance are distinct qualifications. A passing test count is meaningful
only with its revision and environment; there is no hardcoded current count.

4K60 throughput is not automatically a p95 result age below 16.67 ms, and
neither is a camera-to-consumer qualification. All measured runs are soft
real-time. The related SYCL project is an algorithmic provenance reference;
add a timing comparison only after matching its hardware, inputs, configuration
and measurement boundary.
