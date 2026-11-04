# Native subgroup scoring pilot — 2026-11-04

This isolated prototype replaces only the scoring reduction with subgroup
operations and a workgroup merge. It preserves candidate hypotheses, sampling,
thresholds and final LK. Both controls request native `subgroup`; the ordinary
full-pyramid solver remains portable. It was not integrated in the application.

The original UHD tensile pair, translation 32/16, B-spline, two levels and two
failed-only passes produced the same parameters and status mask as the control.
Six alternating blocks of sixty observations gave:

| Reduction | Median | p95 | Engine throughput |
|---|---:|---:|---:|
| Portable workgroup | 14.771 ms | 15.741 ms | 67.32 fields/s |
| Subgroup prototype | 14.826 ms | 15.934 ms | 67.11 fields/s |

No gain is established. A separate private correctness diagnostic found a
difference between two rejection reasons at one point, while the accepted mask
was unchanged. That diagnostic stopped before timing; private inputs and fields
are not published. The prototype is not a new metrology or hardware-coverage
qualification, and the portable production path remains unchanged.

`context.json` identifies the original inputs, control shader and settings.
`report.json` retains raw timing samples and the tested adapter/driver. The
boundary is resident NumPy target to materialized NumPy field/statistics, with
reference preparation excluded and no concurrent tests/builds. `benchmark.py`
is the exact temporary research script: it expects this directory's
`spatial-subgroup.wgsl` copied to `/tmp/live-dic-night-spatial-subgroup.wgsl`
and runs from the repository root with the native environment. It patches only
the research process's device request and shader creation; it is not a library
loader. No browser execution or camera performance is claimed.
