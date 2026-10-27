# WebGL2 IC-GN benchmark against the SYCL implementation

Date: October 27, 2026.

## Common configuration

- GPU: Intel Lunar Lake `[8086:64a0]`, `xe` kernel driver;
- WebGL: Chromium 155.0.8059.79, ANGLE/Mesa Intel LNL OpenGL 4.6;
- SYCL: DPC++ `-O2`, Level Zero backend, same GPU;
- images: raw 512 x 512 luma frames from `seed2026101102_translation`;
- measured pair: `t=0 -> 10`, true translation `(2, -1)` pixels;
- grid: 32 x 32, or 1,024 subsets at 16-pixel spacing;
- subsets: 16 x 16 pixels, with 900 usable interior subsets;
- prefilter: `sigma=1` Gaussian and cubic B-spline coefficients;
- target interpolation: 4 x 4 cubic B-spline;
- six-parameter affine IC-GN with a constant inverse Hessian;
- `1e-3` tolerance, minimum ZNCC `0.8`, at most 30 iterations;
- median consistency test and replacement enabled;
- no H.264 decoding or processing.

## Ported SYCL operations

The reference is prepared once using the same Gaussian filter and cubic
B-spline IIR prefilter. The centered reference, nodal gradients, `H^-1`,
`J^T F`, and `sum(J)` are packed into three WebGL textures. Two GPU passes apply
the same separable radius-15 FIR used by the optimized SYCL configuration to
each target.

The registration shader reads each interpolated target pixel once per
iteration. It accumulates the mean, norm, ZNCC, and `J^T g` together, then
applies `-H^-1 b` and the same inverse-compositional update. A final pass runs
the normalized 3 x 3 median consistency test and replaces flagged subsets.

## WebGL2 performance

Each campaign uses three warm-up runs followed by 30 measured runs. GPU timing
comes from `EXT_disjoint_timer_query_webgl2`. The FIR is measured separately;
solver time is estimated by subtracting it from the complete GPU pipeline to
avoid inserting a timer-query barrier between the two stages.

| Warm campaign | Median FIR GPU | Estimated solver | Median FIR + solver | FIR + solver p95 |
|---|---:|---:|---:|---:|
| 1 | 0.221 ms | 3.556 ms | 3.777 ms | 4.238 ms |
| 2 | 0.222 ms | 3.543 ms | 3.765 ms | 6.810 ms |
| 3 | 0.229 ms | 3.565 ms | 3.795 ms | 4.376 ms |

The representative warm result is **3.78 ms of GPU time per pair**, or about
**265 pairs/s**. Two campaigns have a 4.24-4.38 ms p95; one contains a 6.81 ms
excursion. Compilation plus the first pass takes approximately 79-105 ms with
a populated Chromium cache and 329 ms on the first compilation of the shader.

An additional campaign covering target upload, FIR, solver, and result readback
reports a **17.6 ms median wall time** and **19.1 ms p95**. Reading the two result
textures alone takes approximately 1.4-2.8 ms. Transfers, WebGL/JavaScript
synchronization, and timer-query waits explain most of the difference from GPU
time.

The headless Python interface was validated independently with three warm-ups
and 30 repetitions on the same pair. It measured a **3.908 ms median GPU time**
(**5.702 ms p95**) and a **16.9 ms median wall time** (**19.5 ms p95**). It
returned 900 accepted subsets, a mean translation of
`(1.999993, -0.999995)` pixels, and a `3.819e-5` pixel RMSE. DevTools transport
is outside the GPU timer query.

After the API cleanup, English documentation pass, and added validation, two
30-run non-regression campaigns measured **3.967 ms** and **3.859 ms** median GPU
time, with **16.8 ms** and **16.9 ms** median wall time. Their midpoint is
3.913 ms, effectively unchanged from the 3.908 ms baseline. The displacement,
accepted-subset count, and iteration count remained identical.

## Direct comparison with SYCL

The SYCL binary was rebuilt from `icgn_sycl.cpp`. Thirty independent processes
were measured after three warm-ups, resetting initialization for every
`t=0 -> 10` pair.

| Measurement | Equivalent WebGL2 | SYCL | WebGL/SYCL ratio |
|---|---:|---:|---:|
| GPU registration | 3.55 ms estimated | 2.573 ms | 1.38x |
| GPU prefilter + registration | 3.78 ms | 3.651 ms total SYCL | 1.04x |
| Pipeline p95 | 4.24-4.38 ms typical | 4.280 ms | approximately 1.00x |
| End-to-end with CPU readback | 17.6 ms | 3.651 ms | 4.82x |
| Mean iterations, accepted subsets | 5.957 | 5.763 | 1.03x |
| Accepted subsets | 900/1,024 | 900/1,024 | identical |
| Vector RMSE | `3.82e-5 px` | `1.12e-5 px` | both subpixel |

GPU compute cost is now comparable to SYCL for this case. The browser
end-to-end path remains slower because of synchronization and WebGL readback.
The SYCL `total_ms` includes overlapped upload, prefiltering, registration, and
readback. WebGL GPU timing excludes the initial transfer and final readback, so
the table reports end-to-end time separately.

## WebGL2 accuracy

| Target from t=0 | Truth (px) | Measured mean (px) | RMSE | Accepted | Mean iterations | Replaced |
|---|---:|---:|---:|---:|---:|---:|
| t=1 | (0.20, -0.10) | (0.191, -0.093) | 0.0186 px | 900/1,024 | 3.994 | 0 |
| t=5 | (1.00, -0.50) | (1.000, -0.500) | 0.0170 px | 900/1,024 | 5.197 | 0 |
| t=10 | (2.00, -1.00) | (1.999993, -0.999995) | `3.82e-5 px` | 900/1,024 | 5.957 | 0 |
| t=20 | (4.00, -2.00) | (3.998, -1.997) | 0.0906 px | 634/1,024 | 11.855 | 1 |

The incremental `t=10 -> 20` step, a `(2, -1)` pixel displacement, accepts
900/1,024 subsets in 5.993 mean iterations with a displayed error below
`1e-4 px`.

## Equivalence limits

The implementations are not bit-for-bit equivalent. WebGL2 uses single
precision and does not expose SYCL subgroups, Intel's larger register file, or
Level Zero queues. This port does not yet implement Aitken acceleration,
predictive tolerance, temporal/extrapolated initialization, retry from the last
accepted state, a pyramid, or the H.264 pipeline. These features do not change
the uniform `t=0 -> 10` result used here, but matter for noisy sequences and
larger motions.
