# Sparse-frontier propagation pilots — 2026-11-04

These exploratory pilots are **not product implementations**. They investigate
whether later Jacobi searches can be limited to the 3×3 neighbourhood of
previously re-solved points. Unchanged destinations have unchanged inputs.
A conservative GPU frontier can therefore avoid searches without removing a
changed hypothesis. The ordinary compact LK and dense output remain in use.

Original UHD tensile reference/frame 40, translation 32/16, B-spline, uniform
integration, two fixed levels (8/16), tolerance 0.005, raw fields, two failed-only
passes. A single prepared hardware device alternates variants after every
frame, eight blocks of 100 observations per variant. Resident NumPy target to
materialized field/statistics is timed; preparation, decoding, display and
storage are excluded. Intel LNL, native Vulkan, Mesa 25.2.8.

| Pilot | Control median / p95 | Candidate median / p95 |
|---|---:|---:|
| Separate mark/clear/finalize kernels | 15.416 / 16.314 ms | 15.609 / 16.484 ms |
| Frontier emission fused into compact LK | 15.437 / 16.348 ms | 15.495 / 16.336 ms |
| Packed frontier, unchanged compact LK | 15.267 / 16.205 ms | 15.332 / 16.150 ms |

None establishes a worthwhile improvement on this easy pair, whose passes
queue no new seeds. Added metadata dispatches/clears can cost more than the
saved work. The first and third designs preserve full parameter/status/iteration
arrays on the additional local difficult-pair checks. The fused-solver design
fails exact parity there and requires separate numerical qualification; it was
not integrated. Private images, fields and reports are not distributed.

All three public reports retain raw samples and source hashes as gzip-compressed
original JSON streams, without reformatting. Their metadata are less complete
than the standardized qualification drivers: they do not independently record
all configuration/source files. This table is exploratory evidence, not a
release-performance acceptance. The production empty-work optimization has its
own controlled campaign and repeatability criteria in the neighbouring archive.

`sources.tar.gz` preserves the exact data-free prototype shader/host files;
`source-hashes.json` authenticates every member. Their base is `8bdbad8` (the
subsequent `98002db` only archives results/docs). To inspect/reproduce, use an
isolated checkout at that base, extract into `/tmp`, add `/tmp` and
`benchmarks/` to `PYTHONPATH`, and import `FrontierPyramid`, `FusedPyramid` or
`Frontier2Pyramid` from the respective `live_dic_*.py` module. Their explicit
`frontier=False/True` switch selects the control/candidate. Keep all referenced
shader filenames in `/tmp`; these are deliberately unmaintained experiments,
not an alternate public engine. Do not copy them over canonical shaders.
