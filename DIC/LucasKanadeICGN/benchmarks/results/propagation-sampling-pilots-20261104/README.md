# Scoring width and hardware sampling pilots — 2026-11-04

These additional exploratory controls use the original UHD tensile pair,
translation 32/16, two levels (8/16), tolerance 0.005, B-spline IC-GN and two
failed-only propagation passes. Six alternating blocks per candidate contain
60 resident NumPy image-to-materialized-field observations each; preparation,
decoding, display and storage are excluded. Reports include the samples,
adapter/driver, dirty source provenance, hashes and analytical quality results.

| Scoring workgroup | Median | p95 | Sustained engine throughput |
|---|---:|---:|---:|
| 64 (control) | 15.342 ms | 16.521 ms | 64.99 fields/s |
| 32 | 15.171 ms | 16.467 ms | 65.50 fields/s |
| 16 | 15.253 ms | 16.502 ms | 65.43 fields/s |

The small difference does not establish a useful general gain. A separate
local diagnostic on the original larger private pair found 16 lanes slower;
those inputs, fields and reports remain private. The application keeps its
existing 64-lane scoring path. Scoring width is distinct from the 16-lane LK
solver, image subset width and subset stride.

An additional 64-lane pilot replaced the four explicit bilinear texture loads
with a hardware filtering sampler for **seed scoring only**. Final LK remained
unchanged. Its same-campaign control was 15.035 ms median / 16.058 ms p95,
versus 15.391 / 16.431 ms with hardware filtering. It is not faster here and
the implementation was removed. Hardware filtering can also round samples
differently; this is not an interchangeable numerical default.

The existing coverage/RMSE/p95 gates passed, and translation gradients stayed
exactly zero. Changing floating reduction order need not preserve every bit;
no tolerance was relaxed. The runner required identical status masks for the
width comparison and independently evaluated the analytical gates for hardware
filtering. These smooth-field controls do not validate difficult-scene accuracy
or certify 4K60 camera/display/recording operation.

`experiment.patch` preserves the final exploratory code based on `b8d02ae`.
Apply it in an isolated checkout, then use its `--compare width` or
`--compare hardware` runner and a new output directory. The earlier width report
predates the sampler experiment, so its source hashes differ from that final
superset. A sampler binding remains reflected even when a specialization
constant disables its branch; the experimental native host binds it for both
hardware controls. The patch is not a browser implementation and must not be
substituted for the production shader without updating its host bindings.
