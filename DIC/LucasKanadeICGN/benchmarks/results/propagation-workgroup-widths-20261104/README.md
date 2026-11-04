# Native propagation workgroup controls — 2026-11-04

An explicit width of 32 improves these two native UHD cases. Width 128 is
workload dependent and regresses on the original traction pair. **The default
remains 16.** This is hardware-specific evidence, not width autotuning.

All six runs use one prepared Intel LNL/Vulkan device, Mesa 25.2.8, translation
32/16, B-spline, uniform weights, two fixed levels (8/16), tolerance 0.005,
ZNCC 0.8, two failed-only propagation passes, empty-work stopping and raw final
fields. Scores are recomputed; no score cache or fused rejection collector is
used. Only existing SOLVER_WORKGROUP_SIZE overrides change in fresh, seeded
and compact solves. Buffers, images, reference data and bindings are shared.
Variant selection is outside the timed call. A separate real native constructor
with solver_workgroup_size=32 exactly reproduces the original-pair candidate's
parameter, status and iteration hashes.

| Case | Compared width | Control 16 median | Candidate median / p95 | Candidate fields/s |
|---|---:|---:|---:|---:|
| Original tensile frame 40 | 32 | 18.133 ms | 16.453 / 18.367 ms | 61.02 |
| Original tensile frame 40 | 128 | 17.132 ms | 18.506 / 19.641 ms | 54.33 |
| Seeded translation | 32 | 17.559 ms | 15.616 / 17.318 ms | 64.17 |
| Seeded translation | 128 | 17.374 ms | 16.540 / 20.163 ms | 59.55 |

The boundary is resident NumPy target to materialized field/statistics; it
excludes preparation, decoding, display, storage and variant selection. Each
run has twelve 100-pair blocks with per-frame AB/BA and eight warm-ups per
variant/block. No other project tests, builds or GPU work ran alongside.
Unrelated desktop processes were not controlled. The source revision and
clean state, configuration, command, adapter and every timing are in each
original report.

Reference/reference controls run first for each case. Before either candidate,
the margin is fixed to the 95th percentile of absolute per-block median paired
reference differences. A 10,000-resample block bootstrap (seed 1729) gives the
candidate saving interval; raw criteria and calculations are retained.
Width 32 saves 1.651 ms [1.524, 1.711] on traction and 1.926 ms [1.893, 2.022]
on the seeded case, beyond the corresponding margins. Do not merge absolute
baselines from separate runs into an optimization claim. p95 remains above
16.67 ms; this is not sustained 60 Hz latency, camera or storage qualification.

The traction case retains 100% analytical interior coverage and about 0.01017 px
vector RMSE. The seeded case retains 99.8706% and about 0.00000650 px. Existing
coverage/RMSE/p95 gates are unchanged. Status arrays agree exactly. Width changes
floating reduction order: the maximum parameter difference is below the fixed
0.001 px bound (observed at most 0.000306); a few traction iterations differ.
Local translation gradients remain exactly zero. Work lists and iteration
counts are retained, rather than attributing every effect solely to thread width.

The stress target is a deterministic in-memory np.roll of the original UHD
reference by (dy=-6, dx=14). Coarse 3x3 seed defects use (25,-18) where x%8<3,
y%8<3 and x,y>3. It is an initialization stress control, not a natural crack or
universal robustness proof. Fixtures are neither downloaded nor regenerated.

Compressed JSON files preserve original report bytes. sources.tar.gz retains
the exact development scripts (their ROOT path is machine-specific; point it
to an equivalent checkout for another machine). Their hashes are in the
reports and criteria. Public native API already supports the explicit width;
the browser now exposes 16/32 separately and refuses 128. No default changes.
