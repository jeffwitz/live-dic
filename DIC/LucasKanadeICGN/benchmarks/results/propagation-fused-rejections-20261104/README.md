# Rejected-list construction fused into the initial solve — 2026-11-04

**Exploratory implementation, not integrated.** The initial translation LK
workgroup appends its rejected destination to an atomic list before returning.
Later failed-only searches traverse that same list, skipping newly accepted
points. Because accepted states cannot become failed between these passes,
this removes dense search dispatches without changing eligible candidates or
introducing a separate compaction dispatch. Indirect dimensions support tails
and two-dimensional work lists; the output grid remains dense.

On the original UHD tensile reference/frame 40, full arrays of parameters,
statuses and iterations match the production host. Only 742 fine and 249
coarse initial rejections enter the lists; no improved seed is queued. Settings:
translation 32/16, B-spline, uniform full windows, two levels (8/16), tolerance
0.005, raw outputs, two failed-only passes, empty-work stopping. One prepared
Intel LNL/Vulkan device, Mesa 25.2.8, per-frame AB/BA, eight 100-frame blocks.
Resident NumPy target to materialized field/statistics is timed, excluding
preparation/decoding/display/storage. Other project tests/builds/GPU benchmarks
were stopped during measurement.

| Variant | Median | p95 | Engine fields/s |
|---|---:|---:|---:|
| Dense rejected-point search | 15.197 ms | 16.197 ms | 65.82 |
| Fused initial rejection list | 15.063 ms | 16.025 ms | 66.41 |

The median block-paired saving is 0.1024 ms, with bootstrap 95% interval
[0.0368, 0.1873] ms (10,000 resamples, seed 1729). There is no dedicated
reference/reference acceptance margin for this pilot, and its metadata are
less complete than the standardized campaigns. This small apparent improvement
is insufficient for promotion. A separate local difficult-pair check retains
exact fields but is slower with fusion. Its private inputs/results are not
published. Moving a little more code into LK can change execution cost even
when it avoids a later dispatch; launch counts alone do not establish a gain.

The compressed original public report preserves samples and prototype hashes.
`sources.tar.gz`/`source-hashes.json` preserve the exact data-free shaders and
host; no canonical shader was replaced. Inspect with the repository state at
`ed77da8` (check canonical shader hashes),
extract the four files into `/tmp` and add `/tmp` plus `benchmarks/` to
`PYTHONPATH`. `CollectorPyramid(frontier=False/True)` selects the control or
candidate. The legacy name is an exploratory switch, not a public API.
