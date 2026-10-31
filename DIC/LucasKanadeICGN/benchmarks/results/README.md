# Benchmark evidence index

These archives retain the code revision, configuration, inputs, hardware,
quality populations and raw measurements of each campaign. Read the local
README before comparing reports: protocols and revisions differ. Archived
JSON, NPZ, logs, patches, images and reproduction scripts are evidence, not
current runtime source. Do not reformat or overwrite them during maintenance.

The [performance reference](../../docs/reference/performance.rst) is the
canonical numerical summary and hardware coverage matrix. The
[benchmark protocol](../../BENCHMARK.md) explains how to produce new evidence.
A test count or timing here describes its archived revision and environment,
not an automatically refreshed statement about HEAD.

## Native UHD and temporal campaigns

| Archive | Purpose |
|---|---|
| [Temporal initialization](temporal-init-20261030/README.md) | Zero versus temporal coarse seeds; fixed-reference capture range and stream controls |
| [Real-time optimization](realtime-optimization-20261030/README.md) | Changing-frame controls, fine correction and rejected optimizations |
| [Latency reduction](latency-reduction-20261030/README.md) | Velocity, finest-only stagnation stopping, adaptive/cubic failures and late-convergence audits |
| [Intel width 16/32](workgroup-width-20261030/README.md) | Matched native-UHD temporal campaign; 16 preferred on Lunar Lake |
| [Intel width 16/128](workgroup-128-20261030/README.md) | Matched experimental larger-workgroup control; 128 misses the cadence budget |
| [AMD width 16/32](workgroup-width-amd-20261030/README.md) | Radeon 780M trajectory, precision, throughput and result age |
| [GPU device lifetime](workgroup-width-amd-20261030/device-lifecycle/README.md) | Reproduced AMD resource exhaustion, cleanup correction, stress/parity evidence |

The NVIDIA RTX 3080 Laptop and Intel Tiger Lake fixed-pair measurements are
reported in the performance reference. They are not full temporal-trajectory
or complete regression-suite qualifications. Hardware timings on different
machines/revisions are not a controlled ranking of GPUs.

## ROI and boundary campaigns

| Archive | Purpose |
|---|---|
| [Complete-window ROI](roi-full-20261030/README.md) | Exact selection, compact/fragmented masks and live overhead |
| [Partial-support ROI](roi-partial-20261030/README.md) | Connected boundary supports, conditioning and visibility truncation |
| [Plate with a hole](plate-hole-roi-20261030/README.md) | Native 4K boundary/mask comparisons and von Mises figures |

## Native subgroup migration

The `2026-10-30-intel-arc-130v-*` JSON files qualify the WGSL subgroup
migration on commit `481516a7c3223b8c58be0b9bf45a79cafe2c70d6`. They use the
same 512×512 affine fixture, eight warmups and fifty registrations for forced
portable, forced subgroup and automatic policies. Their workload is distinct
from the UHD full-pyramid campaigns above.

All results are machine- and workload-specific. Software Vulkan can validate
shader correctness but does not qualify hardware real-time performance.
