# Archived benchmark reports

These JSON files are direct outputs from the benchmark programs. Each report
records its Git commit and dirty-worktree state, command, software versions,
platform, input hashes, numerical results, selected solver, and raw timing
samples.

The `2026-10-30-intel-arc-130v-*` reports qualify the native WGSL subgroup
migration on commit `481516a7c3223b8c58be0b9bf45a79cafe2c70d6`. They use the
same 512 × 512 affine fixture, eight warmups, and fifty measured registrations
for the forced portable, forced subgroup, and automatic policies.

These reports are machine- and workload-specific evidence. They are not
portable performance promises and must not be compared directly with the UHD
full-pyramid or pipelined-sequence campaigns.

## GPU hardware coverage

The [hardware coverage matrix](../../docs/reference/performance.rst#gpu-hardware-coverage)
records successful native 4K benchmarks on AMD, Intel and NVIDIA and the
different scope of each experiment. NVIDIA RTX 3080 Laptop and Intel Tiger
Lake fixed-pair measurements are documented in that performance reference.

- [Intel Lunar Lake changing-frame campaign](workgroup-width-20261030/README.md):
  twelve runs, fixed-interior quality gates passed, full GPU suite reported
  passing on that machine.
- [AMD Radeon 780M changing-frame campaign](workgroup-width-amd-20261030/README.md):
  twelve runs and fixed-interior quality gates passed; its initial GPU-suite
  failure was traced to retained GPU resources after `close()`. The lifecycle
  investigation and correction are documented in the campaign archive. Console
  output, raw samples, provenance and final fields are retained in the archive.
