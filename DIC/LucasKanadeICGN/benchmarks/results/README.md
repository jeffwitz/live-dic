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
