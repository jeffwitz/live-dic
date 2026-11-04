# Rejected propagation optimizations — 2026-11-04

Three exploratory UHD controls on Intel xe-2lpg hardware retained the original
inputs, translation 32/16, B-spline, two levels (8/16 iterations), raw output,
two failed-only propagation passes and all existing accuracy gates. Reference
preparation, decoding, display and storage are outside the resident NumPy
image-to-materialized-field boundary. No other tests/builds ran during timing.
The reports retain raw samples, driver/adapter, options, source/input hashes
and dirty revision provenance. These are pilots, not a new 4K60 qualification.

| Pilot | Control median | Candidate median | Decision |
|---|---:|---:|---|
| Compact rejected inputs before scoring | 15.448 ms | 15.727 ms | No gain |
| Reference-pixel workgroup cache, dense scoring | 15.616 ms | 16.319 ms | Slower |
| Exact duplicate-candidate pruning | 15.734 ms | 15.957 ms | No gain |

The second matrix also measured compact scoring at 15.383 ms and compact plus
cache at 16.315 ms. The sign of the compaction difference changes across
campaigns: its small apparent gain is not reproducible here. The extra scan
and indirect setup compete with very cheap uniform exits on accepted points.
Shared-memory caching reduces repeated reference texture loads but adds a
barrier and storage pressure; it did not improve this adapter/workload.
Duplicate pruning adds comparisons and did not pay for itself on this field.

All candidates passed exact equality of parameters, statuses and iterations
against their same-campaign dense control, plus the existing analytical
coverage/error gates. The scoring order and scientific thresholds were kept.
This does not qualify correctness or performance on every scene or device.

None of these branches was enabled in the application. Their code was removed
from the active sources to avoid maintaining unhelpful options. The reversible
`experiment.patch` records the final exploratory implementation based on
`3aaeaa0`; apply it only to an isolated checkout. Its runner is configured for
the final deduplication pair; earlier matrices used the same flags to compare
compaction and reference caching. Earlier reports precede the final patch's
additional disabled experiments, so their source hashes differ. No claim is
made that this final patch reproduces their exact compiler input bytes.

Compaction uses separate read-only and read-write bindings for its input list:
WebGPU forbids binding the indirect argument buffer as writable storage in the
same dispatch that consumes it. The corrected pilot respects that constraint.
