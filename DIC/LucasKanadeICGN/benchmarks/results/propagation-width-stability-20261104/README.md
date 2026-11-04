# Extended 16/32 propagation control — 2026-11-04

Clean measured source `799440ea6eda891f0b0807ad83dbfdfb0b1011d4`. Original native UHD tensile pair,
translation 32/16, B-spline, uniform, two fixed levels (8/16), tolerance 0.005,
raw fields and two failed-only propagation passes with empty-work stopping.
One prepared hardware device and shared resources; only existing solver-width
pipeline constants change. No cache, fused collector, default or numerical gate
is changed. The direct native 32-thread constructor matches the pilot hashes.

180 blocks, 18,000 AB/BA pairs and
36,000 timed fields. Control 16: median
17.895 ms, p95 19.660 ms,
56.02 fields/s. Candidate 32: median
16.235 ms, p95 18.010 ms,
61.62 fields/s. The median per-block
paired saving is 1.591 ms, bootstrap 95%
interval [1.573, 1.609] ms (seed 1729, 10,000 resamples).

The boundary is resident NumPy target to materialized field/statistics, excluding
reference preparation, decoding, display, storage, variant switching and warmups.
A read-only 5 Hz Linux observer runs beside the process; its scheduling/power
perturbation belongs to the protocol. No project tests, builds, repository edits
or other project GPU campaign ran concurrently. Desktop processes were not
controlled. The observer covers the entire process, including preparation and
warmups; its counters cannot be assigned precisely to individual fields.

Existing coverage/RMSE/p95 gates and exact status comparison are checked before
measurement and periodically at block boundaries. Acceptance is checked on
every call. The fixed parameter-agreement bound remains 0.001 px; reductions
can change floating parameters and a few iteration counts. Local translation
gradients remain zero. All paired samples and configurations are archived.
This longer repeated-pair test is not camera/recording qualification, arbitrary
motion recovery or a worst-case latency guarantee. Throughput and the fraction
of calls over 16.67 ms remain separate in summary.json.

The Python pilot retains its bounded timing lists for this fixed-length run.
RSS includes those lists, driver mappings and prepared resources. DRM requested
and resident client sums can share backing and must not be added to RSS as
physical memory. No physical/free-VRAM claim is made. The earlier bounded-batch
hour-long memory control is a separate experiment.

Original JSON is preserved compressed, with complete timings, commands, source
revision, clean state, adapter, limits, work lists, quality and configurations.
monitor.json, monitor.csv.gz and drm.jsonl.gz retain the observer and raw Linux
records. sources.tar.gz contains the exact pilot, launcher and observer. The
pilot's ROOT path is machine-specific; use an equivalent checkout to reproduce.
