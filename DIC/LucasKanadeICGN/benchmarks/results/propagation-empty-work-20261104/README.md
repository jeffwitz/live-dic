# Exact empty-work elimination — 2026-11-04

Clean source `8bdbad87988328934cb45617bbf59090c469837e`, Intel LNL/Mesa 25.2.8,
native Vulkan. Original 3840 × 2160 tensile reference/frame 40; translation
32/16, B-spline, raw fields, fixed two levels (8/16 iterations), two failed-only
propagation passes and one prepared device. Other tests/builds and GPU work
were stopped during timing. Preparation, decoding, display and storage are
outside the resident NumPy image-to-materialized-field/statistics boundary.

| Same propagation budget | Median | p95 | p99 | Engine fields/s |
|---|---:|---:|---:|---:|
| Repeated dense searches | 15.193 ms | 16.302 ms | 16.768 ms | 65.70 |
| Skip searches after an empty work list | 14.983 ms | 16.125 ms | 16.663 ms | 66.51 |

This is a small measured improvement on an easy pair. Every pass in this pair
queues zero new seeds. Once no work is queued, state/image/permissions/prior
remain unchanged and a repeated search cannot improve anything. The first
search of a new frame/level always runs. Integer continuation arguments are
written by the existing GPU finalizer; no additional CPU readback or scoring
approximation is introduced. It does not help a difficult pass that keeps
producing work and does not certify a complete 4K60 camera/recording system.

Before the candidate run, a reference/reference campaign established a 0.09033 ms
margin: the 95th percentile of absolute per-block median paired differences.
`criteria.json` preserves that decision and the control report hash. The
candidate's median per-block paired saving is **0.16941 ms**, bootstrap 95% CI
**[0.12805, 0.23196] ms**. This exceeds that repeatability margin. Separate
aggregate medians and matched-frame savings are different estimators.

Each campaign has twelve blocks with one hundred frames per variant. Cases
alternate after each frame in AB/BA order on the same device and prepared
resources. Variant switching occurs outside the measured registration call;
all pipelines are built beforehand. The arrays of parameters, statuses and
iterations match; interior coverage is 100%, with the unchanged analytical
error gates. Full input/source hashes, configurations, versions, raw timings
and summaries are retained.

The original JSON byte streams are compressed without reformatting in
`repeatability.json.gz` and `candidate.json.gz`; `summary.json` is the short
readable result. Load the raw streams using standard-library `gzip` and `json`.
This avoids thousands of extra formatted sample lines in the active docs.

Reproduction commands are recorded in those reports. The standalone
`benchmark_spatial_chain.py --one-device --interleave` driver exposes the
control-only and candidate-first switches. Output directories must be new.
The browser shares the optimization's WGSL but these timings are native
research-supervisor measurements, not PWA FPS. Separate browser UHD/NPZ
functional checks verify integration.
