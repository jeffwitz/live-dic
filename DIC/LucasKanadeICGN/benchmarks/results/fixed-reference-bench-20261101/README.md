# Fixed-first-reference image-to-field qualification

This campaign predates the sparse live-vector renderer. It compares baseline
`388cc5e` with `e781957`, including the earlier bounded-recording/preview work.
Do not attribute its FPS to the later renderer. The full `python/src` source
tree is identical at both revisions (`c691f0be12a0a0e4d93720628d486321c4f0bfb0`).
`sites.json` retains the frozen site/clock-probe audit. Timing instrumentation
adds no GPU completion fence. All original UHD image hashes remain unchanged.

## Configuration and boundaries

Intel Lunar Lake 8086:64a0, Mesa 25.2.8, Linux 7.0.0-38, performance power
profile, Chromium 155, Python 3.11. Exact runtime/device information is in each
raw report. Affine 16/16, B-spline, median, uniform, two fixed levels 8/16 GN,
0.005 tolerance, zero initialization, no temporal prediction/correction/ROI.
I0 is prepared once. No tests/builds ran during timing.

Native/Python-browser: resident original UHD NumPy reference/frame040 targets
to full NumPy field/statistics, 20 warmups and 300 measurements per process.
The browser includes binary loopback transport. These paths contain no camera,
video decoding, display or database. The 32-pixel interior population has
30,916 selected centres; minimum coverage 99.9806%, RMSE <0.024203 px,
p95 <0.04555 px and gradient RMSE <0.001065/component in every run.

| Path / version | Median ms range | p95 ms range | Sustained FPS range |
|---|---:|---:|---:|
| Native baseline, one slot (4) | 11.347–11.676 | 13.160–13.918 | 86.48–87.33 |
| Native candidate, one slot (2) | 11.476–11.652 | 13.077–13.533 | 86.65–87.84 |
| Python browser baseline (4) | 40.273–40.869 | 45.552–47.120 | 24.30–24.71 |
| Python browser candidate (2) | 40.143–41.931 | 45.056–49.359 | 23.51–24.87 |

Native passes the predeclared baseline-only variability gate. The primary
Python-browser comparison remains **inconclusive**: one slower candidate,
one faster. The predeclared A/B/B/A follow-up retains this classification;
it does not reproduce the isolated slowdown (candidate medians 39.289/39.290 ms,
controls 39.170/39.566 ms). No accuracy tolerance was changed.

Two candidate native two-slot controls give 90.04–92.94 FPS but median result
age 21.49–21.83 ms. Their higher throughput is not lower latency.

## PWA acquisition/recording controls

A static **simulated** 60 Hz UHD camera uses the original reference image.
Identity truth differs from the moving frame040 accuracy case above. This is
not a physical-camera or changing-motion FPS qualification. Each run has
4 seconds warmup and 20 seconds observation, recording off then on. Fields
remain native-sized. All recording cases finish with zero unsaved fields.

| Configuration | Computed FPS, recording off | Committed FPS, recording on |
|---|---:|---:|
| Baseline, every-field native scalar raster (4) | 20.87–22.62 | 11.67–13.04 |
| Candidate, same defaults (2) | 21.06–21.52 | 11.64–12.53 |
| Candidate queue + 1 Hz / 1920 px preview (2) | 35.26–37.31 | 28.84–29.35 |

The two preview controls actually present about 1 Hz. They demonstrate useful
application headroom, not 4K60. Committed throughput and Worker-arrival-to-cache
plus IndexedDB admission latency are separate from computed replies, GPU-only
time and sensor exposure age. The explicit queue is bounded at two CPU slots.

## Audit and reproduction

`criteria.json` and `followup-criteria.json` precede their respective runs.
`run_campaign.py`, `run_followup.py` and `prepare_sites.py` retain exact orchestration;
their temporary paths document the local frozen checkouts/sites and must be
recreated before rerunning. Raw commands, input hashes, versions, per-frame
latencies/quality, iteration histograms, stderr and final drain state are in
`raw/`. Gzip is lossless; `raw-manifest.json` records uncompressed SHA-256 and
size. `tests-before.log` records the 280-test preflight.

Run `python3 summarize.py` to recompute `summary.json` from the raw records.
It asserts existing numerical gates and zero unsaved recording fields; the
primary timing decisions use only the four reference repeats for variability.
Short runs are neither thermal statistics nor a hard real-time guarantee.
