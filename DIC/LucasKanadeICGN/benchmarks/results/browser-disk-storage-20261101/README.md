# Browser field storage qualification — 2026-11-01

The final hardware and software reports both use clean commit `3bd4f4d89d885eba126ac9cefdf572995a378188`.
The six canonical shader hashes and original UHD plate image hashes are retained
in each build record. No numerical shader/default changed during this work.
This is storage/GUI correctness evidence, **not a recording-FPS qualification**.

| Case | Committed fields | Field payload | Resident field cache |
|---|---:|---:|---:|
| Hardware: distinct snapshots of one actual UHD plate result | 200 | 211.84 MiB | 127.86 MiB / 128 MiB |
| SwiftShader: same storage protocol | 200 | 211.84 MiB | 127.85 MiB / 128 MiB |
| Hardware: actually decoded/solved UHD video | 1070 | 1057.98 MiB | 1023.96 MiB / 1024 MiB |

The final video cache contains 1033 fields: older fields remain
committed on browser disk. Cache accounting includes auxiliary arrays such as
`roiSelected`, not just the six parameters, status and iterations. It excludes
catalog/object overhead, GPU/decoder allocations and explicit analysis/export
scratch; it is not a whole-browser RSS limit.

## What was exercised

- Actual IndexedDB transactions, typed arrays, lossless eviction and exact SHA-256
  reload checks on first/middle/last UHD fields after a page reload.
- Lazy read-only session reopening, saved recipes/views, and 200-field history
  extraction with a 16 MiB cache and unchanged values/order.
- Four actual multipart download Blobs, every NPY read by NumPy with
  `allow_pickle=False`, exact raw-array/mask hashes, complete unique field IDs,
  and reimport through the real file picker.
- A failed write retains the completed field in RAM, pauses acquisition and
  immediately disables recording/capture until recovery. The pre-fix failing
  control is retained as `negative-control.json.gz` (commit `0f6d4d9`).
- Interrupted recovery batches commit two fields, retain the other ten and finish
  without duplicates. A deleted session rejects settings saves. Disabled disk
  storage retains unsaved fields and stops at the RAM budget.
- GPU API resource reports before/after snapshot recording remain equal. Final
  video resources are recorded separately. These are API sizes, not physical/free
  VRAM or proof of unified/discrete placement.

The snapshot test explicitly labels replay copies; it does not claim 200 new
GN solves. The optional real-video test uses the unchanged temporary native-size
VP9 movie documented in [the video input campaign](../browser-uhd-video-20261031/README.md),
loops it and performs real frame admissions/solves. Its SHA-256 is retained in
`hardware.json.gz`. Repeated media timestamps are not a physical acquisition
clock or a new deformation-accuracy campaign. The movie is not committed; the
original checked-in PNG fixtures remain authoritative.

## Other validation and scope

The complete native/browser-enabled suite passed **259 tests** on the integration
candidate (`suite.log.gz`); **14 final cache contracts** then passed after the
focused recovery/accounting follow-ups (`cache-tests.log.gz`). This is not a
claim that the complete suite was rediscovered with 14 cache tests. The actual
UHD workbench passed with eight fields and independent NumPy operator error
`1.940721888749053e-17`. Ruff, JavaScript syntax and strict Sphinx passed.

A fresh pure-Python wheel was installed outside the checkout. Its packaged core,
including `gpu-memory.js`, ran a real browser registration/memory report; native
ROI registration also passed (`installed-wheel.json.gz`). No runtime/package
source changed after that wheel check. The standalone UI is not part of the wheel.

Hardware browser identity: {"vendor": "intel", "architecture": "xe-2lpg", "device": "", "description": ""}.
Both tests ran on Linux/Chromium. SwiftShader validates a software GPU path,
not discrete hardware. Windows/macOS/mobile/discrete-GPU execution has not been
qualified here. A separate software-browser CI job is configured; no new remote
CI execution or Pages deployment is claimed in these records.

Two supplementary shell runs were interrupted by SIGTERM and are **not counted
as passes**. The hardware browser was still processing with 952 committed fields,
no unsaved fields and no reported app error; those isolated test browsers were
closed. Both final reports above were rerun to normal zero-exit completion.

## Reproduction

```bash
python3 tools/build_pages.py --output /tmp/live-dic-storage-site
DIC/LucasKanadeICGN/python/.venv/bin/python tools/qualify_browser_storage.py \
  --site /tmp/live-dic-storage-site --output /tmp/live-dic-storage-software --software
DIC/LucasKanadeICGN/python/.venv/bin/python tools/qualify_browser_storage.py \
  --site /tmp/live-dic-storage-site --output /tmp/live-dic-storage-hardware \
  --video /path/visible/to/chromium/uhd-six-seconds.webm
```

The optional movie must retain native 3840 × 2160 dimensions; Snap Chromium must
be able to read its path. The default qualification needs only checked-in images.
Raw reports/logs are losslessly gzipped; `files.json` gives their **uncompressed**
SHA-256 and length. Exported stress archives are described by manifests/hashes
in the reports, rather than committing hundreds of MiB of redundant field copies.
