# Continuous UHD result playback

The implementation is `7318989272a83d34ab4d2cd867f38f6624b64e19`.
Both clean-build workflow reports record that revision: the normal browser
adapter and explicitly forced SwiftShader. Both pass the same UHD assertions. The workbench and
no-Worker controls use the same runtime files from the immediately preceding
dirty build; `verification.json` hashes the four changed production resources.
All numerical Python/core/WGSL sources retain tree
`c691f0be12a0a0e4d93720628d486321c4f0bfb0`.

These are functional checks, not FPS, storage-throughput or latency campaigns.
The images are the original 3840 × 2160 plate-hole reference/frame-095/frame-225
PNGs. Their input hashes and canonical shader hashes remain in each build report.

## Checks

- Three saved UHD fields: exact raster SHA-256, dimensions and opacity are
  unchanged during a held disk load and during a held analysis Worker request.
  A nearby status identifies the retained field; stale exports stay disabled.
- Revisiting the three analyses increments cache hits without incrementing
  `analysisRuns`. A different recipe is calculated; restoring a cached recipe
  reuses its arrays. Slider seeks retain one active and one latest request.
- Raw parameter/status/iteration hashes are unchanged after all navigation.
  Only three target correlations ran, during Solve.
- Region statistics use 16 px text below the field and 14 px on the image.
  The controlled rectangle contains 1,144 valid subset samples.
- The broader workbench check passes eight UHD fields, NPZ/project interchange,
  recipe restoration and independent NumPy operators (maximum error
  1.940721888749053e-17).
- Review without WebGPU or Worker returns 31,474 valid samples and performs
  zero correlations.
- The full local suite passes 284 tests with GPU/browser flags and no skips.
  Ruff, JavaScript syntax, strict Sphinx and repository checks pass. This does
  not report a new remote CI run or qualify another OS/hardware family.
- The analysis LRU has both an eight-entry and 32 MiB typed-buffer limit;
  its CPU test checks eviction, recipe identity, clearing and exclusion of raw
  fields. This additional allocation is separate from raw-field storage.

## Reproduction

From the repository root, with the documented Python environment:

```sh
python3 tools/build_pages.py --output /tmp/live-dic-results-cache-site
python3 tools/qualify_browser_workflow.py \
  --site /tmp/live-dic-results-cache-site --output /tmp/results-cache-hardware
python3 tools/qualify_browser_workflow.py \
  --site /tmp/live-dic-results-cache-site --output /tmp/results-cache-software --software
python3 tools/qualify_browser_workbench.py \
  --site /tmp/live-dic-results-cache-site --output /tmp/results-cache-workbench
python3 tools/qualify_browser_review.py \
  --site /tmp/live-dic-results-cache-site \
  --field /tmp/results-cache-workbench/processed-field.npz \
  --output /tmp/results-cache-review --no-worker
ICGN_RUN_GPU_TESTS=1 ICGN_RUN_BROWSER_TESTS=1 python3 -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests -v
```

Reports use deterministic gzip; `manifest.json` records the uncompressed
SHA-256 and length. `results.png` is the actual Chromium rendering, including
the enlarged region summary. Browser storage contains fields, not source images.
