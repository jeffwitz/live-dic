# First milestone: one prepared first-image reference

This is a functional qualification, **not an FPS campaign**. The owner chose
one persistent first image as the initial milestone: prepare I0 once, retain
its data, and compute I0 -> In for every later target. Prepared-data storage
and saving computed fields to disk are separate operations.

The unchanged original UHD tensile fixtures are used in the sequence
reference, frame 40, reference, frame 40. The reference is never replaced during
the sequence. Native `ICGNWGPUFullPyramid` and the canonical browser `DICEngine`
use affine 16/16, B-spline, two levels, median and their ordinary zero-initialized
defaults. Browser CPU preparation and opt-in GPU preparation are tested
separately. No temporal-history or consecutive-reference substitution is used.

For both browser preparation paths, `setReference` executes exactly once.
SHA-256 hashes of the complete packed reference metadata/Hessian buffers at
both pyramid levels are identical before and after all four targets. All
twelve computed fields (four native, eight browser) are saved to temporary NPZ
files and reread with `allow_pickle=False`; parameters, statuses and iterations
match exactly. The temporary field files are removed after verification; this
archive preserves their qualification report and original input hashes.

The native/browser frame-40 controls retain 99.9806% interior coverage,
approximately 0.02420 px displacement RMSE, 0.04554 px displacement p95 and
0.0010643 gradient RMSE per component. Identity controls have full interior
coverage and displacement RMSE below 1e-6 px. These satisfy the existing fixture
limits, without adjustment. The repeated targets reproduce the same metrics.
Frames 100/199 are not assigned a new two-level convergence guarantee.

The browser regression in `python/tests/test_gpu_reference.py` additionally
exercises translation and affine with both preparation paths on small cases.
It checks a valid target, an unusable flat target and recovery against the same
reference: exactly one preparation, zero reference-preparation GPU passes or
new buffers/textures/pipelines during registration, unchanged buffer identities
and bitwise unchanged metadata/Hessians. This is a numerical/resource contract,
not a timing threshold.

`full-tests.log.gz` records 280 local tests passing with both
`ICGN_RUN_GPU_TESTS=1` and `ICGN_RUN_BROWSER_TESTS=1`, without skips. Ruff and
strict Sphinx checks pass. No new remote CI execution is asserted.

`report.json.gz` is losslessly compressed; `manifest.json` records its
uncompressed hash and canonical browser-source hashes. Production code is at
`1fa909d`; the assembled browser site was built at `1baf913`, whose canonical
core files match byte-for-byte. Pending changes were tests and documentation.
`qualify_fixed_first.py` is the exact local qualification script. To reproduce,
from the repository root with normal Python development dependencies:

```bash
python tools/build_pages.py --output /tmp/live-dic-cadence-site
python DIC/LucasKanadeICGN/benchmarks/results/fixed-first-reference-20261101/qualify_fixed_first.py
```

This creates a temporary static site and writes a new report under `/tmp`;
original images and this archive remain unchanged. Intel Lunar Lake hardware,
Linux/Vulkan, Chromium 155; no physical camera, other OS or new remote CI result
is claimed. Fixed-reference calculation/recording throughput still requires its
own complete-application measurement. Keeping a reference removes the per-pair
preparation cost; it does not make saving/decoding/rendering free.
