# Full-resolution images beyond default browser limits

The implementation revision and source hashes are in `verification.json`.
The hardware report was collected from a dirty assembled build with identical
committed runtime files. This is functional/accuracy evidence, not a timing
campaign or a claim of 60 FPS at these larger dimensions.

## Cause and scope

On the local Intel xe-2lpg Chromium adapter, the advertised storage binding and
buffer limits are 4,294,967,292 bytes, and the texture dimension limit is 16384.
The original device request enabled eight storage bindings but left device
limits at 128 MiB per binding, 256 MiB per buffer and 8192 per texture side.
A 4112 × 3008 reference needs 197,902,336 bytes, hence the artificial rejection.

Only required buffer/texture limit increases are now requested. UHD retains
its exact prior request. Metadata/inverse Hessians for overlapping subsets are
included; injected devices remain caller-owned. Unknown-size external images
can replace an owning device at cold preparation. No resizing or per-image
live-loop negotiation was introduced. Native Python and WGSL equations are
unchanged. Real hardware/memory limits still apply; automatic spatial tiling
is not implemented.

## Measured correctness

A transient analytic grayscale field contains a known (2, -1) px translation.
There is no resampling or tiled/cropped truth. The qualifier opens actual
TIFF/PNG files through the PWA, prepares reference, runs Solve and exports the
complete field. Interior excludes the outermost grid ring, following the
solver's existing interpolation margin. Gates: >90% coverage, <0.05 px RMSE.

| Image | Input | Interior coverage | Translation RMSE |
| --- | --- | ---: | ---: |
| 4112 × 3008 | TIFF | 100% | 0.00100966 px |
| 4113 × 3009 | PNG | 100% | 0.00100967 px |
| 9001 × 1201 | PNG | 100% | 0.00101594 px |

A 64² → 4112 × 3008 external-reference change replaces an owning device and
raises its binding limit from 134,217,728 to 197,902,336 bytes. Injected-device
refusal/ownership is covered by executable GPU tests. All 289 local tests pass
with GPU/browser flags; the eight core tests also pass on SwiftShader. No new
remote CI execution is claimed.

The wheel is pure `py3-none-any`; its 47 packaged files match source. A fresh
Python 3.11 environment outside the checkout imports that installed wheel and
runs the raw browser adapter at 4112 × 3008: 47,430 accepted identity subsets,
finite parameters and 6.8964e-6 px displacement RMSE. The complete grid is
257 × 188. Ruff/format, Node syntax, repository checks and strict Sphinx pass.

## Reproduce

```sh
python3 tools/build_pages.py --output /tmp/live-dic-large-images
python3 tools/qualify_browser_image_sizes.py --site /tmp/live-dic-large-images \
  --output /tmp/large-images-hardware
ICGN_RUN_GPU_TESTS=1 ICGN_RUN_BROWSER_TESTS=1 python3 -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests -v
ICGN_RUN_BROWSER_TESTS=1 ICGN_BROWSER_SOFTWARE=1 python3 -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests -p test_browser_core.py -v
```

JSON reports are deterministically gzipped; their uncompressed hashes/lengths
are recorded in the manifest. Existing analytic UHD fixtures and prior reports
are preserved.
