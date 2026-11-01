# Browser TIFF input qualification

Implementation: `4322e22d0cdb4d66c33cf13f8dc83de3a2113846`.
The normal-adapter report comes from the preceding dirty build with identical
runtime bytes (verified in `verification.json`). The forced-SwiftShader report
uses the clean committed build and passes the same controls. Original analytic fixtures and
all numerical Python/core/WGSL sources are unchanged. The Python source tree is
`c691f0be12a0a0e4d93720628d486321c4f0bfb0`.

## Scope and outcomes

- Original 3840 × 2160 plate-hole reference/frame-095 PNGs are encoded as
  temporary TIFFs, never substituted for the archived fixtures. Eight-bit LZW
  and unsigned 16-bit Deflate samples (`PNG value * 257`) decode to exactly
  the original RGBA pixels. Input hashes are checked independently in Python.
- Actual TIFF file pickers, reference preparation, Solve and export return
  exactly the PNG baseline's status and parameter arrays for both depths.
- A UHD PackBits binary TIFF mask matches the independently constructed mask.
- Dimension errors, missing Workers, cancellation and corrupt Deflate input
  produce explicit diagnostics. Corrupt input tests include pako's string error.
- After actual service-worker activation, the origin server is stopped. A new
  TIFF Worker still imports the local decoder and returns the correct UHD hash.
- Portable pixel-oracle tests cover all four compression families, both byte
  orders, horizontal prediction, grayscale/white-is-zero, binary, RGB/RGBA,
  palette, 16-bit RGB and all eight orientations. Float, stacks and truncation
  are explicitly rejected. These are decoding/functional checks, not TIFF FPS.
- The local full suite passes 287 tests with native GPU and browser tests
  enabled, no skips. Ruff/format, Node syntax, strict Sphinx and repository checks
  pass. A software-browser CI step is configured; no new remote CI result is
  claimed. The numerical solver and Python wheel contents are unchanged.

Sixteen-bit TIFF admission still maps each unsigned sample with
`floor(sample/256)`, fixed across the sequence. It does not preserve the sensor's
low bits or provide a 16-bit solver. Multipage/BigTIFF and unsupported layouts
are refused. See the browser input guide for limits and conversion semantics.

## Reproduce

```sh
python3 tools/build_pages.py --output /tmp/live-dic-tiff-site
python3 tools/qualify_browser_tiff.py --site /tmp/live-dic-tiff-site \
  --output /tmp/tiff-hardware
python3 tools/qualify_browser_tiff.py --site /tmp/live-dic-tiff-site \
  --output /tmp/tiff-software --software
ICGN_RUN_GPU_TESTS=1 ICGN_RUN_BROWSER_TESTS=1 python3 -m unittest discover \
  -s DIC/LucasKanadeICGN/python/tests -v
```

Reports use deterministic gzip. The manifest hashes their uncompressed bytes.
Pinned upstream source hashes, npm integrity values and licences are in
`web-app/vendor/tiff-provenance.json` and the third-party notices.
