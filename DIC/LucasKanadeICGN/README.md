# Affine Lucas-Kanade IC-GN for GPU-IO

This implementation ports the affine inverse-compositional Gauss-Newton
(IC-GN) registration core from the SYCL prototype in
`DIC_CODEC/DIC_FROM_H264` to the repository's WebGL2/GPU-IO framework. It runs
on offline test images and does not require a camera or an H.264 pipeline.

The implementation retains the main algorithmic choices of the SYCL version:

- one six-parameter affine transformation per subset;
- reference-image gradients and a reference-dependent Hessian;
- inverse-compositional warp updates;
- convergence based on the maximum update at the four subset corners;
- normalized correlation to reduce sensitivity to affine brightness changes;
- Gaussian filtering and cubic B-spline coefficients;
- one constant inverse Hessian prepared with the reference image;
- ZNCC rejection and median replacement of spatial outliers.

WebGL2 requires a different memory layout from C++/SYCL. Translation and the
four linear affine terms are stored in two floating-point textures. Reference
quantities (`F`, gradients, `H^-1`, `J^T F`, and `sum(J)`) are prepared once on
the CPU and uploaded to three GPU textures. Each target is filtered on the GPU
with the same radius-15 FIR as the optimized SYCL configuration. The IC-GN
shader accumulates ZNSSD statistics and `J^T g` in one pass before applying the
inverse-compositional update.

The remaining differences are platform constraints: WebGL2 uses single
precision and exposes neither SYCL subgroups nor the same register resources.
This version does not implement Aitken acceleration, predictive tolerance,
temporal/extrapolated initialization, a pyramid, or the H.264 pipeline.

## Test dataset

The five PNG files in `assets/` were extracted directly from the raw YUV file
`validation_inputs/seed2026101102_translation/master.yuv` in the source
repository. The selected frames are `t = 0, 1, 5, 10, 20`. The analytical
motion is `(+0.2, -0.1)` pixel per frame, producing displacements up to
`(+4, -2)` pixels. No H.264 stream is used when loading or preparing these
images. [`assets/source.json`](./assets/source.json) records the source path,
hashes, format, and motion parameters.

## Headless Python interface

The [`python/`](./python/) package is the primary programmatic interface. It
provides:

- a persistent engine for image sequences;
- a one-shot function for a single image pair;
- complete affine fields, status, and iteration arrays as NumPy arrays;
- GPU and wall-clock benchmarking;
- `.npz` export and the `icgn-webgl` command-line tool;
- compatibility with scripts and Jupyter notebooks.

Chromium is used only as an invisible WebGL2 runtime. The headless entry point
does not load Tweakpane, attach the canvas to the DOM, or render a visible UI.
See [`python/README.md`](./python/README.md) for installation and API examples.

## Optional browser demo

Serve the repository root over HTTP and open:

```text
/DIC/LucasKanadeICGN/
```

The page loads `t = 0` as the reference, evaluates all four targets, and runs a
synchronous benchmark. Its controls can select another target or rerun the
campaign. Timings use `gl.finish()` or WebGL timer queries, so GPU completion is
observed without including video decoding.

Default parameters match the SYCL geometry: a 16-pixel grid spacing, 16 x 16
subsets centered at `(8 + 16i, 8 + 16j)`, a `1e-3` pixel tolerance, a minimum
ZNCC of `0.8`, and at most 30 iterations with per-subset early convergence.

Measured accuracy and performance are recorded in
[`BENCHMARK.md`](./BENCHMARK.md).

## Documentation

The Sphinx documentation follows the Diátaxis structure and focuses on using
the Python implementation for real-time DIC. It starts at
[`docs/index.rst`](./docs/index.rst) and includes tutorials, operational
how-to guides, API reference, and explanations of IC-GN and latency boundaries.

Build it locally with:

```bash
python3 -m sphinx -W --keep-going -b html \
  DIC/LucasKanadeICGN/docs \
  DIC/LucasKanadeICGN/docs/_build/html
```
