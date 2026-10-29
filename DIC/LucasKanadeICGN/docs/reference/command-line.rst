Command-line reference
======================

``icgn-webgl`` is the browser-control command used for compatibility,
qualification, and offline file processing. DIC-RT's primary deployment is a
Python supervisor using ``ICGNWGPUFullPyramid`` for resident NumPy streams.
The command-line default must therefore be read as the default *browser*
backend, not as the recommended 4K60 Python data plane. See
:doc:`../explanation/supervision`.

Synopsis
--------

.. code-block:: text

   icgn-webgl [options] reference target [target ...]

Positional arguments
--------------------

``reference``
   One reference image. Its dimensions define the subset grid for every
   backend; GPU-IO requires both dimensions to be multiples of the selected
   subset size.

``targets``
   One or more target images with the same dimensions as the reference.
   ``--benchmark`` requires exactly one.

Solver options
--------------

``--iterations N``
   Maximum IC-GN iterations, from 1 to 30. Default: 30.

``--tolerance PX``
   Positive corner-update convergence threshold in pixels. When omitted, the
   selected backend applies its default: ``1e-3`` for single-scale execution
   and ``5e-3`` at the fine level of the default full pyramid.

``--zncc-min VALUE``
   Minimum accepted ZNCC in ``[0, 1]``. Default: ``0.8``.

``--interpolation {bspline,linear,cubic}``
   Select historical Gaussian/B-spline reconstruction, direct bilinear
   interpolation, or direct Keys cubic convolution. Default: ``bspline``.

``--motion-model {affine,translation,quadratic}``
   Select the default six-parameter affine model, the displacement-only
   translation model with an analytic 2 by 2 Hessian inverse, or the
   twelve-parameter second-order model. Quadratic mode adds the three
   second-order Lagrange terms for each displacement component and is supported
   by ``webgpu-wgsl``; the independent Three.js and deprecated GPU-IO browser
   backends reject it. Default: ``affine``.

``--outlier-policy {replace,flag,disabled}``
   Control the displacement UOD post-pass. ``replace`` detects inconsistent
   ``tx`` or ``ty`` and replaces the complete active motion state. ``flag``
   preserves raw coefficients and sets status 2. ``disabled`` omits the pass.
   Default: ``replace``.

``--subset-size {8,15,16,17,31,32,33}``
   Select the square numerical subset width. Default: ``16``.

``--subset-stride PIXELS``
   Select the distance between adjacent subset centres. It must be an integer
   from 1 through ``--subset-size`` and defaults to the subset size. A smaller
   value creates overlapping subsets and currently requires
   ``--backend webgpu-wgsl``.

``--pyramid-levels {1,2,3}``
   Full-pyramid depth for ``webgpu-wgsl``. The backend default is two levels;
   one selects the original single-scale path.

``--iterations-per-level COUNT [COUNT ...]``
   Coarse-to-fine iteration ceilings for ``webgpu-wgsl``. Supply exactly one
   integer from 1 to 30 per pyramid level.

``--tolerance-per-level PX [PX ...]``
   Coarse-to-fine positive convergence thresholds for ``webgpu-wgsl``. Supply
   exactly one value per pyramid level.

``--truth DX DY``
   Constant true translation used to report vector RMSE.

Benchmark and output options
----------------------------

``--benchmark``
   Run the repeated target prefilter and solver benchmark.

``--repeats N``
   Number of measured runs. Default: 30.

``--warmups N``
   Number of unreported warm-up runs. Default: 3.

``--output PATH``
   Write complete numeric arrays and timing samples to a compressed ``.npz``
   archive.

Runtime options
---------------

``--backend {three-tsl,webgpu-wgsl,gpuio}``
   Select the optimized Three.js WebGPU path, the canonical shared WGSL in raw
   browser WebGPU, or the deprecated historical GPU-IO path. Default:
   ``three-tsl``. Use ``gpuio`` only to reproduce or compare legacy results.

``--force-three-webgl``
   Force ``three-tsl`` to use its serial WebGL2 transform-feedback solver. Use
   this for compatibility qualification on a WebGPU-capable machine.

``--repo-root PATH``
   Override automatic repository discovery.

``--chromium PATH``
   Select a Chromium or Chrome executable.

``--startup-timeout SECONDS``
   Browser startup timeout. Default: 20 seconds.

``--call-timeout SECONDS``
   Timeout for one browser-control or binary-registration operation. Default:
   120 seconds.

``--enable-chromium-sandbox``
   Do not pass ``--no-sandbox`` to Chromium.

``--chromium-arg ARG``
   Add one Chromium argument. Repeat the option to add multiple arguments.

Output
------

Standard output is JSON containing the device name, timer availability,
configuration, accepted and UOD-flagged counts, the legacy ``replaced`` alias,
mean translation, iteration
statistics, and timing summary. Reference construction is reported separately
under ``reference_preparation``. Errors are written to standard error and
return exit status 2.
