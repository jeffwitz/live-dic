Command-line reference
======================

``icgn-webgl`` is the browser-control command used for compatibility,
qualification, and offline file processing. live-dic's primary deployment is a
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
   backend.

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
   by ``webgpu-wgsl``; the independent Three.js browser backend rejects it.
   Default: ``affine``.

``--outlier-policy {replace,flag,disabled,median}``
   Control the displacement UOD post-pass. ``replace`` detects inconsistent
   ``tx`` or ``ty`` and replaces the complete active motion state. ``flag``
   preserves raw coefficients and sets status 2. ``disabled`` omits the pass.
   ``median`` bypasses UOD and repairs rejected subsets from the component-wise
   median of valid neighbours. Coarse repairs can seed the next finer pyramid
   level; their status remains rejected until a local solve is accepted.
   Accepted subsets are unchanged. Default: ``replace``.

``--median-filter-size {3,5,7,9}``
   Width of the square median neighbourhood in subset-grid cells. Default:
   ``3``. Values above 3 require ``--outlier-policy median`` and
   ``--backend webgpu-wgsl``. This selects the neighbour stencil for rejected-
   subset seed repair; it is separate from post-processing applied to saved
   fields.

``--subset-size PIXELS``
   Select a square numerical subset width from 4 through 64. Default: ``16``.
   Arbitrary widths require ``--backend webgpu-wgsl``; Three.js accepts only
   8, 16, or 32.

``--subset-stride PIXELS``
   Select the distance between adjacent subset centres. It must be an integer
   from 1 through ``--subset-size`` and defaults to the subset size. A smaller
   value creates overlapping subsets and currently requires
   ``--backend webgpu-wgsl``.

``--initialization {zero,fourier,global-affine}``
   Opt-in global initialization for ``webgpu-wgsl`` only. Fourier estimates a
   rigid translation; ``global-affine`` fits masked affine motion over
   five CPU scales by default, independently of the selected local model.
   With ``--motion-model translation``, the fit provides per-subset translation
   seeds while the local solver estimates only two displacement parameters. The
   ``--global-affine-levels`` option accepts 5–8 scales. The default is zero
   initialization. See :doc:`../how-to/register-large-shifts`.

``--global-affine-levels {5,6,7,8}``
   Number of coarse-to-fine CPU scales for ``global-affine`` initialization.
   Default: ``5``. Six to eight are experimental settings for large shifts and
   require sufficient image dimensions and masked samples.

``--pyramid-levels {1,2,3,4,5,6}``
   Full-pyramid depth for ``webgpu-wgsl``. Five and six levels are experimental
   large-motion options. The backend default is two levels;
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
   archive. The scalar ``config_json`` entry records the algorithm
   configuration, including the applied outlier policy.

Runtime options
---------------

``--backend {three-tsl,webgpu-wgsl}``
   Select the Three.js/TSL WebGPU implementation or the canonical shared WGSL
   browser implementation. Both require WebGPU. Default: ``three-tsl``.

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
