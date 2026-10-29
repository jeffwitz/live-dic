Backend selection and defaults
==============================

There is no single process-wide backend default.  The default depends on the
public engine class selected by the application.  This distinction matters
because the browser API, the low-latency native API, and the native
coarse-to-fine API intentionally expose different execution graphs.

The project itself does have a primary supervision model: Python owns the
measurement workflow. For NumPy input and NumPy fields, native wgpu is the
recommended data plane. A browser backend can be controlled from Python for
compatibility and qualification, but competitive browser-native streaming
requires a JavaScript supervisor that keeps the per-frame path inside
Chromium. See :doc:`supervision`.

Current recommendation
----------------------

.. list-table:: Backend policy
   :header-rows: 1
   :widths: 20 26 18 36

   * - Use case
     - Construction
     - Default pyramid
     - Policy
   * - Native NumPy coarse-to-fine registration
     - ``ICGNWGPUFullPyramid()``
     - two full-pyramid levels
     - Recommended product path for Python supervision, the validated
       displacement range, bounded NumPy streams, and the 4K60 compute
       objective.
   * - Native single-scale or selective processing
     - ``ICGNWGPU()``
     - one scale
     - Use for automatic subgroup tuning or a one-scale graph. Its optional
       selective recovery is a compatibility/exception mode, not the
       recommended normal pyramid. It also retains its bounded stream API.
   * - Python-controlled browser compatibility
     - ``ICGNWebGL()``
     - one scale
     - Selects Three.js/WebGPU and falls back to Three.js/WebGL2. Useful for
       compatibility and qualification, but not the recommended low-latency
       NumPy data plane.
   * - Python-controlled canonical WGSL in Chromium
     - ``ICGNWebGL(backend="webgpu-wgsl")``
     - two full-pyramid levels
     - Use when browser/native source identity is more important than the
       specialized Three.js subgroup path. The process-copy cost remains.
   * - Historical GPU-IO reproduction
     - ``ICGNWebGL(backend="gpuio")``
     - one scale
     - Deprecated; retained only for regression and historical comparison.

The phrase *two-level default* therefore applies to
``ICGNWGPUFullPyramid`` and ``webgpu-wgsl``.  The phrase *one-scale default*
applies only to ``ICGNWGPU`` and Three.js/TSL.  It must not be interpreted as
a recommendation to use selective recovery instead of the full-pyramid
backend.

Two different native pyramid strategies
---------------------------------------

``ICGNWGPUFullPyramid`` executes every configured level for every frame.  It
uploads the finest target once, constructs reduced targets on the GPU,
prefilters and solves from coarse to fine, transfers affine seeds on the GPU,
and reads back only the final fine field.  Two levels with the ``(8, 16)``
coarse-to-fine iteration ceilings are the current native recommendation.

``ICGNWGPU(pyramid_levels=2 or 3)`` is older selective recovery.  It first
runs the fine single-scale solve, detects failures on the host, constructs
reduced targets on the CPU, solves only the coarse dependency closure, and
retries failed fine subsets.  It preserves already accepted fine results, but
its host orchestration and conditional transfers make it unsuitable as the
normal high-throughput pyramid.  It remains useful for compatibility and for
experiments in which large motion is genuinely exceptional.

.. code-block:: text

   recommended full pyramid                 optional selective recovery

   upload fine target                       fine solve first
          |                                        |
   GPU reductions + prefilters              failures present?
          |                                   /          \
   coarse solve -> GPU seed transfer        no           yes
          |                                  |            |
   fine solve -> one readback              return    CPU reduction +
                                                        coarse retry

Execution mappings are independent of pyramid depth
---------------------------------------------------

Three separate sizes appear in the implementation and should not be
conflated:

* a numerical subset is 8 x 8, 16 x 16, or 32 x 32 pixels, with 16 x 16 as
  the default and published-performance geometry;
* the shared-WGSL FIR uses 32 x 8 horizontal and 8 x 32 vertical output
  tiles, each executed by 256 threads;
* the portable IC-GN solver uses one 16-lane workgroup per subset, while the
  specialized Three.js path fills the measured hardware subgroup (32 lanes on
  the validation GPU).

The full-pyramid reduction passes use 16 x 16 dispatch groups, and affine seed
transfer and median filtering use 256-thread one-dimensional groups. These
dispatch dimensions are scheduling choices and do not change with
``subset_size``. See :doc:`subset-size` for the numerical geometry.
See :doc:`gpu-optimization` for the detailed memory-access diagrams.
