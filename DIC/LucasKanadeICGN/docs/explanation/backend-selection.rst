Backend selection and defaults
==============================

There is no single process-wide backend default.  The default depends on the
public engine class selected by the application.  This distinction matters
because the browser API, the low-latency native API, and the native
coarse-to-fine API intentionally expose different execution graphs.

Current recommendation
----------------------

.. list-table:: Backend policy
   :header-rows: 1
   :widths: 20 26 18 36

   * - Use case
     - Construction
     - Default pyramid
     - Policy
   * - Browser application with automatic compatibility
     - ``ICGNWebGL()``
     - one scale
     - Recommended browser entry point.  It selects Three.js/WebGPU and falls
       back to Three.js/WebGL2.
   * - Browser execution of the canonical shared WGSL
     - ``ICGNWebGL(backend="webgpu-wgsl")``
     - two full-pyramid levels
     - Recommended when browser/native source identity is more important than
       the specialized Three.js subgroup path.
   * - Native NumPy coarse-to-fine registration
     - ``ICGNWGPUFullPyramid()``
     - two full-pyramid levels
     - Recommended native path for the validated displacement range and 4K60
       compute objective.
   * - Native single-scale or bounded sequence processing
     - ``ICGNWGPU()``
     - one scale
     - Use for the lowest isolated-frame latency or ``register_stream``.  Its
       optional selective recovery is a compatibility/exception mode, not the
       recommended normal pyramid.
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
and reads back only the final fine field.  Two levels with the ``(12, 30)``
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

* a numerical subset is always 16 x 16 pixels;
* the shared-WGSL FIR uses 32 x 8 horizontal and 8 x 32 vertical output
  tiles, each executed by 256 threads;
* the portable IC-GN solver uses one 16-lane workgroup per subset, while the
  specialized Three.js path fills the measured hardware subgroup (32 lanes on
  the validation GPU).

The full-pyramid reduction passes use 16 x 16 dispatch groups, and affine seed
transfer and median filtering use 256-thread one-dimensional groups.  These
are scheduling choices for different kernels, not alternative subset sizes.
See :doc:`gpu-optimization` for the detailed memory-access diagrams.
