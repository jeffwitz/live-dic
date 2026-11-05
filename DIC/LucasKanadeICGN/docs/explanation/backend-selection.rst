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
uses the standalone PWA supervisor around the shared raw-WebGPU core.
It keeps frames and display in the browser; see :doc:`supervision` and
:doc:`../how-to/run-in-browser`.

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
     - ``ICGNBrowser()``
     - one scale
     - Selects Three.js/TSL and requires WebGPU. Useful for compatibility and
       qualification, but not the recommended low-latency NumPy data plane.
   * - Python-controlled canonical WGSL in Chromium
     - ``ICGNBrowser(backend="webgpu-wgsl")``
     - two full-pyramid levels
     - Use when browser/native source identity is more important than the
       specialized Three.js subgroup path. The process-copy cost remains.

The phrase *two-level default* therefore applies to
``ICGNWGPUFullPyramid`` and ``webgpu-wgsl``.  The phrase *one-scale default*
applies only to ``ICGNWGPU`` and Three.js/TSL.  It must not be interpreted as
a recommendation to use selective recovery instead of the full-pyramid
backend.

Two different native pyramid strategies
---------------------------------------

With its default fixed policy, ``ICGNWGPUFullPyramid`` executes every configured
level for every frame. It
uploads the finest target once, constructs reduced targets on the GPU,
prefilters and solves from coarse to fine, transfers motion-model seeds on the GPU,
and reads back only the final fine field.  Two levels with the ``(8, 16)``
coarse-to-fine iteration ceilings are the current native recommendation.
The optional native adaptive policy can omit the coarse solve, but fails
the current tensile accuracy check and is not that recommendation. See
:doc:`temporal-initialization` for latency experiments and their limits.

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

* a numerical subset has any integer square width from 4 through 64 in the
  native and raw shared-WGSL paths, with 16 x 16 as the default. Three.js/TSL
  retains the discrete widths 8, 16 and 32. Stride controls centre-to-centre
  distance independently and can create overlapping windows;
* the shared-WGSL FIR uses 32 x 8 horizontal and 8 x 32 vertical output
  tiles, each executed by 256 threads;
* the portable native IC-GN solver uses one configurable 16-, 32- or
  experimental 128-lane workgroup per subset (16 by default). The raw browser
  core retains 16 lanes. The specialized Three.js path
  fills the measured hardware subgroup (32 lanes on the validation GPU).

The full-pyramid reduction passes use 16 x 16 dispatch groups, and polynomial seed
transfer and displacement UOD use 256-thread one-dimensional groups. These
dispatch dimensions are scheduling choices and do not change with
``subset_size`` or ``subset_stride``. See :doc:`subset-size` for the numerical
geometry.
See :doc:`gpu-optimization` for the detailed memory-access diagrams.

Options do not automatically propagate between hosts
----------------------------------------------------

Sharing WGSL preserves the numerical equations, but a host must also prepare
resources and schedule each optional stage. The current boundaries are:

.. list-table:: Current native and raw-browser scope
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Native full pyramid
     - Raw browser core / standalone workbench
   * - Local motion models
     - Translation, affine, quadratic
     - Translation, affine, quadratic
   * - Fixed pyramid depth
     - 1 through 4; default 2
     - 1 through 6; default 2
   * - Global affine initialization
     - Not a public native option
     - Opt-in CPU initializer; independent of the local model
   * - GPU spatial seed propagation
     - Research supervisor under ``benchmarks/``
     - Analysis/Solve uses two passes by default in compatible Translation or
       Affine settings; the direct API defaults to zero
   * - ROI
     - Full by default; partial supports explicitly enabled
     - Full only
   * - Reference preparation
     - CPU, retained across a fixed-reference sequence
     - CPU by default; experimental GPU variant is a separate opt-in

Automatic browser propagation supports local Translation and Affine models and
requires CPU reference preparation, uniform full windows and one slot; it is
available in Analysis/Solve, not the fixed
Live preset. Incompatible settings leave automatic propagation off. Preparation
on CPU does not mean that target correlation runs on CPU. See
:doc:`spatial-initialization` and
:doc:`../reference/js-api` before copying options between languages.
