Interpolation modes
===================

DIC-RT exposes three target-reconstruction choices through the same public
argument:

.. code-block:: python

   engine = ICGNWGPUFullPyramid(interpolation="linear")

The mode is selected when the engine is constructed and applies to every
pyramid level. ``bspline`` remains the default so existing applications keep
their numerical model and subgroup optimization. Changing the interpolation
model also changes reference gradients. Python callers must construct a new
engine; direct JavaScript callers must prepare the reference again with
``setReference``.

Numerical definitions
---------------------

``bspline``
   The historical path first applies the sigma-one Gaussian and inverse cubic
   B-spline FIR. The solver then reconstructs each warped target value from a
   4 x 4 coefficient neighbourhood. Reference intensities and gradients use
   the same filtered spline model.

``linear``
   Bilinear interpolation of the four adjacent unfiltered image samples. The
   reference uses unfiltered intensities and centered finite-difference
   gradients. This mode performs the fewest target reads and is intended for
   applications that value throughput over the last fraction of subpixel
   accuracy.

``cubic``
   Direct 4 x 4 Keys cubic convolution with parameter ``a = -0.5`` (the
   Catmull-Rom member of the Keys family). It is interpolating at integer
   pixels and uses the same raw-sample reference preparation as ``linear``.

The direct modes do not build B-spline coefficients. Their per-frame graph is:

.. code-block:: text

   target upload -> optional pyramid reductions -> IC-GN solves -> optional UOD -> readback

The default graph retains the two separable FIR dispatches at every scale:

.. code-block:: text

   target upload -> optional reductions -> FIR X -> FIR Y -> IC-GN -> optional UOD -> readback

Backend coverage
----------------

.. list-table:: Interpolation implementation by backend
   :header-rows: 1

   * - Backend
     - ``bspline``
     - ``linear`` / ``cubic``
   * - Native ``ICGNWGPUFullPyramid``
     - Shared WGSL portable solver
     - Shared WGSL portable solver, FIR dispatches omitted
   * - Native ``ICGNWGPU``
     - Portable or validated SPIR-V subgroup solver
     - Portable WGSL solver, FIR dispatches omitted
   * - Raw browser ``webgpu-wgsl``
     - Shared WGSL
     - Shared WGSL specialized pipelines
   * - Three.js/TSL WebGPU
     - TSL workgroup or subgroup solver
     - Same TSL solver families with direct sampling
   * - Three.js WebGL2 fallback
     - Serial TSL solver
     - Serial TSL solver with direct sampling
   * - Deprecated GPU-IO
     - Supported for historical reproduction
     - Not supported

The native SPIR-V subgroup binary encodes the historical coefficient sampler.
Consequently ``ICGNWGPU(interpolation="linear")`` and ``"cubic"`` deliberately
select the shared portable WGSL solver. The full-pyramid engine already uses
that portable seeded solver at every level.

Fractional-displacement validation
----------------------------------

The regression input is sampled analytically before and after translation; it
is not produced by shifting an already quantized raster. This prevents the
image generator from silently choosing one of the interpolation modes under
test. The GPU regression requires all modes to recover ``(0.5, -0.375)`` with
finite accepted results. The reproducible benchmark is:

.. code-block:: bash

   python benchmarks/benchmark_interpolation.py \
     --width 512 --height 512 --levels 2 \
     --translation 0.5 -0.375 --warmups 3 --repeats 10

On the validation GPU, two-level native RMSE remained between 0.0086 and
0.0089 px for ``bspline``, 0.0103 and 0.0107 px for ``cubic``, and 0.0165 and
0.0194 px for ``linear`` across the tested fractional translations
``(0.25, -0.25)``, ``(0.5, -0.375)``, ``(0.75, -0.625)``, and
``(1.25, -0.75)``. These values qualify this deterministic texture, not every
experimental image spectrum.

The full-pyramid defaults use ``(8, 16)`` iteration ceilings and
``(5e-3, 5e-3)`` convergence tolerances from coarse to fine. The previous
``(12, 30)`` / ``(5e-3, 1e-3)`` schedule spent additional fine iterations on
changes below the useful precision of the quantized inputs. Across the cases
above, the new schedule preserved or improved RMSE and increased the number of
converged linear subsets. Three.js remains single-scale and keeps its existing
30-iteration, ``1e-3`` defaults; applying the pyramid ceilings there reduced
the number of converged subsets.

Selection guidance
------------------

Keep ``bspline`` when compatibility with previous results, the native subgroup
solver, or the best measured accuracy is the priority. Start with ``linear``
when 4K throughput is limiting and validate its approximately doubled error on
representative images. ``cubic`` is the accuracy-oriented direct option: it
was close to B-spline accuracy, but its 16 image reads and iteration count mean
that eliminating FIR passes does not guarantee a native speed-up.

The interpolation mode is therefore an explicit application trade-off, not a
new universal default.
