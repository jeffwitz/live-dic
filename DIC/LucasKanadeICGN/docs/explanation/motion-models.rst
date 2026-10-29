Motion models
=============

The solver supports two motion models selected when an engine is created:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   engine = ICGNWGPUFullPyramid(motion_model="translation")

``affine`` remains the default. It estimates six parameters per subset in the
public order ``[tx, du/dx, du/dy, ty, dv/dx, dv/dy]``. Use it when local
rotation, strain, shear, or magnification is part of the quantity to measure.

``translation`` estimates only ``tx`` and ``ty``. It is appropriate when the
subset content can be treated as rigid and the desired result is displacement
only. The public result layout does not change: components 1, 2, 4, and 5 are
written as exact zeros. Existing displacement, status, export, pyramid, and
median-filter APIs therefore continue to work.

Analytic translation solve
--------------------------

For a translation, the steepest-descent image contains only the two reference
gradients. The constant Hessian is

.. math::

   H = \begin{bmatrix}
       \sum F_x^2 & \sum F_x F_y \\
       \sum F_x F_y & \sum F_y^2
   \end{bmatrix}.

Reference preparation evaluates its inverse directly:

.. math::

   H^{-1} = \frac{1}{H_{xx}H_{yy}-H_{xy}^2}
   \begin{bmatrix}H_{yy} & -H_{xy} \\ -H_{xy} & H_{xx}\end{bmatrix}.

No general matrix inversion is used in this mode. At run time the kernels also
omit the four affine Jacobian products, the 6 by 6 matrix-vector product, affine
warp composition, and four-corner convergence calculation. The WGSL pipeline
constant and the Three.js graph specialization remove those branches when the
pipeline is built; this is not a per-pixel run-time switch.

Backend coverage
----------------

.. list-table:: Translation model support
   :header-rows: 1

   * - Backend
     - Support
     - Implementation
   * - Native ``ICGNWGPU``
     - yes
     - shared WGSL portable workgroup solver
   * - Native ``ICGNWGPUFullPyramid``
     - yes
     - shared WGSL at every pyramid level
   * - Browser ``webgpu-wgsl``
     - yes
     - the same shared WGSL source
   * - Three.js WebGPU
     - yes
     - TSL subgroup or portable solver specialized at graph construction
   * - Three.js WebGL2 fallback
     - yes
     - the same TSL mathematical helpers in its serial solver
   * - Deprecated GPU-IO
     - no
     - affine-only historical implementation

The native specialized SPIR-V subgroup kernel is affine-only. Requesting
``motion_model="translation"`` therefore selects the portable shared-WGSL
kernel. Explicitly combining translation with ``prefer_subgroups=True`` is
rejected instead of silently running a different model.

Choosing the model
------------------

Translation is a modelling constraint, not merely a faster approximation to
affine DIC. It can improve displacement robustness when a subset really moves
rigidly because four unnecessary degrees of freedom cannot absorb noise. It
cannot return strain: all displacement gradients are zero by construction.

Use ``translation`` for rigid-image motion, motion compensation, or a fast
displacement-only pass. Keep ``affine`` for DIC strain measurement and whenever
rotation, shear, extension, or spatially varying displacement within one
16 by 16 subset is significant.

The measured speed-up depends on the execution graph. It is largest when the
iterative solve dominates. In a two-level full pyramid that converges in about
one iteration per level, image reduction, B-spline prefiltering, upload,
readback, and result construction dominate, so the gain is necessarily
smaller. See :doc:`../reference/performance` for current A/B measurements.

The motion model is fixed at engine construction because reference Hessians,
metadata, and GPU pipelines depend on it. Construct a new engine and call
``set_reference()`` again to change models.
