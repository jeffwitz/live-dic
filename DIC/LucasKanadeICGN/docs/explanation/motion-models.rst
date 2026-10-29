Motion models
=============

The motion model describes the displacement inside each subset. It is fixed
when an engine is created because it changes the reference Jacobian, Hessian,
GPU state layout, and compute pipelines::

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   engine = ICGNWGPUFullPyramid(motion_model="quadratic")

Three models are available:

``translation``
   Estimates only horizontal and vertical translation. Use it when a subset
   can be treated as locally rigid and strain is not required.

``affine``
   The default six-parameter model. It estimates translation and the first
   spatial derivatives of displacement.

``quadratic``
   A twelve-parameter second-order model. It adds the three quadratic Lagrange
   terms of each displacement component and can represent displacement
   curvature inside a subset.

Parameter bases
---------------

Let ``x`` and ``y`` be coordinates relative to the subset centre. The affine
model is

.. math::

   u(x,y) &= t_x + u_x x + u_y y,\\
   v(x,y) &= t_y + v_x x + v_y y.

Its public order is
``[tx, du_dx, du_dy, ty, dv_dx, dv_dy]``.

The quadratic model uses the second-order basis
``[1, x, y, x^2/2, xy, y^2/2]``:

.. math::

   u(x,y) &= t_x + u_x x + u_y y
             + \tfrac12 u_{xx}x^2 + u_{xy}xy + \tfrac12 u_{yy}y^2,\\
   v(x,y) &= t_y + v_x x + v_y y
             + \tfrac12 v_{xx}x^2 + v_{xy}xy + \tfrac12 v_{yy}y^2.

The twelve returned components are::

   [tx, du_dx, du_dy, ty, dv_dx, dv_dy,
    d2u_dx2, d2u_dxdy, d2u_dy2,
    d2v_dx2, d2v_dxdy, d2v_dy2]

The one-half factors make the stored pure quadratic coefficients equal to the
second derivatives at the subset centre. ``RegistrationResult.displacement``
continues to select ``tx`` and ``ty`` without copying data for all models.

Inverse-compositional quadratic update
--------------------------------------

Quadratic mode is not implemented as an additive correction. The two
second-order coordinate maps are lifted to a 6 by 6 operator acting on
``[x^2, xy, y^2, x, y, 1]``. At each iteration the incremental operator is
inverted with pivoted Gauss-Jordan elimination and composed with the current
warp. The updated twelve coefficients are then extracted from the composed
operator. This preserves the inverse-compositional update for the complete
second-order warp.

Reference preparation builds a 12-component steepest-descent image and a
12 by 12 inverse Hessian for each subset. The run-time solver accumulates and
reduces twelve residual products. Quadratic mode therefore consumes more
reference memory and arithmetic than affine mode; it is intended for cases
where within-subset curvature is physically useful, not as a faster affine
approximation.

Analytic translation solve
--------------------------

For translation, the steepest-descent image contains only the two reference
gradients. Its 2 by 2 Hessian is inverted analytically. The GPU specialization
omits the affine and quadratic products, general matrix-vector products, warp
composition, and extended convergence tests. The six-component public layout
is retained, with deformation components set to exact zeros.

Pyramid transfer
----------------

Coarse-to-fine transfer recentres every polynomial at the destination subset
centre. Translation scales with the image scale, first derivatives are
dimensionless, and second derivatives scale inversely with image scale. Both
the native and raw-browser full pyramids perform this conversion on the GPU.

Backend coverage
----------------

.. list-table:: Motion-model support
   :header-rows: 1

   * - Backend
     - Translation
     - Affine
     - Quadratic
   * - Native ``ICGNWGPU`` shared WGSL
     - yes
     - yes
     - yes
   * - Native ``ICGNWGPUFullPyramid``
     - yes
     - yes
     - yes
   * - Browser ``webgpu-wgsl``
     - yes
     - yes
     - yes
   * - Three.js WebGPU and WebGL2 fallback
     - yes
     - yes
     - no
   * - Deprecated GPU-IO
     - no
     - yes
     - no

The native specialized SPIR-V subgroup solver is affine-only. Translation and
quadratic requests use the portable shared-WGSL solver. In Python-supervised
browser operation, request quadratic mode with ``backend="webgpu-wgsl"``;
constructing the independent Three.js backend with this model is rejected.

Choosing a model
----------------

Use translation for rigid image motion and displacement-only work. Use affine
for the usual DIC displacement-and-strain field. Use quadratic when the subset
is large enough and the expected displacement has meaningful curvature within
one subset. A 32 by 32 subset is a safer starting point than 8 by 8 for twelve
unknowns, but conditioning remains image dependent.

Quadratic mode has passed synthetic GPU regression tests that contain known
non-zero first and second displacement derivatives. Those tests qualify the
implementation, not its UHD throughput. Current published 4K performance
tables cover affine and translation only; quadratic measurements must be
reported separately because its 12 by 12 Hessian and exact composition have a
different cost.

The second-order formulation follows the family described by Bai et al.,
"A novel 2nd-order shape function based digital image correlation method for
large deformation measurements", *Optics and Lasers in Engineering*, 90
(2017), DOI: ``10.1016/j.optlaseng.2016.09.010``.
