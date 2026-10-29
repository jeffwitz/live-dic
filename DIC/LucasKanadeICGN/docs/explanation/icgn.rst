Inverse-compositional registration
==================================

Subset warp
-----------

The default affine model uses six parameters:

.. math::

   \mathbf{p} = [t_x, u_x, u_y, t_y, v_x, v_y]^T.

For local coordinates :math:`(x, y)` around the subset center, the displacement
is

.. math::

   \mathbf{d}(x, y; \mathbf{p}) =
   \begin{bmatrix}
   t_x + u_x x + u_y y \\
   t_y + v_x x + v_y y
   \end{bmatrix}.

The target is sampled at the warped position with cubic B-spline interpolation.
Direct bilinear and Keys-cubic interpolation are also selectable. Translation
and quadratic warp bases are defined in :doc:`motion-models`; quadratic mode
adds six second derivatives to this affine prefix.

Reference preparation
---------------------

Inverse-compositional Gauss-Newton keeps the reference Jacobian and Hessian
constant. For each reference pixel, the steepest-descent vector is

.. math::

   \mathbf{J}(x,y) =
   [F_x, F_x x, F_x y, F_y, F_y x, F_y y]^T.

The implementation prepares

.. math::

   \mathbf{H}^{-1} =
   \left(\sum_{x,y}\mathbf{J}\mathbf{J}^T\right)^{-1}

once per subset and stores it in a GPU atlas. ``J^T F`` and ``sum(J)`` are also
precomputed so the target pass needs one reconstructed sample per pixel.
Quadratic mode extends the steepest-descent vector to twelve components and
stores a 12 by 12 inverse Hessian.

Normalized residual
-------------------

The solver centers and normalizes reference and target intensities. This makes
the residual less sensitive to affine brightness changes. A ZNCC threshold is
evaluated at convergence; subsets below the threshold are rejected.

Inverse-compositional update
----------------------------

Each iteration solves for an incremental warp :math:`\Delta\mathbf{p}` using
the constant inverse Hessian. The current warp is updated by composition with
the inverse increment:

.. math::

   W(\mathbf{p}) \leftarrow
   W(\mathbf{p}) \circ W(\Delta\mathbf{p})^{-1}.

Convergence is reached when the maximum displacement induced by
:math:`\Delta\mathbf{p}` over the model's control points falls below the
tolerance. Affine mode uses the four corners; quadratic mode uses a 3 by 3 set
so interior curvature contributes.

Spatial consistency
-------------------

After IC-GN, the median pass compares each accepted subset with predictions from
its accepted direct 3 x 3 neighbours. Affine and quadratic neighbour
polynomials are first recentered at the current subset centre. Each active
coefficient is then tested independently against its neighbour median and
median absolute deviation (MAD). An aberrant coefficient is replaced by its
median without discarding the subset's other, locally consistent coefficients;
any replacement assigns status 2.

The robustness floor is expressed in displacement units. Translation keeps a
0.1-pixel floor. Affine-gradient and quadratic-curvature floors correspond to a
0.001-pixel effect at the subset edge, accounting for their different units and
the configured subset size. A component is replaced when its deviation exceeds
twice the sum of its MAD and that floor. At least four accepted neighbours are
required, so sparse boundaries are left unchanged rather than extrapolated.
The derivation, literature boundary, analytic A/B results, and measured cost
are documented in :doc:`component-wise-median`.

Numerical equivalence
---------------------

The algorithm mirrors the SYCL formulation but is not bit-for-bit equivalent.
WebGL2 uses single-precision shader arithmetic and a different execution and
reduction model. See :doc:`limitations` for the remaining scope differences.
