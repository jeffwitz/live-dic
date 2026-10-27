Affine inverse-compositional registration
=========================================

Subset warp
-----------

Each subset uses six parameters:

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
precomputed so the target pass needs one cubic sample per pixel.

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
:math:`\Delta\mathbf{p}` at the four subset corners falls below the tolerance.
This criterion accounts for both translation and affine gradients.

Spatial consistency
-------------------

After IC-GN, the median pass compares each accepted subset with predictions from
its accepted 3 x 3 neighbors. A normalized deviation above the configured rule
replaces translation and affine gradients with neighbor medians and assigns
status 2.

Numerical equivalence
---------------------

The algorithm mirrors the SYCL formulation but is not bit-for-bit equivalent.
WebGL2 uses single-precision shader arithmetic and a different execution and
reduction model. See :doc:`limitations` for the remaining scope differences.
