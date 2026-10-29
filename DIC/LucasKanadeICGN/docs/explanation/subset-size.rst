Subset size and stride
======================

``subset_size`` selects a square of 8, 15, 16, 17, 31, 32, or 33 pixels.
The odd widths provide an exact centre pixel; the established even widths
remain valid with a half-pixel geometric centre. ``subset_stride``
independently selects the distance, in pixels, between adjacent subset
centres. It defaults to ``None``, which means ``subset_stride=subset_size``
and preserves the historical non-overlapping geometry and fast path.

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   with ICGNWGPUFullPyramid(subset_size=32, subset_stride=16) as engine:
       info = engine.set_reference(reference)
       result = engine.register(target)

   assert info.subset_size == 32
   assert info.subset_stride == 16
   assert result.subset_stride == 16

Geometry contract
-----------------

For subset width :math:`S` and stride :math:`D`:

* every numerical subset contains :math:`S \times S` pixels;
* :math:`1 \le D \le S`;
* grid dimensions are ``1 + floor((width - S) / D)`` by
  ``1 + floor((height - S) / D)``;
* subset centres are ``(S/2 + D*i, S/2 + D*j)``;
* local coordinates range from ``-(S-1)/2`` to ``+(S-1)/2``;
* only fully contained subsets are emitted, so a strip smaller than one
  stride can remain uncovered at the right or bottom edge.

For example, a 3840 x 2160 frame with ``subset_size=32`` and
``subset_stride=16`` produces a 239 x 134 field. With the default stride of
32 it produces 120 x 67 values.

Odd and even centres
--------------------

Subset DIC is commonly written for a ``(2M+1) x (2M+1)`` support so the
control point is an acquired pixel. That convention is convenient, but it is
not required by the continuous warp: an even support is centred halfway
between four pixels and retains symmetric local coordinates. Use an odd width
when an integer-pixel control point matters to the measurement convention;
do not assume that odd support is intrinsically more accurate. Both choices
must be qualified with the intended interpolation, texture, and deformation.

Centred Gaussian integration
-----------------------------

The native shared-WGSL engines optionally minimize a spatially weighted
criterion with::

   ICGNWGPUFullPyramid(
       subset_size=31,
       subset_stride=16,
       spatial_weighting="gaussian",
       gaussian_sigma=8.0,
   )

For local coordinates :math:`(x,y)`, the weight is

.. math::

   w(x,y)=\exp\left[-\frac{x^2+y^2}{2\sigma^2}\right].

The same weights are applied consistently to the reference and target means,
photometric norms, ZNCC numerator, inverse-compositional Hessian, Jacobian
sums, and residual projection. ``spatial_weighting="uniform"`` is the default
and reproduces the previous criterion. If ``gaussian_sigma`` is omitted, it
defaults to one quarter of the subset width.

A small sigma suppresses edge pixels. This can reject peripheral mismatches,
but it also removes leverage for affine gradients and especially quadratic
curvature terms, whose basis magnitude grows with distance from the centre.
Treat sigma as a metrological parameter rather than an unconditional
improvement.

Current analytic qualification
------------------------------

The reproducible quadratic benchmark uses a continuous analytic texture,
inverts a known second-order displacement field, rasterizes both images to
8-bit grayscale, and evaluates every result at its exact subset centre. With
B-spline interpolation, stride 16, a moderate noise-free curvature field, and
``sigma=12``, the validation machine produced:

.. list-table:: Quadratic accuracy at 768 x 768
   :header-rows: 1

   * - Support and integration
     - Displacement RMSE (px)
     - Gradient RMSE
     - Curvature RMSE per component
   * - 32 x 32, uniform
     - **0.00623**
     - **0.000649**
     - **3.97e-5**
   * - 32 x 32, Gaussian
     - 0.00667
     - 0.000719
     - 4.23e-5
   * - 31 x 31, uniform
     - 0.00648
     - 0.000710
     - 4.41e-5
   * - 31 x 31, Gaussian
     - 0.00689
     - 0.000782
     - 4.64e-5

The 32 x 32 uniform window was also best in the strong clean and noisy cases
of this campaign. The experiment therefore does not justify changing the
default. The default ``median_scope="all", median_replacement="flagged"``
reduces the moderate uniform-window
curvature error relative to ``median_scope="displacement"``, but its
RMSE still exceeds the RMS magnitude of the injected curvature. The direct
quadratic coefficients are therefore not yet a qualified curvature
measurement.

Reproduce the comparison from the repository root with::

   PYTHONPATH=DIC/LucasKanadeICGN/python/src \
   python3 DIC/LucasKanadeICGN/benchmarks/benchmark_quadratic_weighting.py

Independent-stride reference preparation
----------------------------------------

With non-overlapping subsets, each reference pixel belongs to one subset and
can store its already centred intensity. Overlapping subsets have different
local means for the same shared pixel. The shared WGSL implementation therefore
stores the raw reference intensity and one mean per subset in solver metadata.
The solver subtracts that mean when it reads the reference. Gradients, inverse
Hessians, photometric energy, and Jacobian sums are still prepared per subset.

This alternate representation is selected only when the stride differs from
the size. The default ``stride == size`` path retains the historical buffers,
shader specialization, and subgroup eligibility.

Backend support
---------------

Independent stride is implemented by the shared WGSL family:

* ``ICGNWGPU`` and ``ICGNWGPUFullPyramid`` through native ``wgpu-py``;
* ``ICGNWebGL(backend="webgpu-wgsl")`` in Chromium.

It works with one to three pyramid levels and with translation, affine, and
quadratic motion models. Three.js/TSL, its WebGL2 fallback, and deprecated
GPU-IO currently require ``subset_stride == subset_size`` and reject a
non-default stride instead of silently changing its geometry.

Odd subset widths are available in native wgpu and the raw browser
``webgpu-wgsl`` path. Centred Gaussian integration is currently exposed by the
native engines; browser engines retain uniform integration. Three.js/TSL,
WebGL2, and GPU-IO remain limited to widths 8, 16, and 32.

The specialized native GLSL/SPIR-V subgroup solver requires both size and
stride to be 16. Other configurations select the portable shared-WGSL solver.

Choosing size and stride
------------------------

Reducing the stride makes the field denser without reducing the local image
support. For a large image, halving the stride in both directions creates
approximately four times as many subsets, so solver and result-transfer costs
can approach four times the default. It does not make one subset cheaper.

``subset_size=16, subset_stride=16``
   Qualified historical default and the geometry used by the published UHD
   performance tables.

``subset_size=32, subset_stride=16``
   Keeps the conditioning and larger support of a 32-pixel window while
   restoring approximately the spatial sampling density of the default.

``subset_size=8, subset_stride=8``
   Provides the smallest support and densest currently supported default grid,
   but may be less robust in weakly textured regions.

Both options change the scientific sampling geometry. Requalify accuracy,
accepted-subset count, output size, and complete NumPy-image-to-field latency
on representative data.
