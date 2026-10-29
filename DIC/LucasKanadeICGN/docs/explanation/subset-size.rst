Subset size and stride
======================

``subset_size`` selects an 8, 16, or 32 pixel square. ``subset_stride``
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
