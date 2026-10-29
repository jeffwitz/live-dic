Subset size and field spacing
=============================

Every public engine accepts ``subset_size=8``, ``16``, or ``32``. The default
is 16 and remains the geometry used by the published performance and accuracy
campaigns.

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   with ICGNWGPUFullPyramid(subset_size=32) as engine:
       info = engine.set_reference(reference)
       result = engine.register(target)

   assert info.subset_size == 32
   assert result.subset_size == 32

Geometry contract
-----------------

For a selected width :math:`S`, the implementation currently couples subset
width and field spacing:

* each numerical subset contains :math:`S \times S` pixels;
* grid dimensions are ``floor(width / S)`` by ``floor(height / S)``;
* subset centres are ``(S/2 + S*i, S/2 + S*j)``;
* local coordinates range from ``-(S-1)/2`` to ``+(S-1)/2``;
* pixels beyond the last complete subset on the right or bottom are uncovered.

This is not yet an independent step-size option. A smaller value therefore
produces both smaller correlation windows and a denser result field; a larger
value produces larger windows and a coarser field.

Execution strategy
------------------

The portable shared-WGSL solver always uses one 16-lane workgroup per subset.
The pixels are distributed evenly over those lanes: 4, 16, or 64 pixels per
lane for subset widths 8, 16, or 32. The default 16 specialization preserves
the original per-lane column traversal. Solver workgroups are dispatched as
``grid_width x grid_height`` rather than one long one-dimensional array; this
keeps the 129,600 subsets produced by an 8-pixel UHD grid below WebGPU's
per-dimension dispatch limit.

Three.js builds a size-specialized TSL graph when its headless runtime starts.
Its subgroup solver distributes the selected number of pixels across the
measured subgroup. The WebGL2 paths scan the selected square serially inside
one invocation per subset.

The native GLSL/SPIR-V subgroup solver is specialized for 16 x 16. Native
sizes 8 and 32 therefore select the portable shared-WGSL solver. Explicitly
requesting ``prefer_subgroups=True`` with a non-default size is rejected.

Choosing a value
----------------

``8``
   Provides the densest field and the smallest local support, but creates four
   times as many subsets as 16 in a two-dimensional image. It may be less
   robust in weakly textured regions.

``16``
   Qualified default and the only size currently covered by the published UHD
   performance tables.

``32``
   Uses more image information per estimate and creates one quarter as many
   subsets as 16. It may improve local conditioning, but reduces spatial
   resolution and increases the arithmetic per subset.

Changing this option changes the scientific sampling geometry. Requalify
accuracy, accepted-subset count, and complete image-to-field latency on the
target images instead of extrapolating the 16 x 16 benchmark.
