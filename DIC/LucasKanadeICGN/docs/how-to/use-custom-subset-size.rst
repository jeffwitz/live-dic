Choose a custom subset size
===========================

The native Python engines accept every integer square width from 4 through
64. The default remains 16, so existing calls keep their current grid and
performance path. A custom width uses the portable WGSL solver; the optional
subgroup specialization remains available only for its qualified 16 x 16
configuration.

For example, this uses 24 x 24 pixel supports sampled every 12 pixels:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   with ICGNWGPUFullPyramid(
       subset_size=24,
       subset_stride=12,
       motion_model="affine",
       device_type="gpu",
   ) as engine:
       engine.set_reference(reference)
       result = engine.register(target)

The subset size controls how many image pixels contribute to each local fit.
The stride controls the distance between neighbouring subset centres. Reducing
the stride creates a denser result grid and more subsets; it does not make an
individual solve cheaper. For image width ``W``, subset width ``S`` and stride
``D``, the grid contains ``1 + floor((W - S) / D)`` columns when ``W >= S``.

Both ``ICGNWGPU`` and ``ICGNWGPUFullPyramid`` support custom widths. The
``ICGNWebGL(backend="webgpu-wgsl")`` adapter and the standalone WebGPU
workbench use the same portable kernel family. Three.js/TSL, its WebGL2 path,
and GPU-IO retain their existing 8, 16 and 32 pixel limits; the Python wrapper
rejects a custom width for those backends instead of silently substituting a
different size.

The 4 through 64 range is a functional kernel range, not a metrological
qualification of every width. The minimum 4 x 4 support provides at least
twelve pixel samples for the twelve-parameter quadratic model. Very small
supports contain little texture information and can produce noisy or invalid
fits. Larger supports can average more information but span a wider deformation
gradient. Evaluate accuracy, coverage, masks, displacement range and
image-to-field latency on the intended specimen. The shipped analytical UHD
qualification uses the default 16 x 16 geometry; it does not validate every
custom width.

See :doc:`../explanation/subset-size` for centre conventions, overlap geometry,
weighting and the relation between subset size and stride.
