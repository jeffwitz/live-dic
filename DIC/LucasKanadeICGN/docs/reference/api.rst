Python API reference
====================

Public package
--------------

.. automodule:: lucas_kanade_icgn

Engine
------

.. autoclass:: lucas_kanade_icgn.ICGNWebGL
   :members: start, info, set_reference, register, register_pair, register_many, benchmark, close

For ``backend="webgpu-wgsl"``, ``pyramid_levels`` accepts one to three levels
and defaults to two. ``iterations_per_level`` and ``tolerance_per_level`` are
coarse-to-fine tuples with exactly one value per level. Pyramid options are
rejected by the independent Three.js/TSL and GPU-IO implementations.

``backend="three-tsl"`` selects WebGPU when available and otherwise falls back
to Three.js' WebGL2 transform-feedback backend. Set ``force_three_webgl=True``
to test that compatibility path explicitly. ``EngineInfo.solver`` then reports
``serial-webgl2`` instead of ``subgroup`` or ``portable-workgroup``.
For WebGPU, ``subgroup_probe_width`` records the lightweight discovery result,
``subgroup_solver_width`` records the value observed in the production solver,
and ``subgroup_validation`` is ``"validated"`` or ``"mismatch"``. A mismatch
automatically selects ``portable-workgroup-subgroup-mismatch`` and does not
enter the per-frame critical path.

``backend="gpuio"`` is deprecated. It remains callable for historical
reproducibility and cross-backend regression tests, but new applications
should use ``three-tsl``. The Three.js fallback provides WebGL2 compatibility
without changing the public result schema.

All active engines accept ``interpolation="bspline"``, ``"linear"``, or
``"cubic"`` at construction. ``bspline`` is the historical default. The two
direct modes skip per-target FIR coefficient construction and require matching
raw-sample reference preparation. The deprecated GPU-IO backend accepts only
``bspline``. See :doc:`../explanation/interpolation`.

All active engines also accept ``motion_model="affine"`` or
``motion_model="translation"`` at construction. ``affine`` remains the
six-parameter default. Translation mode estimates only ``tx`` and ``ty`` with
an analytic 2 by 2 inverse while retaining the six-component result layout;
the four deformation components are exact zeros. The deprecated GPU-IO backend
is affine-only. See :doc:`../explanation/motion-models`.

Native single-scale and streaming engine
----------------------------------------

.. autoclass:: lucas_kanade_icgn.ICGNWGPU
   :members: set_reference, register, register_stream, register_many, close

The native engine is optional and experimental. It returns the same
``ReferenceInfo`` and ``RegistrationResult`` types as the browser engine, but
does not start Chromium or serialize frames as PNG.

All Python backends normalize inputs to 8-bit grayscale. Floating-point arrays
in ``[0, 1]`` are scaled to ``[0, 255]``; floating-point arrays already using
the 8-bit intensity convention are clipped to that range. Integer arrays and
integer-mode Pillow images must already lie in ``[0, 255]``. Higher-bit-depth
scientific images are rejected instead of being silently saturated: convert
the reference and every target explicitly with the same calibrated intensity
range before calling the API.

``register_stream`` and ``register_many`` keep at most two images in flight by
default. The public ``DEFAULT_IN_FLIGHT_FRAMES`` constant records this policy
in one place. ``pipeline_depth`` refers only to concurrent sequence execution
and does not denote an image-pyramid depth. Native pyramid recovery is a
separate opt-in constructor setting: ``pyramid_levels=1`` is the default for
this class, and values ``2`` or ``3`` enable the older selective
coarse-to-fine recovery for failed subsets. This mode is retained for
compatibility and exceptional-motion experiments; it is not the recommended
normal pyramid.

Full-pyramid native engine
--------------------------

.. autoclass:: lucas_kanade_icgn.ICGNWGPUFullPyramid
   :members: set_reference, register, register_stream, close

This is the recommended native coarse-to-fine backend. It runs every configured
scale for every target. Pyramid
reduction, affine seed transfer, and all IC-GN solves remain GPU-resident in one
command submission. Per-level iteration and tolerance tuples are ordered from
coarse to fine. Two levels are the default and recommended native configuration;
three levels are opt-in for a larger expected displacement range.

``interpolation`` applies to every pyramid level. ``linear`` and ``cubic``
omit both FIR passes at every level; ``bspline`` retains the existing graph.
``motion_model`` likewise applies to every level and to every transferred seed.

``register_stream`` keeps two complete target pyramids in flight by default.
Every slot has independent mutable textures, state, seed, and readback
resources, while reference data and pipelines remain shared. Results are
yielded in input order. Set ``pipeline_depth=1`` to minimize result age; use
the default of two when sustained throughput is more important. The timing
dictionary separates ``inputPreparationWallMs``, ``uploadWallMs``,
``encodingWallMs``, ``submissionWallMs``, ``completionWaitWallMs``,
``readbackCopyWallMs``, and ``resultBuildWallMs``. ``resultAgeWallMs`` is the
complete accepted-image to materialized-field age.

See :doc:`../explanation/backend-selection` for a single table distinguishing
the defaults of all public engines.

One-shot registration
---------------------

.. autofunction:: lucas_kanade_icgn.register_pair

Results
-------

.. autoclass:: lucas_kanade_icgn.EngineInfo
   :members:

.. autoclass:: lucas_kanade_icgn.ReferenceInfo
   :members:

.. autoclass:: lucas_kanade_icgn.RegistrationResult
   :members: displacement, accepted_mask, replaced_mask, rmse

.. autoclass:: lucas_kanade_icgn.BenchmarkResult
   :members:

Exceptions
----------

.. autoexception:: lucas_kanade_icgn.ICGNError

Constants and input types
-------------------------

.. autodata:: lucas_kanade_icgn.PARAMETER_ORDER

.. autodata:: lucas_kanade_icgn.MOTION_MODELS

.. autodata:: lucas_kanade_icgn.DEFAULT_IN_FLIGHT_FRAMES

.. autodata:: lucas_kanade_icgn.DEFAULT_PYRAMID_LEVELS

.. autodata:: lucas_kanade_icgn.DEFAULT_FULL_PYRAMID_LEVELS

.. autodata:: lucas_kanade_icgn.ImageInput
