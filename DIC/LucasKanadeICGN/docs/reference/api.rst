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

Native engine
-------------

.. autoclass:: lucas_kanade_icgn.ICGNWGPU
   :members: set_reference, register, register_stream, register_many, close

The native engine is optional and experimental. It returns the same
``ReferenceInfo`` and ``RegistrationResult`` types as the browser engine, but
does not start Chromium or serialize frames as PNG.

``register_stream`` and ``register_many`` keep at most two images in flight by
default. The public ``DEFAULT_IN_FLIGHT_FRAMES`` constant records this policy
in one place. ``pipeline_depth`` refers only to concurrent sequence execution
and does not denote an image-pyramid depth. Native pyramid recovery is a
separate opt-in constructor setting: ``pyramid_levels=1`` is the default
single-scale path, and values ``2`` or ``3`` enable selective coarse-to-fine
recovery for failed subsets.

Full-pyramid native engine
--------------------------

.. autoclass:: lucas_kanade_icgn.ICGNWGPUFullPyramid
   :members: set_reference, register, close

This separate backend runs every configured scale for every target. Pyramid
reduction, affine seed transfer, and all IC-GN solves remain GPU-resident in one
command submission. Per-level iteration and tolerance tuples are ordered from
coarse to fine. Two levels are the default and recommended native configuration;
three levels are opt-in for a larger expected displacement range.

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

.. autodata:: lucas_kanade_icgn.DEFAULT_IN_FLIGHT_FRAMES

.. autodata:: lucas_kanade_icgn.DEFAULT_PYRAMID_LEVELS

.. autodata:: lucas_kanade_icgn.DEFAULT_FULL_PYRAMID_LEVELS

.. autodata:: lucas_kanade_icgn.ImageInput
