Python API reference
====================

Public package
--------------

.. automodule:: lucas_kanade_icgn

Engine
------

.. autoclass:: lucas_kanade_icgn.ICGNWebGL
   :members: start, info, set_reference, register, register_pair, register_many, benchmark, close

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

.. autodata:: lucas_kanade_icgn.ImageInput
