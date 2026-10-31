Migrate from GPU-IO to Three.js
===============================

GPU-IO is deprecated as an application backend. Three.js is now the supported
browser path because it uses WebGPU when available and retains compatibility
through its own WebGL2 transform-feedback solver. GPU-IO remains packaged
temporarily for historical reproducibility and replacement qualification.

This guide concerns browser compatibility only. For the intended
Python-supervised NumPy pipeline, ``ICGNWGPUFullPyramid`` remains the
recommended execution path. A performance-oriented browser-native stream
uses the standalone raw-WebGPU supervisor described in
:doc:`../explanation/supervision`. Three.js remains the compatibility backend.

Update Python code
------------------

Remove the explicit backend selection:

.. code-block:: python

   # Before
   engine = ICGNWebGL(backend="gpuio")

   # After: WebGPU first, Three.js/WebGL2 fallback automatically
   engine = ICGNWebGL()

No changes are required for ``set_reference``, ``register``, ``register_many``,
``benchmark``, or the returned ``ReferenceInfo`` and ``RegistrationResult``
objects. Parameter order, status values, grid orientation, and NumPy array
shapes are unchanged.

Update command lines
--------------------

Replace ``--backend gpuio`` with ``--backend three-tsl`` or omit ``--backend``
because Three.js is the default:

.. code-block:: console

   $ icgn-webgl reference.png target.png

Force WebGL2 only for qualification
------------------------------------

Normal applications should allow automatic backend selection. To verify a
machine that must operate without WebGPU, force the Three.js compatibility
path explicitly:

.. code-block:: python

   with ICGNWebGL(force_three_webgl=True) as engine:
       info = engine.info()
       engine.set_reference(reference)
       result = engine.register(target)

   assert info.backend == "WebGLBackend"
   assert info.solver == "serial-webgl2"

The equivalent command-line option is ``--force-three-webgl``.

Check numerical compatibility
-----------------------------

Before removing GPU-IO from a deployment, run representative images through
both implementations and compare:

* status and accepted masks;
* displacement RMSE against known motion, when available;
* maximum and mean absolute affine-parameter differences;
* warmed GPU and complete browser-wall timings.

Small floating-point differences are expected because the execution mappings
and accumulation orders differ. On the validation machine, the 512 x 512
campaign produced identical status arrays, a maximum absolute parameter
difference of ``1.4913e-4``, and approximately ``3.9e-5`` pixel translation
RMSE for both implementations. See :doc:`../reference/performance` for the
complete timing boundaries.

Retain GPU-IO only when necessary
---------------------------------

Continue to select ``backend="gpuio"`` only when reproducing an historical
result or diagnosing a replacement regression. Report such use explicitly;
GPU-IO no longer represents the recommended WebGL2 execution path and will
not receive new algorithmic features.
