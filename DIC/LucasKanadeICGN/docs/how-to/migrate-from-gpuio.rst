Move from the historical GPU-IO backend
========================================

The retired GPU-IO backend, its WebGL solver and the standalone demonstration
have been removed from this repository. This page maps older calls to the
maintained WebGPU and native wgpu-py implementations.

Choose a supported path
-----------------------

For Python-supervised NumPy input and output, use the native wgpu-py engine:

.. code-block:: python

   from live_dic import ICGNWGPUFullPyramid

   with ICGNWGPUFullPyramid() as engine:
       engine.set_reference(reference)
       result = engine.register(target)

For the Python-controlled browser compatibility API, select one of its two
WebGPU implementations:

.. code-block:: python

   from live_dic import ICGNBrowser

   # Independent Three.js/TSL WebGPU solver
   with ICGNBrowser(backend="three-tsl") as engine:
       engine.set_reference(reference)
       result = engine.register(target)

   # Browser adapter using the shared WGSL kernel
   with ICGNBrowser(backend="webgpu-wgsl") as engine:
       engine.set_reference(reference)
       result = engine.register(target)

Both browser choices require WebGPU. Three.js has no WebGL2 fallback. When
WebGPU is unavailable, use the native Python engine with a compatible software
adapter if one is installed; the browser API does not silently switch to a
different solver.

Update older calls
------------------

Remove ``backend="gpuio"`` from ``ICGNBrowser`` and the CLI. The default
``ICGNBrowser()`` remains the Three.js/TSL WebGPU implementation:

.. code-block:: python

   # Before: no longer supported
   engine = ICGNBrowser(backend="gpuio")

   # After: WebGPU required
   engine = ICGNBrowser(backend="three-tsl")

The command-line equivalent is:

.. code-block:: console

   $ live-dic --backend three-tsl reference.png target.png

Review numerical compatibility
-------------------------------

Historical GPU-IO measurements and the maintained Three.js/TSL, raw browser
WGSL and native wgpu-py paths describe distinct execution implementations.
Requalify any saved comparison or measurement workflow against the selected
maintained engine. Compare accepted
points, displacement error against known motion, all requested model
parameters, and complete image-to-field timings on representative inputs.
Do not treat an old GPU-IO timing or result as a current backend guarantee.

See :doc:`../explanation/backend-selection`, :doc:`../reference/api`, and
:doc:`install` for the supported interfaces and setup requirements.
