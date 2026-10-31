Your first native DIC registration
===================================

This tutorial prepares one reference and registers a target with the native
Python engine. It uses checked-in images with a known translation, so no
camera, Chromium or graphical interface is needed.

Prerequisites
-------------

Complete :doc:`../how-to/install` with Python 3.11 or newer and a working GPU
or software WebGPU adapter. Run this example from the live-dic repository root.
The standard installation includes wgpu-py.

Create ``first_registration.py``:

.. code-block:: python

   import numpy as np
   from PIL import Image
   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   reference = np.array(Image.open("DIC/LucasKanadeICGN/assets/frame_000.png").convert("L"))
   target = np.array(Image.open("DIC/LucasKanadeICGN/assets/frame_010.png").convert("L"))

   with ICGNWGPUFullPyramid() as engine:
       preparation = engine.set_reference(reference)
       result = engine.register(target)
       print("Adapter:", engine.adapter_info)

   print("Preparation (ms):", preparation.total_wall_ms)
   print("Mean translation:", result.mean_translation)
   print("Accepted subsets:", result.accepted)
   print("Mean iterations:", result.mean_iterations)
   print("Translation RMSE (px):", result.rmse((2.0, -1.0)))

Run it using the environment created during installation:

.. code-block:: console

   $ .venv-icgn/bin/python first_registration.py

The mean displacement should be close to ``(2, -1)`` pixels. Inspect the
reported adapter: ``device_type="auto"`` prefers hardware but can use an
installed software adapter. Small numerical differences and iteration counts
can vary by implementation and configuration. This integer-translation fixture
is a smoke check, not a complete subpixel metrology qualification; the native
UHD analytic regressions provide stronger evidence.

Inspect and export the fields
------------------------------

.. code-block:: python

   print(result.displacement.shape)  # (32, 32, 2)
   print(result.parameters.shape)    # (32, 32, 6)
   print(result.parameter_order)

   valid = result.accepted_mask
   print(result.displacement[valid].mean(axis=0))
   result.save_npz("first_registration.npz")

The affine parameter order is ``tx, du/dx, du/dy, ty, dv/dx, dv/dy``. Only
accepted values are measurements; rejected or excluded records must not be
averaged as zero motion. See :doc:`../reference/result-layout`.

The context manager releases GPU resources permanently when it exits. The
returned NumPy fields remain usable. Reuse an open engine for additional
targets, and create a new one after close.

Optional browser compatibility
-------------------------------

If the deployment requires a browser, install Chromium separately and use
``ICGNWebGL`` instead. Three.js selects WebGPU or its WebGL2 fallback; its
algorithmic and ROI coverage differ from the native full pyramid. See
:doc:`../explanation/backend-selection` before comparing results or timings.

Continue with :doc:`realtime-sequence` to process a persistent sequence, or
:doc:`../how-to/use-roi` to select a reference-image region.
