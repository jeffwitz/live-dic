Your first DIC registration
===========================

This tutorial runs one affine DIC registration, inspects the displacement
field, and checks it against a known translation. It uses the images shipped in
the repository, so no camera is required.

Prerequisites
-------------

Complete :doc:`../how-to/install` first. Run the commands below from the
``tests_gpuio`` repository root.

Create a script named ``first_registration.py``:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWebGL


   reference = "DIC/LucasKanadeICGN/assets/frame_000.png"
   target = "DIC/LucasKanadeICGN/assets/frame_010.png"

   with ICGNWebGL() as engine:
       engine_info = engine.info()
       reference_info = engine.set_reference(reference)
       result = engine.register(target)

   print(f"GPU: {engine_info.device}")
   print(f"Reference preparation: {reference_info.preparation_wall_ms:.3f} ms")
   print(f"Mean translation: {result.mean_translation}")
   print(f"Accepted subsets: {result.accepted}")
   print(f"Mean iterations: {result.mean_iterations:.3f}")
   print(f"Translation RMSE: {result.rmse((2.0, -1.0)):.6g} px")

Run it:

.. code-block:: console

   $ .venv-icgn/bin/python first_registration.py

On the validated Intel Lunar Lake system, the important values are close to:

.. code-block:: text

   Mean translation: (1.999993, -0.999995)
   Accepted subsets: 900
   Mean iterations: 5.957
   Translation RMSE: 3.81939e-05 px

Inspect the fields
------------------

``result.displacement`` contains the translation at every grid point:

.. code-block:: python

   print(result.displacement.shape)  # (32, 32, 2)
   print(result.displacement[16, 16])

``result.parameters`` contains all six affine parameters:

.. code-block:: python

   print(result.parameters.shape)  # (32, 32, 6)

The final axis is ordered as ``tx, du/dx, du/dy, ty, dv/dx, dv/dy``.

Use the validity mask before computing statistics over individual subsets:

.. code-block:: python

   valid_displacements = result.displacement[result.accepted_mask]
   print(valid_displacements.mean(axis=0))

What you accomplished
---------------------

You started one persistent GPU runtime, prepared reference-dependent data once,
registered a target, and retrieved the complete affine field in Python. The next
tutorial keeps the same reference and processes a sequence without restarting
the runtime.

Next: :doc:`realtime-sequence`.
