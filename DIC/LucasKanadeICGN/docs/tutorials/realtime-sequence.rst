Build a persistent sequence loop
================================

A real-time-oriented DIC loop must not restart Chromium or rebuild the reference
for every frame. This tutorial reuses one engine and measures the complete
Python call for each target.

Create ``sequence_loop.py`` in the repository root:

.. code-block:: python

   from pathlib import Path
   from time import perf_counter

   import numpy as np

   from lucas_kanade_icgn import ICGNWebGL


   asset_dir = Path("DIC/LucasKanadeICGN/assets")
   reference = asset_dir / "frame_000.png"
   targets = [
       asset_dir / "frame_001.png",
       asset_dir / "frame_005.png",
       asset_dir / "frame_010.png",
       asset_dir / "frame_020.png",
   ]

   call_times_ms = []
   translations = []

   with ICGNWebGL() as engine:
       engine.set_reference(reference)

       for target in targets:
           start = perf_counter()
           result = engine.register(target)
           call_times_ms.append(1_000 * (perf_counter() - start))
           translations.append(result.mean_translation)

   print("Translations:")
   for target, translation in zip(targets, translations, strict=True):
       print(f"  {target.name}: {translation}")

   print(f"Median complete Python call: {np.median(call_times_ms):.3f} ms")

Run the script:

.. code-block:: console

   $ .venv-icgn/bin/python sequence_loop.py

Why the engine stays open
-------------------------

The context manager owns three resources:

* a loopback HTTP server for the WebGL assets;
* a headless Chromium process;
* the GPU textures containing the prepared reference.

Keeping the context open avoids process startup and reference preparation on
every frame. ``register_many(reference, targets)`` provides the same reuse as a
convenience method, but an explicit loop is better when frames arrive over time.

Measure the correct boundary
----------------------------

``result.timing`` reports stages measured inside the browser. The
``perf_counter`` measurement in this tutorial also includes Python image
normalization, PNG encoding, DevTools transport, JavaScript image decoding, and
result transfer back to Python. This outer measurement is the relevant one for
a Python processing loop.

The next step is to replace the file list with NumPy frames produced by an
acquisition source. See :doc:`../how-to/process-live-frames`.

Pipeline native frames for throughput
-------------------------------------

When the input is already a NumPy array, ``ICGNWGPU.register_stream`` can keep
a bounded number of registrations in flight. A depth of two lets GPU work on
the next frame overlap Python handling of the previous packed readback:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPU

   with ICGNWGPU(prefer_subgroups=False) as engine:
       engine.set_reference(reference_frame)
       for result in engine.register_stream(frame_source, pipeline_depth=2):
           consume(result)

Backpressure is automatic: the iterator never has more than
``pipeline_depth`` submissions outstanding, and results remain in input order.
Depth two improves sustained throughput at the cost of approximately one extra
frame of result latency. Use ordinary ``register`` when minimum single-frame
latency matters more than sequence throughput.
