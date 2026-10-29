Build a persistent sequence loop
================================

A real-time-oriented Python supervisor must not rebuild the device or reference
for every frame. This tutorial reuses the recommended native full-pyramid
engine and measures the complete Python call for each target. It represents the
project's intended NumPy deployment, not a browser-kernel microbenchmark.

Create ``sequence_loop.py`` in the repository root:

.. code-block:: python

   from pathlib import Path
   from time import perf_counter

   import numpy as np

   from lucas_kanade_icgn import ICGNWGPUFullPyramid


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

   with ICGNWGPUFullPyramid() as engine:
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

The context manager owns the persistent native resources:

* the wgpu adapter, device, and queue;
* compiled compute pipelines and immutable reference buffers;
* the GPU textures containing the prepared reference and target pyramid.

Keeping the context open avoids device/runtime startup and reference preparation on
every frame. ``register_stream(targets, pipeline_depth=1)`` provides the same
ordered one-frame-at-a-time behavior as a convenience method; its default
two-slot configuration instead improves sustained throughput. An explicit
loop remains useful when frames arrive interactively over time.

``ICGNWebGL`` can replace the native class when browser or WebGL2
compatibility is the deployment requirement. Its binary transport is suitable
for qualification, but every frame and field still crosses the
Python/Chromium boundary. A high-throughput Three.js deployment needs the
JavaScript supervision model described in :doc:`../explanation/supervision`.

Measure the correct boundary
----------------------------

``result.timing`` reports stages measured inside the browser. The
``perf_counter`` measurement in this tutorial also includes Python image
normalization, binary loopback transfer into Chromium, GPU upload, packed
result transfer, and NumPy result construction. Targets are not PNG-encoded
and their fields do not return as JSON. The outer measurement remains the
relevant one for a Python processing loop because the browser and GPU copies
are still real work.

The next step is to replace the file list with NumPy frames produced by an
acquisition source. See :doc:`../how-to/process-live-frames`.

Pipeline native frames for throughput
-------------------------------------

When the input is already a NumPy array,
``ICGNWGPUFullPyramid.register_stream`` can keep a bounded number of complete
coarse-to-fine registrations in flight. A depth of two lets GPU work on the
next frame overlap Python handling of the previous packed readback:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   with ICGNWGPUFullPyramid() as engine:
       engine.set_reference(reference_frame)
       for result in engine.register_stream(frame_source):
           print(f"result age: {result.timing['resultAgeWallMs']:.3f} ms")
           consume(result)

Backpressure is automatic: the iterator keeps at most two images in flight by
default, and results remain in input order. This is an execution-pipeline
setting, not an image-pyramid depth. The default improves sustained throughput at the cost
of approximately one extra processed frame of result latency. Pass
``pipeline_depth=1`` when minimum result age matters more than sequence
throughput, or use ordinary ``register`` for isolated pairs. Each slot owns its
target pyramid, mutable solver states, seeds, and readback; immutable reference
data remain shared. Contiguous grayscale ``uint8`` arrays are uploaded directly
to the native texture and never pass through PNG, Chromium, DevTools, or JSON.

On the validation machine, two slots sustained 93.16 FPS in affine mode and
105.58 FPS in translation mode on the changing UHD sequence. Median result age
was 21.408 and 18.888 ms, respectively. Depth one reduced those ages to 11.478
and 10.331 ms. Throughput and result age are different engineering quantities;
choose the slot count from the needs of the measurement loop.
