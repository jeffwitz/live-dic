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

Keep the first image as the reference
-------------------------------------

The first functional milestone is a persistent first-image reference: prepare
``I0`` once, then obtain ``I0 -> I1``, ``I0 -> I2`` and so on. Reference pyramid,
gradients, subset statistics and inverse Hessians stay in memory and are shared
by the target slots. Only current-image data and solver results change.

For an iterable of acquired NumPy images:

.. code-block:: python

   from pathlib import Path

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   frames = iter(frame_source)
   output = Path("fields")
   output.mkdir(exist_ok=True)

   with ICGNWGPUFullPyramid(outlier_policy="median") as engine:
       engine.set_reference(next(frames))
       for index, result in enumerate(
           engine.register_stream(frames, pipeline_depth=1), start=1
       ):
           result.save_npz(output / f"field_{index:06d}.npz")

``set_reference`` is outside the loop. Every saved displacement and gradient
therefore refers to the first image, rather than the immediately preceding
target. Optional temporal prediction can reuse previous **estimates** as solver
initialization while preserving this reference. A new explicit reference or ROI
preparation starts a new analysis. In the browser workbench choose **Fixed
reference**, capture/load the first image and keep that reference for playback.

Prepared-data storage in GPU memory and field-file storage are distinct.
``save_npz`` above is a synchronous functional example; compression and disk
writes add to application elapsed time. Qualify their cost when measuring
recorded throughput, separately from the numerical engine call.

Measure the correct boundary
----------------------------

For the native engine, the outer ``perf_counter`` measurement includes Python
input normalization, GPU upload, solving, readback and NumPy result/statistics
construction. In the file-based example it also includes target image decoding.
Use already acquired contiguous NumPy images to measure the resident-image
boundary used by the live numerical benchmarks. Reference preparation remains
a separate one-time cost.

When explicitly using ``ICGNWebGL``, additional binary loopback transfer into
Chromium and back is included. Targets are not PNG-encoded and their fields do
not return as JSON. Browser adapter timing must not be described as the native
in-process path.

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

On the validation machine, two slots sustained 95.51 FPS in affine mode and
104.09 FPS in translation mode on the changing UHD sequence. Median result age
was 21.008 and 19.131 ms, respectively. Depth one reduced those ages to 11.354
and 10.496 ms. Throughput and result age are different engineering quantities;
choose the slot count from the needs of the measurement loop.
