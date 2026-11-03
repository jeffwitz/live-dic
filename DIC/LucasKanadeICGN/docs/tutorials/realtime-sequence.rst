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

``ICGNWebGL`` can replace the native class when a browser-controlled WebGPU
deployment is the requirement. Its binary transport is suitable
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

Measured throughput and result age depend on the configuration and hardware.
Use :doc:`../reference/performance` for archived campaigns and their exact
boundaries; the file/export examples here are not FPS benchmarks.

Run a complete file-to-fields example
-------------------------------------

Download :download:`register_sequence.py <../examples/register_sequence.py>`
or run the checked-in copy from the repository root:

.. code-block:: console

   $ .venv-icgn/bin/python DIC/LucasKanadeICGN/docs/examples/register_sequence.py \
       DIC/LucasKanadeICGN/assets/uhd_tensile/reference.png \
       DIC/LucasKanadeICGN/assets/uhd_tensile/frame_040.png \
       --output /tmp/live-dic-fields --device gpu --outlier-policy median

The output directory must be new. The example preserves one fixed reference,
uses the two-level B-spline engine and writes each complete field immediately.
It supports ``--model translation``, custom ``--subset-size`` and
``--subset-stride``, ``--median-size 5`` with ``--outlier-policy median``,
``--initialization temporal`` for chronological targets and ``--roi mask.npy``
for a reference-sized boolean mask. Use ``--help`` for all choices.
A ROI is supported for translation and affine; quadratic with a mask is refused.

The tensile frame above requires the affine model. To try translation, use
``assets/frame_000.png`` as reference and ``assets/frame_010.png`` as target;
the expected displacement is ``(2, -1)`` pixels. The archived UHD frames 100
and 199 are not qualified for full recovery from zero initialization with the
default two levels.

Choose ``--device cpu`` only when a software WebGPU adapter is installed;
``auto`` can select hardware or software and records the actual adapter.
``--pipeline-depth 2`` enables the throughput-oriented ordered stream. The CSV
records available result-age timings, while the printed sequence elapsed time
also includes file decoding, NPZ compression and disk writes. Neither measures
camera exposure latency. An interrupted run retains its manifest and completed
NPZ files; this example has no automatic resume or asynchronous disk writer.
