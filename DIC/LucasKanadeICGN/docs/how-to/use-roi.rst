Select complete subsets with a reference ROI
=============================================

The native ``ICGNWGPU`` and ``ICGNWGPUFullPyramid`` engines accept a boolean
reference-image mask in ``set_reference``. ``True`` means that measurements
may use that pixel. Translation and affine motion models are supported.
The mask is copied during preparation and then shared as immutable data by
the reference, all in-flight slots, and their results.

.. code-block:: python

   import numpy as np
   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   mask = np.ones(reference.shape[:2], dtype=bool)
   mask[:120] = False
   mask[600:800, 900:1100] = False

   with ICGNWGPUFullPyramid(motion_model="affine", device_type="gpu") as engine:
       info = engine.set_reference(reference, roi_mask=mask, roi_policy="full")
       for result in engine.register_stream(frames, pipeline_depth=2):
           measured = result.accepted_mask
           displacement = result.displacement[measured]
           print(result.populations)
       result.save_npz("measurement.npz")

``frames`` is an iterable of NumPy images. Reference preparation, including
the ROI, is outside this loop. Complete a stream before replacing its reference
or ROI; changing a reference while a stream iterator is suspended is unsupported.

Exact selection and coordinates
-------------------------------

``roi_policy="full"`` is the default. A window at its existing origin
``(x0, y0)`` is selected exactly when
``mask[y0:y0+subset_size, x0:x0+subset_size].all()`` is true. One forbidden
pixel, including a hole entirely inside the square, excludes the window.
No hidden erosion, centre shift, resizing, image modification, or additional
geometric margin is applied. Even and odd sizes, overlapping windows, and
all supported strides keep their existing coordinates. Provide an explicitly
eroded mask if the application needs a geometric margin.

.. code-block:: text

   reference mask       nominal windows       full selection
   +-------------+      +-----+-----+          included only if
   | allowed     |      | yes | no  |          every pixel is allowed
   |    hole X   |      +-----+-----+
   |             |      | yes | yes |
   +-------------+      +-----+-----+

The mask belongs to reference coordinates. A valid tracked point may move
outside its original masked location in the target image; its deformed
position is checked against image bounds, not against the fixed ROI.

The retained windows use the existing preprocessing and correlation equations.
This guarantees inclusion of the nominal square, **not strict photometric
isolation**: gradients, interpolation and prefilters can depend indirectly on
pixels outside it. In particular the recursive reference B-spline prefilter
has no exactly isolating finite halo. Masked-out image pixels are never zeroed.

Validation and special masks
----------------------------

Only a two-dimensional NumPy array of boolean dtype and the same height and
width as the grayscale reference is accepted. Conversion, thresholding,
cropping, and resizing are never implicit. Invalid masks and policies fail
before reference GPU execution. ``roi_mask=None`` preserves the historical
path. An all-true mask uses the same GPU resources, pipelines and arithmetic
as no mask. An all-false mask is valid and returns an empty dense field without
per-frame GPU upload, submission, solve, or readback.

``partial`` is not implemented and raises ``ValueError``, including when the
mask is absent. A ROI with the quadratic model is also rejected; quadratic
processing without a ROI is preserved. Browser engines and the browser-only
``register_pair`` helper explicitly reject ROI requests.

Pyramids, seeds, and neighbourhoods
-----------------------------------

Each coarse mask is prepared once. A coarse pixel is allowed only if every
pixel of the reflected 5 x 5 source footprint used by the five-tap reduction
is allowed. Decimation uses source coordinates ``(2*x, 2*y)``. A full-window
selection is then performed on that conservative coarse mask.

Excluded windows cannot supply seeds, participate in UOD/median filtering, or
be activated by selective recovery. A selected fine window with no admissible
coarse seed still runs from zero. A thin ROI can therefore reduce the useful
capture range without removing its fine measurements.

Links are also screened in reference coordinates to prevent averaging across
a crack between selected windows. The implementation conservatively requires
every pixel in the **closed axis-aligned box enclosing the segment between
centres** to be allowed. This includes all pixels touched by the segment,
including corner touches; it can reject additional links near oblique holes.
For cross-level links both centres are mapped to the original full-resolution
mask. Global component labels are not used as a substitute for this check.
These permissions are static. If UOD lacks its existing minimum number of
admissible neighbours it preserves the solver result; the pure median uses
its existing available-neighbour rule. Excluded windows are never filled.

The GPU receives per-window permissions during preparation, not an image mask
per frame. The full-pyramid path keeps its single submission and final readback;
the dense workgroup dispatch exits before correlation iterations for exclusions.
The admissible native subgroup specialization remains available with ROI.

Interpret results and export them
---------------------------------

Three independent fields describe a measurement:

* ``info.roi_selected`` / ``result.roi_selected``: static geometric selection
  (``None`` means the complete historical grid).
* ``reference_valid``: static numerical validity of the prepared reference,
  independent of ROI selection.
* ``result.status``: target solve status; ``-3`` marks a geometric exclusion,
  while existing convergence/rejection statuses retain their meanings.

Arrays keep their dense shape and finite values. Zero parameters at an excluded
position are not zero-displacement measurements. Use ``accepted_mask`` for
plots, errors, and downstream processing. ``populations`` reports the total
grid, selected/excluded positions, numerically admissible selected references,
accepted solves, and rejected selected positions. The success fraction uses
selected positions as denominator. Undefined fractions and RMSE return ``None``
and serialize as JSON ``null``.

``save_npz`` exports the image mask, grid selection, numerical reference validity,
parameters, statuses, iterations, parameter order, strict ``config_json`` and
population metadata. CLI sequence/benchmark exports also preserve ROI arrays
when given native results. Read archives with ``np.load(path, allow_pickle=False)``.
The mask and static arrays are referenced by live results rather than copied
per frame. Serialization occurs only when explicitly exporting.

The configuration records ``roi.policy``, dimensions, hash, and selected counts
per pyramid level (fine to coarse). The canonical SHA256 input is the ASCII
prefix ``live-dic-roi-v1`` followed by a zero byte, two unsigned 64-bit little-endian
integers ``(height, width)``, then one byte (0 or 1) per boolean pixel in C order.
The NPZ contains the actual mask; a hash alone is insufficient for reproduction.

Benchmark a saved mask
----------------------

.. code-block:: console

   $ python DIC/LucasKanadeICGN/benchmarks/benchmark_image_to_field.py \
       --backend native-pyramid --device-type gpu --roi-mask mask.npy \
       --motion-model affine --pipeline-depth 2 --warmups 12 --frames 160

Save a boolean mask with ``np.save("mask.npy", mask)``. For a PNG, explicitly
convert with a chosen threshold, for example
``mask = np.asarray(Image.open("mask.png").convert("L")) >= 128``.
The benchmark warms the actual stream and reports image-to-field latency,
result age, median/p95/p99, sustained throughput, selected population, iteration
distribution and raw samples. Preparation is reported separately. Lavapipe
checks correctness; its times do not qualify hardware GPU real-time operation.
