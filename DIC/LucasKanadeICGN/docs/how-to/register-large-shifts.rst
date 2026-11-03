Register sparse images with large rigid shifts
==============================================

Use the opt-in Fourier initializer when the reference and target are separated
by a rigid translation too large for the local pyramid's capture range. It
estimates a global translation first; the existing IC-GN pyramid then refines
each subset, including its subpixel displacement and affine deformation.

Python
------

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   with ICGNWGPUFullPyramid(
       initialization="fourier", motion_model="affine", device_type="gpu"
   ) as engine:
       engine.set_reference(reference, roi_mask=mask)  # mask is optional
       result = engine.register(current)
       print(result.config["globalTranslation"])
       print(result.timing["fourierWallMs"])

The cached reference spectrum is built once at ``set_reference``. Every target
gets its own estimate, so independent pairs need no chronological history.
``register_stream`` supports bounded slots; an ambiguous target raises an error,
with earlier pending native results delivered first. A rejected target can be
followed by another synchronous registration against the same reference.

The default remains ``initialization="zero"``. Fourier supports translation
and affine; quadratic is explicitly refused. Do not combine it with velocity
prediction or temporal fine correction. ``ICGNWGPU`` single-scale/selective
recovery does not expose this initializer.

Browser workbench
-----------------

1. Load the images and open **Solve**.
2. Check **Fourier initialization · up to 1024 × 1024**.
3. Press **Start calculation**. Pending settings are applied to the retained
   reference before solving; no reference recapture is needed.
4. The last-pair summary shows the seed and
   its correlation. Results playback remains separate from calculation.

The checkbox works with the normal two-level preset and custom settings.
It disables temporal warm start for this preparation. The separate **Try 6
levels for large shifts** button selects six local scales, 20 iterations at
the coarsest scale, and no temporal warm start. Five and six levels are now
available in the browser workbench and the ``webgpu-wgsl`` interface. They are
experimental: added scales can help a local solve follow larger displacement
increments, but do not guarantee convergence. More levels also add reference
preparation and per-image solve work. The existing four-level campaign did not
capture every large-shift case, so the six-level setting still needs validation
on each target sequence.

The JavaScript API uses ``DICEngine.create({initialization:'fourier', ...})``.
Both CPU and experimental GPU reference preparation support it, including
``registerPair``. For Python browser supervision, use
``ICGNWebGL(backend="webgpu-wgsl", initialization="fourier")``; the browser
CLI accepts ``--backend webgpu-wgsl --initialization fourier``. Three.js and
GPU-IO explicitly refuse this option.

Global affine initialization
----------------------------

For sparse pairs whose dominant motion includes scale, shear, or rotation,
the raw ``webgpu-wgsl`` browser engine also offers an experimental global
affine seed. Unlike the local pyramid, this cold CPU step fits one affine
transform over spatially sampled pixels from the entire reference mask. It uses
five image scales by default; the scale count can be increased to eight. That
one transform initializes every subset; the normal
local IC-GN solve then estimates the final field. The local pyramid depth
remains independently configurable.

In the workbench, configure the local solve and initializer separately:

1. Load a reference and target from an image, image sequence, or video file,
   then open **Solve**. This initializer is not available for a live camera.
2. Enable **Use custom settings** and choose **Motion model: Translation** for
   translation-only local solves. Choose **Affine** only when local subset
   deformation should also be estimated.
3. Select **Global affine initialization**, choose **Global affine fit scales**
   (5–8; default: 5), and press **Start calculation**.

Enabling global affine initialization does not change the selected local model
or disable its selector. The global fit is evaluated at each subset centre and
supplies the starting displacement. Translation mode still solves only the
two displacement components; affine mode estimates its six local parameters.
Through the Python-supervised shared core, quadratic mode without an ROI mask
also accepts this seed: it starts with the six affine coefficients from that
fit and zero curvature, then estimates its full local quadratic model. The
workbench currently offers translation and affine local models.
The global-fit scale count is independent of the local optical-flow pyramid
depth. Six global-fit scales can extend the coarse capture range on large
images; seven and eight are experimental. A loaded image or video file is
required.
The workbench carries the input kind through its worker so decoded local files
can use this initializer even though their frames reach WebGPU as external
textures. Live camera streams remain unsupported for this cold CPU fit.
Fourier, temporal prediction, and global affine are mutually exclusive. A
mask is optional; when supplied, only reference samples inside its white ROI
are used by the global fit. The mask is fixed in reference coordinates.

Python supervision can request six scales with
``ICGNWebGL(backend="webgpu-wgsl", initialization="global-affine",
motion_model="translation", global_affine_levels=6)``. Use
``motion_model="affine"`` when the local solve should also estimate
deformation. The ``webgpu-wgsl`` CLI accepts
``--initialization global-affine --motion-model translation
--global-affine-levels 6``. The Python API and CLI accept values from 5 to 8.
Three.js/GPU-IO and
the native ``ICGNWGPUFullPyramid`` do not implement this initializer.

This is not a camera-rate initialization path. It builds the selected
multiscale CPU reference representation once and performs a robust, sampled affine fit for
each target. For a browser file frame, the worker first reads back grayscale
pixels from the external GPU image so the CPU fitter can use them; this extra
readback and fit can add substantial latency. Live camera input is rejected
because doing this for every camera frame is not suitable for real-time
processing. It is
intended to establish a seed for difficult sparse pairs, not to replace the
GPU local solve or to guarantee convergence when the specimen has strongly
non-affine deformation, occlusion, or poor texture. Result metadata records
the center translation, affine terms, correlation, overlap, sample count,
per-level diagnostics, and initialization time. Treat low correlation or
rejected local subsets as a failed registration, not as a recovered field.

Interpretation and limits
-------------------------

The square reference zone is at most 1024 pixels wide. The image-centred patch
is tried first. If its aligned intensity correlation is too low, the initializer
samples a half-window-spaced grid, retains up to nine non-overlapping eligible
patches, and tries them in turn. It caches the first reliable patch for later
targets. This is a bounded search, not an exhaustive image search. The
reference ROI gates this cold selection; the target is not clipped against the
fixed reference mask. A patch with strong texture can still be invalid for a
rigid estimate when its local deformation changes the pattern. A fallback may
read back several target patches before solving, so ambiguous first targets
can take longer.

The implementation averages 2 × 2 pixels, applies a Hann taper and zero pads
for linear phase correlation. Its two-pixel seed is refined by IC-GN. Peak
score, geometric overlap and aligned intensity correlation reject implausible
matches. ``globalTranslation`` records the crop, estimate, score, correlation,
overlap and elapsed initialization time in results and exports. These checks
are guards, not a calibrated confidence interval or proof of a unique match.
Repetitive patterns, excessive shift, rotation, strong deformation, occlusion
or insufficient common texture can still fail.

Fourier executes on CPU: NumPy FFT in Python, dependency-free JavaScript FFT
in the browser Worker. External browser images add a cold-sized target-window
readback before each solve. It is deliberately outside the normal real-time
path. Include this cost in image-to-field latency; do not compare a GPU-only
solve time to a complete Fourier-initialized call. See
:doc:`../reference/performance` for the UHD controls and timing boundary.

For local experiments, five and six levels use coarse-to-fine default
iteration limits ``(12, 8, 8, 8, 16)`` and
``(12, 8, 8, 8, 8, 16)`` respectively. Their tolerances start at ``0.01``
and tighten to ``0.005`` toward the fine levels. Six levels need an image large
enough to contain a subset at 1/32 resolution. Existing one- to four-level
schedules and the two-level default remain unchanged. These schedules are
initial settings, not a qualified solution for every large shift.

A checkbox edit does not alter a field already saved in Results. The saved
field summary shows its actual initialization and accepted subset count. Return
to Solve and calculate a new field after changing settings. Input, mask and
solver edits stay pending; **Start calculation** prepares the selected reference,
mask and settings together before solving.
For a named pair such as ``Ref.tiff`` / ``Def.tiff``, select the reference
explicitly: natural filename ordering places ``Def.tiff`` first. Choosing the
currently previewed image as reference suggests another target when available;
you can still seek back deliberately for an identity check.
