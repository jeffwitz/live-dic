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
3. Capture/select the reference again, or apply the mask, to prepare the option.
4. Start calculation in **Solve**. The last-pair summary shows the seed and
   its correlation. Results playback remains separate from calculation.

The checkbox works with the normal two-level preset and custom settings.
It disables temporal warm start for this preparation. The separate **Try 4
levels for large shifts** button tests a four-level local pyramid without
Fourier; it has a smaller measured capture range and is not a substitute.

The JavaScript API uses ``DICEngine.create({initialization:'fourier', ...})``.
Both CPU and experimental GPU reference preparation support it, including
``registerPair``. For Python browser supervision, use
``ICGNWebGL(backend="webgpu-wgsl", initialization="fourier")``; the browser
CLI accepts ``--backend webgpu-wgsl --initialization fourier``. Three.js and
GPU-IO explicitly refuse this option.

Interpretation and limits
-------------------------

The square reference zone is at most 1024 pixels wide. The centre is preferred;
if it lacks texture/ROI coverage, nine candidate locations are considered.
This is not an exhaustive image search. The reference ROI gates this cold
selection; the target is not clipped against the fixed reference mask.

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

For a local-only experiment, ``pyramid_levels=4`` is also available, with
default iteration limits ``(12, 8, 8, 16)`` and tolerances
``(0.01, 0.01, 0.005, 0.005)`` from coarse to fine. Four levels need an image
large enough to contain a subset at 1/8 resolution. Existing one-, two- and
three-level defaults remain unchanged.
