Register sparse images with large rigid shifts
==============================================

Use the opt-in Fourier initializer when the reference and target are separated
by a rigid translation too large for the local pyramid's capture range. It
estimates a global translation first; the existing IC-GN pyramid then refines
each subset, including its subpixel displacement and affine deformation.

Python
------

.. code-block:: python

   from live_dic import ICGNWGPUFullPyramid

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

Native Python Analysis with WGPU-Matching
-----------------------------------------

``ICGNWGPUFullPyramid`` can use the optional WGPU-Matching Python package to
obtain sparse CPM-HOG displacement seeds before its normal local solve. Install
the pinned source revision into the same Python environment as live-dic. This
WGPU-Matching revision requires Python 3.11 or newer; live-dic's base install
and its other initializers do not acquire this additional requirement:

.. code-block:: console

   python -m pip install "wgpu-matching @ https://github.com/jeffwitz/WGPU-Matching/archive/4dce5d0bbbf2ac7b58b35bfb9824afe4cb200d93.zip"

The base live-dic installation remains unchanged. Then select the initializer
in the native full-pyramid API just like ``fourier`` or ``temporal``:

.. code-block:: python

   from live_dic import ICGNWGPUFullPyramid

   with ICGNWGPUFullPyramid(
       initialization="wgpu-matching",
       motion_model="translation",  # or "affine"
       pyramid_levels=3,
       device_type="gpu",
   ) as engine:
       engine.set_reference(reference, roi_mask=mask)  # mask is optional
       result = engine.register(current)
       print(result.config["wgpuMatching"])

The matcher runs CPM-HOG on up to 512 evenly spaced, reference-valid coarse
subset centres. Only usable CPM translations become active coarse-level seeds;
the existing IC-GN pyramid solves those subsets and propagates their estimates
to finer levels. Affine local solving remains available, but the matcher seeds
translation at each centre and initializes higher-order terms to zero. At least
two pyramid levels are required. If no usable match is found, registration
raises an error rather than silently switching to zero initialization.

``matching_options`` optionally accepts CPM option names as a mapping, for
example to tune search radius or matching support. live-dic supplies the seed
points itself; callers cannot set ``seed_points`` or ``oracle_affine`` there.
The matching patch size and stride default to the local subset size and stride.
The reference ROI limits which coarse centres are requested, but CPM-HOG still
searches and describes image content outside that ROI; it does not provide
strict photometric isolation.

This Python integration is for offline Analysis, not a camera-rate path. It
creates a separate WebGPU matcher device and runs matching for every target,
adding device memory use and per-pair work. The result's ``config`` records the
initializer and seed counts, and ``timing["matchingInitializationWallMs"]``
records its wall time. There is no general accuracy or performance
qualification yet. The ``live-dic`` command-line supervisor does not expose
this Python initializer.

Browser workbench
-----------------

1. Load the images and open **Solve**.
2. In Analysis, select **Fourier** in the **Initialization** radio group.
3. Press **Start calculation**. Pending settings are applied to the retained
   reference before solving; no reference recapture is needed.
4. The last-pair summary shows the seed and
   its correlation. Results playback remains separate from calculation.

The selection works with the normal two-level preset and custom settings.
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
``ICGNBrowser(backend="webgpu-wgsl", initialization="fourier")``; the browser
CLI accepts ``--backend webgpu-wgsl --initialization fourier``. Three.js
explicitly refuses this option.

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
``ICGNBrowser(backend="webgpu-wgsl", initialization="global-affine",
motion_model="translation", global_affine_levels=6)``. Use
``motion_model="affine"`` when the local solve should also estimate
deformation. The ``webgpu-wgsl`` CLI accepts
``--initialization global-affine --motion-model translation
--global-affine-levels 6``. The Python API and CLI accept values from 5 to 8.
Three.js and the native ``ICGNWGPUFullPyramid`` do not implement this initializer.

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

WGPU-Matching in the browser Analysis workspace
-----------------------------------------------

The browser workbench also offers **WGPU-Matching · CPM-HOG (experimental)**
in the Analysis/Solve initialization group. This option is separate from the
local motion model: it estimates one global affine seed from sparse reciprocal
GPU patch matches, then runs the normal live-dic IC-GN solver with the selected
Translation or Affine model. The local correlation and rejection criteria
remain active. A successful matcher fit is not itself a DIC result.

The current browser adapter uses CPM-HOG on up to 512 evenly distributed
subset centres selected inside the reference ROI. It fits the accepted
displacements with a deterministic affine consensus and requires at least 12
usable inliers. If the matcher or consensus is insufficient, the pair reports
an error; it does not silently fall back to zero initialization. Each requested
32-pixel seed support must fit inside the image and the selected ROI. This can
leave too few seeds for a narrow ROI. The CPM descriptor and local affine fit
are still computed over the full image; pixels outside the ROI near a selected
seed can influence its match. The ROI therefore constrains seed selection but
does not provide strict photometric isolation for this initializer.

Select this option only for decoded image or video-file pairs in **Analysis**.
It is not available for a camera stream and requires CPU reference preparation.
Matcher output is read back as grayscale bytes, and CPM builds temporary GPU
descriptor and matching resources for each pair. Expect additional latency;
no real-time performance or general accuracy qualification is claimed yet.
Diagnostics record the requested/usable match counts, affine inlier count,
fit residual, matcher pyramid and timing. The usual DIC acceptance counts
remain the authority for whether local measurements were accepted.

The browser ships a small local source snapshot under
``web-app/vendor/wgpu-matching/`` so GitHub Pages and offline use require no
CDN or Python package installation. The snapshot revision and included files
are recorded in ``PROVENANCE.md``. It is independent of the optional Python
package and uses a browser-specific global affine consensus; do not assume its
seed construction or qualification is identical to the native Python API.

Recover rejected coarse subsets with median seeds
--------------------------------------------------

The shared WGSL full-pyramid engines can try a neighbour-based seed for a
subset rejected at a coarse scale. At each intermediate level, the solver
recentres accepted neighbours onto the rejected subset and takes the median
of each active motion coefficient. With at least four eligible neighbours,
that estimate may be transferred to the next finer level, where IC-GN runs
again from the seed.

For Python, select the median policy and stencil explicitly:

.. code-block:: python

   from live_dic import ICGNWGPUFullPyramid

   with ICGNWGPUFullPyramid(
       motion_model="translation",
       pyramid_levels=3,
       outlier_policy="median",
       median_filter_size=9,
       device_type="gpu",
   ) as engine:
       engine.set_reference(reference)
       result = engine.register(current)

In the browser workbench, choose **Median · repair rejected subsets** in
**Solve**, select a 3 × 3, 5 × 5, 7 × 7 or 9 × 9 stencil, and start a new
calculation. The saved-field smoothing control in **Results** is a separate
post-processing operation.

Accepted coarse subsets are copied through unchanged. A repaired subset's
private seed marker is not a valid measurement: the raw coarse status stays
rejected, and only a later IC-GN solve can make the finer point valid. The
finest-level status also remains rejected if its solve fails. Fewer than four
valid neighbours, an invalid reference window, or an excluded ROI point leaves
the estimate unfilled. This can expand a pyramid's recovery when a local solve
fails, but it does not guarantee capture of arbitrary motion and does not
replace Fourier or global-affine initialization when the starting shift is too
large.

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

Seed a sparse affine solve
--------------------------

For independently obtained initial affine estimates, the native single-scale
engine accepts a six-parameter array and a Boolean active grid::

   import numpy as np
   from live_dic import ICGNWGPU

   engine = ICGNWGPU(motion_model="affine", interpolation="linear",
                     outlier_policy="disabled")
   try:
       info = engine.set_reference(reference)
       # Shape: (grid_height, grid_width, 6), ordered as result.parameters.
       seeds = np.zeros((*info.grid[::-1], 6), dtype=np.float32)
       active = np.zeros(info.grid[::-1], dtype=bool)
       # Fill trusted initial states and mark their active positions here.
       result = engine.register_seeded(target, seeds, active, include_zncc=True)
       # result.zncc is read-only; unavailable values are NaN.
   finally:
       engine.close()

Only active, ROI-selected subsets are solved; inactive results are not
measurements. The reference is unchanged. This experimental host-array path
does not provide pyramid recovery or a GPU-resident interface.

For a subsequent frontier, ``seeds, active = engine.affine_frontier_seeds(
result.parameters, result.status == 1)`` transports the accepted neighbors
to adjacent unresolved subsets. These values are initial guesses: run
``engine.register_seeded(target, seeds, active)`` and apply the application's
consistency and retry policy before treating any new state as reliable. The
method transfers arrays through the host and reuses the prepared region-of-interest
links to avoid crossing masked cracks. It cannot identify an unmasked crack.
A running ``register_stream`` owns the solver until exhausted or explicitly
closed; seeded, frontier and independent registration calls fail while it is
active.
