Run the local browser application
=================================

For a guided first exercise, start with :doc:`../tutorials/browser-experiment`.

The browser companion accepts a camera, local video, local image sequence or an analytic demonstration and keeps
its frame processing on the device. It uses the same portable WGSL solver as
native Python. Earlier GPU-DIC and Three.js/WebGPU prototypes by Pierre Margerit
informed the methodological and browser-GPU exploration; the maintained solver
is a separate live-dic implementation. Python remains the recommended
supervisor for NumPy acquisition and complete NumPy result fields. See
:doc:`../explanation/acknowledgements` for the scope of this acknowledgment.

Prepare, calculate, then replay
-------------------------------

The header switches between **Live** and **Analysis** without resetting the
shared session. On a first visit with no saved workspace choice, Live is
selected when the browser enumerates a camera; otherwise Analysis opens. If a
source or calculation remains active while you review fields in Analysis, the
header keeps an activity badge and a **Stop** control visible.

1. **Input / Mask:** open the source, choose a reference and define the ROI.
   The input slider and previous/next image buttons show image previews; they
   do not request target correlation. Reference preparation builds static
   gradients, statistics and Hessians once.
2. **Solve → Start calculation:** calculate the chosen targets. **Save calculated
   fields for replay** is selected initially. It records full fields through the
   bounded RAM/disk library; turn it off deliberately for preview-only timing.
   For image folders, choose **Current target only** or **Current target to end
   of sequence**. Pause/resume continues after the last completed target; an
   explicit input seek makes that previewed target the next one calculated.
3. **Results:** after a completed finite calculation, or after pausing one,
   stored fields open automatically. The quantity selector, **Play results**,
   previous/next buttons and slider sit directly above the image. Choose
   **Displacement vectors** for a magnitude map with arrows, or choose a scalar
   quantity such as u, v, strain, equivalent strain or status. The vector
   spacing and magnification controls tune the arrows. These controls read saved
   fields and never request a correlation.

Playback keeps the chosen quantity, palette and limits across compatible
fields. Each field retains its stored processing recipe. Use **Apply recipe to matching
fields** explicitly for a consistent post-processing recipe across the sequence.
Computed parameters remain unchanged and applied recipes stay in provenance.
Imported projects/sessions restore their saved initial view. Replay rate is a
maximum of 1–30 fields/s (initially 2), independent of acquisition timestamps.
Slow analysis/loading reduces the actual rate instead of skipping saved fields.
Fast slider drags keep one active disk load and only the newest pending seek.
The last completed map stays visible while the next is loaded or analysed; the
status above it names the field still shown. Inspection and exports wait for
the requested field. Revisiting recent fields with the same recipe reuses a
bounded cache (eight analyses, at most 32 MiB, additional to the raw-field
cache). The Space shortcut plays/pauses saved results only in Results.

Open or serve it
----------------

Open the published `live-dic workbench <https://jeffwitz.github.io/live-dic/>`_.
The HTTPS site runs the calculation on your device; local images, video and
camera frames are not uploaded. New correlations require a supported WebGPU
browser. Saved-field review remains available without WebGPU.

The repository's Pages source is **GitHub Actions**. The existing workflow
publishes application changes on ``main`` and can be run manually. For a new
fork, an owner must first select **Settings → Pages → Source → GitHub Actions**.
The workflow retains an artifact and skips deployment if Pages is not configured;
it never changes that repository setting itself. Internal dotfile build markers
are excluded from the offline cache because GitHub Pages does not publish them.

From the repository root, build, serve and open it in one command:

.. code-block:: bash

   python3 tools/run_browser_app.py

This uses Python's standard library and your normal browser profile. If your
default browser lacks WebGPU, choose an installed supported browser explicitly,
for example ``--browser chromium``. ``--no-open`` prints the URL without opening
a window, ``--port 8001`` chooses another port, and ``--no-build`` serves the
existing assembled site. Ctrl+C stops this server. An occupied port does not
replace the files used by the existing server, and unrelated directories are
refused as build destinations.

To assemble and serve manually:

.. code-block:: bash

   python3 tools/build_pages.py --output _site
   python3 -m http.server 8000 --directory _site

Open ``http://localhost:8000/``. Do not open ``index.html`` as a local file.
Localhost is a secure development context; a phone connecting to another
computer's plain HTTP address generally needs HTTPS instead. The published
site uses HTTPS. The assembler copies canonical shaders, verifies their hashes
and includes the packaged core. No npm install, bundler, CDN or numerical
server is needed.

The source ``web-app/`` is an assembly input. Serve the assembled site, not
that directory alone. Its imports intentionally refer to the copied core and
canonical shader directories.

Try the demonstration first
---------------------------

Choose **Tensile demo**. The application loads a derived 1920 × 1080 tensile
replay, captures its reference and displays norm-coloured displacement vectors over
the associated current image. The
reference, image-formation recipe, raw input pixel hashes and encoded-file
hash are in ``web-app/demo/manifest.json``. The 200-frame stretch history comes
from the archived native-UHD fixture. Demo images are formed from a BOX-reduced
reference and bilinear inverse warping, then VP9 encoded. They are demonstration
assets, not replacements for the lossless UHD regression images.

The live truth error includes image formation, video compression and browser
colour conversion. Near the image edge the analytic motion can leave the
observable image; the local solver can then accept a wrong basin. The HUD
retains these errors and highlights large truth RMSE instead of hiding points.
See :ref:`browser-core-qualification` for the native reproduction. At each loop the temporal history resets. The browser may
drop source frames when a slot is busy; the source does not wait in an
unbounded queue. A 30 Hz movie cannot establish a completed 60 FPS acquisition
pipeline, even if the numerical graph finishes in less than 16.67 ms.

Use automatic or fixed norm colour limits and an independent arrow length
magnification. Inspect device/acquisition details. Store a field and open
Results to examine affine gradients, symmetric shear or equivalent strain.
These are kinematic quantities, not material stresses. Grey crosses mark
unavailable requested subsets; excluded subsets are not zero measurements.

Use a camera
------------

1. Choose a requested resolution and **Camera**, then grant the browser's
   camera permission. Audio is never requested.
2. Read the **actual** size/rate summary next to the camera controls and the
   detailed track settings below. Requested size/rate, initial settings and
   post-lock settings are recorded separately. A negotiated rate is not a
   timestamp-measured delivered rate; requested 60 Hz is a preference.
3. Keep the camera and specimen fixed while capturing a reference. Use a
   textured surface and controlled illumination.
4. Inspect exposure, focus and white-balance lock results. The app requests
   manual modes only when exposed by the camera API; unsupported locks remain
   visible. Watch acceptance and iteration counts when illumination changes.
5. Use **Pause calculation**, **Capture reference** or **Reset history** deliberately.
   Changing resolution restarts camera capture and stages a new reference; GPU
   reference preparation waits until **Start calculation**.
   A disconnected/ended camera changes the calculation action into **Restart camera**, preserving
   the last completed field for export. Restart captures a new reference; reapply
   the mask. This recovery does not manufacture missing measurements.
   A solver, mask or comparison change only updates the pending setup.
   **Start calculation** prepares the staged image, applied mask and selected
   settings together. Loading images, editing/applying a mask and changing solver
   controls do not build a GPU reference. After reference, mask or comparison
   changes, target exports/review remain disabled until a new target field
   arrives; old metrics are cleared. Drop counts restart on source replacement.

Timestamp origins are labelled: a reference captured from the video element
uses its media timeline, whereas track-processor frames use the browser/device's
VideoFrame clock. Do not subtract timestamps from these different origins to
claim camera exposure latency. Engine age starts at worker arrival.

The camera stays local. Frames are transferred to a dedicated worker as
``VideoFrame`` or ``ImageBitmap`` objects and closed after use. Where a track
processor is available it supplies the frames directly; otherwise the app uses
video-frame callbacks and ImageBitmap/VideoFrame capture. Input colour-to-gray
conversion runs on GPU. The cold reference preparation reads grayscale once;
target frames are not read into a CPU pixel array.

Open your own images or video
-----------------------------

Correlation uses browser-decoded 8-bit luminance. This path does not preserve
16-bit/HDR sensor precision; retain original files and use an explicit conversion
when that range matters. Browser decoding and colour management are part of the
image-formation pipeline to qualify.

Choose **Open image folder** or **Select images**. Images are decoded lazily,
one pair at a time, in natural relative-filename order (``frame_2`` before
``frame_10``). TIFF (``.tif`` / ``.tiff``) has a bundled, offline decoder. PNG,
JPEG, WebP, BMP, AVIF and GIF use the browser decoder; animated formats are
still images. Unsupported extensions are listed as ignored. No images are uploaded.

TIFF images and imported TIFF masks can use uncompressed, LZW, Deflate or
PackBits storage. Supported pixels are unsigned 1/8/16-bit grayscale,
8/16-bit grayscale with one alpha channel, 8/16-bit interleaved RGB(A), and
8-bit palette colour. TIFF orientation is honoured;
ICC profiles are not applied. Images are never resized. Choose a folder of
single-page TIFFs; stacks/multipage TIFF and BigTIFF are explicitly refused.
Signed/float samples, planar channels, reversed bit fill order, unsupported
compression and predicted tiles produce an error with the filename. Files
are limited to 256 MiB, dimensions to 16384 per side and 64 × 1024² pixels before pixel allocation. Decoding runs in a cancellable Worker;
there is no compiler, CDN fetch or PNG re-encoding.

The numerical engine still receives **8-bit** intensities. For a 16-bit TIFF,
the fixed mapping is ``floor(sample / 256)`` for every image, including the
reference. White-is-zero grayscale is inverted consistently. There is no
per-frame min/max or histogram adjustment. A 12-bit camera stored in a 16-bit
container uses only 16 output levels under this mapping: convert such inputs
with an explicit, common sensor range before import when precision matters.
Keep originals; TIFF support does not add a 16-bit correlation solver.
Decoded depth, compression, orientation and conversion accompany reference
and target file provenance in exports. Imported masks default to an automatic
luminance threshold of 128 with alpha at least 128, so white/light TIFF pixels
are retained and black/dark pixels are excluded without an extra conversion
step. Choose strict conversion for exact opaque black/white masks, or a custom
threshold when the grayscale separation needs adjustment. The mask editor
keeps its full-resolution binary geometry while displaying a bounded-size
preview on very large images.

In **Input / Mask → Define the specimen**, choose **Create mask from scratch**
to begin with every pixel excluded, then add domains with rectangles, ellipses,
circles or polygons. Nested shapes still define holes and islands. Choose
**Apply mask** to commit the edited geometry locally; **Full image** removes the
ROI. Neither action builds the GPU reference. The mask is prepared together
with the selected image and solver settings when **Start calculation** is pressed.

Images need not match the 720p/1080p/UHD camera presets, be square or have
sizes divisible by the subset size. They keep their original dimensions. A
subset must still fit in the coarsest configured level. Reference preparation
requests higher WebGPU limits when needed and supported by the adapter; there
is no fixed UHD-size ceiling. Very large images can still exceed the real
texture/buffer limits or available memory. They are refused with the required
resource size; automatic spatial tiling is not implemented.

Select a **Reference image**, then **Set selected reference**. The timeline,
**Previous image** and **Next image** select the target. **Start calculation** in Solve processes the selected range sequentially
without skipping a file. The input slider itself never calculates a field. The image rate cap limits replay
speed; zero removes that cap. This replay setting is not an acquisition clock.
An undecodable image or a different image size stops playback with its filename;
there is no implicit resampling. Change the source to begin a different size.

**Open video** uses the browser decoder and a temporary local Blob URL. Use the
timeline to preview while paused and **Start calculation** in Solve to calculate. Slow replay down to ×1/16
is useful for expensive analysis or software adapters; the playback-speed control
affects presentation, not the acquisition timestamps. Codec support depends on
the browser. No codec or decoder is downloaded by the application. The actual
decoded size is used even if another requested camera resolution is selected.

Fixed or consecutive reference
------------------------------

**Fixed reference** remains the default. Every target is compared with the
captured reference. For the first-image workflow, load/capture the first image
once and keep this selection: all fields are ``I0 -> In``. Its gradients,
statistics and inverse Hessians remain prepared on GPU. The experimental GPU
reference-preparation option is not required for this steady-state reuse;
it changes how the initial preparation is performed. Temporal initialization
can reuse a preceding field estimate without changing the reference image.
Explicitly capturing a replacement reference or applying a new ROI starts a
new preparation.

**Consecutive received images** performs a new IC-GN
correlation between the previous input and the current input; it does not
subtract two displacement fields and it never chooses the last successful
result as the next reference. This is an incremental field, not an accumulated
material trajectory. Strains refer to that pair's reference configuration.

For folders the pair is exactly files ``n-1`` and ``n``. For video/camera the
application retains only the latest received predecessor and has one pair in
flight. It drops and counts work when busy and rejects detected gaps; delivered
frames can already be missing upstream. It cannot reconstruct an unreceived
camera frame. Export records both input identifiers and video timestamps when
available. Seeking, pausing or switching modes clears the retained predecessor.
The analytical plate preset uses fixed reference and labels that restriction.

Consecutive mode rebuilds the grayscale reference, gradients, Hessian and ROI
preparation on every pair. Its reference-preparation time is reported separately
and included in worker result age. **This mode is not qualified for 4K60**; the
fixed-reference GPU timings must not be used to describe it.

To diagnose a low camera cadence, first note the actual decoded dimensions,
adapter identity, comparison mode and recording checkbox. **Completed FPS** is
completed fields per second, not the negotiated camera frame rate. The stage
line samples the latest admitted field at about 1 Hz: reference preparation,
observable solver call, and, when recording, extra full readback and main-thread
storage/admission. These are wall times with different boundaries, not hardware
GPU timestamps or exposure-to-display latency. Recording admission includes the
cache/index update and any review selection; it is not isolated disk I/O.

Changing comparison mode clears the age/FPS windows so fixed-reference samples
cannot hide expensive per-pair preparation. For a cadence check, select
**Fixed reference** and temporarily disable recording. This compares each target
to the captured reference; it changes the physical meaning of an incremental
measurement, so it must be an explicit user choice. If only consecutive mode is
slow, try the explicit experimental GPU preparation described below. See :ref:`camera-pipeline-diagnosis` for the measured simulated-camera
control and its limits.

The Solve pane distinguishes applied settings from pending edits. It reports
window/step, overlap, pyramid/iteration schedule, grid dimensions, ROI selection
and prepared-reference admissibility. The last-pair count can change in consecutive
mode; it does not reduce the requested ROI. Reference admissibility is not a
promise of target convergence.

The entire geometric ROI is requested on every pair, including after a failed
or textureless frame. Numerical invalidity and non-convergence are retained as
statuses and do not shrink the ROI. In a saved-field review, choose **Status · entire ROI** to distinguish
excluded, unavailable, rejected, unconverged, valid and flagged/replaced cells.
A failed cell is not a valid zero displacement. The mask remains in image
coordinates when the pair reference changes; it is not advected with material.

Try GPU preparation for consecutive images
------------------------------------------

In **Solve**, check **Experimental GPU reference preparation**, then press
**Start calculation** to apply it to the staged reference. Choose **Consecutive images** and initially leave
recording off to distinguish computation from durable result storage. This
opt-in is independent of the custom-solver checkbox. The applied solver summary
reports CPU or GPU preparation; pending edits do not modify an active analysis.

The GPU mode reuses resources and recomputes intensities, gradients and inverse
Hessians for every pair. The stage HUD shows **GPU reference + observable solver
call** as one combined time. Separate preparation time is unavailable in this
mode. Completed FPS still includes scheduling/rendering and, when enabled,
recording admission. A fast numerical kernel does not establish camera cadence.

The experimental mode supports translation/affine, uniform integration and one
slot. It keeps the applied image-coordinate ROI; changing it needs a new
preparation. Temporal history is discarded on each new pair reference. It does
not accumulate incremental deformation. CPU preparation remains the default;
uncheck the option and capture another reference to return to it.

This option adds persistent scratch textures/buffers, counted by the existing
GPU resource report. It does not use disk recording memory as GPU memory. See
:doc:`../explanation/gpu-reference-preparation` for validation and timing limits.

Overlap recording and control the live preview
----------------------------------------------

In **Solve → Live monitor → Lightweight live preview**, choose display rate and preview
width. The initial GUI values are 10 Hz maximum and 1920 px width. These
settings apply immediately without preparing a reference, resetting seeds or
changing recording. They are independent of recording overlap.

In **Solve**, check **Experimental pipeline overlap**, then press
**Start calculation**. Changes stay pending until Start; a staged reference
can be reused without capturing another image. Synchronous recording remains the default.
This option is independent of experimental GPU preparation and custom settings.

With overlap enabled, the supervisor reserves space before every recorded solve.
At most **two fields**, including a calculation in progress, may be waiting for
library admission. The GPU can compute the next field while the previous CPU
field is being committed to IndexedDB. A full queue applies backpressure. Camera
and video inputs retain their existing explicit drop policy for unadmitted
frames; local image playback waits for space and does not skip a file.

A storage failure pauses acquisition and recording. Both already completed
fields are retained in the results library in RAM, including a field completing
after the first failure. Export them or use **Retry disk save**, then explicitly
re-enable recording. Pending results are drained before pause completes, source
or reference replacement, project export or saved-session replacement. Closing
the page with pending data triggers the existing unsaved-data guard; it cannot
force a browser to finish writes after the page has been terminated.

The preview shows the latest completed field at the selected rate. **0 / Every
computed field** preserves unthrottled presentation. A 1920 px preview of a
3840 × 2160 acquisition is only a display raster: the original numerical images,
ROI coordinates, subset grid, parameter precision and saved fields remain UHD.
The preview never enlarges an image. Mask editing explicitly restores the
native-size reference preview. A paused/exported view is flushed to the latest
completed pair rather than leaving the last throttled preview on screen.

Choose **1 Hz**, **5 Hz** or **10 Hz** for slow visual monitoring independent
of recording cadence. **Suspend live preview · keep computing/recording**
suppresses live field drawing and bitmap transfer; progress still updates at
about 1 Hz. It does not pause acquisition or omit computed fields. Explicit
inspection, reference/mask views and Pause/export can still draw the latest
view. Changes apply immediately. A low display rate is a maximum, not a
target computational rate.

For dark or strongly textured specimens, **High-contrast vector outlines**
adds thin light/dark borders around the norm-coloured glyphs. It is off by
default and applies immediately without recalculating a field or preparing a
reference. Colour still represents the displacement norm; magnification only
changes displayed lengths, and the nearby length key includes the gain. Arrow
tips retain the measured current subset centres. Zero-length vectors remain
dots, unavailable requested points are crosses, and ROI holes remain empty.
The outline adds sparse drawing work, not scalar interpolation or strain fitting.

In **Live**, choose a **Vector colour map** immediately to the right of the
colourbar. **Original** preserves the existing appearance. Viridis, Cividis,
Magma, Plasma and Inferno provide sequential alternatives; colour still encodes
the displacement norm from zero to the displayed maximum. This display-only
choice updates the arrows and their legend together without rebuilding the
reference, solving a pair or altering recorded fields. In Results, use its
separate quantity/colour controls for the saved-field analysis.

The pipeline status distinguishes the displayed input, pending recording slots
and library admissions. Computed FPS may temporarily lead committed FPS by up
to the bounded capacity. Recording benchmarks must report committed throughput
and worker-arrival-to-admission latency alongside computed throughput. These
are not sensor-exposure timestamps. A 30 Hz preview does not promise 30 or 60
computed measurements per second.

The :ref:`browser-recording-overlap` controls compare committed recording rates
and worker-to-admission latency at 1080p and native UHD. They use a simulated
camera; they do not qualify a physical camera or a 60 FPS application.

Build and edit a mask with domains and holes
--------------------------------------------

Choose **Edit mask**. Capture pauses and the viewer shows the fixed reference.
Existing shapes stay in place. Select a tool to add rectangles, ellipses,
circles or polygons; there is no fixed shape-count limit. Drag opposite corners
for a rectangle/ellipse/circle. Click polygon vertices, then double-click or
choose **Finish polygon**.

Start with an outer domain. A shape whose selected reference pixels lie wholly
inside another becomes a **hole** automatically. A shape inside that hole
becomes a kept **island**. Repeat for further nesting. Separate outer domains
are united; overlapping outer domains are also united. Holes are subtracted
after uniting outer domains, so another overlapping outer domain does not fill
a hole silently. Islands are restored at the next depth. The region list names
each domain/hole and indents its depth.

Use **Select / move / resize**, select a region in the list or click it in the
viewer, then drag its body or white handles. **Add centred shape** creates a
rectangle, ellipse or circle without dragging; use its numeric geometry to
position it precisely. The mask toolbar above the image offers Apply/Cancel
without scrolling back to the panel. **Cancel edits** restores the last applied
geometry, including an imported image base, without recalculating a field. Polygon handles move individual
vertices. The numeric left/top/width/height fields use full-resolution reference
pixels; **Update geometry** applies them. A circle keeps equal width and height.
Selecting equal dimensions also makes an ellipse circular. Delete removes the
selected shape; its remaining nested shapes are reclassified. **Undo** restores
previous geometry, including deletion. **Full image** removes the mask.

Choose **Apply mask** when finished. This commits the pending mask geometry;
the GPU reference and static mask are built together when **Start calculation**
is pressed. No geometry work runs per video frame. To adjust it later, choose
**Edit mask** again. Edits remain unapplied until that action. With no shapes,
there is no mask. Rectangular/polygon scanlines use pixel centres and half-open
edges; ellipse/circle interiors are strict. Classification checks the entire
rasterized support, not just centres or corners. The mask does not depend on
canvas antialiasing or drawing order. Equal pixel footprints are duplicates.

The navigation bar immediately below the colourbar provides stepped zoom, a
directional pan pad and **Fit and centre** to restore the initial view. The
zoom keeps the current viewport centre; scrollbars remain available for
fine panning. These controls change presentation only: image coordinates,
masks and the DIC grid remain at the original resolution.

The Analysis workspace defaults to centre-connected partial supports at mask
boundaries, so a small mask can still retain informative windows. Choose
**Full window** to require every nominal subset pixel to lie inside the mask;
holes then exclude complete crossing subsets. Nothing tests the moving target
against the fixed mask. Conservative seed/neighbour links and preprocessing
scope are documented in :doc:`use-roi` and :doc:`use-partial-roi`. Browser
visibility truncation remains unsupported. Export metadata retains both the
full binary mask and the applied vector geometry, allowing the mask to be
reconstructed.

Import or download a black/white mask
-------------------------------------

After capturing a reference, choose **Import mask image** in **Mask**. White
retains pixels and black excludes them. The image must have exactly the reference
width and height; it is never resized. The default exact mode requires opaque,
strictly black/white decoded pixels. Grayscale, coloured or transparent input is
refused, rather than silently thresholded.

For an antialiased or grayscale source, explicitly choose **luminance threshold**
and an integer threshold 0–255 before importing. The conversion keeps pixels
with rounded BT.709 luminance at least that threshold and alpha at least 128.
The filename, file hash and conversion convention are recorded. Inspect the
preview and choose **Apply mask**; importing a mask does not run correlation or
modify saved fields. **Full image** removes both the imported base and vector restrictions.

The imported mask is a fixed base. Vector domains/holes clip it by intersection;
they cannot restore a black base pixel. A full-image rectangle is supplied so
that adding a nested ellipse/circle/polygon makes an extra hole immediately.
The preview shows exclusions in the base as well as vector exclusions. Undo
preserves the base and restores vector edits. To change an existing pixel hole,
edit/reimport the image itself or replace the mask with vectors.

**Download edited mask PNG** exports the current editable net mask as opaque
black/white at original size, including unapplied edits. Apply it locally before
calculation to select it for the next solve. Numerical field exports always keep the **applied**
binary mask and geometry, even while another edit is being previewed. In Python:

.. code-block:: python

   from PIL import Image
   import numpy as np

   mask = np.asarray(Image.open("live-dic-roi-edited.png").convert("L")) == 255
   engine.set_reference(reference, roi_mask=mask)

Original UHD plate with a hole
------------------------------

Choose **Plate with hole · UHD**. This loads the original archived reference
and frame 95 at **3840 × 2160**, without resampling, tiling or video encoding.
The outlined outer rectangle and circular hole are already present and editable.
Choose **Edit mask** to adjust them or add more regions. Applying the mask only
updates the pending ROI; **Start calculation** prepares the reference and mask
and solves the selected pair. **Plate load** selects frame 95
(250/0 MPa) or frame 225 (250/80 MPa). **Capture reference** restores the
original reference and preset for this case.

The circle has radius 112 px. The preset uses a half-pixel vector offset so
that rasterization matches the archived integer-index material mask exactly.
Its four image edges are an observation window in an infinite medium, not free
specimen edges. Only the circular boundary is traction free. The source
manifest, image hashes, material constants and load state are retained offline.

This is a static image-pair demonstration, not a video FPS benchmark. It
performs an explicit full-field readback once to score against the analytical
solution; video/camera targets keep their ordinary small-summary path. The
analytic displacement error and equivalent-strain error use selected centres
at radius at least 288 px, with a 32 px image margin. Coverage and the precise
population appear in the details/export. These interior scores do not qualify
near-hole strain precision. Analytical data are scoring inputs only and never
enter IC-GN. Changing the drawn mask changes the scored population.

Displacement vectors and equivalent strain
------------------------------------------

The live view offers **Norm-coloured displacement vectors** and **Current image
only**. It copies the associated current image and draws sparse GPU glyphs;
it does not create a scalar field raster, interpolate displacements or compute
strain. Adjust subset spacing, vector opacity, norm colour maximum and length
magnification independently. Auto spacing samples near the centre of each
block, keeping roughly 20 columns by 12 rows of glyphs; small fields stay
dense. Fixed spacing is also available. This samples the display only: all
subsets are still calculated and recorded. The colourbar is in measured pixels.

A vector's tip is at its current subset centre: reference centre plus measured
``(u,v)``. Its tail is tip minus magnified displacement. At magnification one,
the segment therefore joins the reference and current centres. The same
magnification applies to both components, without normalization to unit arrows.
The length key scales with zoom; it is separate from the norm colourbar. Auto
magnification fits ordinary arrows within part of a sampled cell, with a
one-pixel floor to avoid amplifying tiny identity-field noise. Very short arrows
are dots. Grey crosses mark unavailable requested cells at reference centres;
ROI-excluded cells have no glyph. A magnified arrow can cross a mask boundary;
its current tip identifies its measurement, not a new valid sample.

To display equivalent strain, open **Results** after calculation, then
**Equivalent strain** in Results. This pauses acquisition and analyses the
stored field. For direct infinitesimal affine gradients and ``εzz=0``:

.. math::

   \varepsilon_{xx}=u_{,x},\quad \varepsilon_{yy}=v_{,y},\quad
   \varepsilon_{xy}=\tfrac12(u_{,y}+v_{,x}),\qquad
   \varepsilon_{VM}=\tfrac23\sqrt{\varepsilon_{xx}^2-
   \varepsilon_{xx}\varepsilon_{yy}+\varepsilon_{yy}^2+3\varepsilon_{xy}^2}.

This is strain, not stress or a general finite-strain invariant. Saved views
use reference coordinates and their own colour scales. Palette, range and vector
styling reuse the completed component values and statistics. Changing the
processing recipe computes a new analysis; it never changes the archived raw
field. Acquisition recording does not run this analysis on every field. It
stores full coefficients/statuses on browser disk with the bounded CPU cache;
it does not archive original camera images. Keep source images separately.

Correlation settings without hidden changes
--------------------------------------------

The numbered **Input / Mask / Solve / Results** navigation keeps the current
step visible. Calculation controls stay in Solve; saved-result playback stays above the image. Capture reference remains beside the viewer. The
quick guide explains the workflow and keyboard shortcuts (Space: play/pause saved results in Results;
R: capture reference; Esc: clear the current measurement). Shortcuts do not
operate while editing an input, modal or mask.

In **Solve**, enable **Use custom settings** to choose translation/affine,
window width, grid step, interpolator, pyramid levels, coarse/fine iteration
limits and solver spatial policy. **Start calculation** prepares the selected
image, applied mask and pending settings together, then solves. Loading an
image, changing a solver setting or applying a mask does not initialize the
GPU. Unchanged starts reuse their preparation; immediate display controls do
not rebuild it. Exported configuration describes the applied settings,
not the pending form. Three-level custom runs use four middle-level iterations;
one level uses the fine limit. These custom combinations do not inherit a
qualified preset's precision/FPS claim. Translation has no direct strain field;
use displacement-neighbourhood strain analysis in the results lab.

The **Motion model** and initialization radio options control different stages.
For a translation-only local field with a global affine seed, enable **Use
custom settings**, select **Translation**, then select **Global affine
initialization**. This fits an affine transform to the loaded image pair and
evaluates its displacement at each subset centre; the local solve still updates
only the two translations. The radio option does not reset the model selection or
disable its menu. The global-fit scale selector is also separate from the local
pyramid-level selector. This cold initializer is intended for sparse loaded
image pairs, not camera-rate processing. See
:doc:`register-large-shifts` for the full procedure and limitations.

GPU seed propagation in Analysis
--------------------------------

In **Analysis → Solve**:

1. The default **Affine** preset already uses propagation. To choose the local
   model, enable **Use custom settings** and select **Translation** or
   **Affine**.
2. Keep **B-spline** interpolation and leave **Experimental GPU reference
   preparation** unchecked. CPU reference preparation is the existing default;
   the subsequent correlation and propagation still run on the GPU.
3. Leave **GPU seed propagation** on **Automatic**. For the default preset and
   these compatible custom settings, the workbench applies two passes per
   level. You can choose
   **Disabled** or explicitly select one to four passes. More attempts cost
   more work and are not a convergence or frame-rate guarantee. The GPU skips
   further searches once an empty work list proves that the state has not
   changed.
4. Keep the desired median/UOD spatial policy. It runs after the propagation
   re-solves. In Affine mode each candidate contains only neighbour tx/ty; the
   retry zeros the four affine derivatives, then estimates the full affine
   state from the destination patch. Global affine or Fourier initialization
   can be selected independently of this local stage.
5. Press **Start calculation**. Editing the option only marks preparation
   pending; it does not start calculation or rebuild immediately.

The GPU tests neighbour hypotheses on rejected subsets and reruns LK only for
an improved seed, using an indirect work list. It does not replace a measured
point with a neighbour vector. **Automatic** uses two passes for Translation or
Affine with B-spline interpolation, CPU reference preparation, uniform full
windows and one processing slot; incompatible settings keep propagation off.
Explicitly requesting propagation with an unsupported combination gives an
error. The fixed Live preset does not use propagation. The direct
``DICEngine`` API keeps its default at zero. Changing the option requires a
new preparation at Start; saved fields and NPZ exports retain their applied
``spatialPropagationPasses`` value.

See :doc:`../reference/propagation` for the illustrated algorithm and
:doc:`../reference/performance` for the qualified timing boundaries. Native
prototype measurements must not be interpreted as browser camera/recording FPS.

Review and process a computed field
-----------------------------------

In **Results**, open **Results** after calculation. Capture pauses and one full
field is read. The saved view uses reference image coordinates and a plain field
background, never an older field overlaid on an unrelated current image.
**Return to live view** restores GPU rendering. A saved field retains its own
model, configuration, ROI and input pair even after changing the active source.

The quantity selector stays above the image. The colour-map and clim controls
sit to the right of the colourbar to keep the Results view compact; they stack
below it on narrow screens. Choose displacement, strain components, equivalent
strain, principal strains, polar rotation, iteration count or acceptance
criteria / solver status.

Choose **Automatic**, **Viridis**, **Cividis**, **Plasma**, **Inferno** or
**Magma** for a perceptually uniform sequential map, or choose the signed
blue–white–red, blue–green–gold or grayscale map. Automatic preserves the
existing signed diverging/nonnegative sequential behaviour. The sequential maps
are bundled lookup tables; the page does not load a plotting package or CDN.

The default clim uses the valid-value **2nd and 98th percentiles**. Change the
minimum and maximum percentile fields directly to choose another range; for
example, 1 and 99. Signed quantities keep symmetric limits around zero. Choose
**Full value range** to include every valid value, or **Manual limits** to type
the lower and upper clim in the units shown by the legend (including µε for
strain components). Outlying values are counted as colour-clipped; computed
measurements and exports are never clipped or changed by these display limits.
Solver status is categorical, so its six status colours do not use a scalar clim.

Palette and clim edits recolour the cached displayed field. They do not rerun
correlation or derived field analysis. The histogram shows the valid value
distribution whenever no line profile is selected.

By default, subsets outside the active acceptance population stay grey. Select
**Show masked estimates** beside the colourbar to display finite solver states
for subsets where at least one iteration ran. The selected display
recipe is applied to those diagnostic estimates; valid displayed values stay
unchanged. ROI-excluded and never-solved subsets remain blank. These values
are diagnostic only: the status map, summary statistics, measurements and
processed CSV continue to use the accepted population. Select **Acceptance
criteria · solver status** to inspect every status category independently of
the scalar field.

Computed estimates are
preserved, including any solver median already applied. Optional displacement
median/Gaussian smoothing and VSG plane-fit strain are separate analysis
operations. **Show computed field** temporarily compares with the computed
values; **Restore processing recipe** returns to the selected operations.
The processing worker keeps these operations off the acquisition worker.
A pending or failed derived view is dimmed, its state is displayed, and derived
exports are disabled until an analysis is available. Resizing or changing another
control while a calculation is pending never presents the previous recipe as new.
Saved-field vectors use processed displacements with explicit magnification;
an optional contour follows the exact binary ROI pixels. Image x points right
and y points down. These displays do not modify computed values.

The quantity list also exposes **Computed local model coefficients**, including
all six displacement second derivatives in an imported native quadratic field.
Translation-only fields do not expose unestimated zero gradients. These diagnostic
quantities always read the stored solver coefficients; Gaussian/median displacement
post-processing and VSG do not change them. A visible source note distinguishes
them from derived strain. Displacement coefficients use the chosen length unit,
first derivatives are dimensionless, and second derivatives use reciprocal length
(e.g. ``1/mm``). This is displacement curvature, not plate-bending curvature or
stress. The native solver's original spatial policy remains part of the data.
Browser acquisition still supports translation/affine, not a new quadratic solver.

Choose an odd analysis window in grid points and inspect its pixel span. More
smoothing trades spatial resolution for lower noise; it is not a robustness
proof. Missing values stay unavailable and neighbourhoods do not cross excluded
ROI pixels. VSG needs non-collinear valid support. Green-Lagrange and infinitesimal
strain are distinct choices. Length units use an explicit units-per-pixel scale,
not camera calibration; strain remains dimensionless. See
:doc:`../explanation/browser-workbench` for equations, rank checks and support.

Click the field or enter reference pixel-index coordinates and choose
**Place measurement**. Image x points right and y points down.
**Point probe** reports a snapped subset's coordinates, status, iterations and
value. For a **Line profile**, click two endpoints and export actual grid samples;
invalid cells remain gaps. **Extensometer** reports endpoint distance change
and engineering gauge strain, distinct from a local strain tensor. It may span a hole: it uses only the two endpoints, not a continuous material
line. It requires
two distinct accepted endpoints and does not infer measurements at invalid points.
**Region statistics** uses two opposite rectangle corners. It reports valid,
selected and total grid counts, mean, standard deviation, extrema and percentiles
of the displayed quantity at subset centres. Holes and invalid points do not
become zeros. Each valid subset has equal weight; this is not a physical area
integral, especially with overlapping windows. Status view reports category
counts instead of averaging status codes.

**Save NPZ + processing recipe** preserves computed arrays, separate processed
arrays, their validity masks, metadata and binary ROI. Open it in Python with:

.. code-block:: python

   import json
   import numpy as np

   with np.load("live-dic-1.npz", allow_pickle=False) as saved:
       parameters = saved["parameters"]   # grid height, grid width, 6 or 12
       strain = saved["processed_exx"]
       valid = saved["strain_valid"].astype(bool)
       recipe = json.loads(str(saved["recipe_json"]))

**Open saved NPZ field** restores a workbench archive or the native CLI
parameters/status/iterations/config_json layout, including compressed NumPy
archives. A stored ROI supplies exact image dimensions. Otherwise a dialog
requires explicit width/height and any missing subset geometry; current-image
values are suggestions to check, never a guessed extent. For sequence archives,
choose the field number explicitly. One field is added per import. The source
archive SHA-256, geometry source and selected index are retained. An archive
that declares an ROI but omits its binary mask is refused. The importer has a 256 MiB uncompressed limit and
rejects pickle/object arrays. **Export displayed values CSV** leaves unavailable
measurements empty and retains solver statuses. **Save figure PNG** includes the
full-resolution field and annotations, a quantitative colour bar or six labelled
status categories, reference/target label, subset geometry, units, processing
convention and attribution. Captions wrap; exceptionally long imported text is
abbreviated with an ellipsis. Complete metadata remains in NPZ/project exports.
The PNG is a presentation artifact, not a replacement for those numerical files.
Images are not included in saved fields; retain your original sequence separately.

Save a results project
-----------------------

**Save results project** pauses acquisition and exports computed fields, shared
ROI masks, provenance and applied recipes. Small libraries use one
``.dicproject`` ZIP. Long libraries open a parts dialog: download each numbered
part and then its index. Where supported, **Write all parts to a chosen folder**
creates a new experiment subfolder. Folder access is optional; ordinary downloads
work without that browser API. Keep original images separately.

Import accepts multiple parts in filename order. Each part is validated before
its fields are added; successful earlier parts remain available if a later part
is invalid. It does not replace the acquisition source or solver settings. Parts
target at most 64 MiB (less with a small cache) and 128 fields. Each part remains
independently readable; shared masks are repeated once per part. Import retains
its 256 MiB uncompressed / 512-field safety limits per archive.

A download request cannot prove that a file reached disk: check your downloads.
NPZ exports of individual fields do not mark the whole project as saved. Use
**Close results view** to begin a separate recording; committed fields remain
in **Saved sessions**. See :doc:`store-long-sequences` for reopening browser
sessions, failure recovery, memory accounting and independent backups.

Browse and reuse processing
---------------------------

Opening Results selects a saved field automatically. Its visible transport bar
provides the quantity selector, timeline slider, previous/next, field number and
Play/Pause controls. Playback uses compatible saved order, waits for completed
analysis and never resumes the acquisition source. Changing quantity or palette
can continue during replay. Dragging the slider pauses replay and loads the
latest requested field; it never recalculates it.

**Apply recipe to matching fields** copies the current applied recipe to that
same compatible population. It changes no computed parameter, status or iteration.
Other analyses keep their recipes. This action is unavailable during a temporary
"Show computed field" comparison. Each field is recomputed on demand when viewed;
the project stores the recipes. Fixed colour limits are useful for comparison.

Extract a measurement through saved fields
-------------------------------------------

Create and name any number of points, lines, rectangular zones and
extensometers. Select a tool, place its geometry with clicks or the coordinate
form, then choose **Add as new**. Select an item in the measurement list to
inspect or edit it; **Update selected** changes that item, while **Delete
selected** and **Delete all** remove measurements. The overlays use reference
coordinates and remain attached to the saved Results view/project.

Choose **Extract all through saved fields** to evaluate every saved measurement
on compatible fields. The application applies the current analysis recipe one
field at a time in a separate worker, so it does not load a whole long recording
into memory. Progress and cancellation are available. Fields from another
analysis epoch, ROI or grid are excluded; matching dimensions alone do not prove
a shared reference. Native sequence fields imported from the same archive share
its source hash. Selecting a measurement after extraction displays its temporal
trace; for a line, the chart shows the middle sample and the CSV retains every
sample along the line for every field.

Choose **Export time history CSV** for a long-format table. Each row records the
saved-field order, source frame/time when available, measurement identity and
type, coordinates/sample index, value and unit. Point histories have one row
per field; lines have one row per sampled subset and field; zones report the
valid-sample mean, spread, bounds and population; extensometers report reference
and deformed lengths, extension and engineering strain. Missing or rejected
values are blank, never zero-filled or interpolated. Measurement definitions
and their selected quantity are also retained in the results project. CSV
contains derived values, not source images.

The chart uses **saved field order**, not an inferred physical acquisition clock.
Exports retain input indices and media timestamps when present. Failed
measurements remain null gaps. Region histories report a mean and the number of
valid samples for every field, because a changing valid population can bias an
average. Neither zero filling nor success-only compaction
is used. Consecutive comparisons remain increments at fixed image locations,
not accumulated material trajectories. The extraction CSV has an accompanying
JSON containing units, input pair, analysis recipe and chosen geometry.

Record fields deliberately
--------------------------

**Save calculated fields for replay** records every completed admitted field. It
adds a full readback and can lower throughput. **Worker input → GPU field** keeps its
observable-GPU boundary; each recorded field separately reports full-readback
time and worker-arrival-to-recorded-field age. These are not camera-to-Python
timings. Video/camera can still drop and count work under backpressure; folders
continue to await every image.

Computed fields are committed to browser-local IndexedDB before the next
recorded field is admitted. A configurable CPU cache retains up to 1 GiB;
automatic mode uses coarse host-memory information when available and a
conservative 128 MiB otherwise. Old committed fields leave the cache and are
reloaded on demand. Their values are not discarded or reduced in precision.
The cache target excludes analysis scratch, metadata objects, GPU/decoder
resources and export buffers; it is not a total process-memory limit.

Disk/quota failures stop recording and acquisition explicitly, preserving the
completed field in RAM. Export it or use **Retry saving RAM-only fields** before
closing. Browser disk quotas, privacy settings and storage eviction still apply:
export project parts for an independent backup. No images or results are
uploaded. Switching sources preserves the library and the save preference; recording resumes only after
you explicitly enable it again.

Review without WebGPU
---------------------

New correlations require WebGPU. Opening saved NPZ fields and results projects
uses CPU analysis and a 2D canvas, so these tools remain available when the
application reports WebGPU unavailable. The Results pane opens automatically.
OffscreenCanvas and worker absence also have explicit main-thread fallbacks for
this review path. Heavy processing can then block interaction temporarily.
This is not a CPU implementation of IC-GN inside the browser.

Profiles and device limits
--------------------------

The **Live** workspace always uses the fixed two-level affine B-spline latency
preset: 16 × 16 windows, stride 16, one processing slot, temporal velocity
initialization, fine correction and the existing median spatial policy. Its
display rate and preview width are separate presentation controls. Fourier,
global-affine initialization, custom solver parameters and the six-level
large-shift preset are available only in **Analysis**, where they can be used
for decoded files without imposing per-frame CPU initialization on a stream.

Analysis starts with the same portable solver preset and can opt into custom
solver settings. The old **Profile** selector was removed because its choices
fed the same solver configuration. Export metadata labels the preset as
``portable`` and custom settings or initializers as ``custom``; this label
does not select a device-specific kernel or guarantee a particular FPS. Core API
defaults remain independent of these browser workspaces.

On startup an on-device zero-motion microbenchmark tries the permitted
resolutions. Its short measurements guide the resolution recommendation; they
do not replace the changing-trajectory qualification or a long thermal run.
Minimal single-level processing stays disabled because it has not passed the
required trajectory gates. Custom one-level and cubic configurations remain
experimental.

An adapter can satisfy the binding limits and still lack memory for UHD.
Choose a lower resolution after allocation/device-loss errors. Missing WebGPU
produces an unsupported-browser message; this application has no WebGL fallback.
The separate Python Three.js backend requires WebGPU and has no WebGL2 fallback.

Export and offline use
----------------------

**Download field CSV + metadata JSON** pauses capture, waits for pending work
and reads the complete field once. CSV rows include coordinates, statuses,
iterations and parameters. Retain the JSON: it records configuration, device,
acquisition settings, frame timestamps, timing samples and the full ROI mask
when present. Some browsers ask permission for multiple downloads.

After the service worker finishes its first installation, the application,
shaders and demonstration can be reopened offline from the same URL. Camera
permissions and browser WebGPU support still apply. Storage quotas and private
browsing may prevent offline installation; the demo is a substantial local
asset. No Python server is needed once a hosted copy has been cached.
A new service worker waits for existing app tabs to close before activating,
so an active numerical session does not mix shader versions. Reopen after
closing the old tabs to update.

Hand-held capture is a demonstration, education and field-screening tool.
There is no current metrology qualification for phone optics, rolling shutter,
exposure stability or hand motion. See :doc:`../explanation/browser-portability`
for tested and untested devices and :doc:`../reference/performance` for the
separate timing boundaries.

Developer reproduction
----------------------

With the Python package and Chromium installed, exercise the same assembled
application under a subpath, including explicit downloads and offline reload:

.. code-block:: bash

   python3 tools/smoke_browser_app.py --site _site --frames 1800 --duration 70 \
     --offline --export --report /tmp/browser-display.json

Run this without other tests/builds or GPU campaigns in parallel. Add
``--software`` for a correctness smoke test at a deliberately slowed demo rate;
software time is not a hardware performance result. The source replay is
30 Hz. The report separates worker-field ages and an explicit full export;
it does not measure camera exposure-to-screen latency.

Large translations in separated images
---------------------------------------

In Analysis / Solve, select **Fourier** in the **Initialization** group, then prepare
the reference again. It supplies a global translation seed for each target;
the regular pyramid still refines the field. It adds CPU/readback latency and
refuses ambiguous matches. See :doc:`register-large-shifts`.

Troubleshoot sparse rejected results
------------------------------------

The top of Results identifies the saved field's solver initialization, window
size/step, input pair and accepted count. A field calculated with temporal
initialization keeps that provenance after Fourier is selected in Solve.
Post-processing, replay and smoothing never rerun correlation or recover
rejected subsets. Calculate a new field after changing settings.

For large independent-pair translations, enable Fourier and then start the
calculation. The applied summary must show ``Initialization: fourier``. A
first temporal estimate has no prior motion to predict. Larger windows (for
example 32/16) may help noisy/fine textures, at the cost of spatial resolution
and processing time; acceptance alone does not establish metrological accuracy.
See :doc:`register-large-shifts`.


Troubleshoot canvas presentation
--------------------------------

A browser can expose ``navigator.gpu`` and select a hardware adapter while
canvas presentation still fails. ``GPUCanvasContext.getCurrentTexture: Canvas
not configured`` concerns the drawing surface; adapter detection alone does
not establish that rendering works. The application configures the WebGPU
canvas after a size change before acquiring its next texture.

Check the build identifier in the downloaded device report and reload the
latest application. Include the browser version, adapter/driver and browser
console messages when reporting a failure. Vulkan image-allocation or swapchain
errors in Firefox's ``about:support`` are useful driver/presentation evidence;
changing solver tolerances cannot repair that stage. Firefox and mobile
hardware coverage remain unqualified in :doc:`../explanation/browser-portability`.

The Python launcher's ``BrokenPipeError`` means a browser closed an HTTP
connection while a resource was being sent, for example during a reload. It
is distinct from a WebGPU validation error; retain the browser console's first
GPU error when diagnosing presentation failures.

Solver workgroup width
----------------------

The browser workbench keeps the solver's qualified default of 16 invocations
per workgroup and does not expose a width selector. This is an implementation
and benchmarking parameter; changing it is useful only when comparing complete
solver runs on a specific adapter. The benchmark and low-level JavaScript and
Python APIs retain the parameter for controlled engineering experiments. See
:doc:`../reference/performance` for the existing measurements.
