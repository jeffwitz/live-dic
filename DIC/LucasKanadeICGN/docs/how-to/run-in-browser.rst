Run the local browser application
=================================

The browser companion accepts a camera, local video, local image sequence or an analytic demonstration and keeps
its frame processing on the device. It uses the same portable WGSL solver as
native Python. Pierre Margerit's foundational GPU work made this project
possible. Python remains the recommended supervisor for NumPy acquisition and
complete NumPy result fields.

Open or serve it
----------------

The Pages workflow is prepared, but enabling or publishing the site is a
repository-owner action. In GitHub, select **Settings → Pages → Source → GitHub
Actions**. After a successful deployment, open the URL displayed by the Pages
job. Builds retain an artifact and skip deployment while Pages is not configured;
the workflow never enables Pages. A prepared workflow alone does not prove that a public site exists.

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
replay, captures its reference and displays displacement magnitude. The
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

Choose the field, use automatic or fixed colour limits, and inspect the device
and acquisition details. Affine gradients and symmetric shear are direct
kinematic coefficients, not a calibrated material stress calculation. Grey
means unavailable or rejected; excluded subsets are not zero measurements.

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
5. Use **Pause**, **Capture reference** or **Reset history** deliberately.
   Changing resolution restarts camera capture and prepares a new reference.
   A profile change takes effect at the next reference capture.

The camera stays local. Frames are transferred to a dedicated worker as
``VideoFrame`` or ``ImageBitmap`` objects and closed after use. Where a track
processor is available it supplies the frames directly; otherwise the app uses
video-frame callbacks and ImageBitmap/VideoFrame capture. Input colour-to-gray
conversion runs on GPU. The cold reference preparation reads grayscale once;
target frames are not read into a CPU pixel array.

Open your own images or video
-----------------------------

Choose **Open image folder** or **Select images**. Images are decoded lazily,
one pair at a time, in natural relative-filename order (``frame_2`` before
``frame_10``). PNG, JPEG, WebP, BMP, AVIF and GIF are admitted when the browser
can decode them; animated image formats are treated as still images. Unsupported
files are listed as ignored. TIFF is not admitted. No images are uploaded.

Select a **Reference image**, then **Set selected reference**. The timeline,
**Previous image** and **Next image** select the target. **Play** processes every
image sequentially without skipping a file. The image rate cap limits replay
speed; zero removes that cap. This replay setting is not an acquisition clock.
An undecodable image or a different image size stops playback with its filename;
there is no implicit resampling. Change the source to begin a different size.

**Open video** uses the browser decoder and a temporary local Blob URL. Use the
timeline to seek while paused and **Play** to resume. Slow replay down to ×1/16
is useful for expensive analysis or software adapters; the playback-speed control
affects presentation, not the acquisition timestamps. Codec support depends on
the browser. No codec or decoder is downloaded by the application. The actual
decoded size is used even if another requested camera resolution is selected.

Fixed or consecutive reference
------------------------------

**Fixed reference** remains the default. Every target is compared with the
captured reference. **Consecutive received images** performs a new IC-GN
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

The entire geometric ROI is requested on every pair, including after a failed
or textureless frame. Numerical invalidity and non-convergence are retained as
statuses and do not shrink the ROI. Choose **Status · entire ROI** to distinguish
excluded, unavailable, rejected, unconverged, valid and flagged/replaced cells.
A failed cell is not a valid zero displacement. The mask remains in image
coordinates when the pair reference changes; it is not advected with material.

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

Choose **Apply mask** when finished. The engine prepares the new static mask
and history; no geometry work runs per video frame. To adjust it later, choose
**Edit mask** again. Edits remain unapplied until that action. With no shapes,
there is no mask. Rectangular/polygon scanlines use pixel centres and half-open
edges; ellipse/circle interiors are strict. Classification checks the entire
rasterized support, not just centres or corners. The mask does not depend on
canvas antialiasing or drawing order. Equal pixel footprints are duplicates.

**Zoom** and scrollbars help inspect or position small shapes; **Centre view**
centres the observation window. Zoom changes presentation only. Numeric
coordinates, masks and the DIC grid remain at the original resolution.

The square of every retained subset must lie entirely inside the mask; holes
exclude complete crossing subsets. Nothing tests the moving target against the
fixed mask. Conservative seed/neighbour links and the preprocessing guarantee
are documented in :doc:`use-roi`. Browser partial ROI and geometric truncation
remain explicitly unsupported. Export metadata retains both the full binary
mask and the applied vector geometry, allowing the mask to be reconstructed.

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
preview and choose **Apply mask**; import alone does not alter the last numerical
field. **Full image** removes both the imported base and vector restrictions.

The imported mask is a fixed base. Vector domains/holes clip it by intersection;
they cannot restore a black base pixel. A full-image rectangle is supplied so
that adding a nested ellipse/circle/polygon makes an extra hole immediately.
The preview shows exclusions in the base as well as vector exclusions. Undo
preserves the base and restores vector edits. To change an existing pixel hole,
edit/reimport the image itself or replace the mask with vectors.

**Download edited mask PNG** exports the current editable net mask as opaque
black/white at original size, including unapplied edits. Apply it separately to
use it for correlation. Numerical field exports always keep the **applied**
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
Choose **Edit mask** to adjust them or add more regions. Applying the mask
recalculates the selected image pair. **Plate load** selects frame 95
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

Enable **Displacement arrows**, then choose their subset spacing and length
magnification. Arrows originate at accepted reference subset centres, read the
GPU field directly, and use the calculated (u,v), in pixels. The same gain
applies to both components; magnification is explicit, not normalization to
unit arrows. Excluded/rejected points have no arrow. A magnified valid arrow
may extend across a mask boundary; its origin identifies the measurement.

Select **von Mises strain εVM (εzz = 0)** for nonnegative equivalent strain.
It uses the same convention as the archived plane-strain plate analysis:

.. math::

   \varepsilon_{xx}=u_{,x},\quad \varepsilon_{yy}=v_{,y},\quad
   \varepsilon_{xy}=\tfrac12(u_{,y}+v_{,x}),\qquad
   \varepsilon_{VM}=\tfrac23\sqrt{\varepsilon_{xx}^2-
   \varepsilon_{xx}\varepsilon_{yy}+\varepsilon_{yy}^2+3\varepsilon_{xy}^2}.

These are direct infinitesimal affine strains with an explicit εzz = 0
assumption, not stress or a general finite-strain invariant. The legend uses
microstrain; manual scales are dimensionless (0.002 means 2,000 microstrain).
Automatic VM scale is a conservative bound derived from the existing maximum
gradient summary, rather than another field readback. **Field opacity** chooses
an image blend, field only or image only. Invalid cells stay grey. Arrows and
VM are visualization operations; they do not modify the calculated parameters.

Correlation settings without hidden changes
--------------------------------------------

The numbered **Input / Mask / Solve / Results** navigation keeps the current
step visible. Play/Pause and Capture reference remain beside the viewer. The
quick guide explains the workflow and keyboard shortcuts (Space: Play/Pause;
R: capture reference; Esc: clear the current measurement). Shortcuts do not
operate while editing an input, modal or mask.

In **Solve**, enable **Use custom settings** to choose translation/affine,
window width, grid step, interpolator, pyramid levels, coarse/fine iteration
limits and solver spatial policy. Edits remain pending until a new reference
capture or Apply mask. Exported configuration describes the applied settings,
not the pending form. Three-level custom runs use four middle-level iterations;
one level uses the fine limit. These custom combinations do not inherit a
qualified preset's precision/FPS claim. Translation has no direct strain field;
use displacement-neighbourhood strain analysis in the results lab.

Review and process a computed field
-----------------------------------

In **Results**, choose **Inspect current field**. Capture pauses and one full
field is read. The saved view uses reference image coordinates and a plain field
background, never an older field overlaid on an unrelated current image.
**Return to live view** restores GPU rendering. A saved field retains its own
model, configuration, ROI and input pair even after changing the active source.

The analysis controls stay beside the saved field. Choose displacement, strain components, equivalent strain, principal strains,
polar rotation, iteration count or solver status. Colour bounds can use robust
2–98 percentiles, full range or explicit fixed limits in the legend units.
Clipped colour values are counted; underlying measurements stay unchanged.
Choose a signed diverging, sequential or grayscale palette. The histogram
shows the valid value distribution whenever no line profile is selected.
Computed estimates are
preserved, including any solver median already applied. Optional displacement
median/Gaussian smoothing and VSG plane-fit strain are separate analysis
operations. **Show computed field** temporarily compares with the computed
values; **Restore processing recipe** returns to the selected operations.
The processing worker keeps these operations off the acquisition worker.
Saved-field vectors use processed displacements with explicit magnification;
an optional contour follows the exact binary ROI pixels. Image x points right
and y points down. These displays do not modify computed values.

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
measurements empty and retains solver statuses. **Save figure PNG** includes
quantity, scale, processing convention and attribution. Images are not included
in saved fields; retain your original input sequence separately.

Save a results project
-----------------------

**Save results project** pauses acquisition and packages all saved computed
fields, their shared ROI masks, provenance and applied processing recipes in one
``.dicproject`` ZIP. Keep original images separately. Importing a project adds
its fields to the current library after validating the entire batch and checking
capacity; it does not silently replace the active source or solver settings.
A browser close/reload warning protects changes not included in the last
requested project download, where the browser permits such warnings. A download
request cannot confirm that the file reached disk: check your downloads. NPZ
exports of individual fields do not mark the complete results project as saved.
Use **Clear results** first if you want a separate library. An oversized or
invalid project leaves the library unchanged. The current stored-data limit is
128 MiB; archive input is limited to 256 MiB uncompressed. Single-field NPZ is
still the simplest direct NumPy interchange format.

Extract a measurement through saved fields
-------------------------------------------

Place a point probe, two extensometer endpoints or two region corners, then
choose **Extract through saved fields**. The application applies the current analysis recipe to compatible
saved fields one at a time in a separate worker. Progress and cancellation are
available. Results from another analysis epoch, ROI or grid are counted as
excluded; matching dimensions alone do not prove a shared reference. Native
sequence fields imported from the same archive share its source hash.

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

**Keep computed fields for review** records every completed admitted field. It
adds a full readback and can lower throughput. **GPU field age** keeps its
observable-GPU boundary; each recorded field separately reports full-readback
time and worker-arrival-to-recorded-field age. These are not camera-to-Python
timings. Video/camera can still drop and count work under backpressure; folders
continue to await every image.

The library is bounded to 128 MiB of stored arrays, distinct masks and metadata
in the current tab. Analysis scratch, browser decoder/GPU memory and export
buffers are additional. At the limit recording and acquisition stop with an
explicit message; existing fields are not discarded. Export or delete results
before continuing. There is no automatic disk persistence: save NPZ before
closing the tab. Switching sources preserves saved fields but disables recording
until you explicitly enable it again. No image stream or result is uploaded.

Profiles and device limits
--------------------------

``portable`` and ``qualified-latency`` use the portable 16-lane solver,
B-spline, affine 16/16 subsets, two fixed levels, one slot and explicit median
smoothing. The latter name identifies the native-qualified latency
configuration: raw-history velocity prediction, fine correction and finest-only
stagnation stopping (3/4). It does not certify a phone's precision or FPS.
Portable use starts at lower resolution. Core API defaults remain the native
FullPyramid defaults; the application explicitly selects this latency profile.

On startup an on-device zero-motion microbenchmark tries the permitted
resolutions. Its short measurements guide the recommendation; they do not
replace the changing-trajectory qualification or a long thermal run. Minimal
single-level processing stays disabled because it has not passed the required
trajectory gates. Adaptive pyramids and cubic are not offered as qualified
profiles.

An adapter can satisfy the binding limits and still lack memory for UHD.
Choose a lower resolution after allocation/device-loss errors. Missing WebGPU
produces an unsupported-browser message; this application has no WebGL fallback.
The separate Python Three.js backend retains its existing WebGL2 fallback.

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
