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

For a local checkout, assemble the assets and serve the resulting directory:

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
2. Read the **actual** track settings in the details. Requested resolution and
   60 Hz are preferences, not a measurement of delivered frames.
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
timeline to seek while paused and **Play** to resume; the playback-speed control
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
viewer, then drag its body or white handles. Polygon handles move individual
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
