Run the local browser application
=================================

The browser companion accepts a camera or an analytic demonstration and keeps
its frame processing on the device. It uses the same portable WGSL solver as
native Python. Pierre Margerit's foundational GPU work made this project
possible. Python remains the recommended supervisor for NumPy acquisition and
complete NumPy result fields.

Open or serve it
----------------

The Pages workflow is prepared, but enabling or publishing the site is a
repository-owner action. In GitHub, select **Settings → Pages → Source → GitHub
Actions**. After a successful deployment, open the URL displayed by the Pages
job. A prepared workflow alone does not prove that a public site exists.

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

Choose **Analytic demo**. The application loads a derived 1920 × 1080 tensile
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

Draw a full-window ROI
----------------------

Pause, choose rectangle or polygon, and select **Draw ROI**. Draw with a mouse,
pen or touch. Finish a polygon by double-clicking or choosing **Apply ROI**.
Apply the mask, then resume. **Full image** removes it. Reference geometry and
temporal history are rebuilt between sequences.

The mask is defined at full reference resolution. A pixel centre inside the
drawn shape becomes True. The square of every retained subset must lie entirely
inside the mask; holes and forbidden pixels exclude it. Nothing tests the
moving target against a fixed mask. Programmatic masks may contain holes;
see :doc:`../reference/js-api` and :doc:`use-roi` for the exact geometry,
conservative seed/neighbour links and preprocessing scope. Browser partial ROI
and geometric truncation are explicitly unsupported.

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
