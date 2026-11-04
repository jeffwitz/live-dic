Browser workbench design and scientific contract
================================================

The browser companion complements the native Python supervisor. Its interface
is organized around a short path: choose input, define the reference and
ROI, run correlation, then inspect and process the resulting fields. Earlier
GPU-DIC and Three.js/WebGPU prototypes by Pierre Margerit informed the
methodological and browser-GPU exploration. The maintained solver is a separate
live-dic implementation; see :doc:`acknowledgements` for details.

Two workspaces
--------------

The header switch separates two user goals while keeping one browser session,
one result library and the same numerical core:

* **Live** is for continuous camera or video streams. It foregrounds acquisition,
  the fixed reference, the specimen mask, frame timing and a lightweight vector
  preview. It does not show the saved-field post-processing panel. Fields can be
  recorded during a run and opened in Analysis afterward.
* **Analysis** is for image folders, image sequences, video files and saved
  projects. It foregrounds solving a selected range, replaying saved fields,
  post-processing, named measurements and exports. The camera control belongs
  to Live.

Switching workspaces changes navigation and presentation only. It does not
change solver settings, discard fields, rebuild the reference or pause an
active calculation. The selected workspace is remembered in the browser. Live
rate and 4K60 goals remain device-specific; the workspace does not certify a
browser or mobile device.

Three distinct user operations
------------------------------

Input preview, correlation and saved-result playback have separate controls.
Input seeks draw one decoded image without a target solve. Reference/mask
preparation remains a cold operation. Solve owns Start/Pause calculation and
its explicit image range; the GUI initially selects full-field recording so a
normal calculation produces replayable results. Custom session protocol and
numerical defaults remain unchanged. Canned diagnostic URLs explicitly request
unrecorded preview calculations; they do not define the interactive workflow.

Results owns a transport beside the image: quantity, Play/Pause, rate and saved
field slider. It pauses acquisition and reads the result library. Navigation
keeps quantity/style for compatible fields; each field retains its saved recipe.
Apply a batch recipe explicitly when uniform post-processing is required;
initial archive/session import restores its saved view. Original coefficients
are immutable. Fast dragging coalesces seeks to one load plus the newest request.
Playback waits for analysis and does not invent acquisition timestamps or skip
stored fields to meet a presentation rate. Source changes suspend recording
until Solve starts again while preserving the user's save preference.

Interface research
------------------

The review used official product documentation on 31 October 2026. It is a
representative comparison, not a claim that every DIC package was tested or
that commercial solvers were executed locally.

* `Ncorr manual 1.2.2 <https://www.ncorr.com/download/ncorrmanual_v1_2_2.pdf>`_:
  staged reference/current/ROI/analysis workflow, a visible strain-fit scale,
  point inspection and separate displacement/strain plots.
* `VIC-2D manual 7 <https://correlated.kayako.com/api/v1/articles/86/attachments/5544/url>`_:
  image/data/extraction organization, timeline navigation, line profiles,
  extensometers and explicit strain/post-processing settings.
* `MatchID software <https://www.matchid.eu/software>`_:
  datasets, result views, extraction/chart tools and reusable analysis recipes.
* `ZEISS CORRELATE <https://www.zeiss.com/metrology/en/software/zeiss-correlate.html>`_:
  persistent evaluation steps, reference systems and virtual extensometers.
* `DICe VSG processor <https://dicengine.github.io/dice/class_d_i_ce_1_1_v_s_g___strain___post___processor.html>`_:
  displacement-neighbourhood strain calculation as an explicit postprocessor.

* `DICe desktop GUI <https://github.com/dicengine/dicegui>`_:
  a separate HTML/JavaScript desktop supervisor around an independent engine.
  Its numerical implementation is not imported into live-dic.
* `Dantec Istra4D <https://www.dantecdynamics.com/new-istra4d-v4-11-software-release/>`_:
  integrated acquisition, correlation and temporal/spatial extraction with
  different virtual gauge geometries.
* `iDICs Good Practices Guide, edition 2 <https://idics.org/guide/>`_:
  keep reference convention, strain calculation and spatial support explicit.
  Using these interface principles is not a claim of certification.

The adopted principles are progressive disclosure, a large central field view,
explicit acquisition/measurement state, reversible processing and named physical
quantities. No proprietary source, icon set or screen layout is copied. Stereo,
material identification, stress reconstruction, instrument synchronization and
commercial metrology certification are outside this application.

Live monitoring and saved-field analysis
-----------------------------------------

The acquisition session imports a dedicated vector renderer. A background pass
reads only the associated current image; an instanced pass reads sparse measured
vectors. Colour encodes their norm, while a separate length key documents gain.
The Live colourbar offers the original palette and five sequential maps. A
small GPU lookup texture and the HTML legend use the same colour tables;
changing a map does not trigger correlation or saved-field analysis. Optional
light/dark outlines improve contrast on textured images while keeping arrow
geometry and norm colours unchanged.
No scalar raster, interpolation or strain postprocessor runs in this path. The
GUI initially limits presentation to 10 Hz and 1920 px width. Rate, size and
suspension are presentation settings, independent of computing and recording.
A presentation flush shows the latest completed pair; mask editing restores
the full reference image. Numerical images, fields and archives remain full size.

The result library caches computed fields in bounded RAM and browser disk.
Selecting a saved field pauses acquisition and applies its explicit processing
recipe. A separate least-recently-used replay cache holds at most eight derived
analyses and 32 MiB of typed-array buffers. Its key includes field identity and
the complete recipe; a matching revisit reuses the analysis without worker
recalculation. Cache misses run the recipe. It retains neither raw fields nor
image masks and is cleared when closing or changing the library. Component
arrays and statistics are cached for the active analysis; palette/range/vector
edits reuse them. No eager processing of an entire recording is performed.

The completed field raster stays visible at full opacity during disk loading
and analysis. The status above the image identifies the displayed field and
the pending request. Its title, legend and raster change together only after
the requested analysis is ready; same-size canvases are not reset between
frames. Inspection and derived exports remain disabled during this wait, so
previous pixels cannot be mistaken for a new measurement. A failed request
keeps the last completed image with an explicit failure status. Region and
probe summaries use larger type both on the image and below it.
Original camera images are not stored with fields. Saved maps therefore use
reference coordinates instead of an unrelated live image background.

Numerical analysis module
-------------------------

``web-app/ui/postprocess.js`` operates on a materialized field. It never changes
the engine parameters, statuses or iterations. "Computed" means exactly the
solver output, including any explicitly selected solver median policy; it does
not mean unsmoothed if that policy was enabled.

Displacement median/Gaussian smoothing is a separate, optional operation. Its
odd 3/5/7/9 window is in grid points. Gaussian sigma is half the window radius,
with a minimum of half a grid step. Invalid points are not filled. Neighbours
must have a clear conservative rectangle in both the valid grid and image ROI;
this avoids averaging across a thin excluded crack. Near boundaries the support
is reduced, not extended across missing data. Smoothing changes spatial
resolution and is not outlier detection or an uncertainty estimate.

Strain is either read from the local affine/quadratic derivatives or recomputed
by a least-squares plane fitted to the displacement neighbourhood (VSG). The
fit uses double precision JavaScript arithmetic, normalized grid coordinates,
centred normal equations and a determinant rank test. At least three
non-collinear admissible points are required. Failed fits retain a separate
invalid strain mask. Translation-only fields require VSG for strain.
Smoothing requires VSG so that processed displacement and strain describe the
same operation. The displacement smoothing span is
``subsetSize + (window - 1) * subsetStride`` pixels. VSG on unsmoothed
displacements has the same span; applying smoothing followed by VSG composes
two neighbourhoods, giving ``subsetSize + 2 * (window - 1) * subsetStride``
for strain. Direct unsmoothed gradients retain the solver subset span. At
16/16 with a 5-point window, smoothing spans 80 px and smoothed VSG strain
spans 144 px nominally; truncated boundary support can be smaller.

Small strain uses the symmetric displacement gradient. Green-Lagrange strain
uses ``E = (F.T F - I)/2``. Principal values are the eigenvalues of that 2D
symmetric tensor. Equivalent strain uses the documented ezz=0 convention;
rotation is the 2D polar angle ``atan2(vx - uy, 2 + ux + vy)``. Length calibration
changes displacement/length units, not dimensionless strain. This is an isotropic
scale, not a lens/perspective calibration. No material stress is inferred.

A two-point extensometer snaps to actual subset centres and requires valid
endpoints. It reports change in endpoint distance divided by original distance,
which is a gauge engineering strain, distinct from a local tensor component.
Rectangle statistics select actual subset centres between two snapped corners,
inclusive. They use equal sample weights and exclude invalid values. Total grid
positions, ROI-selected positions, valid samples and categorical statuses remain
separate. Overlapping windows do not define disjoint material areas: a sample
mean is not an area integral. No interpolation, gap filling or uncertainty claim
is introduced. Status codes have counts, not a scalar average.

Incremental comparisons remain pairwise; they are not summed into trajectories.

Archive contract
----------------

``ui/archive.js`` writes standard uncompressed ZIP/NPY NPZ archives without any
runtime dependency. Computed parameters/status/iterations and optional processed
arrays are separate. Metadata and processing recipe are scalar Unicode JSON;
image dimensions, subset geometry and the binary ROI are included. NumPy can
open these files with ``allow_pickle=False``. Float64 imported parameters are
preserved. The importer checks CRC and sizes, rejects object/pickle arrays and
limits total uncompressed input to 256 MiB. It supports C-order little-endian
numeric/Unicode arrays, including exact-range 64-bit integers. Native CLI
sequence archives require an explicit selected frame. Image extent comes from
stored scalar dimensions, an archived ROI shape, or explicit user confirmation;
it is never inferred from a last grid cell. Subset geometry comes from stored
scalars/configuration or confirmation. Declared ROI without its binary mask is
rejected. The input archive hash and geometry provenance survive re-export.

Qualification
-------------

The independent CPU tests cover an analytical 3840 x 2160 affine field, exact
finite rotation/Green strain, a thin excluded crack, rank-deficient support,
Gaussian weights, unit-aware extensometry, non-mutation and NumPy archive
round trips. These validate the analysis operators, not an experimental noise
floor or the precision of an uncalibrated browser camera. Browser end-to-end
records must be reported separately from this numerical operator coverage.

Presentation is separate from measurement
-----------------------------------------

Saved results use a separate reference-coordinate field canvas. Displaying them
never overlays an old field on an unrelated current camera image. Point and
gauge annotations also appear inside the viewer, while processing controls stay
beside it. The solver's dense exclusions remain visible in status mode.

Colour-map and clim controls sit to the right of the colourbar, stacking below
it on narrow screens. Robust limits use editable lower/upper percentiles of
valid measurements (2 and 98 by default); signed fields retain symmetric limits
around zero. Full-range and
manual clims are explicit options. Alongside the existing automatic,
blue/green/gold, blue/white/red and grayscale maps, the viewer bundles
256-entry Viridis, Cividis, Plasma, Inferno and Magma lookup tables sampled
from Matplotlib's perceptually uniform sequential maps. They require no runtime
Matplotlib dependency. Saturated values are counted but are not removed or
clipped in the data. Histograms use the valid population, not only the
unsaturated range. Status categories retain their fixed discrete colours and do
not use scalar clims. These controls recolour the cached analysis and do not
trigger correlation or derived-field recalculation.

Solver states outside the active acceptance population are masked from scalar
maps by default. The optional **Show masked estimates** view evaluates the
selected display recipe over finite estimates from every ROI-selected subset
where at least one solver iteration ran, while retaining the normal rendering
of already-valid values. This makes partially converged and correlation-
rejected regions visible for diagnosis. ROI exclusions and subsets with no
solver estimate stay blank. Drawing a diagnostic estimate does not make it pass
the solver criteria: the categorical status map, scalar statistics,
measurements and processed CSV retain the original accepted population. The
status map is available as **Acceptance criteria · solver status** in the
quantity selector.

PNG exports use the displayed palette and limits, or the six exact categorical
status colours. A separate wrapped footer records the field label, subset
geometry, applied processing, length scale and measurement text. The field and
annotation pixels retain their original resolution. The browser qualification
decodes each actual PNG and checks its image pixels against the composed field,
caption bounds and legend colours. It covers strain, displacement and status;
this is a presentation/interchange check, not a new metrology qualification.

Saved-field history
-------------------

The acquisition worker assigns a new analysis epoch whenever the user prepares
a reference/ROI. A consecutive pair changes its actual reference image while
retaining that preparation epoch; both input identifiers remain in comparison
metadata. The history tool requires matching epoch (or native archive hash),
image/grid geometry, model, ROI hash and comparison mode. It processes one field
at a time and sends a shared static mask once to its worker. No GPU calculation
or image decoding is repeated. Cancellation releases that worker.

The workbench retains a named collection of point probes, line profiles,
rectangular zones and extensometers in the saved Results view. A probe samples
an actual grid point; a line returns distinct snapped grid samples; a gauge
computes endpoint-distance strain for each field independently. A zone reports
its valid-sample mean and population for each field; changing coverage is
retained rather than hidden. The collection is saved with project view metadata,
not embedded into the numerical field or solver.

Time extraction analyzes each compatible saved field once in a bounded worker,
then evaluates all measurement geometries against that result. Its long-format
CSV uses one row per saved-field/measurement sample: points and gauges produce
one row per field, line profiles produce one row per snapped subset and field,
and zones produce one summary row per field. Coordinates remain fixed in the
reference image. Missing or rejected values stay null and break chart lines;
there is no interpolation or gap filling. The chart shows the selected point,
gauge or zone over saved order; for a line it shows the middle snapped sample,
while the CSV preserves the full profile at every field. Source indices and
timestamps are included when available, but saved order is not an inferred
physical acquisition clock. The tool does not infer regular sampling,
accumulate incremental displacements or silently combine unrelated reference
configurations.

Results projects
----------------

A ``.dicproject`` is a standard ZIP with ``manifest.json``, per-field NPY
parameters/status/iterations, and shared static masks. It retains every computed
field, applied recipe, grid, source-pair provenance and analysis epoch. A mask
shared by a recording is written once. Processed arrays can be regenerated from
the stored recipe; individual NPZ export also includes those arrays explicitly.
No input image, camera permission or File handle is serialized.

Import validates every field in one part before adding that part. Multiple parts
are processed in filename order; a later failure does not remove earlier imports.
The versioned format retains its 256 MiB uncompressed, 512-field and 4096-entry
limits per archive. Long exports split into bounded parts, with a shared mask
once per part. CRC protects entries and ROI hashes are checked. No archive
executes code or changes acquisition settings.

The in-tab array cache is now backed by IndexedDB, not a fixed-capacity library.
Each field is committed first, then eligible for LRU eviction from CPU RAM.
The cache budget is configurable up to 1 GiB; unknown device memory defaults to
128 MiB. Disk persistence is requested but may be denied. A quota/write failure
keeps the completed field in RAM and stops acquisition; retry never duplicates
it. No pending field is silently evicted. See
:doc:`../how-to/store-long-sequences` for the full lifecycle and limits.

Geometry/configuration are interned in the catalog, static masks are shared per
session, and browser-session reopening reads a lightweight index before loading
one selected field. Reopened sessions are read-only for acquisition; view and
recipe settings can be saved separately with revision-conflict detection.
Computed raw arrays remain immutable. Histories and exports load fields in
bounded order. The worker's live numerical graph does not gain a field-history
buffer on the GPU. The supervisor awaits the disk commit before admitting the
next recorded field; this intentional backpressure can affect throughput.

Changes since the last project download request or explicit browser settings
save trigger an unload warning when supported. This is an accidental-loss guard,
not proof of independent backup. The browser database contains fields and masks,
not input images. Clearing site data/profile storage can remove entire sessions.

Mask image editing
------------------

An optional binary base is intersected with the existing exact vector mask.
Imported masks use a default luminance threshold of 128 (white/light is kept),
with strict binary and custom-threshold choices. **Create mask from scratch**
starts from a full-size all-excluded mask; added domains mark the pixels to keep.
Undo snapshots retain an immutable base reference, not a copied full image per edit.
Only paused edits rasterize or compose masks. PNG import/export does not change
IC-GN equations or add a pixel mask in the solver loop. Preparing a new ROI
reuses the core's full-window selection, conservative links and canonical hash.
A failed import leaves the existing geometry/calculation intact. Source and
reference operations share one admission gate, drain pending work and invalidate
stale video seeks before replacing resources.

Saved-result replay uses the same compatibility predicate as history extraction.
It holds at most one analysis request, pauses at missing/unanalysable results and
stops at the end. Its fixed two-hertz presentation rate is independent of the
experiment clock. Batch recipe application changes derived-analysis settings
only; it never recomputes or rewrites IC-GN results.

The saved-field review/analysis path is independent of WebGPU. A missing API
disables new acquisition/correlation while keeping NPZ/project import and CPU
post-processing available. HTML canvas substitutes for OffscreenCanvas; without
workers the same pure analysis functions run on the main thread. This fallback
is not a second correlation solver.

Computed-coefficient diagnostics read the public six/twelve-parameter ordering
directly. They are separate from processed displacements and derived strain.
For isotropic scale ``s`` length units/pixel, displacement scales by ``s``,
first derivatives are unchanged and second derivatives scale by ``1/s``. The
quadratic basis already uses ``x²/2, xy, y²/2``; no extra factor of two applies.
The original solver spatial policy and validity flags remain explicit. This is
not a new quadratic browser solver or a way to smooth curvature implicitly.

Archives now carry the public parameter names. A declared incompatible order
is rejected rather than interpreted as a different motion convention. Raw
parameter units remain image-pixel units; display calibration never rewrites
those arrays. Byte-view normalization also keeps native uncompressed NumPy
archives correct when passed through offset byte views in the Node harness.

A resize observer can run while an analysis response is pending. Chart rendering
therefore requires a current analysis, clears live-view caches and defers size
changes to an animation frame. A deterministic UHD regression holds the real
worker response during a line-profile resize and then releases it. Pending or
failed derived views are visibly dimmed and cannot produce derived exports.
A live-mode library uses an explicit selection placeholder, so its first recorded
field can be opened without creating a duplicate frozen field.

Acquisition, computation and presentation
-----------------------------------------

The normal application already uses a dedicated Worker for image upload,
reference preparation, solving and GPU field rendering. The main thread
receives transferable image bitmaps for presentation and handles controls and
recorded-field admission. Browser video capture/decoding also has its own
implementation scheduling. This is already a form of parallel execution.

The default path deliberately admits one solve at a time and acknowledges a
recorded field after storage admission. The opt-in pipeline keeps one GPU solve
but reserves at most two CPU recording slots, including work currently being
computed. It acknowledges transfer of ownership to that bounded queue, allowing
the next solve to overlap disk admission. Image sequences wait for capacity;
live inputs retain their existing explicit drop semantics. They do not turn a
non-adjacent camera pair into an allegedly consecutive one.

The queue serializes writes. On failure, it stops new reservations and retains
already completed fields in RAM through the same results library. Static ROI
masks are shared; typed parameters/status/iterations retain their precision.
An error during RAM retention keeps the original queue object and blocks export
completion. Source/reference/ROI changes and pause/project operations drain the
queue first. The unsaved-data guard includes pending queue ownership, even before
the first field has appeared in the library. Recovery preserves field order and
does not duplicate the field whose first write failed.

Display and computation have separate clocks. The Worker renders the most
recent completed field when a presentation is due, rather than queueing old
preview frames. A timer/explicit flush presents the final field when inputs
pause. Preview width affects only the canvas raster and shader coordinate
mapping. Mask editing forces native-size reference presentation, while numerical
inputs, subsets and all saved/exported arrays remain unchanged. Custom session
clients retain unthrottled/native-size protocol defaults; the PWA explicitly
chooses 10 Hz/1920 px for lightweight live monitoring.

Slow monitoring can request 1, 5 or 10 Hz. Suspending live preview prevents both
live render calls and bitmap creation; the supervisor still records every
admitted calculation, updates progress and flushes the latest view on explicit
pause/export. There is no queue of old display frames to catch up later. This
control can bound display overhead, but cannot remove reference preparation,
correlation, field readback or database cost. Inspect both committed rate and
age before attributing a whole-application bottleneck to display.

The application keeps a single GPU owner and uses asynchronous browser disk
I/O. Another display Worker/device would require exchanging resources instead
of directly reading the existing field buffer, without removing GPU work. The
short display-removal controls in
``benchmarks/results/browser-display-headroom-20261101/`` do not qualify a new
FPS limit. The new overlap path has its own correctness/performance controls;
computed, presented and committed rates/ages are distinct quantities.

TIFF input decoding
-------------------

TIFF admission is separate from the numerical engine. ``ui/tiff-codec.js``
reads a bounded classic-TIFF directory and validates its layout before the
pinned UTIF.js upstream source snapshot decompresses the selected image. This
snapshot fixes truncated LZW strip output in the published npm bundle. It also
allows interleaved grayscale-plus-alpha samples for imported masks. It deliberately skips
unrelated EXIF directories and refuses multipage stacks rather than treating
a first page as a whole sequence. ``ui/tiff-worker.js`` loads the pinned local
UTIF source; the application service worker caches it with the other assets.
Each decode owns a short-lived Worker and transfers one
ImageBitmap. Abort, timeout and failures terminate that Worker; dimensions are
checked before pixel decoding. This keeps source changes bounded and avoids
blocking the UI on UHD decompression. It is input decoding, not a new GPU solver.

Unsigned 16-bit samples use their high byte, consistently across reference
and targets. This fixed reduction preserves PNG pixels embedded as ``value *
257`` in TIFF, but discards low sensor bits. Reference and target provenance
records the conversion. Existing PNG/browser-decoder semantics are unchanged.
See :doc:`../how-to/run-in-browser` for admitted formats and explicit limits.

Lazy solver preparation
-----------------------

Choosing an input stores a reference image and its dimensions in the page. It
does not create the DIC engine or upload/preprocess the reference on the GPU.
Selecting a different reference, committing a mask, changing the comparison
mode or editing solver settings only updates the pending analysis. Mask
geometry and solver controls therefore remain responsive while an engineer
sets up an experiment.

**Start calculation** is the single preparation boundary. The supervisor sends
the staged reference image, committed mask and current solver options together
to the worker. The worker creates the engine, prepares the reference with the
mask and returns the prepared grid before target solving starts. A failed
preparation leaves the staged inputs available for correction and retry; it
does not substitute an older image or mask.

Solve keeps a fingerprint of the last successfully prepared request. An
unchanged start reuses the engine and analysis epoch. A changed reference,
mask, comparison mode or cold solver option invalidates that fingerprint and
will be prepared at the next **Start calculation**. Presentation controls apply
independently, and no reference is rebuilt in the per-frame loop. Saved fields
retain their original arrays and configuration throughout.

Saved fields retain their original arrays and configuration. Results reports
initialization, input names and solver acceptance from the displayed field,
not from the pending form or a newer reference. Low acceptance is explicitly
identified; neither recipe changes nor smoothing manufacture rejected results.
ROI exclusions are not counted as requested failures. The UHD workflow qualifier
checks these contracts with saved failed/successful fields and an applied mask.
