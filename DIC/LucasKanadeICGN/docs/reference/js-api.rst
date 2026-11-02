JavaScript engine API
=====================

Import the dependency-free modules from the assembled site or from the packaged
``web/core`` directory. URLs are relative to ``import.meta.url``; importing from
a page or dedicated worker uses the same source. The core never reads
``window``, the DOM or ``navigator.mediaDevices``.

.. code-block:: javascript

   import { DICEngine, probeCapabilities } from './web/core/engine.js';

   const capabilities = await probeCapabilities();
   if (!capabilities.supported) throw new Error(capabilities.reasons.join('; '));
   const engine = await DICEngine.create({
     width: 1920, height: 1080,
     motionModel: 'affine', interpolation: 'bspline',
     subsetSize: 16, subsetStride: 16, pyramidLevels: 2,
     initialization: 'temporal', temporalPrediction: 'velocity',
     temporalFineCorrection: true, divergencePatience: 3,
     divergenceMinIterations: 4, outlierPolicy: 'median',
     pipelineDepth: 1, readback: 'none',
   });
   await engine.setReference(referenceFrame, { roiMask });
   const result = await engine.register(currentFrame);
   // Draw from engine.fieldBuffer with engine.device. No full-field readback.
   const completeField = await engine.exportField(); // Explicit, while idle.
   engine.resetTemporalState();
   engine.destroy();

``referenceFrame``/``currentFrame`` can be a contiguous row-major Uint8Array,
ImageBitmap or VideoFrame. Width and height are mandatory for a Uint8Array
reference. The input helper also accepts an HTMLVideoElement in a page; it does not require one in a worker. Keep input bytes and external
frames alive and unchanged until the matching registration promise resolves. The caller owns and then closes external frames.
The PWA worker closes the frames transferred to it.

Capabilities
------------

``probeCapabilities({ benchmark: true })`` acquires an adapter, reports its
identity, features and relevant limits, and benchmarks a few zero-motion frames
at permitted 720p/1080p/UHD resolutions. It returns ``recommendedProfile``,
``recommendedResolution``, ``measurements`` and explanatory ``reasons``.
Use ``benchmark: false`` for a limits-only diagnostic. The recommendation never
uses a vendor-name heuristic. This is an advisory graph benchmark, not camera
or thermal qualification. The minimal profile remains unavailable.

``DICEngine.create`` acquires an eight-storage-buffer device unless a
``GPUDevice`` is injected through ``device``. Injected devices must satisfy
the graph limits. The engine destroys only devices it owns. It does not
require subgroups or compile a subgroup module.

The requested buffer/texture limits follow the reference geometry. UHD keeps
its original eight-storage-buffer request; larger frames request only the
higher limits needed by their packed reference and subset metadata. The engine
checks the adapter limits before allocation. A 4112 × 3008 reference needs
197,902,336 bytes (188.73 MiB) for its reference planes, exceeding the default
128 MiB binding even when the hardware supports it.

For external frames with initially unknown dimensions, ``setReference`` can
replace an owning device when larger limits are required. It occurs only during
reference preparation, while the slot ring is idle. Recreate device-dependent
custom rendering resources after such a replacement; read ``engine.device``
after preparation. A supplied device is never replaced or destroyed and must
have been created with sufficient limits. A genuine adapter limit produces a
resource-specific error; no resizing or fallback to a different solver occurs.

Options and native correspondence
---------------------------------

.. list-table:: Defaults and scope
   :header-rows: 1
   :widths: 30 35 35

   * - JavaScript
     - Python correspondence / default
     - Notes
   * - ``subsetSize``, ``subsetStride``
     - ``subset_size=16``, stride defaults to size
     - Widths 8, 15, 16, 17, 31, 32, 33; integer stride 1..size.
   * - ``motionModel``
     - ``motion_model="affine"``
     - Translation, affine and existing quadratic preparation/solver. ROI is translation/affine only.
   * - ``interpolation``
     - ``interpolation="bspline"``
     - Linear and Keys cubic are explicit alternatives; cubic is not a qualified trajectory profile.
   * - ``pyramidLevels``
     - ``pyramid_levels=2``
     - Fixed 1..3 levels; every configured level runs per frame.
   * - ``iterationsPerLevel``
     - ``iterations_per_level=None``
     - Coarse-to-fine schedules: [30], [8,16], [6,8,16].
   * - ``tolerancePerLevel``
     - ``tolerance_per_level=None``
     - [0.001], [0.005,0.005], [0.01,0.005,0.005].
   * - ``znccMin``
     - ``zncc_min=0.8`` at reference preparation
     - Finite value in [0,1].
   * - ``outlierPolicy``
     - ``outlier_policy="replace"``
     - Also flag, disabled, median. Median is explicit smoothing, not outlier detection.
   * - ``initialization``
     - ``initialization="zero"``
     - Temporal mode is opt-in.
   * - ``temporalPrediction``
     - ``temporal_prediction="none"``
     - Velocity requires temporal initialization and uses raw history.
   * - ``temporalFineCorrection``
     - ``temporal_fine_correction=False``
     - Requires temporal mode and exactly two levels.
   * - ``temporalMaxAge``
     - ``temporal_max_age=None``
     - Null means no finite expiry; otherwise a nonnegative integer age.
   * - ``divergencePatience``, ``divergenceMinIterations``
     - ``divergence_patience=None``, minimum 4
     - Integers 1..30; stopping applies only at the finest level.
   * - ``pipelineDepth``
     - Register-stream slot depth; JS default 1
     - Independent target buffers, shared reference/history, bounded admission.
   * - ``referencePreparation``
     - Browser-only; ``'cpu'`` by default
     - Opt-in ``'gpu'`` prepares translation/affine references on GPU; one slot, uniform weights.
   * - ``readback``
     - Full typed field by default
     - full, decimated, none; native always materializes its NumPy result.

The JS default one-slot register API does not change Python's two-slot
``register_stream`` default. Gaussian integration, partial ROI, ROI truncation
and adaptive pyramid options are rejected. The portable workgroup width is 16;
no vendor-specific solver or workgroup-width autotuning is hidden here.
The application selects its latency options explicitly. Changing an execution
configuration requires a coherent new reference; create a new engine for a
different constructor configuration.

Reference and ROI
-----------------

``await engine.setReference(source, { roiMask, roiPolicy: 'full' })`` prepares
all levels and resets temporal history. ``roiMask`` is null or a Uint8Array of
exactly ``width*height`` bytes, each 0 or 1. It is copied at preparation.
No resize, threshold, geometric margin or moving-target mask test is implicit.
A square is retained only if every nominal pixel is True. All-True uses the
unmasked solver. All-False is valid, produces no measurements and launches no
IC-GN iterations. Partial ROI throws even with an absent mask.

The reference result includes geometry, configuration, preparation costs,
reference validity, ROI selection and mask provenance. Hashes use the same
canonical format as Python: UTF-8 ``live-dic-roi-v1`` plus a zero byte,
uint64 little-endian height and width, then row-major 0/1 bytes. Conservative
5 × 5 mask reduction and closed-box neighbour/seed permissions are prepared
once. See :doc:`../how-to/use-roi` for the preprocessing guarantee.

Finish all pending registrations before replacing a reference or resetting
history. Inputs and output static ROI arrays must not be mutated by a consumer.
Modification of the caller's original mask after preparation does not affect
GPU computation. ``roiSelected`` is static data shared by results, not copied
per frame.

Consecutive pairs with GPU preparation (experimental)
-----------------------------------------------------

With ``referencePreparation: 'gpu'``, a cold ``setReference`` prepares geometry,
ROI permissions, pipelines and resources. It also returns the initial numeric
validity mask. Then call:

.. code-block:: javascript

   const engine = await DICEngine.create({
     width: 1920, height: 1080,
     referencePreparation: 'gpu', motionModel: 'affine',
     interpolation: 'bspline', pyramidLevels: 2,
     pipelineDepth: 1, readback: 'full',
   });
   await engine.setReference(firstFrame, { roiMask });
   const field = await engine.registerPair(previousFrame, currentFrame);
   // Caller retains/closes both frames after the promise resolves.
   engine.destroy();

``registerPair`` updates all reference levels, gradients, Hessians and inverse
Hessians, then solves the target in one GPU submission. It reuses the geometry,
ROI and allocations; it performs no intermediate CPU image readback or numeric
reference preparation. The returned field uses the same parameter/status layout
as ``register``. The optional ``referenceAdmissible`` count is read with the
final field; the changing dense numeric validity mask stays on GPU. After a
pair, ``engine.reference.referenceValid`` is null and ``referenceValidity`` is
``'gpu-resident'``. Static ``roiSelected`` remains available.

Every pair discards temporal history from the previous reference. All configured
pyramid levels still solve; no previous field is interpreted as a seed in the
new reference coordinates. ``timing.gpuReferenceIncluded`` is true and the wall
time includes both preparation and solving. There is no separately measured
zero-cost preparation time. ``timing.gpuMs`` remains null on this path; these
wall measurements are not GPU timestamps.

Translation/affine, the three interpolators, existing subset sizes/strides and
full ROI are supported. Quadratic and depth greater than one are rejected.
``registerPair`` on a CPU-preparation engine is rejected. Input dimensions are
validated before mutation. A later execution failure invalidates the reference;
call ``setReference`` again to recover. A completely excluded ROI submits no
pair work. ``register`` can still reuse a GPU-prepared fixed reference.

For cold numerical diagnostics only, ``await engine.exportReference(level)``
returns copied reference pixels, metadata, inverse Hessians, numeric validity
and its count. It allocates and synchronizes, requires an idle GPU-preparation
engine, and must not be called in a live loop. See
:doc:`../explanation/gpu-reference-preparation` for arithmetic and limitations.

Ordering and lifetime
---------------------

``engine.register(source)`` admits at most ``pipelineDepth`` work items.
A full ring rejects immediately with a backpressure error; the caller decides
whether to drop. GPU submissions and temporal updates follow call order.
Each slot has its own image, scratch, state and mapped readback; reference and
history are shared. Promises deliver results in submission order.

``resetTemporalState()`` resets current, previous and coarse histories in queue
order while idle. It preserves the prepared reference and pyramid. Expiry and
velocity counters are returned when temporal mode is active; the fixed graph
does not enable adaptive-only expiry reporting.

``destroy()`` is permanent and idempotent. Finish pending work first. Typed CPU
results remain usable after destruction; GPU buffers and textures do not.
``fieldBuffer`` and ``imageTexture`` refer to the latest completed field/frame,
not a transferable snapshot belonging to every returned promise. Render them
on ``engine.device``; GPUDevice/GPUBuffer objects are not transferred to another
worker. Handle a device-loss event by ending the sequence and creating a new
engine/reference.

Results and timing
------------------

The packed layout is :doc:`result-layout`: translation and affine use 8 floats
per subset, quadratic uses 16. Translation/affine public parameters remain
``[tx, du/dx, du/dy, ty, dv/dx, dv/dy]``; translation-only gradients are zero.
Status -3 is geometric exclusion, not an optimizer failure. Parameters in an
excluded cell do not represent a zero displacement measurement.

* ``readback: 'full'`` returns typed ``parameters``, ``status`` and ``iterations``
  arrays, geometry, statistics, configuration and timing.
* ``readback: 'none'`` keeps the complete field on GPU. It reads a small
  per-workgroup summary and optional four temporal counters for the HUD.
  No complete field array is returned; this is not a zero-byte-readback claim.
* ``readback: 'decimated'`` also returns packed sampled states and
  ``sampleIndices`` into the original dense grid. Samples are selected on GPU;
  arrays are not falsely labelled as a full grid.

``exportField()`` reads the full latest field on demand while idle.
``setAnalyticTruth([ex, ey])`` optionally enables the demo-only diagonal-affine
truth summary, centred on the image centre. It affects diagnostics only.

``timing.wallMs`` covers upload/encoding and observable GPU graph completion;
``timing.totalWallMs`` includes result materialization. The PWA measures from
frame availability in the worker through GPU completion and its small HUD
summary. Acquisition, main-to-worker delivery, display presentation and export
readback are distinct boundaries. Timestamp queries remain disabled until a
wall-time sanity test qualifies them. Retain raw age samples; FPS and result
age are not interchangeable.

Application mask and rendering options
--------------------------------------

The standalone app keeps editable rectangle/ellipse/circle/polygon geometry in
reference pixels. Complete containment defines nested holes/islands; rasterized
outer domains unite before holes are removed. The resulting binary mask is
passed to the existing ``setReference(..., {roiMask})`` full-policy contract.
Vector geometry is a UI representation, not a new IC-GN support model. Export
``roiGeometry`` contains the applied shape list and composition version along
with the binary ``roi`` data. Unapplied edits are not exported as applied data.

``web-app/render/live-vectors.js`` is the live presentation adapter, separate
from ``DICEngine``. It accepts mode 2 (vectors) or 8 (image only), ``arrowStep``
(positive integer grid spacing, or null for an automatic overview), ``arrowGain`` (positive length magnification
or null for automatic), ``scale`` (positive norm maximum in image pixels or
null for automatic), and ``opacity`` (0..1 for glyphs). It reads the existing
GPU image and field, with no full-field readback or derived strain map.
``vectorLegend`` describes actual norm maximum, gain, image/preview dimensions
and current-tip anchoring. The live colourbar and length key use this metadata.
The legacy ``render/field.js`` scalar renderer is retained as a diagnostic
oracle; it is not loaded by the acquisition session. Scalar review uses the
separate saved-results analysis pipeline.

The static plate demo maps its complete field to compare with analytic truth.
Its selected far-field population is labelled in ``acquisition.analyticScore``;
its scoring readback must not be confused with the camera/video small-summary
path. The analytical helper and material constants never enter the solver.

Application local-source contract
---------------------------------

The PWA supervisor (not ``DICEngine`` defaults) adds local File-based image and
video sources and ``fixed`` / ``consecutive`` comparison modes. Each exported
``comparison`` records reference and target input metadata, the geometric
``requested`` count, ``referenceAdmissible`` and ``fixedImageCoordinateRoi``.
``pairReferencePreparationWallMs`` is zero for a normal fixed-reference frame;
for a consecutive pair it includes ``setReference`` on the predecessor. Worker
age includes this preparation. Image filenames, sizes, modification timestamps
and the sequence manifest are retained in application export metadata.

Image folders use lazy, sequential natural ordering. Files of incompatible size
are rejected rather than resized. The playback-rate cap does not define physical
time. Video timestamps use decoded media time; a missing timestamp remains null.
The application does not accumulate incremental fields or infer strain rates.

Application workbench options
------------------------------

Custom solver controls pass the existing camelCase options to a newly prepared
engine; they never reconfigure an in-flight field. The default application
preset and core/Python defaults remain unchanged. The app offers translation
and affine because its mask editor uses full ROI. Quadratic remains supported
by the core without ROI, not implicitly enabled by the application.

The separate analysis recipe contains ``smoothing`` (none/median/gaussian),
``window`` (3/5/7/9 grid points), ``strainSource`` (direct/vsg), ``strainMeasure``
(small/green), ``excludeFlagged`` (boolean), ``unit`` (px/mm/µm/m) and positive
``unitsPerPixel``. Pixel units require scale one; smoothing and translation-only
fields require VSG strain. Unsupported recipes throw. Analysis returns separate
``valid`` and ``strainValid`` masks and finite arrays; missing statistics are
null. It never changes solver status or computed parameters.

Optional recording performs full ``exportField`` readback after each admitted
solve. Its metadata contains ``recordingReadbackWallMs`` and
``workerArrivalToRecordedFieldMs``. The normal unrecorded live path retains its
small-summary readback. These are application options, not new solver defaults.

Application history and native archives
---------------------------------------

Reference preparation assigns an ``analysisEpoch`` to exported/recorded fields.
History extraction requires the same epoch (or source ``archiveSha256`` for
native imports), ROI, geometry, model and comparison mode. It preserves null
measurements and plots saved order. Native NPZ import accepts the documented
parameters/status/iterations/config_json arrays; sequence frame and any absent
image/subset dimensions require explicit confirmation. Import provenance records
``frameIndex``, ``frameCount``, geometry and its source. Integer values outside
JavaScript's exact range, inconsistent shapes and missing declared ROI data fail.

Results-project schema
----------------------

``.dicproject`` is ZIP/NPY plus a versioned JSON manifest
(``format="live-dic-results"``, ``version=1``). Each field records image/subset/grid
geometry, actual solver configuration, metadata, recipe and a shared mask path.
No source image or live permission is embedded. Import validates shapes, finite
parameters, documented statuses, integer iteration counts, recipes, mask hashes
and library capacity before adding a batch. It never changes an active solver.

Application mask-image and display contract
-------------------------------------------

The mask editor accepts an optional immutable binary image base, intersected
with its vector ROI. Exact import accepts only opaque black/white decoded pixels;
explicit threshold import uses rounded BT.709 luminance and alpha >= 128.
Dimensions must match. File hash and conversion are retained with applied vector
geometry; exported binary masks are sufficient to reconstruct the numerical ROI.
Mask edits remain unapplied until preparation completes. Cold source/reference
operations are serialized and invalidate stale seeks; comparison-mode changes
reuse applied solver options rather than pending form edits.

The results view can draw processed displacement vectors and a binary-pixel ROI
contour. Pixel-index measurements use integer pixel centres; canvas coordinates
include the corresponding +0.5 offset. Display controls are saved separately
from processing recipes. Application exports also include the assembled build's
revision/content version and canonical shader hashes when available.

``ui/postprocess.js`` exposes ``regionSummary(analysis, component, a, b)``.
It snaps two reference pixel-index coordinate pairs to grid centres and returns
null outside the grid. The inclusive rectangle report separates ``total``,
``selected`` and valid ``count``; its mean, population standard deviation and
percentiles use equal weights. Status quantities return category counts with
null scalar statistics. The Results workbench stores multiple named point,
line, zone and extensometer definitions in its project view metadata.

``ui/history.js`` exports ``sampleMeasurement(analysis, measurement,
component)`` and extends ``extractHistory(entries, {measurements, recipe,
component, ...})`` for multi-measurement time histories. Each field is analyzed
once; point/gauge/zone rows have one sample per field and line rows have one
sample per snapped subset. Region rows record their valid population and
statistics. Measurements use fixed reference coordinates; histories do not
accumulate consecutive-frame increments into material trajectories. The PWA's
**Export time history CSV** is long format and leaves unavailable values blank.
Status codes cannot be averaged through a region history.

Application review does not require a WebGPU device. The core ``DICEngine`` still
does. Saved-result replay and batch recipes share ``sameAnalysis`` compatibility;
neither changes live engine options or computed field arrays. Presentation
advances only after the current derived analysis finishes. Camera acquisition
metadata keeps requested/initial/actual settings and explicit timestamp-clock
labels; target export is unavailable after a new preparation until a target runs.

``componentValues(analysis, 'computed-pN')`` reads an estimated public model
coefficient, ``N=0..5`` or ``0..11`` for quadratic imports. Translation permits
only 0 and 3. Its source and unit are explicit; computed coefficients use
``analysis.valid`` and are unaffected by smoothing/VSG. Second derivatives use
reciprocal selected length units. These diagnostics do not add processed
coefficient arrays to NPZ: the original parameters remain the authoritative data.

NPZ exports include ``parameter_order`` as a Unicode array and raw parameter
units in metadata. Raw coefficients retain pixels, dimensionless gradients and
reciprocal pixels independently of display calibration. CSV keeps its existing
``pN`` columns with this explicit mapping in its metadata JSON. Import rejects a
declared order that conflicts with the public model contract. Older live-dic
archives without an order use that established contract. Results manifests carry
the same optional, checked order.

Application memory accounting
-----------------------------

``engine.memoryUsage()`` returns tracked buffer/texture counts, requested buffer
bytes, mapped-buffer bytes, known texture payloads and unknown texture formats.
It walks live resources only when called, outside the registration hot path.
``knownResourceBytes`` excludes driver padding, pipelines, canvas swapchains and
other browser allocations. ``physicalVramBytes`` and ``freeVramBytes`` are
``null``; WebGPU does not expose these. Mapped buffers may be host-backed, and
unified GPUs share host RAM. This is an allocation diagnostic, not a free-memory
or FPS guarantee.

The standalone workbench's ``ui/field-storage.js`` is a separate supervisor
module, not a ``DICEngine`` option. It stores full fields/masks in IndexedDB,
keeps a configurable CPU array cache, and loads archived fields lazily. It does
not retain GPU resources. See :doc:`../how-to/store-long-sequences`.


Standalone PWA supervisor settings
----------------------------------

The application Worker accepts ``options.supervision`` in its reference/ROI
preparation message. It is separate from the numerical ``DICEngine`` API;
passing it directly to ``DICEngine.create`` remains an unsupported option.
For a custom supervisor around the application session:

.. code-block:: javascript

   worker.postMessage({
     type: 'reference', id: 1, frame: referenceFrame,
     options: {
       profile: 'portable', referencePreparation: 'gpu',
       supervision: {
         overlapRecording: true,
         displayRate: 30,
         previewMaxWidth: 1920,
       },
     },
   }, [referenceFrame]);

For custom session clients omitting ``supervision``, the protocol defaults remain
``{overlapRecording:false, displayRate:0, previewMaxWidth:0, previewEnabled:true}``.
``overlapRecording`` must be boolean; the recording capacity is fixed at two,
including reserved computations. ``displayRate`` is finite in 0..240 Hz, with
zero meaning every field. ``previewMaxWidth`` is an integer in 0..16384, zero
meaning native width; presentation preserves aspect ratio and never upsamples.
``previewEnabled`` is boolean. False suppresses live field drawing and bitmap
transfer while computation and recording continue. It does not change the
meaning of ``displayRate:0``. Reference capture, explicit display/inspection and
``flush`` still present a view; suspension is not a renderer-free engine mode.
Unknown keys and invalid types throw before reference replacement. The GUI
sets 10 Hz and 1920 px preview explicitly, with synchronous recording initially.
The ``presentation`` message accepts only ``displayRate``, ``previewMaxWidth``
and ``previewEnabled``. It applies these immediately, without reference
preparation or history reset. Recording-overlap changes still require preparation.

Numerical result dimensions always describe the original image. With an
explicit supervisor configuration, a bitmap message's ``presentation`` object
reports its actual width/height, the result ID/input pair, reference-preview
flag and worker timing origin. Skipped preview updates do not skip computations
or recorded fields. The explicit ``flush`` session message waits for current
work and presents the latest pending field. The PWA uses it outside the live
loop during pause/export/source transitions.

The session ``report`` reply contains presentation draw/bitmap counts and CPU
API wall-time totals, with a Worker snapshot timestamp. These counters are
active for explicit supervision settings. They help distinguish presented and
computed rates; draw-submission wall time is not elapsed GPU render time.

The bounded ``RecordingQueue`` owns full typed fields until library admission.
A disk failure blocks new reservations and routes completed fields to explicit
RAM retention. Unexpected retention failure keeps the original field object in
the queue and rejects draining; it must not be treated as a successful export.
Default recording acknowledgements remain synchronous. The opt-in path
acknowledges computation after handing ownership to the queue, so committed
throughput and age must be reported separately. IndexedDB I/O is asynchronous;
this does not create a second GPU device or transfer WebGPU buffers to another
Worker.

Application workflow controls
------------------------------

The standalone GUI starts target correlation from Solve. Its input timeline
uses a separate image-preview canvas, outside the numerical session. Results
playback navigates stored typed fields and never sends ``frame`` requests.
The GUI initially selects saving; ``options.supervision`` and all numerical
constructor defaults stay unchanged. Canned ``?smoke=1`` scenarios explicitly
start an unrecorded preview; ``?smoke=1&case=manual`` exposes diagnostics while
retaining the ordinary user-controlled workflow.

Saved-result replay supports maximum rates 1, 2, 5, 10, 20 and 30 fields/s.
A changed quantity/palette applies to following compatible results. Initial
project/session import restores its saved view. Each field retains its saved processing recipe. Explicit batch application
sets a uniform recipe across compatible fields. Raw coefficients are unchanged. Slider admission is bounded to one load and the
latest pending request.

Large rigid shifts
------------------

``initialization: 'fourier'`` enables a per-target CPU phase-correlation seed
on a <=1024-square reference crop. Supported models are translation and affine;
CPU/GPU reference preparation and bounded slots are supported. Velocity and
temporal fine correction require the separate temporal mode. Full exported
fields preserve ``config.globalTranslation`` and ``timing.fourierWallMs``.
``memoryUsage().cpuInitializationBytes`` counts the reference FFT cache and
slot seed arrays separately from GPU resources. External target conversion
reads back only the cropped window; the ordinary path adds no such readback.
See :doc:`../how-to/register-large-shifts` for rejection and latency limits.
``pyramidLevels`` accepts 1–4; its default remains two.
