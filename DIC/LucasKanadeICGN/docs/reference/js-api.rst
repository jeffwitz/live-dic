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

The renderer accepts mode 7 for equivalent strain (plane strain, εzz = 0),
``arrows``, ``arrowStep`` (grid samples) and ``arrowGain`` (pixels per pixel of
measured displacement). These are display settings, not ``DICEngine``
constructor options. ``opacity`` controls image blending; it does not affect
correlation. Display settings are included in the application metadata.
See :doc:`../how-to/run-in-browser` for the formula and editing workflow.

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
