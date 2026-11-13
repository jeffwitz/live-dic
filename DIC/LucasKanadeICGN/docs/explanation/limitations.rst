Current limitations
===================

Hardware coverage
-----------------

The native full-pyramid solver has completed 3840 x 2160 benchmarks on AMD
Radeon 780M, Intel Lunar Lake, Intel UHD Graphics (Tiger Lake GT1), and NVIDIA
GeForce RTX 3080 Laptop GPUs. This provides measured evidence of portability
across three vendors, integrated and discrete hardware, and the tested Linux
Vulkan configurations. The :ref:`gpu-hardware-coverage` matrix separates
successful benchmark execution, numerical checks and full-suite status.

The AMD 780M campaign completed all twelve runs and passed the fixed-interior
accuracy gates. Its initial full-suite device-loss failure was traced to
closed engines retaining GPU pipelines and devices, exhausting the driver's
4 GiB GTT budget. Native ``close()`` now releases those references, destroys
the owning device and collects cyclic wrappers. Shared pyramid levels do not
destroy their parent's device. See :ref:`gpu-hardware-coverage` for the
reproduction, correction and subsequent validation. A passing finite test run
does not guarantee stability under every workload or system load.

Coverage does not extend to every option, browser backend, operating system,
driver or recent GPU. NVIDIA and Tiger Lake evidence is a fixed-pair
experiment; it does not qualify the changing temporal trajectory. Functional
portability also does not imply 60 FPS, identical floating-point fields or a
hard deadline on every adapter. Qualify accuracy, stability and the complete
image-to-field timing boundary on the intended deployment configuration.

Backend lifecycle
-----------------

The maintained project supports native wgpu-py, Three.js/TSL WebGPU, and raw
shared-WGSL WebGPU. The former GPU-IO runtime and WebGL2 solver have been
removed from the repository. Three.js requires WebGPU and does not fall back
to WebGL2.

Python remains the primary supervisor for NumPy acquisition and complete NumPy
fields. The standalone PWA now supplies a JavaScript supervisor for raw WebGPU
camera/demo acquisition, bounded scheduling and GPU display. It is separate
from Three.js. ``ICGNBrowser`` still includes the Python/Chromium process boundary;
neither direct Three.js GPU times nor PWA worker timings are full Python-stream
times. See :doc:`browser-portability` for the browser qualification matrix.

Geometry and input
------------------

* All current backends support dynamic frame dimensions and have been validated
  at 3840 x 2160. Reference and target dimensions must match.
* Native and raw browser shared-WGSL accept integer subset widths from 4 to
  46,340 (the solver's signed 32-bit pixel-index limit). Actual image size,
  pyramid depth, adapter limits, and memory constrain practical choices.
  ``subset_stride`` defaults to the selected size and can be
  reduced to create overlapping subsets. Three.js/TSL supports widths 8, 16,
  and 32 and requires stride equal to size.
* Centred Gaussian subset integration is currently exposed by the native
  engines. Browser hosts use uniform integration.
* Solver-time median repair defaults to a 3 by 3 subset-grid stencil and
  rejected-only scope. The shared-WGSL native and raw-browser paths accept 5,
  7, or 9; the largest stencil is bounded at 81 samples. The opt-in all-subset
  scope also smooths accepted parameters after each pyramid level, while
  preserving solver statuses. Both scopes can provide seeds at finer scales;
  neither makes a final rejected status valid. Larger stencils and all-subset
  smoothing have not received general performance or metrology qualification.
  Three.js retains a separate fixed 3 by 3 post-pass. Results display smoothing
  is a different operation.
* Grid dimensions are ``1 + floor((width - subset_size) / subset_stride)`` by
  ``1 + floor((height - subset_size) / subset_stride)``; only fully contained
  subsets are emitted.
* Inputs are converted to 8-bit grayscale.
* Three.js and raw WGSL browser targets use a persistent binary loopback
  transport; per-frame PNG and JSON field serialization have been removed.
  Chromium still owns separate CPU memory, so the frame crosses a process
  boundary before its GPU upload and the result crosses back. Native
  ``ICGNWGPU`` avoids the browser process and uploads contiguous 8-bit
  grayscale bytes in-process.
* The one-time browser reference setup still uses the encoded image path. It is
  excluded from steady-state target registration measurements.

Registration range
------------------

IC-GN is a local optimizer. The direct ``t=0 -> 20`` analytical displacement of
``(4, -2)`` pixels lies outside the robust local basin for many subsets in a
single-scale solve. The incremental ``t=10 -> 20`` step of ``(2, -1)`` pixels
returns 900 accepted subsets at 512 x 512. Native users should select
``ICGNWGPUFullPyramid`` for the normal GPU-resident coarse-to-fine path; it
defaults to two levels. ``ICGNWGPU(pyramid_levels=2 or 3)`` remains an older
optional selective-retry mode for exceptional failures, not the recommended
native pyramid. The raw ``webgpu-wgsl`` browser backend runs the same
full-pyramid WGSL graph and defaults to two levels. Three.js/TSL remains
single-scale.

``ICGNWGPUFullPyramid(initialization="temporal")`` projects the last accepted
fine field onto the coarsest grid, then solves every configured level on every
image under the default fixed policy. It is opt-in and also supports a single
level. The standalone JS core now has equivalent temporal controls; the Python
browser compatibility interface does not expose them. Native full-pyramid constant-velocity
prediction is opt-in; no backend provides other extrapolated initialization,
Aitken acceleration, predictive tolerance, or automatic retry from the last
accepted state. Temporal history requires
ordered inputs and cannot guarantee recovery after discontinuous motion or
accepted wrong-basin solutions. See :doc:`temporal-initialization`.

The browser Analysis workbench additionally offers experimental
WGPU-Matching CPM-HOG initialization. It seeds a global affine transform from
up to 512 reciprocal patch matches and requires a minimum robust consensus.
This path is implemented only in the standalone browser workbench, adds
per-pair descriptor construction and grayscale readback, and has not yet been
qualified on the analytical UHD fixtures. Failure to obtain a consensus stops
the pair; the application does not substitute another initializer. It is not
a camera-rate path and has no general accuracy or speed guarantee. See
:doc:`../how-to/register-large-shifts`.

The shared browser WGSL core has an experimental
``initialization="global-affine"`` for sparse image pairs. It fits one robust
affine transform over sampled pixels in the reference mask using five CPU
image scales by default. Configure 5–8 scales with ``global_affine_levels``
in Python or ``globalAffineLevels`` in JavaScript; six to eight are
experimental and require sufficient image dimensions and masked samples.
This setting is independent of the local solve pyramid. The fit then seeds
the local subsets. The workbench supports loaded
image and video files by reading grayscale pixels back from their external GPU
image; live camera streams are rejected. The per-target readback and CPU fit
can take substantial time, making this unsuitable for camera-rate processing.
It can initialize local translation, affine, or quadratic solves. With
translation, each subset receives only the fitted displacement at its centre,
so local deformation is not estimated. Quadratic solves receive the affine
seed with zero curvature coefficients; quadratic mode currently requires no
ROI mask. It is not implemented by the native engine or Three.js. A single
global affine model cannot represent strongly
non-affine specimen deformation; the following local solve may still reject
subsets. See :doc:`../how-to/register-large-shifts`.

``temporal_fine_correction=True`` is an additional two-level
option. It preserves fine history while applying the measured coarse change,
using the absolute seed if complete age-matched history is unavailable. It is
not a motion predictor or a recovery guarantee. The native-UHD performance
fine-correction-only campaign exceeds 60 FPS in all three measured repeats, but one repeat still
exceeds 16.67 ms at p95 with one slot. Two slots improve throughput while
increasing result age; neither mode establishes a hard real-time guarantee.
The newer fixed B-spline profile with finest-only conservative stopping passes
the three-run p95 gate; see :ref:`latency-reduction-profile`. That finite result
does not remove system-load, hardware or thermal qualification limits.

Native latency controls are experiments with explicit tradeoffs. Velocity
assumes consecutive equal-time steps, stores raw measurements even under
``replace``, and can amplify estimation noise or accepted wrong solutions.
Stagnation stopping can reject late convergence and reports status 0. Adaptive
mode prepares two levels but can solve only the fine level; it serializes host
feedback, so requesting two or three slots does not overlap images in that
mode. Default ``pyramid_policy="fixed"`` still solves every configured level.
The latency campaign qualifies the affine native-UHD tensile case; it does not
establish universal accuracy or hard real-time deadlines for other motions,
hardware, noise levels, discontinuities or long-duration thermal loads.
The initial acceptance-only adaptive prototype fails the analytic displacement and
gradient criteria on the native-UHD tensile trajectory despite keeping a high
accepted fraction. It remains experimental and is not recommended for this
measurement. A numerical-only replay after restricting stopping to the finest
level still reaches 0.874 px displacement RMSE and fails the gate. Cubic interpolation must also be evaluated independently of that
failure; the report includes fixed-pyramid and interpolation-only controls.

Runtime
-------

* Browser backends require Chromium as their WebGPU execution
  environment. Chromium is a system dependency and is not distributed in the
  Python wheel.
* Chromium is not required by ``ICGNWGPU``, but that native backend requires
  Python 3.11 or newer and a wgpu-native-supported graphics driver.
* One engine instance is not thread-safe.
* Native sequence pipelining improves throughput by keeping bounded work in
  flight; the default of two adds approximately one processed frame of result
  latency. This setting is unrelated to an image pyramid.
* Selective recovery on ``ICGNWGPU`` is deliberately off by default and is not
  the recommended native pyramid. It currently performs reduced-image
  construction and recovery orchestration on the CPU.
* The recommended ``ICGNWGPUFullPyramid`` backend removes those per-frame CPU
  reductions and intermediate readbacks and exposes bounded sequence
  pipelining. It still uses the portable WGSL solver rather than the native
  subgroup variant.
* Browser GPU shader arithmetic is single precision.
* The fastest Three.js solver requires a WebGPU subgroup capable of covering
  the subset and sizes its workgroup to the measured subgroup width. Reference
  preparation confirms the width inside the actual solver pipeline and selects
  the validated 16-lane workgroup-memory fallback if it differs from the
  discovery probe. This validation is performed for each rebuilt runtime, not
  in the per-target loop. Other adapters use the portable workgroup solver
  directly. There is no Three.js WebGL fallback.
* The raw ``webgpu-wgsl`` browser backend prioritizes source sharing and uses
  the portable workgroup solver at every pyramid level. It is slower than the
  specialized Three.js subgroup solver for the comparable single-scale case.
  Its default two-level graph requires an adapter exposing at least ten storage
  buffers per compute stage.
* Maximum image size depends on the adapter's WebGPU storage-buffer and
  texture limits.
* Python ``register`` calls materialize complete parameter/status fields.
  The standalone JS API separately offers full, decimated or small-summary
  readback modes.
* The native backend includes portable and subgroup solvers, but subgroup
  support does not imply higher performance. Reference preparation benchmarks
  both and selects the faster valid pipeline for the current adapter.
  The specialized WGSL path is limited to affine B-spline solving with uniform
  weighting, size 16, and stride 16.
* Current timing is soft real-time and provides no operating-system deadline
  guarantee.
* Native ``solver_workgroup_size=128`` is experimental and must be explicitly
  selected. A larger workgroup does not guarantee lower latency. The default
  remains 16; browser hosts and the specialized subgroup solver are unchanged.

The solver's HTML and JavaScript resources are packaged locally. Network access
is not a runtime requirement once the Python dependencies and Chromium are
installed.

Acquisition
-----------

The native Python package accepts NumPy-compatible frames but does not own a camera, trigger,
timestamp synchronization, or codec pipeline. Those components must preserve
their own timing and provenance and must be included in end-to-end qualification.

Scientific interpretation
-------------------------

Accepted status indicates numerical convergence and the configured ZNCC/spatial
tests. It is not by itself a complete uncertainty estimate. Experiments should
retain calibration, image quality, subset texture, timing, and rejected-region
information alongside displacement fields.

Standalone browser companion
----------------------------

The PWA requires WebGPU and a secure context. It has no WebGL fallback,
CDN/runtime toolchain, remote solver or SharedArrayBuffer dependency. Limits
are checked and the seeded layout fits eight storage buffers, but formal
limits do not guarantee adequate memory or sustained UHD speed on a phone.
The portable width is 16; subgroups are not required.

Full and partial ROI support translation and affine in the native Python and
raw browser WebGPU cores. Browser partial ROI requires CPU reference
preparation; visibility truncation and Gaussian integration remain unsupported.
Adaptive pyramids are explicitly rejected. The app
uses an explicit median/temporal latency configuration; core and Python
defaults have not changed. Minimal and cubic trajectory profiles are not
presented as qualified alternatives.

Phone capture is not qualified for metrology. Rolling shutter, optics, motion,
illumination and exposure locks require separate validation. Demo video is a
derived compressed asset; lossless UHD fixtures remain the scientific tests.
Only desktop Linux Chromium execution has been exercised locally. Android,
iOS/iPadOS Safari, desktop Safari and Firefox remain untested. Portable Python
CI is not evidence of camera or GPU qualification on those platforms.

Local sequences and incremental browser fields
----------------------------------------------

Local image decoding and video codecs depend on browser support; TIFF image
sequences are unsupported. Consecutive comparison rebuilds the numerical
reference for each pair and has no 4K60 qualification. Its ROI stays in image
coordinates, not material coordinates. Input frame loss cannot be reconstructed;
known gaps and busy-pair drops are counted. Incremental fields are not cumulative
trajectories. Playback speed does not provide physical acquisition time.

Results workbench
-----------------

Results recording is opt-in, adds full-field readback and has no live 60 Hz
qualification. Fields are written to browser-local IndexedDB with a configurable
CPU array cache up to 1 GiB (128 MiB on unknown memory). This excludes analysis
scratch, catalog/object overhead, GPU/decoder memory and export buffers. Browser
quotas, private browsing and eviction can limit or remove that storage; export
project parts for an independent backup. Disk errors retain the completed field
in RAM and pause recording. Raw images are not recorded. Analysis is a reversible
post-processing view, not a new convergence test. Smoothing reduces spatial
resolution. VSG can lose valid strain support near gaps and narrow regions;
invalid values are not filled. Result palettes and clims are display-only: they
never change stored parameters, statuses or exported measurements. The optional
masked-estimate rendering shows finite solver states only where an iteration
ran; it does not recover a failed solve. ROI exclusions and never-solved subsets
stay blank. The diagnostic colours do not change the acceptance mask used by
statistics, measurements or processed CSV exports. Use the categorical solver
status map to distinguish non-convergence, correlation rejection and geometry
exclusion. Percentile
clims describe only the valid displayed population and can hide extreme values
from view. Signed fields use a symmetric interval for percentile limits; a
sequential palette still maps negative through positive values monotonically
and does not mark zero as a visual centre. Use the diverging palette when zero
or sign is important. Unit scaling is isotropic and does not correct
lens distortion, perspective or out-of-plane motion. Gauge strain is an endpoint
distance measure. No automatic cumulative trajectory or physical acquisition
clock is inferred from folder replay. Native CLI/workbench archive layouts are supported. Missing image geometry
requires confirmation, and sequence import requires an explicit frame. Other
legacy layouts are not guessed. A history uses saved order and matching analysis
provenance; it does not supply an acquisition clock or material trajectory.
Multiple saved point, line, zone and extensometer definitions can be extracted
together, but their coordinates remain fixed in the reference image. A line
history chart shows its middle sampled subset; the long-format CSV retains all
line samples at every field. These derived values are not raw images or an
uncertainty estimate.

Results projects preserve computed fields and recipes but exclude input images
and live source permissions. Project import adds validated parts to the disk-backed library; it does not resume an acquisition automatically. Smoothing and
VSG compose their support sizes: two 5-point operations on a 16/16 grid have a
nominal 144 px strain span, not an 80 px span. Smaller windows can preserve more
spatial detail; numerical operator tests are not a material-specific choice of
strain gauge size.

Mask-image input is a reference-preparation operation. Exact mode rejects
nonbinary/transparent decoded images; explicit threshold conversion is a user
choice. Imported black pixels cannot be restored by adding vector domains;
replace/reimport that base to change them. Preview/export of unapplied edits is
distinct from the applied mask carried by a numerical field. The result canvas
has no retained source image; its vector and contour overlays use reference
coordinates and are presentation only.

Region statistics are equal-weight subset sample statistics, not area
integrals; overlapping subsets remain overlapping measurements. A region mean
can change when its valid population changes. History exports retain its sample
count and null gaps. Browser unload warnings may be suppressed by platform
policy and cannot replace saving and checking a project file.

Saved-field replay is a bounded presentation of compatible stored results, not
a physical-time movie or a numerical performance benchmark. Browser CPU review
works without WebGPU; new correlations still require WebGPU. Main-thread review
can stall interaction on large analyses. Camera media and track-processor clocks
can have different origins and must not be combined into exposure-latency claims.

Computed-coefficient plots are solver output diagnostics, including the solver's
configured spatial policy. Workbench smoothing/VSG do not alter those coefficients.
Native quadratic imports can display displacement second derivatives; browser
acquisition remains translation/affine. Their noise and estimator bias are not
removed by switching display or length units.

GPU memory diagnostics count application-owned buffer sizes and known texture
payloads, excluding driver padding, pipelines and canvas swapchains. WebGPU does
not expose physical/free VRAM or reliable unified/discrete placement. Mapped
buffers may be host-backed. A unified GPU shares the physical RAM pool; do not
add CPU and GPU figures as if they were separate free-memory budgets. These
diagnostics do not prove that a long recording meets a live FPS target.

The camera stage diagnostic is sampled from the latest admitted field at about
1 Hz; it is not a distribution of exposure-to-display latency. Consecutive
reference construction includes CPU preparation on every pair by default;
the experimental GPU option below changes that path explicitly.
Resetting the timing window on mode changes improves reporting, not its speed.


Experimental browser GPU reference preparation
----------------------------------------------

``referencePreparation='gpu'`` is an explicit JavaScript/PWA option; Python
native preparation and all existing defaults are unchanged. It uses portable
float32 WGSL, while the existing CPU reference preparation uses float64.
Consequently it does not promise bitwise-identical gradients or inverse
Hessians, nor identical decisions for numerically marginal textures. Tests
compare healthy reference quantities, flat rejections, analytical fractional
motions and field accuracy without loosening existing scientific gates.

The initial implementation supports one slot, uniform full windows,
translation and affine. It does not implement quadratic, partial masks or
Gaussian weighting. GPU memory is higher because reference scratch and
independent input textures persist. Consecutive pairs discard cross-reference
temporal history and use every configured pyramid level. They are incremental
measurements, not cumulative material tracking. Recorded fields still require
an explicit readback and durable storage admission.

Hardware/browser observations qualify only that tested configuration. A
software browser validates execution and numerical results, never live FPS.
The simulated static camera controls are not physical-camera tests. See
:doc:`gpu-reference-preparation` and the current performance campaign.


Bounded browser recording overlap and preview
---------------------------------------------

The experimental pipeline checkbox is opt-in. It preserves one GPU computation
at a time and permits a two-field CPU recording queue to overlap IndexedDB
admission. This hides some waiting; it cannot increase GPU capacity or sustain
an arrival rate permanently above disk throughput. A full queue slows admission,
and a failed write retains completed data and pauses capture. Computed FPS may
lead durable throughput temporarily and must not be presented as committed FPS.

A reduced-rate/size preview is approximate presentation, not a resampling of the
DIC input or a new field resolution. Numerical exports stay at the original
geometry/precision. Preview timing begins at worker availability, not camera
exposure. Explicit drain barriers protect source/reference/ROI/project changes.
Browser termination cannot be made transactional with pending writes; heed the
unsaved-data guard and export independent backups. Device-loss or unexpected
CPU retention failures stop admission rather than discarding a pending field.
See :doc:`browser-workbench` and :doc:`../how-to/run-in-browser` for ownership and
recovery rules. No 60 FPS or all-platform claim follows from this option alone.
The live preview can run at 1/5/10 Hz or be suspended. Explicit inspection and
pause/export flushes still draw; suppressing live presentation does not remove
numerical preparation, full-field readback or storage. Its maximum display rate
must not be used as an acquisition or computed-field FPS claim.

Live preview versus results analysis
-------------------------------------

The live companion shows sparse norm-coloured vectors over the current image,
or the image only. ``vectorOutline`` adds optional high-contrast borders as a renderer display
setting, off by default, not a numerical-engine option. It adds two sparse draw
calls and cannot make an unavailable vector valid or alter stored fields.
``vectorColourMap`` changes only the Live norm palette and legend, with the
original palette retained by default. The sequential alternatives do not add
spatial interpolation, recover rejected points or improve measurement accuracy.
Their device execution is checked on the available hardware/software adapters;
this does not qualify mobile displays or camera-to-display throughput.
It does not compute scalar/strain maps in the live renderer.
Preview decimation and its maximum rate are display choices, not changes to
full-resolution fields or a guarantee of 4K60 recording. Saved-field analysis
pauses acquisition; palette and range edits reuse cached component arrays and
statistics. Changing the recipe recomputes analysis. Camera images are not
archived by field recording.

Calculation and replay
-----------------------

Input previews do not calculate target fields. Start calculation explicitly in
Solve; save fields to use Results playback. Saving is initially selected in the
GUI and adds full readback/disk admission cost. A preview-only run has no saved
sequence to replay. Video/camera admission retains its explicit dropped-frame
policy; use an image folder when every input file must be calculated.
Playback rate is a presentation maximum, never an acquisition FPS measurement.
It waits for disk loading and derived analysis, keeping the last completed
map visible and explicitly labelled during the wait. The eight-entry/32 MiB
derived replay cache is additional to the configurable raw-field RAM target;
it is not a promise to cache a whole recording or sustain the selected rate.
Inspection and derived exports remain unavailable until the new analysis is
ready. Raw images are not recorded
with fields, so saved scalar maps use reference coordinates.

TIFF inputs
------------

Browser TIFF support remains an 8-bit input path. Unsigned 16-bit samples are
mapped with ``floor(value / 256)`` independently of image content; low bits are
lost, and low-dynamic-range data require an explicit common-range conversion
before import. TIFF ICC profiles are ignored. Multipage/BigTIFF, signed or
floating-point samples, planar channels, reversed bit order and unqualified
compressions/layouts are refused rather than partially decoded. TIFF import
requires a Worker, including when correlation is unavailable. File/dimension
limits and the 30-second decode timeout bound resource use; they do not promise
real-time TIFF decompression. See :doc:`../how-to/run-in-browser`.

Image dimensions beyond UHD
---------------------------

Browser reference preparation negotiates the buffer and texture limits needed
by the original image, including nonstandard and odd dimensions. This removes
the default 128 MiB storage-binding ceiling when the adapter permits a larger
binding. It does not remove physical allocation limits or the adapter's
maximum texture dimension. There is no automatic spatial tiling or silent
resizing. A supplied GPUDevice must have sufficient limits; only an owned
device may be recreated during a cold reference change. Minimum/coarsest subset
constraints still apply. See :doc:`browser-portability`.

Sparse pairs and global initialization
--------------------------------------

The opt-in Fourier mode is a global translation seed, not a replacement for
local affine/translation correlation. It requires overlapping texture in a
<=1024-square crop and may fail on repeated textures, rotation or strong
nonrigid motion. Score/overlap/correlation rejection is heuristic. CPU FFT and
browser target-window readback add latency; this mode has no 4K60 claim.
Quadratic, velocity prediction and temporal fine correction are unsupported
combinations. Four or more local pyramid levels alone do not guarantee capture
of large shifts; extended schedules remain experimental and have no general
performance qualification. Available local pyramid depths differ by host:

.. list-table:: Current large-shift capabilities by execution path
   :header-rows: 1

   * - Execution path
     - Local pyramid depth
     - Global-affine initializer
   * - Shared browser core (PWA and Python browser shim)
     - 1–6 levels
     - Available for decoded image/video pairs
   * - Native ``ICGNWGPUFullPyramid``
     - 1–4 levels
     - Not implemented
   * - Native ``ICGNWGPU`` selective-recovery engine
     - 1–3 levels
     - Not implemented
   * - Three.js browser engine
     - Backend-specific; no global-affine initializer
     - Not implemented

These are implementation limits, not a claim of numerical or performance
parity. The global-affine initializer is browser-core-only and is not intended
for camera-rate streaming. See :doc:`../how-to/register-large-shifts` for its
scope and current qualification.


Spatial candidate research
--------------------------

:doc:`spatial-initialization` describes a CPU diagnostic oracle for proposing
spatial seeds to the unchanged native solver. It is not a live backend or a
browser option. Its extra readbacks and scoring costs are not qualified for
real-time use; no optical-flow smoothing or candidate is returned as a measured
state without IC-GN re-estimation and its existing validity checks.

GPU seed propagation
--------------------

The native ``benchmarks/gpu_spatial_seed.py`` host remains development tooling.
It binds one prepared slot and supports synchronous ``register`` only;
``register_stream`` is explicitly refused, including with depth one. Prepare it
with ``prepare_research`` before solving and create a fresh host to replace its
reference. These restrictions do not change the public
``ICGNWGPUFullPyramid`` streaming or reference-replacement contracts.
The browser's ``spatialPropagationPasses`` setting uses the same kernels for
rejected-point search. Analysis/Solve defaults to two passes when the local
model is Translation or Affine and the setup uses B-spline interpolation,
uniform full subsets, CPU reference preparation and one slot. In affine mode,
propagation supplies only tx/ty and resets local derivatives before the affine
IC-GN retry. Automatic mode leaves it off for incompatible configurations;
explicitly requesting an unsupported setup is rejected. It is available in
Analysis/Solve, not the fixed Live preset. The direct ``DICEngine`` API still
defaults to zero.
Exact score caching remains a native research option; it is not automatically
activated by the browser. It needs 176 additional GPU bytes per grid position
and level, and falls back to recomputation at device buffer limits. The UHD
controls have not established a useful gain, so its memory/speed trade-off
must not be presented as an accepted live optimization.
Failed-point-only
selection cannot correct a bad basin already accepted by the solver. Coarse
median controls initialize the fine solve and do not validate an interpolated
or filled point as a final measurement. See :doc:`spatial-initialization` and
:doc:`../reference/performance`; timing on one analytic UHD pair does not
qualify camera capture, recording, discontinuities or general 4K60 recovery.

Opt-in browser workgroup width
------------------------------

``solverWorkgroupSize=32`` is an explicit constructor choice; 16 remains the
default and the fixed Live preset. It changes reduction order, so numerical
comparisons use the existing scientific error gates rather than assuming
bit identity between widths. Default-width UHD fingerprints are checked
separately for exact preservation. The 32-thread choice is not a universal
speed recommendation, mobile qualification or a sustained 4K60 guarantee.
The browser refuses width 128 and does not perform width autotuning.

Native seeded affine solves
---------------------------

``ICGNWGPU.register_seeded`` currently accepts host arrays for the affine
model and does not run pyramid recovery. It uploads and reads data; it is not
a GPU-resident interface. Its output status is the existing solver status,
not a guarantee of physical correctness. An optional final correlation pass on ``register`` and ``register_seeded``
currently supports only affine motion and uniform full windows; ``register``
requires a single pyramid level. Other integration configurations are
rejected explicitly. The score is not proof that a correspondence is correct.

``affine_frontier_seeds`` uses host arrays, performs one snapshot pass, and
does not enforce displacement or gradient consistency, track provenance, or
identify material discontinuities absent from the reference mask. Prepared
region-of-interest links prevent transport across known masked cracks. Its
median policy differs from the
two-neighbor policy used by some WGPU-Matching qualifications. It is opt-in
and does not change the browser or default propagation behavior.

An active ``ICGNWGPU.register_stream`` owns the solver until the iterator is
exhausted, closed, or terminates with an error. The standalone registration,
seeded solve, and affine-frontier calls reject concurrent use with
``ICGNError``. This does not imply arbitrary concurrent reference or engine
lifecycle operations are supported.
