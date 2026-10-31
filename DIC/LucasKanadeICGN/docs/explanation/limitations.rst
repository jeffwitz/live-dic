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

GPU-IO is deprecated as an application backend. It is still packaged so that
historical measurements can be reproduced and the Three.js/WebGL2 replacement
can be tested on additional adapters. No removal release is scheduled until
that compatibility matrix has been exercised. New numerical features are not
planned for GPU-IO.

Three.js/TSL is the supported browser path. Its automatic WebGL2 fallback
removes the need to select GPU-IO merely because WebGPU is unavailable.

Python remains the primary supervisor for NumPy acquisition and complete NumPy
fields. The standalone PWA now supplies a JavaScript supervisor for raw WebGPU
camera/demo acquisition, bounded scheduling and GPU display. It is separate
from Three.js. ``ICGNWebGL`` still includes the Python/Chromium process boundary;
neither direct Three.js GPU times nor PWA worker timings are full Python-stream
times. See :doc:`browser-portability` for the browser qualification matrix.

Geometry and input
------------------

* All current backends support dynamic frame dimensions and have been validated
  at 3840 x 2160. Reference and target dimensions must match.
* GPU-IO dimensions must be exact multiples of the selected subset size because its texture grid is
  allocated when the private headless page starts. A dimension change rebuilds
  that page automatically.
* Native and raw browser shared-WGSL support subset widths 8, 15, 16, 17, 31,
  32, and 33. ``subset_stride`` defaults to the selected size and can be
  reduced to create overlapping subsets. Three.js/TSL, WebGL2, and GPU-IO
  support widths 8, 16, and 32 and still require stride equal to size.
* Centred Gaussian subset integration is currently exposed by the native
  engines. Browser hosts use uniform integration.
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
full-pyramid WGSL graph and defaults to two levels. Three.js/TSL and GPU-IO
remain single-scale.

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

* Browser backends require Chromium as their WebGL2 or WebGPU execution
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
  in the per-target loop. Other adapters use the portable fallback directly.
  Browsers without WebGPU use the serial Three.js/WebGL2 transform-feedback
  solver instead; this compatibility path is not a 4K60 implementation.
* The raw ``webgpu-wgsl`` browser backend prioritizes source sharing and uses
  the portable workgroup solver at every pyramid level. It is slower than the
  specialized Three.js subgroup solver for the comparable single-scale case.
  Its default two-level graph requires an adapter exposing at least ten storage
  buffers per compute stage.
* Maximum image size depends on the adapter's WebGPU storage-buffer or WebGL2
  texture limits.
* Full parameter and status fields are read back for every ``register`` call.
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

The package accepts NumPy-compatible frames but does not own a camera, trigger,
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

Full ROI supports translation and affine. Browser partial supports, truncation,
Gaussian integration and adaptive pyramids are explicitly rejected. The app
uses an explicit median/temporal latency configuration; core and Python
defaults have not changed. Minimal and cubic trajectory profiles are not
presented as qualified alternatives.

Phone capture is not qualified for metrology. Rolling shutter, optics, motion,
illumination and exposure locks require separate validation. Demo video is a
derived compressed asset; lossless UHD fixtures remain the scientific tests.
Only desktop Linux Chromium execution has been exercised locally. Android,
iOS/iPadOS Safari, desktop Safari and Firefox remain untested. Portable Python
CI is not evidence of camera or GPU qualification on those platforms.
