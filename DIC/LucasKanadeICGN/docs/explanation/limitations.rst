Current limitations
===================

Backend lifecycle
-----------------

GPU-IO is deprecated as an application backend. It is still packaged so that
historical measurements can be reproduced and the Three.js/WebGL2 replacement
can be tested on additional adapters. No removal release is scheduled until
that compatibility matrix has been exercised. New numerical features are not
planned for GPU-IO.

Three.js/TSL is the supported browser path. Its automatic WebGL2 fallback
removes the need to select GPU-IO merely because WebGPU is unavailable.

The supported application supervisor is currently Python. The repository does
not provide a persistent JavaScript supervisor for browser-native acquisition,
bounded frame scheduling, and local result consumption. Consequently,
``ICGNWebGL`` performance includes the Python/Chromium process boundary. Direct
Three.js GPU timings must not be presented as complete Python-stream timing.

Geometry and input
------------------

* All current backends support dynamic frame dimensions and have been validated
  at 3840 x 2160. Reference and target dimensions must match.
* GPU-IO dimensions must be exact multiples of 16 because its texture grid is
  allocated when the private headless page starts. A dimension change rebuilds
  that page automatically.
* Grid spacing and subset size are fixed at 16 pixels.
* Grid dimensions are ``floor(width / 16)`` by ``floor(height / 16)``;
  uncovered right or bottom pixels are not assigned to a subset.
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

No backend currently provides
temporal initialization, extrapolated initial guesses, Aitken acceleration,
predictive tolerance, or retry from the last accepted state. These features
matter for larger, noisier, or faster motion.

Runtime
-------

* Browser backends require Chromium as their WebGL2 or WebGPU execution
  environment. Chromium is a system dependency and is not distributed in the
  Python wheel.
* Chromium is not required by ``ICGNWGPU``, but that optional backend requires
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
* The PyO3 module currently qualifies adapter creation, shared-shader
  compilation, and direct texture upload. Full solver orchestration still
  lives in the wgpu-py host.
* Current timing is soft real-time and provides no operating-system deadline
  guarantee.

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
