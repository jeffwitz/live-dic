Current limitations
===================

Geometry and input
------------------

* All four backends support dynamic frame dimensions and have been validated
  at 3840 x 2160. Reference and target dimensions must match.
* GPU-IO dimensions must be exact multiples of 16 because its texture grid is
  allocated when the private headless page starts. A dimension change rebuilds
  that page automatically.
* Grid spacing and subset size are fixed at 16 pixels.
* Grid dimensions are ``floor(width / 16)`` by ``floor(height / 16)``;
  uncovered right or bottom pixels are not assigned to a subset.
* Inputs are converted to 8-bit grayscale.
* Every Python frame is encoded as PNG before browser transfer. The native
  ``ICGNWGPU`` path instead uploads contiguous 8-bit grayscale bytes directly.

Registration range
------------------

IC-GN is a local optimizer. The direct ``t=0 -> 20`` analytical displacement of
``(4, -2)`` pixels lies outside the robust local basin for many subsets in the
current single-scale implementation. The incremental ``t=10 -> 20`` step of
``(2, -1)`` pixels returns 900 accepted subsets.

The current implementation has no image pyramid, temporal initialization,
extrapolated initial guess, Aitken acceleration, predictive tolerance, or retry
from the last accepted state. These features matter for larger, noisier, or
faster motion.

Runtime
-------

* Browser backends require Chromium as their WebGL2 or WebGPU execution
  environment. Chromium is a system dependency and is not distributed in the
  Python wheel.
* Chromium is not required by ``ICGNWGPU``, but that optional backend requires
  Python 3.11 or newer and a wgpu-native-supported graphics driver.
* One engine instance is not thread-safe.
* Browser GPU shader arithmetic is single precision.
* The fastest Three.js solver requires a WebGPU subgroup containing all 16
  active lanes. Other adapters use the validated workgroup-memory fallback.
* The raw ``webgpu-wgsl`` browser backend prioritizes source sharing and uses
  only the portable workgroup kernel; it is slower than the specialized
  Three.js subgroup solver on the validation machine.
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
