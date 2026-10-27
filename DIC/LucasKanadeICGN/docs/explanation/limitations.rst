Current limitations
===================

Geometry and input
------------------

* Frame size is fixed at 512 x 512 pixels.
* Grid spacing and subset size are fixed at 16 pixels.
* The grid is fixed at 32 x 32 subsets.
* Inputs are converted to 8-bit grayscale.
* Every Python frame is currently encoded as PNG before browser transfer.

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

* Chromium is required as the WebGL2 execution environment.
* One engine instance is not thread-safe.
* WebGL2 shader arithmetic is single precision.
* Full parameter and status fields are read back for every ``register`` call.
* Current timing is soft real-time and provides no operating-system deadline
  guarantee.

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
