Acknowledgements and provenance
===============================

Pierre Margerit's foundational contribution
--------------------------------------------

Pierre Margerit's `tests_gpuio project
<https://github.com/MargeritPierre/tests_gpuio>`_ established the original
GPU-IO/WebGL DIC prototype. His later `Strobox project
<https://github.com/MargeritPierre/Strobox>`_ contains a Three.js/WebGPU camera
prototype that informed the browser path developed here.

The affine IC-GN solver, headless Python interface, benchmarks, and documentation
in this repository continue that foundational work. Presentations and reuse
should credit Pierre and distinguish both project stages.

GPU-IO
------

The standalone historical browser demo uses `GPU-IO
<https://github.com/amandaghassaei/gpu-io>`_, created by Amanda Ghassaei, as
its WebGL computation framework. The maintained Python package no longer
includes or exposes GPU-IO as a backend.

Implementation provenance
-------------------------

The affine inverse-compositional formulation is ported from the associated SYCL
DIC prototype into Pierre's original GPU-IO framework. His later Three.js/WebGPU
experiments in Strobox informed the browser execution direction. The supplied
validation images are direct luma-frame extractions from the recorded raw YUV test sequence; their
source path, format, hashes, and analytical motion are stored in
``assets/source.json``.

The twelve-parameter second-order warp follows the established quadratic DIC
formulation described by Bai et al., "A novel 2nd-order shape function based
digital image correlation method for large deformation measurements",
*Optics and Lasers in Engineering*, 90 (2017),
DOI: ``10.1016/j.optlaseng.2016.09.010``. The implementation in this
repository uses a shared WGSL kernel and an exact lifted inverse-compositional
composition specialized for its documented parameter order.
