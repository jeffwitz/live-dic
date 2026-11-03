Acknowledgements and provenance
===============================

Methodology acknowledgement — Pierre Margerit
----------------------------------------------

Earlier GPU-based DIC experimentation in Pierre Margerit's `tests_gpuio project
<https://github.com/MargeritPierre/tests_gpuio>`_ and his later `Strobox project
<https://github.com/MargeritPierre/Strobox>`_ informed the methodological and
browser-GPU exploration in live-dic. Strobox documents a Three.js/WebGPU camera
prototype. This acknowledgment describes methodological influence; it does not
claim that Pierre authored the maintained live-dic solver or that the solver
reuses Strobox source code. The active Three.js/TSL solver was implemented as a
separate live-dic backend. Third-party software licenses and notices are listed
separately in the project-root ``THIRD_PARTY.md`` file.

Implementation provenance
-------------------------

The maintained solver and host adapters are developed in this repository. The
affine inverse-compositional formulation follows the associated SYCL DIC
prototype and the established IC-GN method; the old GPU-IO runtime has been
removed. The supplied validation images are direct luma-frame extractions from
the recorded raw YUV test sequence; their source path, format, hashes, and
analytical motion are stored in ``assets/source.json``.

The twelve-parameter second-order warp follows the established quadratic DIC
formulation described by Bai et al., "A novel 2nd-order shape function based
digital image correlation method for large deformation measurements",
*Optics and Lasers in Engineering*, 90 (2017),
DOI: ``10.1016/j.optlaseng.2016.09.010``. The implementation in this
repository uses a shared WGSL kernel and an exact lifted inverse-compositional
composition specialized for its documented parameter order.
