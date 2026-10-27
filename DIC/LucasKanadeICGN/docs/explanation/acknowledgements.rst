Acknowledgements and provenance
===============================

Pierre Margerit's foundational contribution
--------------------------------------------

This Python real-time DIC work exists because of `Pierre Margerit's original
tests_gpuio project <https://github.com/MargeritPierre/tests_gpuio>`_. Pierre
created the repository, assembled its GPU-IO experiments, and established the
WebGL framework on which this implementation is built.

The affine IC-GN solver, headless Python interface, benchmarks, and documentation
in this fork should therefore be understood as a continuation of Pierre's work,
not as an independent starting point. Any presentation or reuse of this work
should preserve that provenance and credit the original project.

GPU-IO
------

The repository uses `GPU-IO <https://github.com/amandaghassaei/gpu-io>`_,
created by Amanda Ghassaei, as its WebGL computation framework.

Implementation provenance
-------------------------

The affine inverse-compositional formulation is ported from the associated SYCL
DIC prototype into Pierre's GPU-IO framework. The supplied validation images are
direct luma-frame extractions from the recorded raw YUV test sequence; their
source path, format, hashes, and analytical motion are stored in
``assets/source.json``.
