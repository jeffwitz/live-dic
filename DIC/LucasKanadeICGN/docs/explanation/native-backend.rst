Native WebGPU and PyO3 path
============================

Why a native path exists
------------------------

The browser backends preserve the original GPU-IO implementation and provide
portable WebGPU execution, but a Python frame crosses several representation
boundaries: NumPy, PNG, the DevTools protocol, JavaScript, and finally a GPU
resource. ``ICGNWGPU`` removes that browser transport. It uses `wgpu-py
<https://wgpu-py.readthedocs.io/>`_ to submit the project's WGSL kernels
directly through wgpu-native.

The native data path is:

.. code-block:: text

   contiguous NumPy uint8 frame
                |
                | direct buffer-protocol view
                v
          r8unorm GPU texture
                |
                +--> radius-15 FIR, horizontal then vertical
                |
                +--> one 16-lane workgroup per subset
                |    all affine IC-GN iterations in one dispatch
                |
                +--> spatial median consistency pass
                |
                v
      packed 8-float state per subset
                |
                | one binary readback
                v
         zero-copy NumPy result views

There is no Chromium process, canvas, PNG encoding, JSON conversion, or
DevTools socket on this path. The input must still be copied into a GPU-visible
resource: WebGPU does not make an arbitrary NumPy allocation into a texture.
The optimization removes avoidable representation and process transfers; it
does not claim that host-to-device transfer has disappeared.

Algorithm mapping
-----------------

The native backend uses the same 16 x 16 subsets, affine parameter order,
radius-15 target prefilter, inverse-compositional update, ZNCC test, and median
replacement as the browser implementations. Reference-dependent values are
prepared once by ``set_reference`` and uploaded as persistent buffers.

The current WGSL solver is the portable workgroup-memory implementation. Each
subset owns one workgroup with 16 active lanes. A lane processes one 16-pixel
row and the lanes reduce their partial sums through shared memory. All
iterations remain fused in one dispatch. A subgroup-specialized native kernel
is a future optimization; numerical qualification does not depend on it.

Python use
----------

Install the optional dependency and select the native engine explicitly:

.. code-block:: console

   $ python3 -m pip install -e 'DIC/LucasKanadeICGN/python[native]'

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPU

   with ICGNWGPU() as engine:
       preparation = engine.set_reference(reference)
       result = engine.register(target)

``RegistrationResult`` has the same parameter, displacement, status, and
iteration fields as the browser API. The native backend is experimental and
is not yet the package default.

Rust/PyO3 milestone
-------------------

``native/`` contains a second host implementation built with `Rust wgpu
<https://docs.rs/wgpu/>`_, `PyO3 <https://pyo3.rs/>`_, and Maturin. It already:

* creates and retains a native adapter, device, and queue;
* embeds and validates the exact WGSL source used by ``ICGNWGPU``;
* borrows a contiguous NumPy ``uint8`` array without an intermediate Python
  byte string;
* uploads it to an ``r8unorm`` texture; and
* reports queue-write and synchronized upload boundaries separately.

The full IC-GN host orchestration still lives in ``ICGNWGPU``. The Rust module
is deliberately a measured integration milestone, not a second incomplete
solver. The next Rust step is to move the already-qualified allocation,
dispatch, and packed readback sequence behind the same Python result contract,
without forking the WGSL kernels.

Measured effect
---------------

On the validation machine, the 512 x 512 native complete Python call measured
1.882 ms median and 2.382 ms p95 over 30 runs. At 3840 x 2160 it measured
16.160 ms median and 16.569 ms p95, including direct target upload, computation,
packed result readback, and NumPy result construction. The UHD campaign is just
inside one 60 Hz period, but it excludes camera acquisition and downstream
consumers. See :doc:`../reference/performance` for the exact boundaries and
the browser/SYCL comparison.

The initial PyO3-only UHD upload probe measured 1.783 ms median. It does not
yet demonstrate an advantage over the wgpu-py upload in the complete solver.
Its value at this stage is architectural: the Rust boundary is working,
offline-buildable, and shares the validated shader. Further migration should
be accepted only when the complete-call distribution improves.

