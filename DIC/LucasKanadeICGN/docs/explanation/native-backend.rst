Native WebGPU and PyO3 path
============================

Why a native path exists
------------------------

The browser backends preserve the original GPU-IO implementation and provide
portable WebGPU execution, but a Python frame crosses several representation
boundaries: NumPy, PNG, the DevTools protocol, JavaScript, and finally a GPU
resource. ``ICGNWGPU`` removes that browser transport. It uses `wgpu-py
<https://wgpu-py.readthedocs.io/>`_ to submit the project's WGSL and SPIR-V
kernels directly through wgpu-native.

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

The portable solver assigns one 16-lane workgroup to each subset. A lane owns
one column and processes its 16 pixels; the lanes reduce their partial sums
through shared memory. Its reference, inverse-Hessian, metadata, and state
buffers use aligned ``vec4`` records. All iterations remain fused in one
dispatch. Raw browser WebGPU loads this same source; see
:doc:`shared-kernel` for the binding contract.

An optional first-party GLSL solver performs the same reductions with subgroup
collectives. It is compiled to SPIR-V at build time because the Naga frontend
bundled with wgpu-py 0.32 cannot parse the WGSL subgroup extension. Both the
GLSL source and generated SPIR-V are included in the wheel; no shader compiler
or network access is needed at runtime.

By default, ``set_reference`` benchmarks both valid solver pipelines on a
representative shifted frame and retains the faster one. This tuning belongs
to preparation, not the per-target loop. Pass ``prefer_subgroups=False`` to
force the portable path or ``prefer_subgroups=True`` to force the subgroup
path for qualification. On the validation Intel driver, native subgroup
collectives were slower, so automatic selection correctly retained WGSL.

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

Bounded sequence execution
--------------------------

``register`` uses one target resource set and synchronizes its readback before
returning. ``register_stream`` instead keeps a bounded ring of independent
target textures, FIR intermediates, solver state, and mapped readback buffers.
Reference, inverse-Hessian, metadata, and FIR-weight buffers remain shared.

With ``pipeline_depth=2``, Python retires the oldest packed result while the GPU
can process the next submitted target. Queue ordering preserves input order,
and the iterator applies backpressure whenever both slots are occupied. The
design improves steady-state period rather than the latency of an individual
registration. See :doc:`../reference/performance` for the measured throughput
and latency trade-off.

Rust/PyO3 milestone
-------------------

``native/`` contains a second host implementation built with `Rust wgpu
<https://docs.rs/wgpu/>`_, `PyO3 <https://pyo3.rs/>`_, and Maturin. It already:

* creates and retains a native adapter, device, and queue;
* embeds and validates the portable WGSL source used by ``ICGNWGPU``;
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

On the validation machine, the median of three 512 x 512 campaigns measured a
native complete Python call of 1.812 ms median and 2.187 ms p95. At 3840 x
2160 the median campaign measured 14.621 ms median and 15.748 ms p95, including direct target upload,
computation, packed result readback, and NumPy result construction. The UHD
median and p95 are inside one 60 Hz period, but both exclude
camera acquisition and downstream consumers. See
:doc:`../reference/performance` for the exact boundaries and the browser/SYCL
comparison.

The initial PyO3-only UHD upload probe measured 1.783 ms median. It does not
yet demonstrate an advantage over the wgpu-py upload in the complete solver.
Its value at this stage is architectural: the Rust boundary is working,
offline-buildable, and shares the validated shader. Further migration should
be accepted only when the complete-call distribution improves.

