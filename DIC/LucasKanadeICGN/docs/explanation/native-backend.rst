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
                +--> tiled separable FIR, horizontal then vertical
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
31-value radius-15 target filter vector, inverse-compositional update, ZNCC
test, and median replacement as the browser implementations. The WGSL runtime
omits only the six outer taps whose combined absolute weight is below ``7e-8``
and stages the remaining reads in axis-oriented workgroup tiles.
Reference-dependent values are
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

Python use: single-scale and streaming engine
---------------------------------------------

Install the optional dependency and select the native engine explicitly:

.. code-block:: console

   $ python3 -m pip install -e 'DIC/LucasKanadeICGN/python[native]'

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPU

   with ICGNWGPU() as engine:
       preparation = engine.set_reference(reference)
       result = engine.register(target)

``RegistrationResult`` has the same parameter, displacement, status, and
iteration fields as the browser API. ``ICGNWGPU`` is the native low-latency and
bounded-sequence engine; it is not the recommended coarse-to-fine engine.

Bounded sequence execution
--------------------------

``register`` uses one target resource set and synchronizes its readback before
returning. ``register_stream`` instead keeps a bounded ring of independent
target textures, FIR intermediates, solver state, and mapped readback buffers.
Reference, inverse-Hessian, metadata, and FIR-weight buffers remain shared.

With the default ``pipeline_depth=2``, Python retires the oldest packed result while the GPU
can process the next submitted target. Queue ordering preserves input order,
and the iterator applies backpressure whenever both slots are occupied. The
design improves steady-state period rather than the latency of an individual
registration. See :doc:`../reference/performance` for the measured throughput
and latency trade-off.

Optional selective pyramid recovery
-----------------------------------

.. warning::

   This is the older compatibility/exception path, not the recommended native
   pyramid.  Use ``ICGNWGPUFullPyramid`` for normal coarse-to-fine processing.

``pyramid_levels=1`` is the default and executes the original registration
graph without coarse engines, seed buffers, or retry dispatches. Values ``2``
and ``3`` prepare half- and optionally quarter-resolution references once.

For every target, the fine solver remains the first observation. If all useful
subsets converge, no pyramid work is performed. Otherwise the native host:

#. reduces the target with the SYSCL five-tap ``[1, 4, 6, 4, 1] / 16`` filter;
#. computes the four-source interpolation closure required by failed subsets;
#. solves only that active closure at the coarse levels;
#. transfers affine seeds from coarse to fine coordinates; and
#. retries only failed fine subsets before the spatial consistency pass.

The final merge restores the parameters, status, and iteration count of every
subset accepted by the first fine solve bit for bit. Pyramid recovery therefore
extends capture locally; it does not silently revise observations that were
already published as accepted.

The current native implementation constructs reduced target images on the CPU
and uses one retained wgpu engine per coarse scale. This is acceptable for an
exception path, but the active two-scale 4K path is not a 60 FPS path. Moving
the five-tap reduction and level orchestration onto one device is the next
optimization if recovery frequency becomes significant.

GPU-resident full pyramid
-------------------------

``ICGNWGPUFullPyramid`` is the recommended native backend when a pyramid is
part of the normal registration path. It deliberately leaves the low-latency
and streaming-oriented ``ICGNWGPU`` behaviour unchanged.

The full-pyramid class defaults to two levels with the ``(8, 16)``
coarse-to-fine iteration schedule. This is the recommended native path. After
axis-oriented FIR tiling, both two and three levels retained a p95 margin under
the 16.667 ms period on the validation GPU; two levels remain the default
because they use the smaller graph and already cover the validated motion.
Select three levels explicitly when a larger displacement capture range is
required.

Reference preparation builds and retains the reference pixels, gradients,
inverse Hessians, and metadata for every scale. For each target, the backend
then records this graph into one WebGPU command buffer:

.. code-block:: text

   upload finest target once
             |
             v
   5-tap GPU reductions + per-level spline prefilters
             |
             v
   coarsest IC-GN solve
             |
             v
   GPU affine seed transfer -> next solve -> ... -> finest solve
             |
             v
   finest median filter -> one packed readback

The graph uses separate compute passes rather than one monolithic shader. Pass
ordering provides the required device-wide dependencies while all images,
coefficients, states, and transferred seeds remain on the same GPU. This avoids
redundant pyramid construction inside each subset and keeps the numerical
solver in the canonical ``icgn.wgsl`` file.

Iterations and tolerances are specified from coarse to fine. For example:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   with ICGNWGPUFullPyramid() as engine:
       engine.set_reference(reference)
       result = engine.register(target)

The target reduction uses the same separable ``[1, 4, 6, 4, 1] / 16`` filter
and uint8 truncation as the prepared reference pyramid. A one-level full
pyramid executes the same registration graph as the portable one-scale native
backend and has been verified bit for bit. Sequence pipelining is not yet
exposed by this new backend.

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

On the validation machine, the latest 50-run UHD campaign measured the
recommended two-level full pyramid at 10.863 ms median and 12.915 ms p95 for
the complete Python call. It accepted 31,724 of 32,400 subsets on the difficult
``(4, -2)`` pair with ``0.001873`` px truth RMSE. Three levels measured 11.354
ms median and 13.172 ms p95. These boundaries include direct target upload,
GPU pyramid construction and solving, packed result readback, and NumPy result
construction, but exclude camera acquisition and downstream consumers.

See :doc:`../reference/performance` for the current timing boundaries,
reproduction commands, and browser comparison.

The PyO3 upload probe does not yet demonstrate an advantage in a complete
solver. Its value at this stage is architectural: the Rust boundary is
working, offline-buildable, and shares the validated shader. Publish a Rust
performance claim only after complete solver orchestration is implemented and
rerun against the same current workload and timing boundary.

