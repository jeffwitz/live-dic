Native WebGPU with wgpu-py
==========================

New to GPU programming? :doc:`webgpu-fundamentals` explains wgpu-py, WGSL,
device resources, and synchronization before this implementation-level tour.
The runnable :doc:`../tutorials/first-gpu-kernel` shows the API objects in use.

Why a native path exists
------------------------

The native engines can execute on hardware GPU adapters or on a software CPU
adapter using the same WGSL. Select ``device_type="cpu"`` or ``"gpu"``
explicitly; the default ``"auto"`` prefers enumerated hardware. See
:doc:`../how-to/run-on-cpu` for the selection contract, diagnostics, and
software-driver requirements.

DIC-RT is intended to be supervised from Python. When acquisition produces
NumPy arrays and downstream code consumes NumPy fields, the GPU runtime should
therefore remain in the Python process. This is the product boundary for which
the native backend exists.

The browser backends preserve the original GPU-IO implementation and provide
portable WebGPU execution, but a Python frame crosses several representation
boundaries: NumPy, a binary loopback response, a Chromium ``Uint8Array``, and
finally a GPU resource. The binary browser path no longer encodes each target
as PNG or carries the field through DevTools/JSON, but the inter-process copies
remain. ``ICGNWGPU`` removes that browser transport. It uses `wgpu-py
<https://wgpu-py.readthedocs.io/>`_ to compile and submit the project's WGSL
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
                +--> one configurable 16-, 32-, or 128-lane workgroup per subset
                |    all IC-GN iterations in one dispatch
                |
                +--> optional displacement UOD pass
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

The native backend uses selectable widths 8, 15, 16, 17, 31, 32, and 33 and an
independent stride from 1 through the selected size. Odd widths provide an
exact centre pixel; stride defaults to size. Uniform least-squares integration
is the default, with centred Gaussian integration and sigma available as
engine options.
It retains the affine parameter order, 31-value radius-15 target filter vector,
inverse-compositional update, ZNCC
test, and displacement UOD as the browser implementations. The WGSL runtime
omits only the six outer taps whose combined absolute weight is below ``7e-8``
and stages the remaining reads in axis-oriented workgroup tiles.
Reference-dependent values are
prepared once by ``set_reference`` and uploaded as persistent buffers.

The portable solver assigns one workgroup to each subset. Its default
``solver_workgroup_size=16`` distributes the 256 pixels across 16 lanes. The
optional ``solver_workgroup_size=32`` distributes them across 32 lanes and
halves the per-lane pixel count. Experimental ``solver_workgroup_size=128``
assigns two pixels per lane for a 16 by 16 subset. All widths reduce partial
sums through shared memory. The reference, inverse-Hessian, metadata, and state buffers use
aligned ``vec4`` records. All iterations remain fused in one dispatch. Native
engines accept these three widths; raw browser WebGPU uses the shader's 16-lane
default. See :doc:`shared-kernel` for the binding contract.

Set the portable workgroup width when constructing either native engine:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   with ICGNWGPUFullPyramid(solver_workgroup_size=32) as engine:
       preparation = engine.set_reference(reference)
       result = engine.register(target)

The selected value is available as ``engine.solver_workgroup_size`` and in
``preparation.config["solverWorkgroupSize"]``. A 32- or 128-invocation solve uses the
portable reduction; ``prefer_subgroups=True`` is incompatible because the
separate subgroup shader is specialized for 16 invocations.

An optional first-party WGSL solver performs the same reductions with subgroup
collectives. ``wgpu-py`` 0.32 exposes the native capability as ``subgroup``.
The backend requests that feature when the configuration and selection policy
allow it, then compiles and executes a small WGSL qualification probe. The
probe verifies both the reported width and a vector ``subgroupAdd`` result
before the complete specialized solver can be constructed.

The native Naga frontend accepts the subgroup built-ins and operations without
the browser-side ``enable subgroups;`` directive. The two native-only WGSL
files therefore omit that directive explicitly; no general source rewriting
is performed. They are compiled by the wgpu-native library already contained
in the wgpu-py wheel. Cargo, Maturin, ``glslc``, a Vulkan SDK, and network
access are not needed to build, install, import, or execute DIC-RT.

The specialized source is limited to affine motion, B-spline interpolation,
uniform integration, 16 x 16 subsets, and a 16-pixel stride. Other
configurations remain independent of subgroup support and use the portable
WGSL solver.

By default, ``set_reference`` benchmarks both valid solver pipelines on a
representative shifted frame and retains the faster one. This tuning belongs
to preparation, not the per-target loop. Pass ``prefer_subgroups=False`` to
force the portable path or ``prefer_subgroups=True`` to force the subgroup
path for qualification. On the validation Intel driver, native subgroup
collectives were slower, so automatic selection correctly retained the
portable workgroup pipeline.
If the specialized candidate raises during validation or timing, automatic
selection records the exception in ``solver_validation`` and retains the
portable pipeline. Forced subgroup mode raises an explicit ``ICGNError``;
failure of the portable reference itself remains fatal.
``prefer_subgroups=True`` therefore requires both ``subset_size=16`` and
``subset_stride=16``.

Python use: single-scale and streaming engine
---------------------------------------------

Install the package and select the native engine explicitly:

.. code-block:: console

   $ python3 -m pip install -e DIC/LucasKanadeICGN/python

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPU

   with ICGNWGPU() as engine:
       preparation = engine.set_reference(reference)
       result = engine.register(target)

``RegistrationResult`` has the same parameter, displacement, status, and
iteration fields as the browser API. ``ICGNWGPU`` is the native single-scale
engine; it is not the recommended coarse-to-fine engine. Both native engines
now expose bounded sequence execution.

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
#. transfers motion-model seeds from coarse to fine coordinates; and
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
part of the normal registration path. It leaves the single-scale and selective
recovery behaviour of ``ICGNWGPU`` unchanged while adding its own bounded
full-pyramid stream.

The full-pyramid class defaults to two levels with the ``(8, 16)``
coarse-to-fine iteration schedule. This is the recommended native path. After
axis-oriented FIR tiling, both two and three levels retained a p95 margin under
the 16.667 ms period on the validation GPU; two levels remain the default
because they use the smaller graph and already cover the validated motion.
Select three levels explicitly when a larger displacement capture range is
required.

Reference preparation builds and retains the reference pixels, gradients,
inverse Hessians, and metadata for every scale. With the default fixed policy,
for each target the backend
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
   GPU polynomial seed transfer -> next solve -> ... -> finest solve
             |
             v
   finest optional UOD pass -> one packed readback

The native graph records ordered compute dispatches in one compute pass,
rather than one monolithic shader. Buffer clears precede that pass and packed
readback copies follow it. Dispatch ordering provides the dependencies while all images,
coefficients, states, and transferred seeds remain on the same GPU. This avoids
redundant pyramid construction inside each subset and keeps the numerical
solver in the canonical ``icgn.wgsl`` file.

Native temporal prediction and finest-level stagnation stopping are opt-in
controls described in :doc:`temporal-initialization`. They add no intermediate
readback. The qualified low-age profile retains both scales and complete
coarse solves; its stopping rule applies only at the finest level. The
acceptance-only adaptive experiment is a separate, currently unqualified
policy. See :ref:`latency-reduction-profile` for numerical and timing gates.

Iterations and tolerances are specified from coarse to fine. For example:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWGPUFullPyramid

   with ICGNWGPUFullPyramid() as engine:
       engine.set_reference(reference)
       result = engine.register(target)

The native target reduction evaluates the same separable
``[1, 4, 6, 4, 1] / 16`` filter as one tiled integer 5 x 5 kernel, with the
same uint8 truncation as the prepared reference pyramid. It eliminates the
horizontal intermediate texture. The shared shader retains the original
two-pass entry points for the browser. A one-level full
pyramid executes the same registration graph as the portable one-scale native
backend and has been verified bit for bit.

For sustained sequences, ``register_stream`` allocates a bounded ring of the
complete mutable graph: one finest target texture, reduced targets,
solver states, transferred seeds, and final
readback per slot. Reference images, inverse Hessians, metadata, FIR weights,
and pipelines remain shared. Queue order preserves input order and the
iterator applies backpressure when all slots are occupied.

``pipeline_depth=1`` minimizes result age. The default depth of two overlaps
GPU processing with Python retirement of the previous packed field and is the
recommended throughput compromise. A third slot is supported but adds little
throughput on the validation machine while increasing age. This slot depth is
independent of the number of pyramid levels.

For every result, timings distinguish normalization, direct NumPy-to-texture
upload, command encoding, submission, host completion wait, mapped readback
copy, and NumPy field construction. ``resultAgeWallMs`` spans the full input
acceptance to materialized-field boundary. ``completionWaitWallMs`` is only
the time for which the host blocks when retiring that slot; work may already
have executed while earlier results were being retired, so it is not a GPU
kernel duration.

Measured effect
---------------

On the validation machine, the changing-frame UHD synchronous campaign
measured the recommended two-level full pyramid at 11.571 ms median and 12.185
ms p95 in affine mode. Translation mode measured 10.992 / 11.397 ms. In the
four-run, 50-frame bounded campaign, two slots sustained 95.51 FPS affine and
104.09 FPS translation, with median result ages of 21.008 and 19.131 ms. This
boundary starts with a resident ``uint8`` NumPy image and ends with a
materialized NumPy field; it includes direct target upload, GPU pyramid
construction and solving, packed result readback, and result statistics.
Camera acquisition and downstream consumers remain excluded.

See :doc:`../reference/performance` for the current timing boundaries,
reproduction commands, and browser comparison.

The subgroup migration was qualified separately on the same Intel Arc 130V at
commit ``481516a``. On the 512 x 512 affine fixture with 8 warmups and 50
registrations, forced portable WGSL measured 1.836 ms median and 2.694 ms p95;
forced subgroup WGSL measured 1.816 ms median and 2.208 ms p95. Both accepted
900 subsets and reported a displacement RMSE below ``2.7e-5`` pixel. The
automatic preparation-time comparison measured 1.722 ms portable and 1.994 ms
subgroup and therefore retained the portable solver; its subsequent 50 calls
measured 1.721 ms median. The forced medians are close enough to demonstrate
run-to-run variability rather than a universal winner. These figures qualify
selection on this adapter and do not predict subgroup performance elsewhere.
The complete samples and provenance are preserved in the :download:`portable
report <../../benchmarks/results/2026-10-30-intel-arc-130v-portable.json>`,
:download:`subgroup report
<../../benchmarks/results/2026-10-30-intel-arc-130v-subgroup.json>`, and
:download:`automatic report
<../../benchmarks/results/2026-10-30-intel-arc-130v-auto.json>`.

The retired SPIR-V specialization was not used as a numerical performance
reference. On the starting commit its two-dimensional host dispatch was
flattened with only ``gl_WorkGroupID.x`` in the shader: on a 128 x 128 check it
accepted 0 subsets while the portable solver accepted 36. The automatic policy
therefore rejected it, as designed. The new WGSL specialization corrected that
indexing during the port and matched all 36 portable results, with a parameter
RMSE of ``1.5e-7`` and a maximum absolute difference of ``8.4e-7``.

The recommended full-pyramid path does not request or load subgroup support.
Its UHD figures above measure a different GPU-resident, two-level workload and
must not be compared directly with these 512 x 512 single-scale solver reports.

