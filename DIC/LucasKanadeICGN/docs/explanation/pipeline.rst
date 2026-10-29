Pipeline architecture
=====================

The Python API controls an optimized Three.js/TSL backend that selects WebGPU
or falls back to WebGL2, a raw WebGPU adapter using the canonical WGSL, or the
native wgpu-py backend. The deprecated GPU-IO/WebGL2 backend is
retained only for historical comparison and migration qualification.
Chromium is an execution environment
for the first three, not a user interface:
each headless page loads only its numerical runtime, creates no controls, and
never attaches a canvas to the visible document. The native path does not
start Chromium.

Supervision boundary
--------------------

The intended application supervisor is Python. It owns acquisition
integration, reference changes, configuration, bounded scheduling, timing, and
the returned NumPy fields. The native wgpu path is therefore the recommended
data plane for resident NumPy images.

The browser adapters expose the same numerical work to Python, but Chromium is
a separate process. A future performance-oriented Three.js deployment should
move per-frame supervision into JavaScript so that browser-native inputs and
result consumers stay beside the GPU runtime. Python may remain a control and
monitoring plane. See :doc:`supervision` for the exact boundary and decision
rule.

Data flow
---------

.. code-block:: text

   Python reference image
           |
           v
   grayscale PNG data URL -----> headless Chromium
                                      |
                                      v
                              reference preparation
                         F, gradients, H^-1, J^T F, sum(J)
                                      |
                                      v
                         persistent GPU textures or buffers

   Python target frame
           |
           v
   contiguous grayscale uint8
           |
           v
   same-origin binary long poll -----> target upload
                                      |
                                      v
                    selected reconstruction preparation
                   B-spline: FIR X + FIR Y; direct: none
                                      |
                                      v
                              selected IC-GN iterations
                                      |
                                      v
                              median consistency pass
                                      |
                                      v
                          parameter/status GPU readback
                                      |
                                      v
                            packed binary state
                                      |
                                      v
                               NumPy result views

The native path removes the browser process and loopback stage with a direct
buffer-protocol upload:

.. code-block:: text

   contiguous NumPy uint8 --> r8unorm texture --> WGSL compute
             ^                                      |
             |                                      v
             +-------- packed binary readback -- NumPy views

Reference stage
---------------

The reference stage is intentionally separate from target processing. It
computes the centered reference values, gradients, inverse Hessian, ``J^T F``,
and ``sum(J)`` for every subset. These quantities remain valid while the
reference and solver geometry remain unchanged.

Calling ``set_reference`` again replaces this state. Keeping one reference for
many targets is essential for sustained processing.

Target stage
------------

Every target is converted to grayscale and uploaded. The default ``bspline``
mode filters it into cubic B-spline coefficients; ``linear`` and ``cubic``
sample the uploaded target directly and skip both FIR passes. Three.js runs up
to 30 IC-GN iterations inside
one compute dispatch and fills the measured subgroup when available (32 lanes
on the validation GPU); its portable fallback uses 16 lanes. GPU-IO fuses the
same iteration loop into one fragment per subset and scans the selected square
serially. Raw browser WebGPU and native wgpu-py load the same WGSL solver, with
one 16-lane workgroup per subset and the same fused iteration structure. In
every case, converged or rejected subsets stop expensive target sampling. A
final GPU pass detects spatial outliers and can replace them with neighbor
medians. :doc:`gpu-optimization` describes the GPU mappings,
:doc:`shared-kernel` defines the common numerical contract, and
:doc:`native-backend` describes the native path.

For normal native coarse-to-fine processing, ``ICGNWGPUFullPyramid`` uploads
the finest target once and records reductions, per-level FIR passes, seeded
solves, affine transfers, the final median pass, and one readback into one
ordered command graph. The older selective ``ICGNWGPU`` option instead runs a
fine solve first and performs CPU-orchestrated recovery only after failures;
it is an optional compatibility path. See :doc:`backend-selection` for the
public-engine policy.

The reference intensities and gradients follow the selected reconstruction
model. This is why interpolation is an engine-level choice rather than a
per-call switch. :doc:`interpolation` defines the three models and their
measured trade-offs.

Control path
------------

Python uses the Chromium DevTools Protocol to start and configure the headless
page and to prepare the reference. Per-frame targets do not cross DevTools.
The page instead maintains a same-origin binary long poll against the private
loopback asset server. Python exposes the contiguous ``uint8`` storage without
PNG encoding; Chromium receives an ``Uint8Array`` and posts one interleaved
packed state field after registration (eight floats per subset for translation
and affine, sixteen for quadratic). Options and small timing metadata
travel in HTTP headers or a short JSON prefix, never as image or field JSON.

Raw WebGPU writes the received bytes directly to its ``r8unorm`` texture.
Three.js keeps four input pixels in each ``uint32`` upload word, then a small
compute pass expands them into its float source buffer. This avoids a UHD-sized
JavaScript scalar conversion and reduces the Three.js upload from about 33 MB
to about 8.3 MB. WebGL2 cannot use that WebGPU unpack pass and retains the
scalar conversion after the binary transfer.

The browser still runs in another process: the input crosses the loopback
boundary, a CPU-to-GPU upload remains, and the result crosses back. Binary
transport removes serialization, not those necessary copies.

``ICGNWGPU`` calls wgpu-native in-process. It uploads the grayscale NumPy
buffer to a texture and reads one packed binary state buffer. This also removes
the browser process and loopback copies, while retaining the required GPU
resource copy.

Concurrency model
-----------------

One engine instance owns one browser process or native device/queue and is not
thread-safe. A real-time application should keep GPU calls on one processing
thread and use a bounded queue to exchange frames and results with acquisition
and persistence workers. This describes the current Python supervisor. A
browser-native implementation needs the equivalent bounded queue and failure
policy in a persistent JavaScript supervisor; simply wrapping each frame in a
Python browser call preserves the inter-process bottleneck.
