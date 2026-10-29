Pipeline architecture
=====================

The Python API controls an optimized Three.js/TSL backend that selects WebGPU
or falls back to WebGL2, a raw WebGPU adapter using the canonical WGSL, or the
experimental native wgpu-py backend. The deprecated GPU-IO/WebGL2 backend is
retained only for historical comparison and migration qualification.
Chromium is an execution environment
for the first three, not a user interface:
each headless page loads only its numerical runtime, creates no controls, and
never attaches a canvas to the visible document. The native path does not
start Chromium.

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
   grayscale PNG data URL -----> target upload and FIR prefilter
                                      |
                                      v
                            cubic B-spline coefficients
                                      |
                                      v
                              affine IC-GN iterations
                                      |
                                      v
                              median consistency pass
                                      |
                                      v
                          parameter/status GPU readback
                                      |
                                      v
                               NumPy result arrays

The native path replaces the PNG and DevTools stages with a direct buffer-
protocol upload:

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

Every target is converted to grayscale, uploaded, and filtered into cubic
B-spline coefficients. Three.js runs up to 30 affine IC-GN iterations inside
one compute dispatch and fills the measured subgroup when available (32 lanes
on the validation GPU); its portable fallback uses 16 lanes. GPU-IO fuses the
same iteration loop into one fragment per subset and scans the 16 x 16 tile
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

Control path
------------

Python communicates with the headless page through the Chromium DevTools
Protocol over a loopback WebSocket. The returned JavaScript arrays are converted
to NumPy without changing their floating-point values. This design preserves
the existing browser implementations, but PNG serialization and DevTools transfer
are visible costs in the complete Python latency.

``ICGNWGPU`` calls wgpu-native in-process. It uploads the grayscale NumPy
buffer to a texture and reads one packed binary state buffer. This removes PNG,
WebSocket, JavaScript-array, and JSON boundaries, while retaining the required
GPU resource copy.

Concurrency model
-----------------

One engine instance owns one browser process or native device/queue and is not
thread-safe. A real-time application should keep GPU calls on one processing
thread and use a bounded queue to exchange frames and results with acquisition
and persistence workers.
