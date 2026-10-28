Pipeline architecture
=====================

The Python API controls either a persistent GPU-IO/WebGL2 backend or an
independent Three.js/TSL WebGPU backend. Chromium is an execution environment,
not a user interface: each headless page loads only its numerical runtime,
creates no controls, and never attaches a canvas to the visible document.

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
B-spline coefficients. The selected GPU solver then runs up to 30 affine IC-GN
iterations per subset inside one fused compute dispatch. Subsets stop doing the
expensive interpolation and update after convergence or rejection. The fast
path uses 16 active lanes and hardware subgroup reductions; a startup probe
selects the portable workgroup-memory implementation when one subgroup cannot
contain those lanes. A final GPU pass detects spatial outliers and can replace
them with neighbor medians.

Control path
------------

Python communicates with the headless page through the Chromium DevTools
Protocol over a loopback WebSocket. The returned JavaScript arrays are converted
to NumPy without changing their floating-point values. This design preserves
the existing browser implementations, but PNG serialization and DevTools transfer
are visible costs in the complete Python latency.

Concurrency model
-----------------

One ``ICGNWebGL`` instance owns one browser process and is not thread-safe. A
real-time application should keep GPU calls on one processing thread and use a
bounded queue to exchange frames and results with acquisition and persistence
workers.
