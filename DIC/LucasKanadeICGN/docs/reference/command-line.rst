Command-line reference
======================

Synopsis
--------

.. code-block:: text

   icgn-webgl [options] reference target [target ...]

Positional arguments
--------------------

``reference``
   One reference image. Its dimensions define the subset grid for either
   backend; GPU-IO requires both dimensions to be multiples of 16.

``targets``
   One or more target images with the same dimensions as the reference.
   ``--benchmark`` requires exactly one.

Solver options
--------------

``--iterations N``
   Maximum IC-GN iterations, from 1 to 30. Default: 30.

``--tolerance PX``
   Positive corner-update convergence threshold in pixels. Default: ``1e-3``.

``--zncc-min VALUE``
   Minimum accepted ZNCC in ``[0, 1]``. Default: ``0.8``.

``--truth DX DY``
   Constant true translation used to report vector RMSE.

Benchmark and output options
----------------------------

``--benchmark``
   Run the repeated target prefilter and solver benchmark.

``--repeats N``
   Number of measured runs. Default: 30.

``--warmups N``
   Number of unreported warm-up runs. Default: 3.

``--output PATH``
   Write complete numeric arrays and timing samples to a compressed ``.npz``
   archive.

Runtime options
---------------

``--repo-root PATH``
   Override automatic repository discovery.

``--chromium PATH``
   Select a Chromium or Chrome executable.

``--startup-timeout SECONDS``
   Browser startup timeout. Default: 20 seconds.

``--call-timeout SECONDS``
   Timeout for one DevTools operation. Default: 120 seconds.

``--enable-chromium-sandbox``
   Do not pass ``--no-sandbox`` to Chromium.

``--chromium-arg ARG``
   Add one Chromium argument. Repeat the option to add multiple arguments.

Output
------

Standard output is JSON containing the device name, timer availability,
configuration, accepted and replaced counts, mean translation, iteration
statistics, and timing summary. Reference construction is reported separately
under ``reference_preparation``. Errors are written to standard error and
return exit status 2.
