Choose the scope of solver-time median filtering
================================================

The ``median`` spatial policy has two scopes on the native WGSL engines and
the raw ``webgpu-wgsl`` browser core:

* ``rejected`` (default) repairs failed subsets from accepted neighbours.
  Accepted measurements pass through unchanged.
* ``all`` performs that same repair and also smooths accepted parameter states
  after every pyramid level, beginning at the coarsest level. The filtered
  coarse state seeds the next finer solve.

This is a component-wise median of the active motion parameters on the subset
grid. It is distinct from smoothing saved fields in the Results workbench.

Use the whole-field scope in Python
-----------------------------------

Select it when constructing either native solver:

.. code-block:: python

   from live_dic import ICGNWGPUFullPyramid

   with ICGNWGPUFullPyramid(
       motion_model="affine",
       pyramid_levels=3,
       outlier_policy="median",
       median_filter_size=5,
       median_filter_scope="all",
       device_type="gpu",
   ) as engine:
       engine.set_reference(reference)
       result = engine.register(current)

``ICGNWGPU`` accepts the same options for a single-scale solve. A scope other
than ``rejected`` requires ``outlier_policy="median"``. In the Python browser
wrapper, ``median_filter_scope="all"`` is available only with
``backend="webgpu-wgsl"``; Three.js does not implement it.

Use the workbench
-----------------

In Analysis, open **Solve**, enable custom settings, select **Median** under
**Solver spatial policy**, then choose **All subsets** under **Median
application**. This setting is applied when calculation starts and reference
preparation runs. The default remains **Rejected subsets only**.

Interpret the result carefully
------------------------------

For each accepted target, the filter gathers its own parameter state and
eligible accepted neighbours in the configured stencil. Neighbour states are
recentered at the target before taking a median for each active coefficient.
The target needs at least four eligible values; otherwise its state is left
unchanged. ROI permissions prevent filtering across excluded regions.

The filter does not change convergence status, iterations, or correlation
scores. Repaired failed subsets remain marked rejected until a subsequent
IC-GN solve accepts them. All-subset scope is explicit smoothing: it can blur
localized motion or strain and bias estimates near discontinuities. Larger
stencils affect more of the field and cost more. This option has not received
general performance or metrology qualification, so compare displacement,
derivative features, coverage and timing against the default on the intended
specimen before using it for measurements.

See :doc:`../explanation/displacement-outliers` for the detection and repair
rules, and :doc:`../reference/api` for the supported engines and defaults.
