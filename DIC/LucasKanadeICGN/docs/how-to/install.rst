Install the Python runtime
==========================

System requirements
-------------------

The browser backends require:

* Python 3.10 or newer;
* Chromium or Google Chrome with WebGL2 and WebGPU support;
* a GPU and driver accepted by Chromium.

The package embeds GPU-IO, Three.js r186, the IC-GN solvers, and both headless pages.
It does not require a repository checkout or a network connection at runtime.

Install the package
-------------------

From the repository root:

.. code-block:: console

   $ python3 -m venv .venv-icgn
   $ .venv-icgn/bin/pip install -e DIC/LucasKanadeICGN/python

Verify the command-line interface:

.. code-block:: console

   $ .venv-icgn/bin/icgn-webgl --help

Verify the headless GPU runtime:

.. code-block:: console

   $ .venv-icgn/bin/icgn-webgl \
       DIC/LucasKanadeICGN/assets/frame_000.png \
       DIC/LucasKanadeICGN/assets/frame_010.png \
       --truth 2 -1

The JSON output should name the GPU, report a 32 x 32 grid, and contain 900
accepted subsets for this test pair.

Select a compute backend
------------------------

Three.js/TSL is the default backend. A standard engine therefore selects it:

.. code-block:: python

   engine = ICGNWebGL()

Select the previous implementation with ``ICGNWebGL(backend="gpuio")`` or the
command-line option ``--backend gpuio``. On Linux, the default Python driver
supplies the native headless Vulkan flags required by Chromium. Confirm that
``engine.info().device`` names the physical adapter rather than ``swiftshader``
before reporting performance.

Prepare an offline installation
-------------------------------

On a connected machine with the same operating system, architecture, and
Python version as the offline machine, build a local wheelhouse:

.. code-block:: console

   $ python3 -m pip wheel DIC/LucasKanadeICGN/python --wheel-dir wheelhouse

This downloads wheels for NumPy, Pillow, and websockets and builds the IC-GN
wheel with its HTML and JavaScript resources. Copy ``wheelhouse/`` to the
offline machine, where Chromium must already be installed, then run:

.. code-block:: console

   $ python3 -m pip install \
       --no-index \
       --find-links wheelhouse \
       lucas-kanade-icgn-webgl

No CDN or package index is contacted during subsequent registrations.

Select a browser explicitly
---------------------------

If automatic browser discovery fails, pass the executable path:

.. code-block:: python

   from lucas_kanade_icgn import ICGNWebGL

   engine = ICGNWebGL(chromium="/usr/bin/chromium")

The equivalent command-line option is ``--chromium``.

Chromium sandbox
----------------

``ICGNWebGL`` passes ``--no-sandbox`` by default for container compatibility.
On a system with a working Chromium sandbox, prefer:

.. code-block:: python

   engine = ICGNWebGL(no_sandbox=False)

Use ``--enable-chromium-sandbox`` on the command line.

Build the documentation locally
-------------------------------

.. code-block:: console

   $ .venv-icgn/bin/pip install -r DIC/LucasKanadeICGN/docs/requirements.txt
   $ .venv-icgn/bin/sphinx-build \
       -W --keep-going \
       -b html \
       DIC/LucasKanadeICGN/docs \
       DIC/LucasKanadeICGN/docs/_build/html

Open ``DIC/LucasKanadeICGN/docs/_build/html/index.html`` in a browser.
