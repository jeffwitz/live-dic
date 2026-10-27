Install the Python runtime
==========================

System requirements
-------------------

The current backend requires:

* Python 3.10 or newer;
* Chromium or Google Chrome with WebGL2 support;
* a GPU and driver accepted by Chromium;
* the ``tests_gpuio`` checkout, because the runtime serves ``gpu-io.js`` and
  the IC-GN shader directly from the repository.

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
