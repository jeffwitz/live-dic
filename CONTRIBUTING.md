# Developing live-dic

The maintained Python project is under `DIC/LucasKanadeICGN/`. Keep the
distribution `lucas-kanade-icgn-webgl`, import `lucas_kanade_icgn` and CLI
`icgn-webgl` stable. Preserve Pierre Margerit's foundational attribution.

## Environment

Use Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv-icgn
.venv-icgn/bin/python -m pip install -e 'DIC/LucasKanadeICGN/python[dev]'
.venv-icgn/bin/python -m pip install -r DIC/LucasKanadeICGN/docs/requirements.txt
```

The examples below use `python3`, `ruff` and `node` from the intended environment.
Activate it or use the corresponding executable paths. On Windows, activate
`.venv-icgn\Scripts\Activate.ps1` and use `python` instead of `python3`.
Native execution needs a GPU/software-adapter driver; only browser integration
needs Chromium. Node is needed for JavaScript syntax and CPU geometry/queue contract tests.

## Required local checks

```bash
python3 -m unittest discover -s DIC/LucasKanadeICGN/python/tests
ICGN_RUN_GPU_TESTS=1 python3 -m unittest discover -s DIC/LucasKanadeICGN/python/tests
ruff check DIC/LucasKanadeICGN/python DIC/LucasKanadeICGN/benchmarks tools DIC/LucasKanadeICGN/docs
ruff format --check DIC/LucasKanadeICGN/python DIC/LucasKanadeICGN/benchmarks tools DIC/LucasKanadeICGN/docs
python3 tools/check_repository.py
python3 -m sphinx -W --keep-going -b html DIC/LucasKanadeICGN/docs DIC/LucasKanadeICGN/docs/_build/html
```

In PowerShell set `$env:ICGN_RUN_GPU_TESTS = "1"` before the GPU command, and
remove that variable for a portable-only run. A skipped GPU test is not proof
of execution. With GPU tests enabled, missing adapters or shader errors are
failures. Lifecycle tests also run on a software adapter. Performance campaigns
must explicitly use hardware and record the actual selected adapter.

Browser execution is separately enabled with `ICGN_RUN_BROWSER_TESTS=1`.
The native and browser flags can be combined. For Chromium SwiftShader use
`ICGN_BROWSER_SOFTWARE=1`; its times are correctness evidence only. The
`browser-software` CI job also compares sixty chronological frames per model
and slot depth, then exercises the assembled demo under a subpath.

Build the static companion with `python3 tools/build_pages.py --output _site`.
`tools/smoke_browser_app.py` checks it through the existing Chromium harness;
see the browser guide for the complete command and timing boundaries. The
Pages workflow is prepared; enabling publication is a repository-owner action.

For changed JavaScript, run `node --check` on each affected module. At minimum,
check the packaged GPU-IO, Three.js and raw-WebGPU entry modules when changing
shared browser resources. Do not rewrite vendor bundles to satisfy local style.

The root Ruff configuration extends the package configuration to active
benchmarks, tools and documentation examples. Archived `benchmarks/results/`
scripts are intentionally excluded: they describe the measured revision.

## Wheel and installed-package validation

Build a pure-Python wheel, including its WGSL, browser resources and licenses:

```bash
python3 -m pip wheel --no-deps DIC/LucasKanadeICGN/python --wheel-dir /tmp/live-dic-wheel
python3 tools/check_repository.py --wheel-dir /tmp/live-dic-wheel
python3 -m venv /tmp/live-dic-install
/tmp/live-dic-install/bin/python -m pip install /tmp/live-dic-wheel/*.whl
```

Use a fresh output directory containing one wheel. Then change to a directory
outside the checkout and import the installed package, inspect its packaged
resources, run `icgn-webgl --help`, and execute an available native registration
plus NPZ export. For release validation, also run the portable suite with that
environment's interpreter using the absolute path to `python/tests`.

The project wheel must remain `py3-none-any`; platform runtimes belong to
normal dependency wheels. Do not add first-party OS/architecture binaries or
an external compiler to installation. Offline wheelhouses must match the target
Python/OS/architecture. See the [installation guide](DIC/LucasKanadeICGN/docs/how-to/install.rst).

## Documentation and evidence

Write English code, comments, documentation and commit messages. Native Python
supervision is the primary product boundary. Keep examples native-first;
document browser CLI and one-shot behavior explicitly without changing it.
New options need API, how-to, explanation and limitations documentation.

Keep detailed timing tables in the [performance reference](DIC/LucasKanadeICGN/docs/reference/performance.rst),
and link them from landing pages. Every campaign must identify its revision,
inputs, hardware, parameters, timing boundary and evaluated populations.
Preserve raw reports and negative controls; do not replace an archived result
with a new run or silently change its scientific tolerances.

Portable GitHub-hosted OS jobs validate installation and Python contracts.
The Linux lavapipe job executes shaders using a software Vulkan adapter.
Neither proves hardware performance or all architectures. Consult the
[hardware matrix](DIC/LucasKanadeICGN/docs/reference/performance.rst#gpu-hardware-coverage)
before making a compatibility claim. Numerical fixtures and performance
protocols are described in [BENCHMARK.md](DIC/LucasKanadeICGN/BENCHMARK.md).

## Repository hygiene

Do not commit caches, environments, generated HTML or local wheelhouses.
Do not casually edit analytic fixtures, vendor assets, archived patches or
logs; their bytes and hashes provide provenance. Preserve original browser
demonstrations outside the maintained package. Keep commits focused, preserve
existing user work, and push only when requested.
