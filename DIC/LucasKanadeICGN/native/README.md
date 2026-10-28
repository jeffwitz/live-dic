# Experimental Rust wgpu/PyO3 host

This crate is the second native-backend milestone. It currently:

- creates a persistent native `wgpu` Vulkan device;
- compiles the exact WGSL source used by the `wgpu-py` backend;
- exposes adapter provenance to Python;
- borrows contiguous NumPy `uint8` frames through PyO3 and measures direct
  `r8unorm` texture uploads without PNG, JSON, or a browser;
- reports the Rust queue-write boundary separately from synchronized upload
  completion.

The complete IC-GN dispatch currently lives in `ICGNWGPU` while this crate
qualifies the Rust/Python boundary. The next step is moving the already
validated buffer allocation, dispatch, timestamp, and readback host code from
Python into `NativeRuntime` without changing `shaders/icgn.wgsl`.

Build and install in a virtual environment:

```bash
maturin develop --release --manifest-path DIC/LucasKanadeICGN/native/Cargo.toml
```

Then inspect the adapter and run the upload probe:

```python
import _wgpu_native
import numpy as np

runtime = _wgpu_native.NativeRuntime()
print(runtime.adapter_info())
samples = runtime.benchmark_upload(
    np.zeros((2160, 3840), dtype=np.uint8),
    30,
)
print(np.median(samples["enqueue_ms"]))
print(np.median(samples["synchronized_ms"]))
```

On the Intel Lunar Lake validation GPU, the initial synchronized UHD upload
probe measured 1.783 ms median. This is a boundary qualification, not yet a
solver speedup: the complete IC-GN pipeline remains hosted by `wgpu-py`, whose
measured transfer and complete-call results are documented in the Sphinx
performance page.

For offline deployment, build a wheel on a connected machine with the same
platform and Python ABI:

```bash
maturin build --release \
  --manifest-path DIC/LucasKanadeICGN/native/Cargo.toml \
  --out wheelhouse
```

The resulting wheel installs offline without Cargo or a Rust toolchain.
