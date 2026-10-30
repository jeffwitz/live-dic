"""Machine-local exact comparison to arrays captured before temporal changes.

Run from any directory with the current package installed. This is a same-
device/runtime qualification, not a cross-driver bitwise requirement.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lucas_kanade_icgn import ICGNWGPUFullPyramid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python/tests"))
from test_gpu_regressions import _fractional_pair


def main():
    baseline = np.load(
        Path(__file__).with_name("before-default.npz"), allow_pickle=False
    )
    arrays = {}

    def check(prefix, result):
        for name in ("parameters", "status", "iterations"):
            key = f"{prefix}-{name}"
            array = getattr(result, name)
            np.testing.assert_array_equal(array, baseline[key], err_msg=key)
            arrays[key] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                "exact_match": True,
            }

    for model in ("translation", "affine", "quadratic"):
        for levels in (1, 2, 3):
            reference, target = _fractional_pair(128, 128, (0.3, -0.2))
            with ICGNWGPUFullPyramid(
                motion_model=model, pyramid_levels=levels
            ) as engine:
                engine.set_reference(reference)
                check(f"{model}-{levels}", engine.register(target))
    with Image.open(ROOT / "assets/uhd_tensile/reference.png") as image:
        reference = np.array(image)
    with Image.open(ROOT / "assets/uhd_tensile/frame_040.png") as image:
        target = np.array(image)
    with ICGNWGPUFullPyramid() as engine:
        engine.set_reference(reference)
        check("uhd", engine.register(target))
        adapter = engine.adapter_info
    report = {
        "before_commit": "67e264f0f028c7cd60382270f55a3562b3d68d9c",
        "kernel_host_commit": "0e9d088",
        "adapter": adapter,
        "arrays": arrays,
    }
    Path(__file__).with_name("default-parity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"{len(arrays)} arrays match exactly, including archived native UHD frame 40")


if __name__ == "__main__":
    main()
