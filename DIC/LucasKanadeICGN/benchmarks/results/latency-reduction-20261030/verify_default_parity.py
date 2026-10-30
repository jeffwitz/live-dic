"""Compare literal defaults against the pre-change native-UHD/three-model oracle."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lucas_kanade_icgn import ICGNWGPUFullPyramid

ROOT = Path(__file__).resolve().parents[3]


def main():
    sys.path.insert(0, str(ROOT / "python/tests"))
    from test_gpu_regressions import _fractional_pair

    arrays = {}
    for model in ("translation", "affine", "quadratic"):
        for levels in (1, 2, 3):
            reference, target = _fractional_pair(128, 128, (0.3, -0.2))
            with ICGNWGPUFullPyramid(motion_model=model, pyramid_levels=levels) as engine:
                engine.set_reference(reference)
                result = engine.register(target)
            for name in ("parameters", "status", "iterations"):
                arrays[f"{model}_{levels}_{name}"] = getattr(result, name)
    reference = np.array(Image.open(ROOT / "assets/uhd_tensile/reference.png"))
    target = np.array(Image.open(ROOT / "assets/uhd_tensile/frame_040.png"))
    with ICGNWGPUFullPyramid(outlier_policy="median", device_type="gpu") as engine:
        engine.set_reference(reference)
        result = engine.register(target)
        for name in ("parameters", "status", "iterations"):
            arrays[f"uhd_{name}"] = getattr(result, name)
    with np.load(Path(__file__).with_name("baseline-default.npz"), allow_pickle=False) as baseline:
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, baseline[key], err_msg=key)
    print(json.dumps(dict(arrays_identical=len(arrays), includes_native_uhd_frame40=True)))


if __name__ == "__main__":
    main()
