"""Record correctness at a crack tip after a full 3840x2160 dense solve.

This is not a timing campaign. The image-to-field benchmark is separate.
"""

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import lucas_kanade_icgn
import numpy as np
from lucas_kanade_icgn import ICGNWGPU

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / "python/tests"))
from uhd_partial_cases import TRANSLATION, crack_pair

reference, target, mask, point = crack_pair()
cases = []
for model in ("translation", "affine"):
    for truncation in ("none", "visibility"):
        with ICGNWGPU(
            motion_model=model,
            subset_size=17,
            interpolation="linear",
            prefer_subgroups=False,
            outlier_policy="disabled",
            device_type="gpu",
        ) as engine:
            info = engine.set_reference(
                reference,
                roi_mask=mask,
                roi_policy="partial",
                roi_truncation=truncation,
                tolerance=0.005,
            )
            result = engine.register(target)
            cases.append(
                {
                    "motion_model": model,
                    "truncation": truncation,
                    "adapter": dict(engine.adapter_info),
                    "config": result.config,
                    "field_shape": list(result.parameters.shape),
                    "populations": result.populations,
                    "tip_grid_index_yx": point,
                    "tip_center_xy": [point[1] * 17 + 8, point[0] * 17 + 8],
                    "expected_tip_displacement": TRANSLATION,
                    "tip_parameters": result.parameters[point].tolist(),
                    "tip_status": int(result.status[point]),
                    "tip_iterations": int(result.iterations[point]),
                    "tip_support_fraction": float(info.roi.coverage[point]),
                    "tip_displacement_error_px": float(
                        np.linalg.norm(result.displacement[point] - TRANSLATION)
                    ),
                    "finite_complete_field": bool(np.isfinite(result.parameters).all()),
                    "reference_sha256": hashlib.sha256(reference).hexdigest(),
                    "target_sha256": hashlib.sha256(target).hexdigest(),
                    "mask_sha256": info.roi.mask_hash,
                }
            )
report = {
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    ),
    "resolution": [3840, 2160],
    "command": sys.argv,
    "python": sys.version,
    "wgpu_version": importlib.metadata.version("wgpu"),
    "scope": "One controlled crack-tip window in a full UHD field; no physical fracture model, global metrology or timing qualification claimed.",
    "fixture_helper_sha256": hashlib.sha256(
        (root / "python/tests/uhd_partial_cases.py").read_bytes()
    ).hexdigest(),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "source_hashes": {
        str(path.relative_to(Path(lucas_kanade_icgn.__file__).parent)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(
            [
                *Path(lucas_kanade_icgn.__file__).parent.glob("*.py"),
                *Path(lucas_kanade_icgn.__file__).parent.glob("shaders/*.wgsl"),
            ]
        )
    },
    "cases": cases,
}
output = Path(__file__).with_name("uhd-crack.json")
output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
print(output)
