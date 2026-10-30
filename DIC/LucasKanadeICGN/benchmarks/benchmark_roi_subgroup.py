"""Image-to-field stream qualification of the optional native subgroup + ROI."""

import argparse
import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import lucas_kanade_icgn
import numpy as np
from benchmark_image_to_field import _fixtures, _measure
from lucas_kanade_icgn import ICGNWGPU


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roi-mask", type=Path)
    parser.add_argument(
        "--solver", choices=("portable", "subgroup"), default="subgroup"
    )
    parser.add_argument("--frames", type=int, default=160)
    args = parser.parse_args()
    assets = Path(__file__).resolve().parents[1] / "assets"
    reference, targets = _fixtures(assets, 3840, 2160)
    mask = None if args.roi_mask is None else np.load(args.roi_mask, allow_pickle=False)
    with ICGNWGPU(
        device_type="gpu",
        prefer_subgroups=args.solver == "subgroup",
        motion_model="affine",
        outlier_policy="median",
    ) as engine:
        info = (
            engine.set_reference(reference)
            if mask is None
            else engine.set_reference(reference, roi_mask=mask)
        )
        measurements = [
            _measure(
                engine,
                targets,
                frames=args.frames,
                warmups=12,
                pipeline_depth=2,
                use_stream=True,
            )
            for _ in range(3)
        ]
        source = Path(lucas_kanade_icgn.__file__).parent
        report = {
            "command": sys.argv,
            "python": sys.version,
            "wgpu_version": version("wgpu"),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=source, text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=no"],
                    cwd=source,
                    text=True,
                ).strip()
            ),
            "source_hashes": {
                str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted([*source.glob("*.py"), *source.glob("shaders/*.wgsl")])
            },
            "adapter": dict(engine.adapter_info),
            "solver": engine.solver,
            "solver_validation": engine.solver_validation,
            "preparation_ms": info.total_wall_ms,
            "reference_sha256": hashlib.sha256(reference).hexdigest(),
            "target_sha256": [digest for _, _, digest in targets],
            "measurements": measurements,
        }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
