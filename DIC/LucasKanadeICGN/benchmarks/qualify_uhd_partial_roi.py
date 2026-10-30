"""Qualify partial ROI only on native 3840x2160 inputs, including subpixel motion.

Run from a checkout with the installed package or its Python source on PYTHONPATH.
Timing is the public NumPy-image to NumPy-field stream, with changing frames.
Small geometry/oracle unit tests are not substituted for these UHD results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import lucas_kanade_icgn
from benchmark_image_to_field import _aggregate_campaigns, _measure
from lucas_kanade_icgn import ICGNWGPU, ICGNWGPUFullPyramid

# Reuse exactly the same offline full-resolution fixtures as the regressions.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python/tests"))
from uhd_partial_cases import TRANSLATION, boundary_mask, translation_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene", choices=("analytic-wave", "archived-speckle"), required=True
    )
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--campaigns", type=int, default=3)
    args = parser.parse_args()
    if args.frames < 2 or args.campaigns < 1:
        parser.error("frames must be >=2 and campaigns >=1")
    reference, target = translation_pair(args.scene)
    mask = boundary_mask()

    def digest(image):
        return hashlib.sha256(image).hexdigest()

    targets = [
        (reference, (0.0, 0.0), digest(reference)),
        (target, TRANSLATION, digest(target)),
    ]
    profiles = []
    configurations = (
        [(size, "translation", "cubic", "disabled", False) for size in (8, 16)]
        + [
            (size, "translation", "bspline", policy, True)
            for policy in ("disabled", "median")
            for size in (8, 16)
        ]
        + [(16, "affine", "bspline", "median", True)]
    )
    for size, model, interpolation, policy, pyramid in configurations:
        factory = ICGNWGPUFullPyramid if pyramid else ICGNWGPU
        with factory(
            motion_model=model,
            subset_size=size,
            interpolation=interpolation,
            outlier_policy=policy,
            device_type="gpu",
        ) as engine:
            info = engine.set_reference(
                reference,
                roi_mask=mask,
                roi_policy="partial",
                **({} if pyramid else {"tolerance": 0.005}),
            )
            campaigns = [
                _measure(
                    engine,
                    targets,
                    frames=args.frames,
                    warmups=12,
                    pipeline_depth=2,
                    use_stream=True,
                )
                for _ in range(args.campaigns)
            ]
            profiles.append(
                {
                    "engine": factory.__name__,
                    "subset_size": size,
                    "motion_model": model,
                    "interpolation": interpolation,
                    "outlier_policy": policy,
                    "solver": engine.solver if not pyramid else engine._root.solver,
                    "adapter": dict(engine.adapter_info),
                    "preparation_ms": info.total_wall_ms,
                    "measurement": _aggregate_campaigns(campaigns),
                }
            )
        print(
            f"{args.scene}: {size}/{model}/{interpolation}/{policy}/{factory.__name__}",
            flush=True,
        )
    source = Path(lucas_kanade_icgn.__file__).parent
    helpers = Path(__file__).resolve().parents[1] / "python/tests"
    report = {
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
        "command": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "wgpu_version": importlib.metadata.version("wgpu"),
        "resolution": [3840, 2160],
        "scene": args.scene,
        "translation": TRANSLATION,
        "image_formation": "continuous analytic waves sampled independently"
        if args.scene == "analytic-wave"
        else "original archived UHD reference; NumPy bilinear inverse translation, uint8 rounding, reflected edges",
        "reference_sha256": digest(reference),
        "target_sha256": digest(target),
        "mask_sha256": info.roi.mask_hash,
        "source_hashes": {
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted([*source.glob("*.py"), *source.glob("shaders/*.wgsl")])
        },
        "qualification_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "fixture_helper_hashes": {
            name: hashlib.sha256((helpers / name).read_bytes()).hexdigest()
            for name in ("uhd_partial_cases.py", "test_gpu_regressions.py")
        },
        "profiles": profiles,
        "scope": "Resident NumPy input to full NumPy output; acquisition, decoding, setup and post-timing quality evaluation excluded. No universal deadline or non-regression claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
