"""Capture or compare exact native-UHD fields across a resource-layout change."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from lucas_kanade_icgn import ICGNWGPUFullPyramid

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    reference = np.array(Image.open(ROOT / "assets/uhd_tensile/reference.png"))
    targets = [
        np.array(Image.open(ROOT / f"assets/uhd_tensile/{name}"))
        for name in ("reference.png", "frame_040.png", "frame_100.png")
    ]
    mask = np.zeros(reference.shape, dtype=bool)
    mask[160:2000, 160:3680] = True
    mask[880:1200, 1800:2040] = False
    fields = {}
    cases = []
    expected = np.load(args.baseline, allow_pickle=False) if args.baseline else None
    try:
        for model in ("translation", "affine", "quadratic"):
            for levels in (1, 2, 3):
                for roi in ("none",) if model == "quadratic" else ("none", "full", "partial"):
                    for profile in (
                        ("zero", "temporal", "latency") if levels == 2 else ("zero", "temporal")
                    ):
                        name = f"{model}-l{levels}-{roi}-{profile}"
                        options = dict(
                            motion_model=model,
                            pyramid_levels=levels,
                            device_type="gpu",
                            outlier_policy="median",
                        )
                        if profile != "zero":
                            options["initialization"] = "temporal"
                        if profile == "latency":
                            options.update(
                                temporal_prediction="velocity",
                                temporal_fine_correction=True,
                                divergence_patience=3,
                                divergence_min_iterations=4,
                            )
                        with ICGNWGPUFullPyramid(**options) as engine:
                            engine.set_reference(
                                reference,
                                roi_mask=None if roi == "none" else mask,
                                roi_policy="full" if roi == "none" else roi,
                            )
                            images = (
                                targets
                                if model != "translation"
                                else [np.roll(reference, (-i, 2 * i), (0, 1)) for i in range(3)]
                            )
                            for frame, field in enumerate(
                                engine.register_stream(images, pipeline_depth=1)
                            ):
                                for key in ("parameters", "status", "iterations"):
                                    array = getattr(field, key)
                                    label = f"{name}-f{frame}-{key}"
                                    assert np.isfinite(array).all(), label
                                    if expected is not None:
                                        np.testing.assert_array_equal(
                                            array, expected[label], err_msg=label
                                        )
                                    else:
                                        fields[label] = array.copy()
                            cases.append(
                                dict(
                                    name=name,
                                    config=field.config,
                                    accepted=field.accepted,
                                    adapter=engine.adapter_info,
                                )
                            )
                        print(name, "exact" if expected is not None else "captured", flush=True)
    finally:
        if expected is not None:
            expected.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.baseline is None:
        np.savez_compressed(args.output.with_suffix(".npz"), **fields)
    report = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        image_sha256=[hashlib.sha256(image).hexdigest() for image in (reference, *targets)],
        mask_sha256=hashlib.sha256(mask).hexdigest(),
        cases=cases,
        arrays=len(cases) * 9,
        bit_identical=args.baseline is not None,
        scope="UHD resource-layout parity, including rejected states; not a capture-range claim for late frames.",
        unsupported="Quadratic ROI is rejected by contract and is not treated as supported.",
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
