"""Register files against one persistent reference and export complete NumPy fields.

Run from any directory after installing live-dic. File paths are explicit;
the scientific fixtures are available in the source checkout, not the wheel.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from live_dic import ICGNWGPUFullPyramid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("targets", type=Path, nargs="+", help="Processed in the supplied order")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument(
        "--roi", type=Path, help="Reference-sized boolean .npy, True retains pixels"
    )
    parser.add_argument("--device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--model", choices=("translation", "affine", "quadratic"), default="affine")
    parser.add_argument("--subset-size", type=int, default=16)
    parser.add_argument("--subset-stride", type=int, default=None)
    parser.add_argument("--solver-workgroup-size", type=int, choices=(16, 32, 128), default=16)
    parser.add_argument(
        "--outlier-policy", choices=("replace", "flag", "disabled", "median"), default="replace"
    )
    parser.add_argument("--median-size", type=int, choices=(3, 5, 7, 9), default=3)
    parser.add_argument("--initialization", choices=("zero", "temporal", "fourier"), default="zero")
    parser.add_argument("--pipeline-depth", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    mask = None if args.roi is None else np.load(args.roi, allow_pickle=False)

    with ICGNWGPUFullPyramid(
        device_type=args.device,
        motion_model=args.model,
        subset_size=args.subset_size,
        subset_stride=args.subset_stride,
        solver_workgroup_size=args.solver_workgroup_size,
        interpolation="bspline",
        outlier_policy=args.outlier_policy,
        median_filter_size=args.median_size,
        initialization=args.initialization,
    ) as engine:
        preparation = engine.set_reference(args.reference, roi_mask=mask, roi_policy="full")
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "reference": str(args.reference.resolve()),
            "targets": [str(path.resolve()) for path in args.targets],
            "adapter": engine.adapter_info,
            "config": dict(preparation.config),
            "preparationWallMs": preparation.total_wall_ms,
            "pipelineDepth": args.pipeline_depth,
            "timingBoundary": "sequence includes file decoding, registration, NPZ compression and writes",
        }
        # Write provenance before processing: completed files survive a later failure.
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        start = perf_counter()
        with (args.output / "fields.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("target", "field", "selected", "accepted", "flagged", "result_age_ms"))
            results = engine.register_stream(args.targets, pipeline_depth=args.pipeline_depth)
            for index, result in enumerate(results, start=1):
                filename = f"field_{index:06d}.npz"
                result.save_npz(args.output / filename)
                writer.writerow(
                    (
                        str(args.targets[index - 1]),
                        filename,
                        result.populations["roiSelected"],
                        result.accepted,
                        result.flagged,
                        result.timing.get("resultAgeWallMs"),
                    )
                )
                stream.flush()
        print(
            f"Saved {len(args.targets)} fields in {perf_counter() - start:.3f} s to {args.output}"
        )


if __name__ == "__main__":
    main()
