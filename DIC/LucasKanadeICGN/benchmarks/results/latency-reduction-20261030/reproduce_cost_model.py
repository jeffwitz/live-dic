"""Reproduce the indicative static UHD cost table, not accuracy qualification."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from lucas_kanade_icgn import ICGNWGPUFullPyramid

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference = np.array(Image.open(ROOT / "assets/uhd_tensile/reference.png"))
    target = np.array(Image.open(ROOT / "assets/uhd_tensile/frame_040.png"))
    rows = []
    for interpolation, levels, caps in (
        ("bspline", 2, None),
        ("bspline", 2, (1, 1)),
        ("bspline", 1, (1,)),
        ("cubic", 1, (1,)),
    ):
        with ICGNWGPUFullPyramid(
            device_type="gpu",
            interpolation=interpolation,
            pyramid_levels=levels,
            iterations_per_level=caps,
        ) as engine:
            engine.set_reference(reference)
            list(engine.register_stream([target] * 8, pipeline_depth=1))
            results = []
            start = time.perf_counter()
            for result in engine.register_stream([target] * 60, pipeline_depth=1):
                results.append(result)
            elapsed = time.perf_counter() - start
            ages = [r.timing["resultAgeWallMs"] for r in results]
            rows.append(
                dict(
                    interpolation=interpolation,
                    levels=levels,
                    iterations=engine.iterations_per_level,
                    adapter=engine.adapter_info,
                    fps=60 / elapsed,
                    median_ms=float(np.median(ages)),
                    p95_ms=float(np.percentile(ages, 95)),
                    timing_samples=[r.timing for r in results],
                )
            )
    report = dict(
        commit=subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        boundary="resident NumPy target to materialized NumPy field and statistics",
        samples=60,
        warmups=8,
        image_sha256=hashlib.sha256(target).hexdigest(),
        measurements=rows,
        note="Single-iteration variants are cost probes, not valid accuracy alternatives.",
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
