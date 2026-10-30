"""Compare partial ROI with the full-only checkpoint on the same hardware.

Run with the project interpreter and PYTHONPATH pointing to the candidate:
    python run_campaigns.py /tmp/partial-masks /path/to/full-checkpoint/python/src

Masks are deterministic and generated locally. Reports contain raw samples and
source/image/mask hashes. Each baseline is measured before and after the absent
mask candidate; temporal drift remains visible rather than being discarded.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    reports = Path(__file__).resolve().parent
    benchmark = reports.parents[1] / "benchmark_image_to_field.py"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("masks", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--model", choices=("translation", "affine"))
    args = parser.parse_args()
    masks, baseline = args.masks.resolve(), args.baseline.resolve()
    masks.mkdir(parents=True, exist_ok=True)
    y, x = np.indices((2160, 3840))
    boundary = ((x - 1920) / 1600) ** 2 + ((y - 1080) / 900) ** 2 <= 1
    boundary[400:1205, 1930:1932] = False
    np.save(masks / "boundary.npy", boundary, allow_pickle=False)
    np.save(masks / "all-true.npy", np.ones(boundary.shape, bool), allow_pickle=False)
    jobs = (
        ("before", True, None, "full", "none"),
        ("absent", False, None, "full", "none"),
        ("before-recheck", True, None, "full", "none"),
        ("all-true", False, "all-true", "partial", "none"),
        ("full", False, "boundary", "full", "none"),
        ("partial", False, "boundary", "partial", "none"),
        ("visibility", False, "boundary", "partial", "visibility"),
    )
    for model in (args.model,) if args.model else ("translation", "affine"):
        for name, old, mask, policy, truncation in jobs:
            environment = os.environ.copy()
            if old:
                environment["PYTHONPATH"] = str(baseline)
            command = [
                sys.executable,
                str(benchmark),
                "--backend",
                "native-pyramid",
                "--device-type",
                "gpu",
                "--motion-model",
                model,
                "--outlier-policy",
                "median",
                "--pipeline-depth",
                "2",
                "--frames",
                "160",
                "--warmups",
                "12",
                "--campaigns",
                "3",
                "--fixture-set",
                "uhd-translation" if model == "translation" else "uhd-tensile",
            ]
            if mask:
                command.extend(
                    [
                        "--roi-mask",
                        str(masks / f"{mask}.npy"),
                        "--roi-policy",
                        policy,
                        "--roi-truncation",
                        truncation,
                    ]
                )
            path = reports / f"{model}-{name}.json"
            with path.open("w", encoding="utf-8") as stream:
                subprocess.run(command, env=environment, stdout=stream, check=True)
            if name == "before":
                campaigns = json.loads(path.read_text())["measurements"][
                    "native-pyramid"
                ]["raw_campaigns"]
                margin = {
                    key: max(c[key] for c in campaigns) / min(c[key] for c in campaigns)
                    - 1
                    for key in ("stream_period_ms", "latency_p95_ms", "latency_p99_ms")
                }
                (reports / f"{model}-predeclared-margins.json").write_text(
                    json.dumps(margin, indent=2, allow_nan=False), encoding="utf-8"
                )
            print(path.name, flush=True)


if __name__ == "__main__":
    main()
