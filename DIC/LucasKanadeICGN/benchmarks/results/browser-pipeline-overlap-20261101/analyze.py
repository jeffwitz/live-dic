"""Recheck archived ownership/quality gates and summarize both measured repeats."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        data = gzip.decompress((HERE / name).read_bytes())
        assert hashlib.sha256(data).hexdigest() == expected["uncompressed_sha256"], name

    summary = {}
    for resolution in ("1920x1080", "3840x2160"):
        variants = {}
        for variant in ("synchronous", "queue", "preview"):
            repeats = []
            for path in sorted(HERE.glob(f"{resolution}-*-{variant}.json.gz")):
                report = json.loads(gzip.decompress(path.read_bytes()))
                case = report["cases"][0]
                quality = case["quality"]
                assert quality["finite"] and quality["parameterType"] == "Float32Array"
                assert [quality["width"], quality["height"]] == list(
                    map(int, resolution.split("x"))
                )
                assert quality["accepted"] / quality["total"] >= 0.9
                assert quality["displacementRmse"] < 0.005
                assert case["adjacent_pairs"] and not case["stop"]["errors"]
                assert case["stop"]["storage"]["unsavedFields"] == 0
                queue = case["pipeline"]["recording"]
                assert queue["peakPending"] <= 2 and not queue["blocked"]
                admissions = case["admission_samples"]
                assert admissions
                if variant != "synchronous":
                    assert not any(row["retained"] for row in admissions)
                ages = [row["workerToAdmissionMs"] for row in admissions]
                repeats.append(
                    {
                        "file": path.name,
                        "committed_fps": case["committed_fps"],
                        "computed_fps": case["completed_fps"],
                        "admission_sample_count": len(ages),
                        "worker_to_admission_ms_p50_p95_p99": np.percentile(
                            ages, [50, 95, 99]
                        ).tolist(),
                        "accepted": quality["accepted"],
                        "total": quality["total"],
                        "identity_displacement_rmse_px": quality["displacementRmse"],
                        "peak_pending": queue["peakPending"],
                    }
                )
            assert len(repeats) == 2
            variants[variant] = {"repeats": repeats}
        baseline_max = max(
            row["committed_fps"] for row in variants["synchronous"]["repeats"]
        )
        for name in ("queue", "preview"):
            variants[name]["passes_predeclared_short_control"] = all(
                row["committed_fps"] > baseline_max + 2 / 20
                for row in variants[name]["repeats"]
            )
        summary[resolution] = variants
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
