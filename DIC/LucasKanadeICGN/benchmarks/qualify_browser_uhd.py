"""Native-UHD browser/native full-ROI parity on archived tensile and plate fixtures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from lucas_kanade_icgn import ICGNWebGL, ICGNWGPUFullPyramid

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    criteria = dict(
        parameter_rmse_max=0.002,
        parameter_max_abs_max=0.02,
        status_mismatch_fraction_max=0.005,
        tensile=dict(displacement_rmse_max=0.05, displacement_p95_max=0.10, coverage_min=0.90),
        scope="UHD tensile frame 40 and plate-hole frame 95; same zero-initialized two-level affine/median configuration; geometric exclusions exact",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name(args.output.stem + "-criteria.json").write_text(
        json.dumps(criteria, indent=2) + "\n"
    )
    y, x = np.ogrid[:2160, :3840]
    plate = (x - 1920) ** 2 + (y - 1080) ** 2 >= 112**2
    tensile = np.zeros((2160, 3840), bool)
    tensile[160:2000, 160:3680] = True
    tensile[880:1200, 1800:2040] = False
    rows = []
    with ICGNWebGL(backend="webgpu-wgsl", outlier_policy="median") as browser:
        for fixture, target_name, mask in [
            ("uhd_tensile", "frame_040.png", tensile),
            ("uhd_plate_hole", "frame_095.png", plate),
        ]:
            root = ROOT / "assets" / fixture
            reference = np.array(Image.open(root / "reference.png"))
            target = np.array(Image.open(root / target_name))
            with ICGNWGPUFullPyramid(outlier_policy="median", device_type="gpu") as native:
                native.set_reference(reference, roi_mask=mask)
                expected = native.register(target)
                adapter = native.adapter_info
            info = browser.set_reference(reference, roi_mask=mask)
            actual = browser.register(target)
            selected = info.roi_selected
            assert np.all(actual.status[~selected] == -3)
            assert np.isfinite(actual.parameters).all()
            common = actual.accepted_mask & expected.accepted_mask
            delta = actual.parameters[common] - expected.parameters[common]
            row = dict(
                fixture=fixture,
                reference_sha256=hashlib.sha256(reference).hexdigest(),
                target_sha256=hashlib.sha256(target).hexdigest(),
                mask_sha256=info.roi.mask_hash,
                native_adapter=adapter,
                browser_device=info.device,
                config=actual.config,
                selected=int(selected.sum()),
                accepted_native=expected.accepted,
                accepted_browser=actual.accepted,
                status_mismatches=int(np.count_nonzero(actual.status != expected.status)),
                iteration_mismatches=int(
                    np.count_nonzero(actual.iterations != expected.iterations)
                ),
                parameter_rmse=float(np.sqrt(np.mean(delta**2))),
                parameter_max_abs=float(np.max(np.abs(delta))),
            )
            row["passes"] = (
                row["parameter_rmse"] <= criteria["parameter_rmse_max"]
                and row["parameter_max_abs"] <= criteria["parameter_max_abs_max"]
                and row["status_mismatches"] / selected.size
                <= criteria["status_mismatch_fraction_max"]
            )
            if fixture == "uhd_tensile":
                with (root / "machine_truth.csv").open() as stream:
                    truth = list(csv.DictReader(stream))[40]
                ex, ey = float(truth["stretch_x"]) - 1, float(truth["stretch_y"]) - 1
                gy, gx = np.indices(selected.shape)
                cx = gx * 16 + 7.5
                cy = gy * 16 + 7.5
                population = selected & (cx >= 224) & (cx < 3616) & (cy >= 64) & (cy < 2096)
                measured = population & actual.accepted_mask
                distance = np.linalg.norm(
                    actual.displacement[measured]
                    - np.stack((ex * (cx - 1920), ey * (cy - 1080)), -1)[measured],
                    axis=-1,
                )
                row["truth"] = dict(
                    coverage=float(measured.sum() / population.sum()),
                    rmse=float(np.sqrt(np.mean(distance**2))),
                    p95=float(np.percentile(distance, 95)),
                )
                row["passes"] &= (
                    row["truth"]["coverage"] >= 0.9
                    and row["truth"]["rmse"] < 0.05
                    and row["truth"]["p95"] < 0.1
                )
            rows.append(row)
            print(fixture, row, flush=True)
    report = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        criteria=criteria,
        rows=rows,
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not all(row["passes"] for row in rows):
        raise SystemExit("UHD parity criterion exceeded; full report retained")


if __name__ == "__main__":
    main()
