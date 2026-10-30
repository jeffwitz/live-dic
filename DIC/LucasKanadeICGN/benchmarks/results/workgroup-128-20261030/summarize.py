"""Recompute the matched native-UHD width comparison from archived reports."""

import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parent
    baseline = json.loads((root / "runs/w16-d1-r0.json").read_text())
    groups = []
    for width in (16, 128):
        for depth in (1, 2):
            runs = [
                json.loads((root / f"runs/w{width}-d{depth}-r{i}.json").read_text())
                for i in range(3)
            ]
            for run in runs:
                for key in (
                    "git_commit",
                    "source_hashes",
                    "target_pixel_sha256",
                    "adapter",
                    "versions",
                ):
                    assert run[key] == baseline[key], key
                assert run["profile_before"] == run["profile_after"] == "performance"
                assert run["frames"] == 180
                for key, percentile in (
                    ("age_median_ms", 50),
                    ("age_p95_ms", 95),
                    ("age_p99_ms", 99),
                ):
                    np.testing.assert_allclose(
                        run[key],
                        np.percentile(run["age_samples_ms"], percentile),
                        rtol=0,
                        atol=1e-12,
                    )
                assert run["options"] == {**baseline["options"], "solver_workgroup_size": width}
                assert run["effective_solver_config"] == {
                    **baseline["effective_solver_config"],
                    "solverWorkgroupSize": width,
                }
                assert all(q["levels_used"] == 2 for q in run["quality"])
                assert run["quality"] == runs[0]["quality"]
            row = dict(width=width, pipeline_depth=depth)
            for key in (
                "throughput_fps",
                "age_median_ms",
                "age_p95_ms",
                "age_p99_ms",
                "preparation_ms",
            ):
                values = [run[key] for run in runs]
                row[key] = dict(values=values, median=float(np.median(values)))
            ages = [age for run in runs for age in run["age_samples_ms"]]
            row["pooled_age_ms"] = {f"p{p}": float(np.percentile(ages, p)) for p in (50, 95, 99)}
            quality = runs[0]["quality"]
            row["worst_fixed_quality"] = {
                key: float((min if "coverage" in key else max)(q[key] for q in quality))
                for key in (
                    "fixed_coverage",
                    "fixed_displacement_rmse_px",
                    "fixed_displacement_p95_px",
                    "fixed_strain_rmse",
                )
            }
            row["whole_grid"] = dict(
                minimum_coverage=min(q["own_coverage"] for q in quality),
                maximum_displacement_rmse_px=max(q["own_displacement_rmse_px"] for q in quality),
                maximum_strain_rmse=max(q["own_strain_rmse"] for q in quality),
            )
            row["last_frame"] = quality[-1]
            row["quality_pass"] = all(
                q["fixed_coverage"] >= 0.9
                and q["fixed_displacement_rmse_px"] < 0.05
                and q["fixed_displacement_p95_px"] < 0.1
                and q["fixed_strain_rmse"] < 0.002
                for q in quality
            )
            row["latency_pass"] = (
                all(run["age_p95_ms"] < 16.67 and run["throughput_fps"] >= 60 for run in runs)
                and row["pooled_age_ms"]["p95"] < 16.67
            )
            diagnostics = json.loads((root / f"runs/w{width}-d1-r0.json").read_text())[
                "untimed_diagnostics"
            ][20:]
            row["untimed_iterations"] = {}
            for level in ("fine", "coarse"):
                values = [q[level] for q in diagnostics]
                row["untimed_iterations"][level] = dict(
                    mean_all_per_frame=float(np.mean([q["mean_all"] for q in values])),
                    would_converge_late_events=sum(q.get("would_converge_late", 0) for q in values),
                    would_converge_late_fixed_events=sum(
                        q.get("would_converge_late_fixed", 0) for q in values
                    ),
                )
            groups.append(row)
    comparisons = []
    for depth in (1, 2):
        for repeat in range(3):
            with (
                np.load(root / f"runs/w16-d{depth}-r{repeat}.npz", allow_pickle=False) as a,
                np.load(root / f"runs/w128-d{depth}-r{repeat}.npz", allow_pickle=False) as b,
            ):
                for width, field in ((16, a), (128, b)):
                    assert json.loads(str(field["config_json"]))["solverWorkgroupSize"] == width
                    for key in ("parameters", "status", "iterations"):
                        assert np.isfinite(field[key]).all()
                y, x = np.indices(a["status"].shape) * 16 + 7.5
                fixed = (x >= 224) & (x < 3616) & (y >= 64) & (y < 2096)
                row = dict(depth=depth, repeat=repeat)
                for name, population in (("fixed", fixed), ("whole", np.ones_like(fixed))):
                    common = population & (a["status"] > 0) & (b["status"] > 0)
                    delta = b["parameters"][common].astype(np.float64) - a["parameters"][common]
                    row[name] = dict(
                        common_accepted=int(common.sum()),
                        status_differences=int(
                            np.count_nonzero(population & (a["status"] != b["status"]))
                        ),
                        parameter_component_rmse=np.sqrt(np.mean(delta**2, axis=0)).tolist(),
                        parameter_component_max_abs=np.max(np.abs(delta), axis=0).tolist(),
                        displacement_vector_rmse_px=float(
                            np.sqrt(np.mean(np.sum(delta[:, (0, 3)] ** 2, axis=-1)))
                        ),
                    )
                comparisons.append(row)
    report = dict(
        groups=groups,
        final_field_comparisons=comparisons,
        source_commit=baseline["git_commit"],
        source_hashes=baseline["source_hashes"],
        scope="Three matched finite runs per cell; no universal deadline or hardware claim.",
        quality_scope="Fixed interior of 26,924 points; whole-grid failures retained separately.",
    )
    (root / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in groups:
        print(
            row["width"],
            row["pipeline_depth"],
            row["throughput_fps"]["median"],
            row["pooled_age_ms"],
            "quality",
            row["quality_pass"],
            "latency",
            row["latency_pass"],
        )
    print("Final fields:", comparisons[0])


if __name__ == "__main__":
    main()
