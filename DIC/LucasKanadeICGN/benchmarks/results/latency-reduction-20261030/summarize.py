"""Recompute the declared latency/quality gates from every archived run."""

import json
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).resolve().parent
    groups = []
    cases = [
        (collection, variant)
        for collection, variants in (
            ("runs", ("baseline", "velocity", "abort", "adaptive", "cubic")),
            ("interpolation-control", ("abort", "cubic-fixed")),
            ("followup-control", ("abort", "abort-fast", "cubic-only")),
            ("counter-control", ("old", "new")),
            ("finest-only-control", ("abort", "abort-fast")),
        )
        for variant in variants
    ]
    for collection, variant in cases:
        for depth in (1, 2):
            runs = [
                json.loads((root / collection / f"{variant}-d{depth}-r{i}.json").read_text())
                for i in range(3)
            ]
            assert all(r["source_hashes"] == runs[0]["source_hashes"] for r in runs)
            assert all(r["quality"] == runs[0]["quality"] for r in runs)
            quality = runs[0]["quality"]
            ages = [age for run in runs for age in run["age_samples_ms"]]
            row = dict(
                collection=collection,
                variant=variant,
                options=runs[0]["options"],
                source_hashes=runs[0]["source_hashes"],
                source_commit=runs[0]["git_commit"],
                requested_depth=depth,
                effective_depth=1 if variant in ("adaptive", "cubic") else depth,
            )
            for key in ("throughput_fps", "age_median_ms", "age_p95_ms", "age_p99_ms"):
                values = [r[key] for r in runs]
                row[key] = dict(
                    values=values,
                    median=float(np.median(values)),
                    minimum=min(values),
                    maximum=max(values),
                )
            row["pooled_age_ms"] = {f"p{p}": float(np.percentile(ages, p)) for p in (50, 95, 99)}
            row["worst_fixed_quality"] = {
                key: float((min if "coverage" in key else max)(r[key] for r in quality))
                for key in (
                    "fixed_coverage",
                    "fixed_displacement_rmse_px",
                    "fixed_displacement_p95_px",
                    "fixed_strain_rmse",
                )
            }
            row["last_frame"] = quality[-1]
            row["levels_used"] = [q["levels_used"] for q in quality]
            row["quality_pass"] = all(
                q["fixed_coverage"] >= 0.9
                and q["fixed_displacement_rmse_px"] < 0.05
                and q["fixed_displacement_p95_px"] < 0.1
                and q["fixed_strain_rmse"] < 0.002
                for q in quality
            )
            row["latency_pass"] = (
                all(r["age_p95_ms"] < 16.67 and r["throughput_fps"] >= 60 for r in runs)
                and row["pooled_age_ms"]["p95"] < 16.67
            )
            row["qualified_one_slot"] = depth == 1 and row["latency_pass"] and row["quality_pass"]
            diagnostic_path = root / collection / f"{variant}-d1-r0.json"
            diagnostic = json.loads(diagnostic_path.read_text())["untimed_diagnostics"][20:]
            row["untimed_iterations"] = {}
            for level in ("fine", "coarse"):
                values = [r[level] for r in diagnostic]
                row["untimed_iterations"][level] = dict(
                    mean_all_per_frame=float(np.mean([r["mean_all"] for r in values])),
                    executed_frames=sum(r["executed"] for r in values),
                    aborted_events=sum(r.get("aborted", 0) for r in values),
                    would_converge_late_events=sum(r.get("would_converge_late", 0) for r in values),
                    would_converge_late_fixed_events=sum(
                        r.get("would_converge_late_fixed", 0) for r in values
                    ),
                )
            groups.append(row)
    parity = []
    for depth in (1, 2):
        for repeat in range(3):
            base = root / "counter-control"
            old = json.loads((base / f"old-d{depth}-r{repeat}.json").read_text())
            new = json.loads((base / f"new-d{depth}-r{repeat}.json").read_text())
            assert old["quality"] == new["quality"]
            keys = (
                "temporallySeededCoarseSubsets",
                "temporalAcceptedUpdates",
                "velocityPredictedSubsets",
            )
            for a, b in zip(old["component_timings"], new["component_timings"], strict=True):
                assert all(a[k] == b[k] for k in keys)
            with (
                np.load(base / f"old-d{depth}-r{repeat}.npz", allow_pickle=False) as a,
                np.load(base / f"new-d{depth}-r{repeat}.npz", allow_pickle=False) as b,
            ):
                for key in ("parameters", "status", "iterations"):
                    np.testing.assert_array_equal(a[key], b[key])
                    assert np.isfinite(b[key]).all()
                    parity.append(dict(depth=depth, repeat=repeat, array=key))
    report = dict(
        criteria_file="criteria.json",
        groups=groups,
        counter_optimization_parity=dict(
            all_quality_rows_equal=True, all_counts_equal=True, final_arrays_equal=parity
        ),
        statistical_scope="Three finite runs, not a universal hard real-time bound.",
    )
    (root / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in groups:
        print(
            row["collection"],
            row["variant"],
            row["requested_depth"],
            f"{row['throughput_fps']['median']:.2f} FPS",
            f"p95 {row['age_p95_ms']['minimum']:.3f}..{row['age_p95_ms']['maximum']:.3f}",
            f"pooled {row['pooled_age_ms']['p95']:.3f}",
            f"quality={row['quality_pass']} latency={row['latency_pass']}",
        )


if __name__ == "__main__":
    main()
