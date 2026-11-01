"""Recompute this frozen campaign's compact tables from lossless raw reports."""

import gzip
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
rows = []
for path in sorted((ROOT / "raw").glob("*.json.gz")):
    report = json.loads(gzip.decompress(path.read_bytes()))
    if not isinstance(report, dict):
        continue
    if "measurements" in report:
        for backend, values in report["measurements"].items():
            quality = values["interior_quality"]["populations"]["complete"]
            assert quality["accepted_fraction_min"] >= .90
            assert quality["displacement_rmse_max_px"] < .05
            assert quality["displacement_p95_max_px"] < .10
            assert quality["gradient_component_rmse_max"] < .002
            rows.append(dict(run=path.name.removesuffix(".json.gz"), backend=backend,
                             median_ms=values["latency_median_ms"], p95_ms=values["latency_p95_ms"],
                             p99_ms=values["latency_p99_ms"], fps=values["throughput_fps"], quality=quality))
    elif "cases" in report:
        for case in report["cases"]:
            assert case["final_state"]["storage"]["unsavedFields"] == 0
            rows.append(dict(run=path.name.removesuffix(".json.gz"), backend="pwa",
                             recording=case["recording"], computed_fps=case["completed_fps"],
                             committed_fps=case["committed_fps"], display_fps=case.get("presentation", {}).get("draw_fps")))

decisions = {}
for backend in ["native-pyramid", "webgpu-wgsl"]:
    primary = [r for r in rows if r["run"].startswith(f"python-{backend}-")]
    metrics = {}
    for key in ["median_ms", "p95_ms", "p99_ms", "fps"]:
        reference = [r[key] for r in primary if r["run"].endswith("baseline")]
        centre, spread = median(reference), max(reference) - min(reference)
        candidates = [r[key] for r in primary if r["run"].endswith("candidate")]
        worse = sum(value < centre-spread if key == "fps" else value > centre+spread for value in candidates)
        metrics[key] = dict(reference_median=centre, reference_range=spread, candidates=candidates,
                            decision="reproducible regression" if worse == 2 else "inconclusive" if worse else "within measured reference range")
    decisions[backend] = metrics

(ROOT / "summary.json").write_text(json.dumps(dict(rows=rows, primary_decisions=decisions), indent=2, allow_nan=False)+"\n")
