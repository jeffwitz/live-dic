"""Recompute both presentation matrices and verify their recorded quality gates."""

import gzip
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    index = int(position)
    return values[index] + (values[min(index + 1, len(values) - 1)] - values[index]) * (position - index)


manifest = json.loads((ROOT / "raw-manifest.json").read_text())
for name, description in manifest.items():
    payload = gzip.decompress((ROOT / "raw" / name).read_bytes())
    assert hashlib.sha256(payload).hexdigest() == description["sha256_uncompressed"]
    assert len(payload) == description["bytes_uncompressed"]

rows = []
for path in sorted((ROOT / "raw").glob("*.json.gz")):
    if not any(path.name.endswith(suffix) for suffix in ["-baseline.json.gz", "-candidate.json.gz"]):
        continue
    report = json.loads(gzip.decompress(path.read_bytes()))
    for case in report["cases"]:
        assert case["final_state"]["storage"]["unsavedFields"] == 0
        assert case["final_state"]["pipeline"]["recording"]["pending"] == 0
        assert case["pipeline"]["recording"]["peakPending"] <= 2
        assert all(sample["referencePreparationMs"] == 0 for sample in case["samples"])
        assert all(sample["acceptance"] >= .95 for sample in case["samples"])
        if case["recording"]:
            quality = case["quality"]
            assert quality["finite"] and quality["parameterType"] == "Float32Array"
            assert [quality["width"], quality["height"]] == [3840, 2160]
            assert quality["grid"] == [240, 135]
            assert quality["accepted"] / quality["total"] >= .95
            assert quality["displacementRmse"] < .005
        else:
            quality = None  # Do not invent a readback in the recording-off window.
        ages = [s["workerToAdmissionMs"] for s in case["admission_samples"] if s.get("workerToAdmissionMs") is not None]
        rows.append(dict(run=path.name.removesuffix(".json.gz"),
                         matrix="overview" if path.name.startswith("overview-") else "initial",
                         candidate="candidate" in path.name, recording=case["recording"],
                         commit=report["build"]["commit"], computed_fps=case["completed_fps"],
                         committed_fps=case["committed_fps"], display_fps=case["presentation"]["draw_fps"],
                         admission_ms={str(p):percentile(ages, p/100) for p in [50,95,99]}, quality=quality))

decisions = []
for matrix in sorted({r["matrix"] for r in rows}):
    for recording in [False, True]:
        group = [r for r in rows if r["matrix"] == matrix and r["recording"] == recording]
        controls = [r for r in group if not r["candidate"]]
        candidates = [r for r in group if r["candidate"]]
        for metric in ["computed_fps"] + (["committed_fps"] if recording else []):
            baseline = [r[metric] for r in controls]
            spread = max(baseline)-min(baseline)
            failures = sum(r[metric] < median(baseline)-spread for r in candidates)
            decisions.append(dict(matrix=matrix, recording=recording, metric=metric,
                                  reference_median=median(baseline), reference_spread=spread,
                                  candidates=[r[metric] for r in candidates],
                                  result="both candidates outside reference spread" if failures == 2 else "inconclusive" if failures else "within observed reference spread"))

(ROOT / "summary.json").write_text(json.dumps(dict(rows=rows, decisions=decisions), indent=2, allow_nan=False)+"\n")
