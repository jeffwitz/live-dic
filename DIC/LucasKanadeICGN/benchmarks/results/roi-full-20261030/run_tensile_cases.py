"""Qualify the original UHD affine fixture using the same stream benchmark.

Arguments: mask directory, baseline Python source directory.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

reports = Path(__file__).resolve().parent
benchmark = reports.parents[1] / "benchmark_image_to_field.py"
for case in ("baseline", "none", "full", "large", "small", "fragmented"):
    environment = os.environ.copy()
    if case == "baseline":
        environment["PYTHONPATH"] = sys.argv[2]
    command = [sys.executable, str(benchmark), "--backend", "native-pyramid",
               "--device-type", "gpu", "--fixture-set", "uhd-tensile",
               "--motion-model", "affine", "--outlier-policy", "median",
               "--frames", "160", "--warmups", "12", "--campaigns", "3",
               "--pipeline-depth", "2"]
    if case not in ("baseline", "none"):
        command.extend(["--roi-mask", str(Path(sys.argv[1]) / f"{case}.npy")])
    path = reports / f"tensile-{case}.json"
    with path.open("w") as stream:
        subprocess.run(command, stdout=stream, check=True, env=environment)
    if case == "baseline":
        samples = json.loads(path.read_text())["measurements"]["native-pyramid"]["raw_campaigns"]
        margins = {key: max(c[key] for c in samples) / min(c[key] for c in samples) - 1
                   for key in ("stream_period_ms", "latency_p95_ms", "latency_p99_ms")}
        (reports / "tensile-predeclared-margins.json").write_text(json.dumps(margins, indent=2))
    print(path.name, flush=True)
