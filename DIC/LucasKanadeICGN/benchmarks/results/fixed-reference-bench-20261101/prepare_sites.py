"""Add a clock metadata field to the old site; never alter numerical work."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path.cwd()
    sites = {
        name: Path(f"/tmp/live-dic-fixed-bench-{name}-site")
        for name in ("baseline", "candidate")
    }
    report = {"sites": {}}
    for name, site in sites.items():
        path = site / "ui/session.js"
        source = path.read_text()
        original = source
        if "workerArrivalEpochMs:" not in source:
            needle = "workerArrivalToRecordedFieldMs: performance.now()-available,"
            assert source.count(needle) == 1
            source = source.replace(
                needle,
                "workerArrivalEpochMs: performance.timeOrigin+available,\n                " + needle,
            )
            path.write_text(source)
        report["sites"][name] = {
            "build": json.loads((site / "build.json").read_text()),
            "worker_epoch_added": source != original,
            "session_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "clock_patch": "Only old Worker record metadata gains performance.timeOrigin+available; no GPU pass/fence added. Both sites use the same CLI storage probe.",
        }
    prefix = "DIC/LucasKanadeICGN/python/src"
    baseline = subprocess.check_output(["git", "rev-parse", f"388cc5e:{prefix}"], text=True).strip()
    candidate = subprocess.check_output(["git", "rev-parse", f"HEAD:{prefix}"], text=True).strip()
    assert baseline == candidate
    report["common_source_tree"] = {"baseline": baseline, "candidate": candidate, "identical": True}
    report["candidate_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    Path("/tmp/live-dic-fixed-bench-sites.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
