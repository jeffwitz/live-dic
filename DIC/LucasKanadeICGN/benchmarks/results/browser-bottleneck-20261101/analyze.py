"""Verify raw hashes/quality and recompute the main wall-time attribution."""

import collections
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads(gzip.decompress((HERE / (name + ".gz")).read_bytes()))


def main():
    manifest = json.loads((HERE / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert (
            hashlib.sha256(gzip.decompress((HERE / name).read_bytes())).hexdigest()
            == expected["sha256_uncompressed"]
        )
    core = load("core-cause.json")
    assert all(row["gpuMs"] == 0 for row in core["browser"]["gpu"]["passes"])
    print(
        "GPU timestamp capture invalid: all durations zero; excluded from attribution"
    )
    for case in core["native"]:
        print(
            "Native temporal",
            case["temporal"],
            "wall p50/p95",
            np.percentile(case["wallMs"], [50, 95]),
            "sustained FPS",
            1000 / np.mean(case["wallMs"]),
        )
    for case in core["browser"]["cases"]:
        rows = [row for row in case["samples"] if not row["profile"]]
        assert len(rows) == 30
        assert all(
            row["finite"] and row["accepted"] == 31654 and row["rmse"] < 0.005
            for row in rows
        )
        print(
            case["name"],
            {
                key: float(np.median([row[key] for row in rows]))
                for key in ("callMs", "exportMs", "totalMs")
            },
        )
    reference = load("reference-steps.json")["browser"]
    assert reference["quality"]["finite"] and reference["quality"]["rmse"] < 0.005
    for kind in ("bytes", "external"):
        rows = reference[kind]["samples"]
        print(
            "Reference",
            kind,
            "whole/fence/copy completion",
            {
                key: float(np.median([row[key] for row in rows]))
                for key in ("wholeMs", "emptyFenceMs", "copyCompletionMs")
            },
        )
        for index, stage in enumerate(rows[0]["steps"]):
            print(
                stage["label"],
                float(np.median([row["steps"][index]["wallMs"] for row in rows])),
            )
    for case in load("camera-stages.json")["cases"]:
        assert case["presentation"]["draws"] == case["presentation"]["bitmaps"] == 0
        assert (
            case["stop"]["storage"]["unsavedFields"] == 0 and not case["stop"]["errors"]
        )
        if case["recording"]:
            assert (
                case["quality"]["finite"]
                and case["quality"]["displacementRmse"] < 0.005
            )
        stages = collections.defaultdict(list)
        for row in case["worker_stages"] + case["main_stages"]:
            if "wallMs" in row:
                stages[row["stage"]].append(row["wallMs"])
            if "recordingReadbackMs" in row:
                stages["recorded-field readback"].append(row["recordingReadbackMs"])
        print(
            "Camera",
            case["mode"],
            "recording",
            case["recording"],
            "computed/committed FPS",
            case["completed_fps"],
            case["committed_fps"],
        )
        print({name: float(np.median(values)) for name, values in stages.items()})


if __name__ == "__main__":
    main()
