"""Read-only Linux process/DRM/sysfs sampling beside a bounded benchmark.

No GPU API, root access or system setting changes. DRM counters are per client,
not physical free VRAM: https://docs.kernel.org/gpu/drm-usage-stats.html
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def parse_clients(records):
    """Deduplicate shared DRM file descriptors by device and client identity."""
    clients = {}
    for descriptor, text in records:
        fields = {}
        for line in text.splitlines():
            key, separator, value = line.partition(":")
            if separator and key.startswith("drm-"):
                fields[key] = value.strip()
        if "drm-driver" not in fields:
            continue
        identity = (fields.get("drm-pdev"), fields.get("drm-client-id", descriptor))
        clients[identity] = fields
    return list(clients.values())


def size_bytes(value):
    """DRM memory quantities use bytes or explicitly stated binary units."""
    parts = value.split()
    factors = {"B": 1, "KiB": 1024, "MiB": 1024**2}
    try:
        if len(parts) == 1:
            return int(parts[0])
        if len(parts) == 2 and parts[1] in factors:
            return int(parts[0]) * factors[parts[1]]
    except ValueError:
        pass
    return None


def memory_totals(clients):
    """Client sums may share backing storage; never call these physical usage."""
    totals = {}
    for prefix, label in (("drm-total-", "requestedBytes"), ("drm-resident-", "residentBytes")):
        values = [
            size_bytes(value)
            for client in clients
            for key, value in client.items()
            if key.startswith(prefix) and not key.startswith("drm-total-cycles-")
        ]
        valid = [value for value in values if value is not None]
        totals[label] = sum(valid) if valid else None
    return totals


def process_stat(text):
    """Names may contain spaces/parentheses; fields after the closing ')' are fixed."""
    words = text.rsplit(")", 1)[1].split()
    return {
        "state": words[0],
        "cpuTicks": int(words[11]) + int(words[12]),
        "startTicks": int(words[19]),
    }


def read_number(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def sensors():
    """Cold discovery. Missing facilities remain missing, without guessed values."""
    selected = {}
    reference = {}
    for card in Path("/sys/class/drm").glob("card[0-9]*"):
        if "-" in card.name:
            continue
        for path in card.glob("device/tile*/gt*/freq*/*"):
            if path.name in {"act_freq", "cur_freq", "rpa_freq"}:
                selected[str(path)] = path
            elif path.name in {"rp0_freq", "min_freq", "max_freq"}:
                reference[str(path)] = read_number(path)
        for path in card.glob("device/tile*/gt*/freq*/throttle/*"):
            if path.is_file() and read_number(path) is not None:
                selected[str(path)] = path
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            if (zone / "type").read_text().strip() == "x86_pkg_temp":
                selected["cpuPackageMilliCelsius"] = zone / "temp"
        except OSError:
            continue
    for frequency in Path("/sys/devices/system/cpu/cpufreq").glob("policy*/scaling_cur_freq"):
        selected[frequency.parent.name + "KHz"] = frequency
    return selected, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--max-seconds", type=float, default=7200)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This optional diagnostic requires Linux /proc and /sys")
    if args.pid <= 0 or not 0.05 <= args.interval <= 60 or not 0 < args.max_seconds <= 43200:
        parser.error("Use a positive PID, interval 0.05..60 s and maximum duration up to 12 h")
    process = Path("/proc") / str(args.pid)
    initial = process_stat((process / "stat").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    channels, references = sensors()
    report = dict(
        command=sys.argv,
        pid=args.pid,
        startTicks=initial["startTicks"],
        sourceSha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        intervalSeconds=args.interval,
        clockTicksPerSecond=os.sysconf("SC_CLK_TCK"),
        sensorPaths={key: str(path) for key, path in channels.items()},
        sensorReferences=references,
        notes="Read-only asynchronous observer; sampling can perturb scheduling/power. DRM is per-client accounting, may include shared backing; RSS is not free physical VRAM. Raw sysfs units are retained. Counters may temporarily decrease; derive rates with monotonic high-water marks.",
        samples=0,
        state="sampling",
    )
    started = time.monotonic()
    columns = [
        "elapsedSeconds",
        "epochSeconds",
        "cpuTicks",
        "rssBytes",
        "requestedBytes",
        "residentBytes",
        *channels,
    ]
    try:
        with (
            (args.output / "samples.csv").open("x", newline="") as stream,
            gzip.open(args.output / "drm.jsonl.gz", "wt", encoding="utf-8") as raw,
        ):
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            while time.monotonic() - started < args.max_seconds:
                now = time.monotonic()
                try:
                    state = process_stat((process / "stat").read_text())
                except (OSError, ValueError, IndexError):
                    report["state"] = "target-exited"
                    break
                if state["startTicks"] != initial["startTicks"] or state["state"] == "Z":
                    report["state"] = "target-exited"
                    break
                records = []
                for descriptor in (process / "fdinfo").glob("*"):
                    try:
                        records.append((descriptor.name, descriptor.read_text()))
                    except OSError:
                        continue
                clients = parse_clients(records)
                row = dict(
                    elapsedSeconds=now - started,
                    epochSeconds=time.time(),
                    cpuTicks=state["cpuTicks"],
                    rssBytes=None,
                    **memory_totals(clients),
                )
                try:
                    for line in (process / "status").read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            row["rssBytes"] = int(line.split()[1]) * 1024
                except OSError:
                    pass
                row.update({key: read_number(path) for key, path in channels.items()})
                writer.writerow(row)
                raw.write(
                    json.dumps(
                        {
                            "elapsedSeconds": row["elapsedSeconds"],
                            "epochSeconds": row["epochSeconds"],
                            "clients": clients,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                report["samples"] += 1
                if report["samples"] % 25 == 0:
                    stream.flush()
                    raw.flush()
                time.sleep(max(0, args.interval - (time.monotonic() - now)))
            else:
                report["state"] = "duration-limit"
    except BaseException as error:
        report["state"] = "incomplete"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["elapsedSeconds"] = time.monotonic() - started
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
