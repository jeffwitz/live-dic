"""Run three alternating 16/128-lane comparisons with the same UHD trajectory."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-cache", type=Path, default=Path("/tmp/live-dic-rt-trajectory.npy")
    )
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    benchmark = directory.parents[1] / "benchmark_latency_reduction.py"
    for repeat in range(3):
        for width in (16, 128) if repeat % 2 == 0 else (128, 16):
            for depth in (1, 2):
                output = directory / "runs" / f"w{width}-d{depth}-r{repeat}.json"
                command = [
                    sys.executable,
                    str(benchmark),
                    "--variant",
                    "abort",
                    "--target-cache",
                    str(args.target_cache),
                    "--pipeline-depth",
                    str(depth),
                    "--solver-workgroup-size",
                    str(width),
                    "--output",
                    str(output),
                ]
                if repeat == 0 and depth == 1:
                    command.append("--diagnostics")
                print(f"width {width}, depth {depth}, repeat {repeat}", flush=True)
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
