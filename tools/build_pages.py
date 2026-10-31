"""Assemble the static PWA without a bundler, npm or runtime network dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "DIC/LucasKanadeICGN"
PACKAGE = PROJECT / "python/src/lucas_kanade_icgn"


def assemble(output: Path) -> dict:
    output = output.resolve()
    if output == ROOT or ROOT in output.parents and output.name not in ("_site", "_browser-site"):
        raise ValueError("Use an external destination or a dedicated _site directory")
    if output.exists():
        if any(output.iterdir()) and not (output / ".live-dic-site").is_file():
            raise ValueError(
                "Refusing to replace a nonempty directory not created by this assembler"
            )
        shutil.rmtree(output)
    shutil.copytree(PROJECT / "web-app", output)
    shutil.copytree(PACKAGE / "web/core", output / "web/core")
    shutil.copytree(PACKAGE / "shaders", output / "shaders")
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    shaders = {}
    for original in sorted((PACKAGE / "shaders").glob("*.wgsl")):
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        if hashlib.sha256((output / "shaders" / original.name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Canonical shader copy differs: {original.name}")
        shaders[original.name] = digest
    demo = json.loads((output / "demo/manifest.json").read_text())
    for name, digest in demo["files"].items():
        if hashlib.sha256((output / "demo" / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Demo asset hash mismatch: {name}")
    report = dict(
        commit=commit,
        dirty=bool(
            subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)
        ),
        shaders=shaders,
        demo_manifest_sha256=hashlib.sha256(
            (output / "demo/manifest.json").read_bytes()
        ).hexdigest(),
    )
    if report["dirty"]:
        fingerprint = hashlib.sha256()
        for file in sorted(output.rglob("*")):
            if file.is_file():
                fingerprint.update(file.relative_to(output).as_posix().encode())
                fingerprint.update(file.read_bytes())
        report["cache_version"] = commit + "-dirty-" + fingerprint.hexdigest()[:12]
    else:
        report["cache_version"] = commit
    (output / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / ".nojekyll").touch()
    (output / ".live-dic-site").touch()
    assets = ["./"] + [
        "./" + p.relative_to(output).as_posix()
        for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "sw.js"
    ]
    service_worker = (
        (output / "sw.js")
        .read_text()
        .replace("__COMMIT__", report["cache_version"])
        .replace("__PRECACHE__", json.dumps(assets))
    )
    (output / "sw.js").write_text(service_worker)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.output), indent=2))


if __name__ == "__main__":
    main()
