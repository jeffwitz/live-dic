"""Check active Markdown links and the completeness of a built pure-Python wheel.

Archived campaigns are immutable evidence, so their historical links and source
snapshots are not rewritten or treated as current documentation by this check.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "DIC/LucasKanadeICGN/python/src/lucas_kanade_icgn"


def check_links() -> list[str]:
    paths = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        )
        .decode()
        .split("\0")
    )
    errors = []
    checked = 0
    for name in sorted(set(paths)):
        archived = "/benchmarks/results/" in name and not name.endswith("/results/README.md")
        if not name.endswith(".md") or archived:
            continue
        path = ROOT / name
        if not path.is_file():
            continue
        text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        for target in re.findall(r"\]\(([^\s)]+)", text):
            target = target.strip("<>")
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            checked += 1
            if not (path.parent / unquote(url.path)).exists():
                errors.append(f"{name}: missing local link target {target}")
    print(f"Checked {checked} active local Markdown links (fragments are not validated).")
    return errors


def check_wheel(path: Path) -> list[str]:
    errors = []
    with ZipFile(path) as wheel:
        names = set(wheel.namelist())
        metadata = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(metadata) != 1:
            return [f"{path}: expected exactly one WHEEL metadata file"]
        tags = wheel.read(metadata[0]).decode()
        if "Root-Is-Purelib: true" not in tags or "Tag: py3-none-any" not in tags:
            errors.append(f"{path}: the project wheel must remain platform-independent")
        checked = 0
        for source in PACKAGE.rglob("*"):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            name = f"lucas_kanade_icgn/{source.relative_to(PACKAGE).as_posix()}"
            checked += 1
            if name not in names:
                errors.append(f"{path}: missing package file {name}")
            elif wheel.read(name) != source.read_bytes():
                errors.append(f"{path}: packaged file differs from its source: {name}")
        for name in names:
            if name.endswith((".so", ".pyd", ".dll", ".dylib", ".spv", ".exe")):
                errors.append(f"{path}: unexpected compiled project artifact {name}")
    print(f"Checked {checked} package files against {path.name}.")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="also validate this built wheel")
    parser.add_argument(
        "--wheel-dir", type=Path, help="directory containing exactly one project wheel"
    )
    args = parser.parse_args()
    if args.wheel is not None and args.wheel_dir is not None:
        parser.error("select --wheel or --wheel-dir, not both")
    wheel = args.wheel
    if args.wheel_dir is not None:
        wheels = list(args.wheel_dir.glob("*.whl"))
        if len(wheels) != 1:
            parser.error("--wheel-dir must contain exactly one project wheel")
        wheel = wheels[0]
    errors = check_links()
    if wheel is not None:
        errors.extend(check_wheel(wheel))
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
