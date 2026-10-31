"""Build and serve the local DIC workbench with Python's standard library."""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import subprocess
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from build_pages import assemble

ROOT = Path(__file__).resolve().parents[1]


def validate_destination(site: Path, *, rebuild: bool) -> None:
    """Never replace an unrelated directory while preparing a development site."""

    if site == ROOT or site in ROOT.parents:
        raise ValueError("The site directory cannot be the checkout or one of its parents")
    marker = site / "build.json"
    if marker.exists() and (site / ".live-dic-site").is_file():
        record = json.loads(marker.read_text(encoding="utf-8"))
        if (
            isinstance(record.get("shaders"), dict)
            and "icgn.wgsl" in record["shaders"]
            and record.get("cache_version")
        ):
            return
    if not rebuild:
        raise ValueError("No assembled live-dic site found; omit --no-build to create it")
    if site.exists() and any(site.iterdir()):
        raise ValueError(
            "Refusing to replace a non-build directory; choose an empty --site directory"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=ROOT / "_site")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="0 chooses an available port")
    parser.add_argument("--no-build", action="store_true", help="serve an already assembled site")
    parser.add_argument(
        "--no-open", action="store_true", help="do not launch an interactive browser"
    )
    parser.add_argument(
        "--browser", help="optional browser executable; its normal profile/settings are retained"
    )
    args = parser.parse_args()
    if args.port < 0 or args.port > 65535:
        parser.error("--port must be between 0 and 65535")
    site = args.site.resolve()
    try:
        validate_destination(site, rebuild=not args.no_build)
        handler = functools.partial(SimpleHTTPRequestHandler, directory=str(site))
        # Bind first: an occupied port must not replace files used by an existing server.
        with ThreadingHTTPServer((args.host, args.port), handler) as server:
            if not args.no_build:
                assemble(site)
            host = "localhost" if args.host in {"127.0.0.1", "0.0.0.0"} else args.host
            url = f"http://{host}:{server.server_port}/"
            print(f"live-dic workbench: {url}", flush=True)
            print(
                "Local processing. Ctrl+C stops this server. Save results projects before closing the tab.",
                flush=True,
            )
            if not args.no_open:
                if args.browser:
                    subprocess.Popen([args.browser, url])
                else:
                    webbrowser.open(url)
            with contextlib.suppress(KeyboardInterrupt):
                server.serve_forever()
    except (OSError, ValueError) as error:
        parser.exit(
            2, f"Cannot launch the workbench: {error}\nChoose another --port if it is occupied.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
