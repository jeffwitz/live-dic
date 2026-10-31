"""The one-command server must serve the chosen build and preserve unrelated files."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[4] / "tools/run_browser_app.py"


class BrowserLauncherTests(unittest.TestCase):
    def test_launcher_refuses_to_replace_an_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            document = directory / "experiment.txt"
            document.write_text("preserve me")
            result = subprocess.run(
                [
                    sys.executable,
                    str(LAUNCHER),
                    "--site",
                    str(directory),
                    "--no-open",
                    "--port",
                    "0",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("Refusing to replace", result.stderr)
            self.assertEqual(document.read_text(), "preserve me")

    def test_launcher_serves_an_existing_build_without_opening_a_browser(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            site = directory / "site"
            site.mkdir()
            (site / ".live-dic-site").write_text("generated")
            (site / "build.json").write_text(
                json.dumps(dict(shaders={"icgn.wgsl": "fixture"}, cache_version="fixture"))
            )
            (site / "index.html").write_text("<title>Local launcher contract</title>")
            logfile = directory / "server.log"
            with logfile.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "import os,socket,runpy,sys; "
                        "socket.getfqdn=lambda *args: (_ for _ in ()).throw(AssertionError('No reverse DNS for a local server')); "
                        "launcher=sys.argv[1];sys.argv=sys.argv[1:];sys.path.insert(0,os.path.dirname(launcher));runpy.run_path(launcher,run_name='__main__')",
                        str(LAUNCHER),
                        "--site",
                        str(site),
                        "--no-build",
                        "--no-open",
                        "--port",
                        "0",
                    ],
                    stdout=log,
                    stderr=log,
                )
                try:
                    deadline = time.monotonic() + 60
                    url = None
                    while time.monotonic() < deadline:
                        text = logfile.read_text(encoding="utf-8")
                        match = re.search(r"workbench: (http://\S+/)", text)
                        if match:
                            url = match[1]
                            break
                        if process.poll() is not None:
                            self.fail(text)
                        time.sleep(0.05)
                    self.assertIsNotNone(url, logfile.read_text())
                    with urllib.request.urlopen(url, timeout=5) as response:
                        self.assertIn("Local launcher contract", response.read().decode("utf-8"))
                finally:
                    process.terminate()
                    process.wait(timeout=5)
            self.assertEqual(
                (site / "index.html").read_text(), "<title>Local launcher contract</title>"
            )
