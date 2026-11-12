"""Exercise workspace mode clicks in an actual Chromium DOM."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHROMIUM = (
    shutil.which("google-chrome")
    or shutil.which("chromium")
    or shutil.which("chromium-browser")
    or next(
        (
            str(path)
            for path in (Path("/snap/bin/chromium"), Path("/usr/bin/chromium"))
            if path.exists()
        ),
        None,
    )
)
RUN_BROWSER_DOM_TESTS = os.environ.get("ICGN_RUN_BROWSER_DOM_TESTS") == "1"


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.server.result = json.loads(self.rfile.read(length))
        self.send_response(204)
        self.end_headers()
        self.server.completed.set()


@unittest.skipUnless(
    CHROMIUM and RUN_BROWSER_DOM_TESTS,
    "set ICGN_RUN_BROWSER_DOM_TESTS=1 to run the Chromium DOM test",
)
class WorkspaceDomTests(unittest.TestCase):
    def test_background_click_does_not_reapply_mode_and_mode_button_fires_once(self):
        with tempfile.TemporaryDirectory(prefix="live-dic-workspace-dom-") as temporary:
            root = Path(temporary)
            (root / "workspace.js").write_text(
                (ROOT / "web-app/ui/workspace.js").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "index.html").write_text(
                """<!doctype html>
<html><body data-workspace-mode="analysis">
  <nav>
    <button id="live" data-workspace-mode="live">Live</button>
    <button id="analysis" data-workspace-mode="analysis">Analysis</button>
    <button data-pane="source">Source</button>
  </nav>
  <main id="background">
    <p id="live-only" data-workspace-only="live" hidden>Live controls</p>
    <p id="analysis-only" data-workspace-only="analysis">Analysis controls</p>
    <div id="pane-source"></div>
  </main>
  <button id="help">Help</button><dialog id="quick-guide"></dialog>
  <select id="outlier-policy"><option value="disabled" selected>Disabled</option></select>
  <select id="median-filter-size"><option value="3">3</option></select>
  <output id="result">pending</output>
  <script type="module">
    import { installWorkspace } from './workspace.js';
    const modeCalls = [];
    const workspace = installWorkspace({
      shortcuts: { play() {}, reference() {}, clear() {} },
      onModeChange: mode => modeCalls.push(mode),
    });
    const initialCalls = modeCalls.slice();
    const initialVisibility = {
      live: document.getElementById('live-only').hidden,
      analysis: document.getElementById('analysis-only').hidden,
    };
    document.getElementById('background').click();
    const afterBackground = { mode: workspace.mode, calls: modeCalls.slice() };
    document.getElementById('live').click();
    const afterLiveButton = {
      mode: workspace.mode,
      calls: modeCalls.slice(),
      livePressed: document.getElementById('live').getAttribute('aria-pressed'),
      analysisPressed: document.getElementById('analysis').getAttribute('aria-pressed'),
      preferenceRecorded: workspace.hasSavedMode,
      liveControlsHidden: document.getElementById('live-only').hidden,
      analysisControlsHidden: document.getElementById('analysis-only').hidden,
    };
    document.getElementById('background').click();
    const result = {
      initialCalls, initialVisibility, afterBackground, afterLiveButton,
      afterSecondBackground: { mode: workspace.mode, calls: modeCalls.slice() },
    };
    await fetch('/result', { method: 'POST', body: JSON.stringify(result) });
  </script>
</body></html>
""",
                encoding="utf-8",
            )

            handler = partial(QuietHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            server.completed = threading.Event()
            server.result = None
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/"
                # Completion comes from the page, independent of Chromium's
                # --dump-dom virtual-time and shutdown behaviour on CI runners.
                with tempfile.TemporaryFile(mode="w+b") as log:
                    browser = subprocess.Popen(
                        [
                            CHROMIUM,
                            "--headless",
                            "--no-sandbox",
                            "--disable-gpu",
                            "--disable-dev-shm-usage",
                            "--no-first-run",
                            "--disable-background-networking",
                            f"--user-data-dir={root / 'profile'}",
                            url,
                        ],
                        stdout=log,
                        stderr=log,
                    )
                    try:
                        finished = server.completed.wait(timeout=30)
                    finally:
                        browser.terminate()
                        try:
                            browser.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            browser.kill()
                            browser.wait(timeout=5)
                    log.seek(0)
                    diagnostic = log.read().decode("utf-8", errors="replace")[-3000:]
                    self.assertTrue(finished, f"Workspace page did not finish: {diagnostic}")
                    actual = server.result
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

        self.assertEqual(actual["initialCalls"], ["analysis"])
        self.assertEqual(actual["afterBackground"], {"mode": "analysis", "calls": ["analysis"]})
        self.assertEqual(
            actual["afterLiveButton"],
            {
                "mode": "live",
                "calls": ["analysis", "live"],
                "livePressed": "true",
                "analysisPressed": "false",
                "preferenceRecorded": True,
                "liveControlsHidden": False,
                "analysisControlsHidden": True,
            },
        )
        self.assertEqual(actual["initialVisibility"], {"live": True, "analysis": False})
        self.assertEqual(
            actual["afterSecondBackground"],
            {"mode": "live", "calls": ["analysis", "live"]},
        )


if __name__ == "__main__":
    unittest.main()
