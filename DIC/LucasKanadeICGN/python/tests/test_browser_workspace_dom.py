"""Exercise workspace mode clicks in an actual Chromium DOM."""

from __future__ import annotations

import json
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
    shutil.which("chromium")
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


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass


@unittest.skipUnless(CHROMIUM, "Chromium is required for the real-DOM workspace test")
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
    document.getElementById('result').textContent = JSON.stringify({
      initialCalls, initialVisibility, afterBackground, afterLiveButton,
      afterSecondBackground: { mode: workspace.mode, calls: modeCalls.slice() },
    });
  </script>
</body></html>
""",
                encoding="utf-8",
            )

            handler = partial(QuietHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/"
                completed = subprocess.run(
                    [
                        CHROMIUM,
                        "--headless",
                        "--no-sandbox",
                        "--disable-gpu",
                        "--disable-dev-shm-usage",
                        f"--user-data-dir={root / 'profile'}",
                        "--virtual-time-budget=1000",
                        "--dump-dom",
                        url,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

        output = completed.stdout.split('<output id="result">', 1)[1].split("</output>", 1)[0]
        actual = json.loads(output)
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
