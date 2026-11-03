# Repository maintenance

The maintained live-dic Python project is in `DIC/LucasKanadeICGN/`.
Read its [AGENTS.md](DIC/LucasKanadeICGN/AGENTS.md) before changing it.
[CONTRIBUTING.md](CONTRIBUTING.md) contains the development and validation commands.

- Write code, documentation and commit messages in English.
- Preserve a prominent, precise methodological acknowledgment of Pierre
  Margerit's prior GPU-DIC and Three.js/WebGPU prototype work on landing pages.
  Do not imply that the current solver is his code or that live-dic reuses
  Strobox source unless that has been verified. Keep software-license notices
  separate from methodological acknowledgments.
- Keep the repository focused on the maintained live-dic Python engine and
  browser workbench. Do not restore retired standalone experiments unless
  explicitly requested.
- Do not rename the Python distribution, module or CLI as part of cleanup.
- Preserve user edits and existing local commits. Fetch before integrating
  remote changes; never force-push or rewrite published history.
- Commit completed work in focused local commits. Push only when requested.
- Treat vendor bundles, analytic image fixtures and archived campaign reports
  as provenance-bearing data. Do not reformat or regenerate them casually.
- Ignore generated caches and local environments rather than committing them.

Prefer the codebase-memory MCP graph for code discovery. If it is unavailable
or insufficient, use targeted text searches. Use the local RTK wrapper for
shell commands when it is available.
