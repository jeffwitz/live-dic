"""Sphinx configuration for the Python-supervised live-dic documentation."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

DOCUMENTATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DOCUMENTATION_ROOT.parent
PYTHON_SOURCE = PROJECT_ROOT / "python" / "src"
sys.path.insert(0, str(PYTHON_SOURCE))

project = "live-dic"
author = "live-dic contributors; foundational work by Pierre Margerit"
copyright = "2026, Real-Time DIC contributors"
project_metadata = tomllib.loads(
    (PROJECT_ROOT / "python/pyproject.toml").read_text(encoding="utf-8")
)
release = project_metadata["project"]["version"]
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosectionlabel_prefix_document = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "live-dic: Python-supervised real-time DIC"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "titles_only": False,
}

pygments_style = "sphinx"
todo_include_todos = False
