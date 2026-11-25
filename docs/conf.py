# Copyright (c) 2024 Q-Selector Contributors
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sphinx configuration file for Q-Selector documentation."""

from __future__ import annotations

import warnings
from importlib import metadata

try:
    version = metadata.version("q-selector")
except metadata.PackageNotFoundError:
    warnings.warn(
        "Package should be installed to produce documentation! "
        "Assuming a modern git archive was used for version discovery.",
        stacklevel=1,
    )
    version = "1.0.0"

release = version.split("+")[0]

project = "Q-Selector"
author = "Q-Selector Contributors"
language = "en"
project_copyright = "2024 Q-Selector Contributors"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "Q-Selector"

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}

# -- Options for napoleon ----------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
