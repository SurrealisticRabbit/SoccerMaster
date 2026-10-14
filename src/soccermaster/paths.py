"""Standard directory locations for SoccerMaster data, weights, and outputs.

All locations are overridable via the ``SOCCERMASTER_HOME`` environment
variable. In an editable (development) install the default is the repository
root; in a regular install the default is ``~/.cache/soccermaster``.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "SOCCERMASTER_HOME"


def _is_editable_install() -> bool:
    """Best-effort detection of an editable install."""
    pkg_file = Path(__file__).resolve()
    # src/soccermaster/paths.py -> repo root is parents[2]
    repo = pkg_file.parents[2]
    return (repo / "pyproject.toml").is_file() and (repo / "src").is_dir()


def repo_root() -> Path:
    """Return the repository root (only meaningful for editable installs)."""
    return Path(__file__).resolve().parents[2]


def home() -> Path:
    """Root directory for data, weights, and outputs."""
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    if _is_editable_install():
        return repo_root()
    return Path.home() / ".cache" / "soccermaster"


def data_dir() -> Path:
    """Directory containing ``raw``/``frames``/``annotations`` datasets."""
    return home() / "data"


def weights_dir() -> Path:
    """Directory containing pretrained model weights."""
    return home() / "weights"


def outputs_dir() -> Path:
    """Directory where pipeline and training outputs are written."""
    return home() / "outputs"
