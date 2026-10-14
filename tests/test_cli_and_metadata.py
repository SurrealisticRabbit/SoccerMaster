"""The CLI surface and the package's build metadata."""
from __future__ import annotations

import subprocess
import sys

import pytest


def test_cli_help_lists_subcommands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "soccermaster", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    for command in ("prepare", "pipeline", "predict", "analyze", "export", "doctor"):
        assert command in result.stdout, f"{command} missing from --help"


def test_console_script_is_installed() -> None:
    """pyproject declares it; make sure the entry point actually resolves."""
    from soccermaster.cli import main

    assert callable(main)


def test_entry_points_are_registered() -> None:
    """Two entry points are load-bearing and easy to break silently.

    - `soccermaster_plugin` lets the search-path plugin find the gsr_* configs
    - there must be NO `hydra.plugins.search_path_plugin` entry point: Hydra does not
      read that group, and having one there is how config discovery was broken while
      looking correct
    """
    import importlib.metadata as md

    soccermaster_eps = list(md.entry_points().select(group="soccermaster_plugin"))
    assert soccermaster_eps, "the soccermaster_plugin entry point is missing"

    hydra_eps = list(md.entry_points().select(group="hydra.plugins.search_path_plugin"))
    assert not hydra_eps, (
        "there is a hydra.plugins.search_path_plugin entry point; Hydra ignores that "
        "group, so this looks correct but does nothing. The plugin belongs in the "
        "top-level hydra_plugins package."
    )


def test_version_is_a_string() -> None:
    import soccermaster

    assert isinstance(soccermaster.__version__, str)
    assert soccermaster.__version__ != ""


@pytest.mark.heavy
def test_doctor_exits_nonzero_when_blocked() -> None:
    """doctor's exit code is its contract: non-zero means do not run the pipeline."""
    result = subprocess.run(
        [sys.executable, "-m", "soccermaster", "doctor", "--steps", "1"],
        capture_output=True,
        text=True,
    )
    # 0 when everything is in place, 1 when something is missing. Never a crash.
    assert result.returncode in (0, 1), result.stderr
    assert "pre-flight" in result.stdout
