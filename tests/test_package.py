"""The package imports, including the pieces whose absence caused real breakage.

Each test here corresponds to something that actually failed during the refactor:

- the package must import with no third-party packages, so a notebook can tell the
  install landed
- hydra_plugins must be importable, because Hydra discovers config-search plugins
  only from a top-level package with that literal name
- soccermaster.tracklab.wrappers imports every tracker, so one broken tracker (a
  missing deep/models.py upstream) used to make the whole step-1 pipeline unimportable
"""
from __future__ import annotations

import importlib

import pytest

LIGHT_SUBPACKAGES = [
    "soccermaster.paths",
    "soccermaster.downloads",
    "soccermaster.cli",
    "soccermaster.pipeline",
    "soccermaster.export",
    "soccermaster.preflight",
    "soccermaster.api",
]

# Need torch/pandas/hydra, so only exercised where the pipeline extra is installed.
# The tuple lists the import that proves the dependency set is present, so a box
# without it skips rather than failing: a missing optional dep is not a code fault.
HEAVY_SUBPACKAGES = [
    ("soccermaster.model", ("torch",)),
    ("soccermaster.model.models.multi_task", ("transformers", "accelerate")),
    ("soccermaster.tracklab", ("pandas",)),
    ("soccermaster.tracklab.wrappers", ("yacs",)),
    ("soccermaster.soccernet_gamestate", ("pandas",)),
    ("soccermaster.sam2_refine", ("pandas",)),
]


def test_package_imports() -> None:
    import soccermaster

    assert soccermaster.__version__


@pytest.mark.parametrize("name", LIGHT_SUBPACKAGES)
def test_light_subpackage_imports(name: str) -> None:
    assert importlib.import_module(name)


def test_hydra_plugins_package_exists() -> None:
    """Hydra only finds search-path plugins in a package literally named hydra_plugins."""
    module = importlib.import_module("hydra_plugins.soccermaster_searchpath_plugin")
    assert hasattr(module, "SoccerMasterSearchPathPlugin")


@pytest.mark.heavy
@pytest.mark.parametrize("name,requires", HEAVY_SUBPACKAGES)
def test_heavy_subpackage_imports(name: str, requires: tuple[str, ...]) -> None:
    for dependency in requires:
        pytest.importorskip(dependency)
    assert importlib.import_module(name)


def test_public_api_is_lazy() -> None:
    """`import soccermaster` must not drag in torch.

    The notebook's "verify the install" cell relies on this: it imports the package
    before any dependency is installed, to confirm the install landed. Checked in a
    subprocess, because this test module has already imported things.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, soccermaster; "
            "print('torch' in sys.modules or 'yaml' in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", "importing soccermaster pulled in a dependency"

    import soccermaster

    assert "load_model" in soccermaster.__all__
