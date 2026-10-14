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
import subprocess
import sys

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
    # `wrappers` pulls in every tracker and dataset wrapper, so it needs the pipeline
    # extra's soccernet on top of the core set. Gate on a pipeline-only module: gating
    # on `yacs` looked right (it is only listed as a pipeline dep elsewhere) but it is a
    # core dependency, so the guard never fired and this failed on
    # `No module named 'SoccerNet'` in the core CI job.
    ("soccermaster.tracklab.wrappers", ("SoccerNet",)),
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


@pytest.mark.heavy
def test_detector_backend_imports() -> None:
    """`ultralytics` 8.0.100 imports `pkg_resources` at module level (yolo/utils/checks.py).

    `pkg_resources` lives inside setuptools, Python 3.12+ no longer installs setuptools
    into a fresh environment, and setuptools 81 removed `pkg_resources`. A clean
    interpreter therefore fails here while the same install works on Colab (which
    preinstalls setuptools) -- hence the bounded `setuptools<81` in pyproject. This is
    the core-only guard for that regression: it does not need the pipeline extra.
    """
    pytest.importorskip("ultralytics")
    importlib.import_module("ultralytics")


@pytest.mark.heavy
def test_torchreid_cmap_lookup_works() -> None:
    """torchreid calls `matplotlib.cm.get_cmap` at import time; matplotlib 3.9 removed it.

    Check the shim both installs and still behaves like the function it replaces, since
    it has to be in place before torchreid is imported.
    """
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")

    import matplotlib.cm as cm

    module = importlib.import_module("soccermaster.tracklab.utils.matplotlib_compat")
    module.patch_legacy_cmap()

    assert hasattr(cm, "get_cmap"), "the matplotlib compatibility shim did not install"
    assert cm.get_cmap("hsv").name == "hsv"
    assert cm.get_cmap("jet", 8).N == 8


# The modules step 1 actually resolves its `detector`, `track` and `reid` entries to,
# each paired with the import that proves its dependency set is present.
STEP1_TARGETS = [
    ("soccermaster.soccernet_gamestate.detect_multiple.yolov8_person_api", "ultralytics"),
    ("soccermaster.tracklab.wrappers", "SoccerNet"),
    ("soccermaster.soccernet_gamestate.reid.prtreid_api", "prtreid"),
]


@pytest.mark.heavy
@pytest.mark.parametrize("module,requires", STEP1_TARGETS)
def test_step1_targets_import_in_a_fresh_process(module: str, requires: str) -> None:
    """Import a step-1 target in a cold process, the way `soccermaster doctor` does.

    Doing this in-process is not equivalent, and that is not hypothetical: the
    target-resolution test in test_pipeline_configs.py passed while `doctor` reported
    step 1's reid target as unresolvable, because a *previous test in the same process*
    had already imported soccermaster.tracklab.wrappers, which installs the
    matplotlib.cm.get_cmap shim globally. Only a fresh interpreter has the import state
    `doctor` starts from.
    """
    pytest.importorskip(requires)

    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"importing {module} in a fresh process failed (exit {result.returncode})\n"
        f"{result.stderr}"
    )


@pytest.mark.heavy
@pytest.mark.parametrize("package", ["soccermaster.tracklab", "soccermaster.soccernet_gamestate"])
def test_package_import_installs_the_cmap_shim(package: str) -> None:
    """Importing either package must install the shim before any submodule can need it.

    The shim has to be in place before torchreid is imported, and torchreid is imported
    by deeper modules in both trees, so the guarantee has to come from the package
    __init__ rather than from a line near the eventual import. Checked cold, since a
    warm interpreter may already have it.
    """
    pytest.importorskip("matplotlib")

    code = (
        "import sys, matplotlib.cm as cm;"
        f"import {package};"
        "sys.exit(0 if hasattr(cm, 'get_cmap') else 3)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"importing {package} did not install the matplotlib.cm.get_cmap shim "
        f"(exit {result.returncode})\n{result.stderr}"
    )
