"""Hydra configs compose and every pipeline _target_ resolves.

This is the single most valuable check in the suite. It would have caught, before any
long run:

- the missing hydra_plugins package, which made gsr_step_1_example unfindable
- the mutable-dataclass-default error, which made every step-1 module unimportable on
  Python 3.11+
- the bare `import posetrack21` / `yolo_old` imports that took down the track module

`_target_` resolution is done without instantiating: instantiating would pull weights
and need a GPU. Importing the module and checking the attribute exists is exactly the
failure mode we hit.
"""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("torch")
pytest.importorskip("hydra")

from hydra import compose, initialize  # noqa: E402

from soccermaster import paths  # noqa: E402

TRACKLAB_CONFIG_PKG = "pkg://soccermaster.tracklab.configs"
CONFIGS = ["gsr_step_1_example", "gsr_step_3_example_accelerate"]

BASE_OVERRIDES = [
    "data_dir=/tmp/sm-data",
    "model_dir=/tmp/sm-weights",
    "dataset.dataset_path=/tmp/sm-ds",
    "dataset.eval_set=sn500",
    "num_cores=1",
]


def _compose(name: str, overrides=()):
    with initialize(version_base=None, config_path=TRACKLAB_CONFIG_PKG):
        return compose(config_name=name, overrides=list(BASE_OVERRIDES) + list(overrides))


@pytest.mark.heavy
@pytest.mark.parametrize("name", CONFIGS)
def test_config_composes(name: str) -> None:
    cfg = _compose(name)

    assert cfg.get("pipeline"), f"{name} has an empty pipeline"
    assert cfg.get("num_cores")


@pytest.mark.heavy
@pytest.mark.parametrize("name", CONFIGS)
def test_every_pipeline_target_resolves(name: str) -> None:
    """Import each module named in the pipeline and check the attribute exists.

    Skipped unless the full dependency set is installed: a target that fails to import
    because pandas is absent is not the failure this test is looking for. Where the
    deps are present (the release workflow installs the pipeline extra) this runs, and
    it is the check that catches missing modules like posetrack21 or yolo_old.
    """
    # Gate on a module that only the `pipeline` extra provides. Guarding on `yacs` and
    # `transformers` was wrong: both are core dependencies, so they are always present,
    # the guard never fired, and this ran in the core CI job where it failed on
    # `SoccerNet`/`skimage` -- dependencies that job deliberately does not install.
    pytest.importorskip("SoccerNet")
    pytest.importorskip("transformers")

    cfg = _compose(name)
    problems = []

    for module_name in cfg["pipeline"]:
        node = cfg.get("modules", {}).get(module_name)
        if node is None or "_target_" not in node:
            problems.append(f"{module_name}: no _target_")
            continue

        target = str(node["_target_"])
        module_path, _, attribute = target.rpartition(".")
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001 - the message is the point
            problems.append(f"{module_name}: {type(exc).__name__}: {exc}")
            continue
        if not hasattr(module, attribute):
            problems.append(f"{module_name}: {module_path} has no {attribute}")

    assert not problems, "unresolvable pipeline modules:\n  " + "\n  ".join(problems)


@pytest.mark.heavy
def test_worker_override_is_applied() -> None:
    """doctor warns when num_cores exceeds the CPU count; the pipeline overrides it."""
    cfg = _compose("gsr_step_1_example", ["num_cores=1"])
    assert cfg["num_cores"] == 1


@pytest.mark.heavy
def test_reid_scratch_overrides_are_config_keys() -> None:
    """prtreid creates save_path/job_id with os.makedirs and no exist_ok.

    The pipeline points save_path at an absolute per-run path and randomises job_id so
    the directory never pre-exists. Both must be real config keys or Hydra errors.
    """
    from pathlib import Path

    from soccermaster.pipeline import _reid_scratch_overrides

    overrides = _reid_scratch_overrides(Path("/tmp/run"))
    cfg = _compose("gsr_step_1_example", overrides)

    assert Path(cfg.modules.reid.save_path).is_absolute()
    assert isinstance(cfg.modules.reid.job_id, int)


@pytest.mark.heavy
def test_preflight_runs_without_weights() -> None:
    """doctor must work on a bare checkout: it is the pre-run gate."""
    from soccermaster import preflight

    # 0 = no blocking failures, 1 = something is missing. Either is fine here; what
    # must not happen is an exception.
    assert preflight.run(steps="1") in (0, 1)
