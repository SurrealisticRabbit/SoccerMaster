"""Config loading and path resolution.

Covers the bug where `api.load_config` built its config directory from the literal
string "foundation" after the package was renamed to "model", which made
load_config, load_model, predict and foundation-demo all raise FileNotFoundError.
"""
from __future__ import annotations

import pytest

pytest.importorskip("yaml")

from soccermaster import api, paths  # noqa: E402


def test_config_dir_exists() -> None:
    assert api.CONFIG_DIR.is_dir(), f"missing config dir: {api.CONFIG_DIR}"
    assert (api.CONFIG_DIR / "pretrain.yaml").is_file()


def test_load_config_has_expected_keys() -> None:
    cfg = api.load_config("pretrain")

    assert len(cfg) > 100, "the super config does not appear to have been merged"
    for key in ("NUM_FRAMES", "AUG_MAX_SIZE", "DATASETS_TO_HEADS", "CKPT_PATH"):
        assert key in cfg, f"{key} missing from the merged config"
    assert cfg["DATASETS_TO_HEADS"]


def test_weight_paths_are_retargeted() -> None:
    """CKPT_PATH must point inside weights_dir, not at the old relative location."""
    cfg = api.load_config("pretrain")
    weights = paths.weights_dir()

    for key in ("CKPT_PATH", "TEXT_ENCODER_CKPT_PATH"):
        value = cfg[key]
        assert value.startswith(str(weights)), f"{key}={value} is not under {weights}"


def test_missing_config_raises() -> None:
    with pytest.raises(FileNotFoundError):
        api.load_config("does-not-exist")


def test_paths_honour_home_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SOCCERMASTER_HOME", str(tmp_path))

    # paths reads the environment on every call, so no reload is needed.
    assert paths.home() == tmp_path
    assert paths.weights_dir() == tmp_path / "weights"
    assert paths.data_dir() == tmp_path / "data"
    assert paths.outputs_dir() == tmp_path / "outputs"
