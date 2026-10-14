"""Regression tests for the step-3 VLM loading kwargs and video frame extraction.

Both areas held values that only worked on the machine they were written on:

- ``attn_implementation="flash_attention_2"`` hardcoded in three modules, while flash-attn
  is an optional extra that the pipeline install does not bring in -- so step 3 raised at
  ``from_pretrained`` before loading anything. It cannot be fixed by insisting on
  flash-attn either, since FlashAttention-2 needs compute capability 8.0+.
- ``dtype=`` in two of those modules, which the pinned transformers (>=4.49,<4.52) does
  not read: it only pops ``torch_dtype``, and ``PretrainedConfig`` has no ``dtype`` alias
  to absorb it. The argument was therefore ignored and a 7B model loaded in float32.
- an ffmpeg ``timeout=300``, which failed silently on any video needing longer to decode.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytest.importorskip("torch")

from soccermaster.soccernet_gamestate import vlm_compat  # noqa: E402


# --------------------------------------------------------------------------- #
# attention backend
# --------------------------------------------------------------------------- #


def test_attn_implementation_is_one_transformers_accepts() -> None:
    implementation = vlm_compat.resolve_attn_implementation()
    assert implementation in {"sdpa", "flash_attention_2", "eager"}


def test_attn_falls_back_when_flash_attn_is_absent() -> None:
    """The whole point: no flash-attn must not mean no step 3."""
    import importlib.util

    if importlib.util.find_spec("flash_attn") is not None:
        pytest.skip("flash-attn is installed here; the fallback is not on this path")
    assert vlm_compat.resolve_attn_implementation() == "sdpa"


def test_attn_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(vlm_compat.ATTN_ENV, "eager")
    assert vlm_compat.resolve_attn_implementation() == "eager"


def test_turing_gpu_falls_back_even_with_flash_attn(monkeypatch: pytest.MonkeyPatch) -> None:
    """flash-attn imports fine on a T4 and then fails at kernel launch."""
    monkeypatch.delenv(vlm_compat.ATTN_ENV, raising=False)
    monkeypatch.setattr(vlm_compat, "torch", _FakeTorch(available=True, capability=(7, 5)))
    monkeypatch.setitem(__import__("sys").modules, "flash_attn", object())
    assert vlm_compat.resolve_attn_implementation() == "sdpa"


def test_ampere_gpu_uses_flash_attn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(vlm_compat.ATTN_ENV, raising=False)
    monkeypatch.setattr(vlm_compat, "torch", _FakeTorch(available=True, capability=(8, 9)))
    monkeypatch.setitem(__import__("sys").modules, "flash_attn", object())
    assert vlm_compat.resolve_attn_implementation() == "flash_attention_2"


class _FakeCuda:
    def __init__(self, available: bool, capability: tuple[int, int]) -> None:
        self._available = available
        self._capability = capability

    def is_available(self) -> bool:
        return self._available

    def get_device_capability(self) -> tuple[int, int]:
        return self._capability


class _FakeTorch:
    def __init__(self, available: bool, capability: tuple[int, int]) -> None:
        self.cuda = _FakeCuda(available, capability)


# --------------------------------------------------------------------------- #
# dtype keyword
# --------------------------------------------------------------------------- #


def test_dtype_kwarg_is_the_key_from_pretrained_actually_reads() -> None:
    """Guards the silent-fp32 bug: the key must match the installed transformers."""
    transformers = pytest.importorskip("transformers")

    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    key = next(iter(vlm_compat.resolve_dtype_kwarg()))

    if (major, minor) < (4, 56):
        assert key == "torch_dtype"
        # The reason `dtype=` was a no-op: nothing absorbs it, so it never reaches the
        # loader and the model quietly comes back in float32.
        from transformers import PretrainedConfig

        assert not hasattr(PretrainedConfig, "dtype")
    else:
        assert key == "dtype"


def test_dtype_kwarg_carries_the_requested_dtype() -> None:
    import torch

    value = next(iter(vlm_compat.resolve_dtype_kwarg(torch.float16).values()))
    assert value is torch.float16


def test_vlm_load_kwargs_has_both_decisions() -> None:
    kwargs = vlm_compat.vlm_load_kwargs()
    assert "attn_implementation" in kwargs
    assert len(kwargs) == 2, kwargs
    assert set(kwargs) & {"dtype", "torch_dtype"}


# --------------------------------------------------------------------------- #
# frame extraction
# --------------------------------------------------------------------------- #


class _CapturingSubprocess:
    """Stands in for the subprocess module so the ffmpeg argv can be asserted."""

    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self) -> None:
        self.cmd: list[str] | None = None
        self.timeout: float | None = None

    def run(self, cmd, **kwargs):  # noqa: ANN001, ANN003 - mirrors subprocess.run
        self.cmd = cmd
        self.timeout = kwargs.get("timeout")

        class _Result:
            returncode = 0
            stderr = ""
            stdout = ""

        return _Result()


def _extract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs):
    from soccermaster.model.data import extract_frames

    capture = _CapturingSubprocess()
    monkeypatch.setattr(extract_frames, "subprocess", capture)
    monkeypatch.setattr(extract_frames, "probe_duration", lambda _path: 120.0)
    result = extract_frames.extract_frames_from_video(
        "match.mp4", str(tmp_path / "seq"), **kwargs
    )
    return capture, result


def test_every_frame_is_kept_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture, result = _extract(tmp_path, monkeypatch)
    assert "-vf" not in capture.cmd
    assert result["success"] is True


def test_fps_decimates_while_decoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The control that makes an 11-minute match tractable."""
    capture, _ = _extract(tmp_path, monkeypatch, fps=5)
    assert "-vf" in capture.cmd
    assert "fps=5" in capture.cmd


def test_timeout_scales_with_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fixed 300 s quietly failed on long videos, leaving a half-filled img1/."""
    capture, _ = _extract(tmp_path, monkeypatch)
    assert capture.timeout is not None and capture.timeout >= 1200


def test_explicit_timeout_is_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture, _ = _extract(tmp_path, monkeypatch, timeout=42)
    assert capture.timeout == 42


def test_probe_duration_returns_none_instead_of_raising() -> None:
    from soccermaster.model.data.extract_frames import probe_duration

    assert probe_duration("/definitely/not/a/video.mp4") is None
