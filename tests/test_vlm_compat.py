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


class _RecordingStream:
    """Stands in for the ffmpeg runner so the argv and timeout can be asserted.

    Replaces an earlier fake of ``subprocess.run``: extraction now uses ``Popen`` and
    streams progress, so the seam moved to the helper that owns that call.
    """

    def __init__(self) -> None:
        self.cmd: list[str] = []
        self.timeout: float | None = None
        self.expected_frames: int | None = None
        self.returncode = 0
        self.tail = ""

    def __call__(self, cmd, timeout, expected_frames=None, report_every=5.0):  # noqa: ANN001, ARG002
        self.cmd = list(cmd)
        self.timeout = timeout
        self.expected_frames = expected_frames
        return self.returncode, self.tail


def _extract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs):
    from soccermaster.model.data import extract_frames

    capture = _RecordingStream()
    monkeypatch.setattr(extract_frames, "_stream_ffmpeg", capture)
    monkeypatch.setattr(extract_frames, "probe_duration", lambda _path: 120.0)
    result = extract_frames.extract_frames_from_video(
        "match.mp4", str(tmp_path / "seq"), **kwargs
    )
    return capture, result


def test_every_frame_is_kept_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture, result = _extract(tmp_path, monkeypatch)
    assert "-vf" not in capture.cmd
    assert "-frames:v" not in capture.cmd
    assert result["success"] is True


def test_fps_decimates_while_decoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The control that makes an 11-minute match tractable."""
    capture, _ = _extract(tmp_path, monkeypatch, fps=5)
    assert "-vf" in capture.cmd
    assert "fps=5" in capture.cmd


def test_max_frames_takes_a_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Native-rate testing on a long video, without decimating.

    ``--fps`` cannot do this: an ffmpeg fps filter still decodes every frame, so it saves
    the downstream steps but nothing on decode, and it makes the tracker's job
    unrealistic by stretching the gap between consecutive frames.
    """
    capture, _ = _extract(tmp_path, monkeypatch, max_frames=600)
    assert "-frames:v" in capture.cmd
    assert capture.cmd[capture.cmd.index("-frames:v") + 1] == "600"


def test_max_frames_precedes_the_output_pattern(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """-frames:v is an output option; after the output path ffmpeg would ignore it."""
    capture, _ = _extract(tmp_path, monkeypatch, max_frames=600)
    assert capture.cmd.index("-frames:v") < len(capture.cmd) - 1
    assert capture.cmd[-1].endswith("%06d.jpg")


def test_progress_is_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Decode used to be silent: ffmpeg's output was captured and thrown away."""
    capture, _ = _extract(tmp_path, monkeypatch)
    assert "-progress" in capture.cmd
    assert "pipe:1" in capture.cmd
    assert "-nostats" in capture.cmd


def test_expected_frames_is_estimated_for_the_progress_percentage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, _ = _extract(tmp_path, monkeypatch, fps=2)
    assert capture.expected_frames == 240, "120s duration at 2 fps"


def test_expected_frames_is_capped_by_max_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, _ = _extract(tmp_path, monkeypatch, fps=2, max_frames=50)
    assert capture.expected_frames == 50


def test_timeout_scales_with_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fixed 300 s quietly failed on long videos, leaving a half-filled img1/."""
    capture, _ = _extract(tmp_path, monkeypatch)
    assert capture.timeout is not None and capture.timeout >= 1200


def test_explicit_timeout_is_respected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture, _ = _extract(tmp_path, monkeypatch, timeout=42)
    assert capture.timeout == 42


def test_ffmpeg_failure_reports_the_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero ffmpeg must surface its output, not a bare code."""
    from soccermaster.model.data import extract_frames

    stream = _RecordingStream()
    stream.returncode = 1
    stream.tail = "Invalid data found when processing input"
    monkeypatch.setattr(extract_frames, "_stream_ffmpeg", stream)
    monkeypatch.setattr(extract_frames, "probe_duration", lambda _path: 120.0)

    result = extract_frames.extract_frames_from_video(
        "match.mp4", str(tmp_path / "seq")
    )

    assert result["success"] is False
    assert "Invalid data" in result["reason"]


def test_probe_duration_returns_none_instead_of_raising() -> None:
    from soccermaster.model.data.extract_frames import probe_duration

    assert probe_duration("/definitely/not/a/video.mp4") is None


# --------------------------------------------------------------------------- #
# ffmpeg progress streaming
# --------------------------------------------------------------------------- #


class _FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


class _FakeProcess:
    """A Popen stand-in that yields ffmpeg's -progress key=value blocks."""

    def __init__(self, lines: list[str], returncode: int = 0) -> None:
        self.stdout = _FakeStdout(lines)
        self.returncode = returncode
        self.killed = False

    def wait(self, timeout=None) -> int:  # noqa: ANN001, ARG002
        return self.returncode

    def kill(self) -> None:
        self.killed = True

    def terminate(self) -> None:
        self.killed = True


def _patch_popen(monkeypatch: pytest.MonkeyPatch, lines: list[str], returncode: int = 0):
    from soccermaster.model.data import extract_frames

    process = _FakeProcess(lines, returncode)
    monkeypatch.setattr(extract_frames.subprocess, "Popen", lambda *a, **k: process)
    return process


def test_progress_blocks_are_summarised_not_dumped(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Several blocks a second is noise; the summary is the useful part."""
    from soccermaster.model.data import extract_frames

    lines = [
        "frame=100\n",
        "fps=25.0\n",
        "speed=4.2x\n",
        "progress=continue\n",
        "frame=200\n",
        "speed=4.5x\n",
        "progress=end\n",
    ]
    _patch_popen(monkeypatch, lines, returncode=0)

    # report_every=0 so both blocks report, and the count is deterministic.
    code, _ = extract_frames._stream_ffmpeg(["ffmpeg"], timeout=60, report_every=0)

    out = capsys.readouterr().out
    assert code == 0
    assert "decode" in out
    assert "200" in out, "the final frame count should be reported"
    assert "frame=200" not in out, "raw key=value pairs should not be dumped"


def test_progress_shows_a_percentage_when_the_total_is_known(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from soccermaster.model.data import extract_frames

    _patch_popen(monkeypatch, ["frame=500\n", "progress=end\n"], returncode=0)
    extract_frames._stream_ffmpeg(
        ["ffmpeg"], timeout=60, expected_frames=1000, report_every=0
    )
    assert "50.0%" in capsys.readouterr().out


def test_stderr_lines_are_shown(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """stderr is merged into the same stream, so ffmpeg errors must still surface."""
    from soccermaster.model.data import extract_frames

    _patch_popen(monkeypatch, ["Error opening input file\n"], returncode=1)
    code, tail = extract_frames._stream_ffmpeg(["ffmpeg"], timeout=60)
    assert code == 1
    assert "Error opening input file" in tail
    assert "Error opening input file" in capsys.readouterr().out
