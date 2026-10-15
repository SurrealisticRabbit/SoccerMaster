"""`_run` must stream a step's output live, and still explain failures.

It used ``subprocess.run(..., capture_output=True)``, which held the whole step's output
in memory and echoed it only after the process exited. A twenty-minute step therefore
printed nothing for twenty minutes -- the progress bar and ``Tracking videos 1/1`` all
appeared at the very end. In a notebook that was survivable because you can look for
other signs of life; in a log-only environment it makes a working run indistinguishable
from a hung one, which is exactly how it presented in practice.

The obvious fix is wrong, which is why the streaming test is timing-based: switching to
``Popen`` but still calling ``read(n)`` reproduces the bug, because ``read`` on a pipe
blocks until it has n characters or hits EOF. Measured against a child printing every
0.7 s for 2.1 s, that delivered all three lines at the end. ``read1`` is what streams.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

pipeline = pytest.importorskip("soccermaster.pipeline")

_run = pipeline._run


class _Recorder:
    """Captures what ``_run`` prints, from another thread, without pytest's help."""

    def __init__(self) -> None:
        self.text = ""
        self._lock = threading.Lock()

    def write(self, data: str) -> None:
        with self._lock:
            self.text += data

    def flush(self) -> None:
        pass

    def seen(self, needle: str) -> bool:
        with self._lock:
            return needle in self.text


def _child(source: str) -> list[str]:
    return [sys.executable, "-c", source]


def test_streams_output_before_the_child_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression: output must arrive while the step is still running.

    Timing-based on purpose. Any test that only asserts "the output eventually appeared"
    passes with the buffered implementation too, which is how this survived so long.
    """
    recorder = _Recorder()
    monkeypatch.setattr(sys, "stdout", recorder)

    worker = threading.Thread(
        target=_run,
        args=(_child("import time; print('FIRST', flush=True); time.sleep(2.0); print('SECOND', flush=True)"), Path(".")),
        daemon=True,
    )
    worker.start()

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if recorder.seen("FIRST"):
            break
        time.sleep(0.02)

    assert recorder.seen("FIRST"), "no output within 1s of a child that printed immediately"
    assert worker.is_alive(), (
        "the child had already finished, so its output was buffered rather than streamed"
    )

    worker.join(timeout=15)
    assert not worker.is_alive(), "the step did not finish"


def test_child_runs_unbuffered(monkeypatch: pytest.MonkeyPatch) -> None:
    """PYTHONUNBUFFERED, so the child's own prints are not held in an 8 KB block buffer."""
    recorder = _Recorder()
    monkeypatch.setattr(sys, "stdout", recorder)
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)

    _run(_child("import os; print('V=' + os.environ.get('PYTHONUNBUFFERED', 'unset'))"), Path("."))

    assert "V=1" in recorder.text


def test_stdout_and_stderr_both_reach_the_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    """stderr is merged into the single pipe, so step logs are not swallowed."""
    recorder = _Recorder()
    monkeypatch.setattr(sys, "stdout", recorder)

    _run(
        _child("import sys; print('to-stdout'); print('to-stderr', file=sys.stderr)"),
        Path("."),
    )

    assert "to-stdout" in recorder.text
    assert "to-stderr" in recorder.text


def test_large_output_does_not_deadlock(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single pipe drained by the reader, so neither stream can fill and block."""
    recorder = _Recorder()
    monkeypatch.setattr(sys, "stdout", recorder)

    source = (
        "import sys;"
        "sys.stdout.write('a' * 300_000); sys.stdout.flush();"
        "sys.stderr.write('b' * 300_000); sys.stderr.flush();"
        "print('DONE')"
    )
    _run(_child(source), Path("."))

    assert "DONE" in recorder.text


def test_failure_raises_with_the_tail_of_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing step must still say why, which is the other half of what _run is for."""
    monkeypatch.setattr(sys, "stdout", _Recorder())

    with pytest.raises(RuntimeError) as excinfo:
        _run(
            _child("import sys; [print(f'line{i}') for i in range(60)]; sys.exit(3)"),
            Path("."),
        )

    message = str(excinfo.value)
    assert "exit 3" in message
    assert "--- last output ---" in message
    assert "line59" in message, "the tail should include the last line"
    assert "line0" not in message, "the tail should be bounded, not the whole log"


def test_successful_step_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stdout", _Recorder())
    assert _run(_child("print('ok')"), Path(".")) is None


# --------------------------------------------------------------------------- #
# heartbeat
# --------------------------------------------------------------------------- #


def test_heartbeat_reports_while_the_child_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A step can be legitimately silent for minutes; silence must not look like a hang.

    The interval is shrunk rather than the child being made to wait a minute -- hence
    ``HEARTBEAT_SECONDS`` being module-level.
    """
    recorder = _Recorder()
    monkeypatch.setattr(sys, "stdout", recorder)
    monkeypatch.setattr(pipeline, "HEARTBEAT_SECONDS", 0.5)

    _run(_child("import time; time.sleep(2.0)"), Path("."))

    assert "still running" in recorder.text, "a silent step produced no heartbeat"
    assert "no output for" in recorder.text


def test_heartbeat_stays_quiet_when_the_child_keeps_talking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chatty step must not be told it has gone quiet.

    Counting ticks instead of measuring time since the last output gets this wrong: the
    counter only ever rises, so continuous output still triggers the warning.
    """
    recorder = _Recorder()
    monkeypatch.setattr(sys, "stdout", recorder)
    monkeypatch.setattr(pipeline, "HEARTBEAT_SECONDS", 0.5)

    _run(
        _child(
            "import time\n"
            "for i in range(20):\n"
            "    print(f'working {i}', flush=True)\n"
            "    time.sleep(0.15)\n"
        ),
        Path("."),
    )

    assert "working 19" in recorder.text
    assert "still running" not in recorder.text, (
        "the heartbeat fired even though the child never stopped printing"
    )


def test_step_duration_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(sys, "stdout", recorder)

    _run(_child("import time; time.sleep(0.4); print('done')"), Path("."))

    assert "step finished in" in recorder.text


# --------------------------------------------------------------------------- #
# device plumbing
# --------------------------------------------------------------------------- #
#
# `device` means two different things in the pipeline -- which torch device the child
# should use, and which GPU indices to expose -- and passing it straight through conflated
# them. The default value is "cuda", which is not a device-index list, so all three steps
# were setting CUDA_VISIBLE_DEVICES=cuda for their children. That only ever worked because
# the Modal app happens to pass "0"; any caller using the default was contradicting itself.


@pytest.mark.parametrize("device", ["cuda", "cpu", None, ""])
def test_cuda_and_cpu_do_not_become_a_visible_devices_value(device: str | None) -> None:
    assert pipeline._device_env(device) is None, (
        f"device={device!r} should leave CUDA_VISIBLE_DEVICES unset, not set it to a "
        f"torch device name that is not an index list"
    )


@pytest.mark.parametrize("device", ["0", "0,1", "1", "0,1,2,3"])
def test_index_lists_are_passed_through(device: str) -> None:
    assert pipeline._device_env(device) == {"CUDA_VISIBLE_DEVICES": device}


@pytest.mark.parametrize("device", ["cuda", "cpu", None, ""])
def test_gpu_list_falls_back_to_zero_for_non_index_devices(device: str | None) -> None:
    """Step 2 does `int(x)` per element, so 'cuda' was a hard crash on its default path."""
    assert pipeline._gpu_index_list(device) == "0"


@pytest.mark.parametrize("device", ["0", "0,1", "0,1,2,3"])
def test_gpu_list_passes_indices_through(device: str) -> None:
    assert pipeline._gpu_index_list(device) == device


def test_gpu_list_is_always_parseable_as_ints() -> None:
    """The property that matters: sam2_refine's int() must never see a device name."""
    for device in ("cuda", "cpu", None, "0", "0,1"):
        value = pipeline._gpu_index_list(device)
        assert [int(part) for part in value.split(",")], f"unparseable for {device!r}"
