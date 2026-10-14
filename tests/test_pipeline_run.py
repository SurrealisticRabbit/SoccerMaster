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
