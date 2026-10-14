"""The weight cache must heal itself instead of poisoning the volume.

A download interrupted part-way used to leave a truncated file, and the md5 check then
*raised* rather than refetching. Because the cache is a persistent volume, one aborted run
broke every subsequent one: the same file failed the same check forever, and the only
recovery was knowing which file to delete by hand. That is what happened in practice --
a killed run left a partial `prtreid-soccernet-baseline.pth.tar` and the next run died on
"MD5 checksum mismatch for file ..." before the pipeline had started.

Downloads now land in `<name>.part` and are moved into place only once verified, so a file
at the final path is either complete or absent.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

download = pytest.importorskip("soccermaster.tracklab.utils.download")

GOOD = b"pretrained weights" * 1000
GOOD_MD5 = hashlib.md5(GOOD).hexdigest()


class _FakeResponse:
    #: The fake streams in its own small chunks regardless of the chunk_size the caller
    #: asks for, so an interruption can happen mid-payload. The production chunk size is
    #: 1 MiB, which would otherwise make a short fixture arrive in a single chunk and
    #: leave fail_after with nothing to interrupt.
    _chunk = 1024

    def __init__(self, payload: bytes, fail_after: int | None = None) -> None:
        self._payload = payload
        self._fail_after = fail_after
        self.headers = {"content-length": str(len(payload))}
        self.status_code = 200

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self._payload), self._chunk):
            if self._fail_after is not None and start >= self._fail_after:
                raise ConnectionError("connection reset mid-download")
            yield self._payload[start : start + self._chunk]


def _patch_get(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> list[str]:
    calls: list[str] = []

    def fake_get(url, stream=False, timeout=None, headers=None):  # noqa: ANN001, ARG001
        calls.append(url)
        return response

    monkeypatch.setattr(download.requests, "get", fake_get)
    return calls


def test_existing_valid_file_is_reused_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "w.pth"
    target.write_bytes(GOOD)
    calls = _patch_get(monkeypatch, _FakeResponse(b"should not be fetched"))

    assert download.download_file("http://x/w.pth", str(target), md5=GOOD_MD5) == str(target)
    assert calls == [], "a valid cached file must not be re-downloaded"


def test_corrupt_existing_file_is_refetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression: a mismatch on an existing file used to raise and poison the cache."""
    target = tmp_path / "w.pth"
    target.write_bytes(b"truncated garbage")
    _patch_get(monkeypatch, _FakeResponse(GOOD))

    result = download.download_file("http://x/w.pth", str(target), md5=GOOD_MD5)

    assert result == str(target)
    assert target.read_bytes() == GOOD
    assert download.check_md5(target, GOOD_MD5), "the file should now be valid"


def test_interrupted_download_keeps_the_partial_for_resuming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real path stays clean, and the partial survives so the retry can resume.

    Discarding the partial on a network error would throw away however many slow
    megabytes already arrived -- and on a link that drops repeatedly, that is the
    difference between finishing and never finishing. ``attempts=1`` so this asserts one
    clean failure rather than exercising the retry loop.
    """
    target = tmp_path / "w.pth"
    _patch_get(monkeypatch, _FakeResponse(GOOD, fail_after=len(GOOD) // 2))

    with pytest.raises(ConnectionError):
        download.download_file("http://x/w.pth", str(target), md5=GOOD_MD5, attempts=1)

    assert not target.exists(), "the real path must be untouched"
    assert (tmp_path / "w.pth.part").exists(), "the partial must be kept for resume"


def test_md5_mismatch_on_a_fresh_download_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "w.pth"
    _patch_get(monkeypatch, _FakeResponse(b"not what was promised"))

    with pytest.raises(ValueError, match="MD5 checksum mismatch"):
        download.download_file("http://x/w.pth", str(target), md5=GOOD_MD5)

    assert not target.exists()
    assert not (tmp_path / "w.pth.part").exists()


def test_success_leaves_no_part_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "w.pth"
    _patch_get(monkeypatch, _FakeResponse(GOOD))

    download.download_file("http://x/w.pth", str(target), md5=GOOD_MD5)

    assert target.exists()
    assert not (tmp_path / "w.pth.part").exists()


def test_download_without_md5_still_lands_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "w.bin"
    _patch_get(monkeypatch, _FakeResponse(GOOD))

    download.download_file("http://x/w.bin", str(target))

    assert target.read_bytes() == GOOD


def test_existing_file_without_md5_is_reused(tmp_path: Path) -> None:
    """No md5 means nothing to verify, so the cached file is trusted."""
    target = tmp_path / "w.bin"
    target.write_bytes(b"whatever")
    assert download.download_file("http://x/w.bin", str(target)) == str(target)


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #


class _RangedResponse(_FakeResponse):
    """Answers a ranged request the way Zenodo does: 206 with just the remainder."""

    def __init__(self, payload: bytes, start: int) -> None:
        super().__init__(payload[start:])
        self.status_code = 206


class _IgnoresRangeResponse(_FakeResponse):
    """A server that answers a ranged request with the whole body and status 200."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.status_code = 200


def _patch_get_status(monkeypatch: pytest.MonkeyPatch, factory) -> list[dict]:
    """Capture the request headers so the Range header can be asserted."""
    seen: list[dict] = []

    def fake_get(url, stream=False, timeout=None, headers=None):  # noqa: ANN001, ARG001
        seen.append(headers or {})
        start = 0
        if headers and headers.get("Range"):
            start = int(headers["Range"].split("=")[1].rstrip("-"))
        return factory(start)

    monkeypatch.setattr(download.requests, "get", fake_get)
    return seen


def test_partial_download_is_resumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A slow link must not cost the whole transfer again on every retry."""
    target = tmp_path / "w.pth"
    already = len(GOOD) // 3
    (tmp_path / "w.pth.part").write_bytes(GOOD[:already])

    seen = _patch_get_status(monkeypatch, lambda start: _RangedResponse(GOOD, start))

    result = download.download_file("http://x/w.pth", str(target), md5=GOOD_MD5)

    assert seen[0].get("Range") == f"bytes={already}-", "should ask for the remainder only"
    assert result == str(target)
    assert target.read_bytes() == GOOD
    assert download.check_md5(target, GOOD_MD5)


def test_resume_survives_a_restart_midway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The realistic case: killed at 40%, resumed, completes.

    The resumed hash must cover the whole file, not just the appended part -- otherwise
    the md5 check fails on a file that is byte-for-byte correct.
    """
    target = tmp_path / "w.pth"
    cut = int(len(GOOD) * 0.4)
    (tmp_path / "w.pth.part").write_bytes(GOOD[:cut])

    _patch_get_status(monkeypatch, lambda start: _RangedResponse(GOOD, start))
    download.download_file("http://x/w.pth", str(target), md5=GOOD_MD5)

    assert target.read_bytes() == GOOD


def test_server_ignoring_the_range_restarts_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Appending a full-body 200 response would corrupt the file; it must restart."""
    target = tmp_path / "w.pth"
    (tmp_path / "w.pth.part").write_bytes(b"stale bytes from a previous attempt")

    _patch_get_status(monkeypatch, lambda start: _IgnoresRangeResponse(GOOD))
    download.download_file("http://x/w.pth", str(target), md5=GOOD_MD5)

    assert target.read_bytes() == GOOD, "the stale prefix must not survive"
    assert download.check_md5(target, GOOD_MD5)


def test_resume_can_be_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "w.pth"
    (tmp_path / "w.pth.part").write_bytes(GOOD[:50])

    seen = _patch_get_status(monkeypatch, lambda start: _FakeResponse(GOOD))
    download.download_file("http://x/w.pth", str(target), md5=GOOD_MD5, resume=False)

    assert seen[0] == {}, "resume=False should not send a Range header"
    assert download.check_md5(target, GOOD_MD5)


def test_network_failure_keeps_the_partial_for_resuming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discarding a partial on a network error would throw away slow progress."""
    target = tmp_path / "w.pth"
    _patch_get(monkeypatch, _FakeResponse(GOOD, fail_after=len(GOOD) // 2))

    with pytest.raises(ConnectionError):
        download.download_file("http://x/w.pth", str(target), md5=GOOD_MD5, attempts=1)

    assert not target.exists(), "the real path must stay untouched"
    assert (tmp_path / "w.pth.part").exists(), "the partial should be kept for resume"


class _FlakyGet:
    """Fails the first ``fail_times`` calls, then behaves like Zenodo."""

    def __init__(self, payload: bytes, fail_times: int) -> None:
        self._payload = payload
        self._fail_times = fail_times
        self.calls = 0
        self.ranges: list[str | None] = []

    def __call__(self, url, stream=False, timeout=None, headers=None):  # noqa: ANN001, ARG001
        self.calls += 1
        rng = (headers or {}).get("Range")
        self.ranges.append(rng)
        if self.calls <= self._fail_times:
            raise ConnectionError("connection broken mid-transfer")
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        return _RangedResponse(self._payload, start)


def test_a_dropped_connection_is_retried_and_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live failure: a break at ~28 MB of 396 MB must not lose the 28 MB.

    Retries resume, so progress accumulates rather than restarting -- without this the
    download never completes on a link that drops, which is what happened.
    """
    target = tmp_path / "w.pth"
    cut = len(GOOD) // 3
    (tmp_path / "w.pth.part").write_bytes(GOOD[:cut])

    flaky = _FlakyGet(GOOD, fail_times=2)
    monkeypatch.setattr(download.requests, "get", flaky)

    result = download.download_file(
        "http://x/w.pth", str(target), md5=GOOD_MD5, retry_backoff=0
    )

    assert flaky.calls == 3, "two failures then a success"
    assert flaky.ranges[0] == f"bytes={cut}-", "the first attempt resumes"
    assert result == str(target)
    assert target.read_bytes() == GOOD
    assert download.check_md5(target, GOOD_MD5)


def test_dropped_connection_without_a_partial_still_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "w.pth"
    flaky = _FlakyGet(GOOD, fail_times=1)
    monkeypatch.setattr(download.requests, "get", flaky)

    download.download_file("http://x/w.pth", str(target), md5=GOOD_MD5, retry_backoff=0)

    assert flaky.calls == 2
    assert download.check_md5(target, GOOD_MD5)


def test_giving_up_after_all_attempts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "w.pth"
    flaky = _FlakyGet(GOOD, fail_times=99)
    monkeypatch.setattr(download.requests, "get", flaky)

    with pytest.raises(ConnectionError):
        download.download_file(
            "http://x/w.pth", str(target), md5=GOOD_MD5, attempts=3, retry_backoff=0
        )

    assert flaky.calls == 3, "should stop after the configured attempts"
    assert not target.exists()


def test_a_corrupt_partial_is_refetched_not_appended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong prefix must not survive: resuming onto it fails md5 forever."""
    target = tmp_path / "w.pth"
    (tmp_path / "w.pth.part").write_bytes(b"garbage prefix" + GOOD[:100])

    _patch_get_status(monkeypatch, lambda start: _RangedResponse(GOOD, start))
    download.download_file("http://x/w.pth", str(target), md5=GOOD_MD5, retry_backoff=0)

    assert download.check_md5(target, GOOD_MD5)


def test_chunk_size_is_not_tiny() -> None:
    """8 KiB meant ~48,000 iterations and syscalls for one 378 MiB checkpoint."""
    assert download.CHUNK_SIZE >= 1 << 20
