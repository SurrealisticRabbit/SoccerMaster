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
    def __init__(self, payload: bytes, fail_after: int | None = None) -> None:
        self._payload = payload
        self._fail_after = fail_after
        self.headers = {"content-length": str(len(payload))}

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self._payload), chunk_size):
            if self._fail_after is not None and start >= self._fail_after:
                raise ConnectionError("connection reset mid-download")
            yield self._payload[start : start + chunk_size]


def _patch_get(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> list[str]:
    calls: list[str] = []

    def fake_get(url, stream=False, timeout=None):  # noqa: ANN001, ARG001
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


def test_interrupted_download_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-written file must never be visible at the final path."""
    target = tmp_path / "w.pth"
    _patch_get(monkeypatch, _FakeResponse(GOOD, fail_after=len(GOOD) // 2))

    with pytest.raises(ConnectionError):
        download.download_file("http://x/w.pth", str(target), md5=GOOD_MD5)

    assert not target.exists(), "the real path must be untouched"
    assert not (tmp_path / "w.pth.part").exists(), "the partial file must be cleaned up"


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
