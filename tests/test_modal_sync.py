"""The outputs sync must be incremental, and idempotent.

It replaced a call to ``modal volume get <volume> / <dir>``, which failed outright the
second time it ran: the CLI refuses an existing output directory, and its documented
escape hatch (``--force``) re-downloads everything. On a tree containing gigabytes of
annotated video neither behaviour is acceptable -- either nothing is fetched or all of it
is. The realistic sequence, running fetch after each of several jobs, did the first.

These drive the real function against a fake volume, so the comparison logic, the skip and
the self-healing path are all covered without contacting a real account.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


def _load_modal_app(monkeypatch: pytest.MonkeyPatch):
    """Import modal_app with a stub `modal`, returning the module."""
    fake = types.ModuleType("modal")

    class _Any:
        def __init__(self, *a, **k):  # noqa: ANN002, ANN003
            pass

        def __call__(self, *a, **k):  # noqa: ANN002, ANN003
            return self

        def __getattr__(self, _name):  # noqa: ANN001, ANN204
            return _Any()

    class _App:
        def __init__(self, *a, **k):  # noqa: ANN002, ANN003
            pass

        def function(self, *a, **k):  # noqa: ANN002, ANN003
            return lambda fn: fn

        def local_entrypoint(self, *a, **k):  # noqa: ANN002, ANN003
            return lambda fn: fn

    fake.App = _App
    fake.Image = _Any()
    fake.Volume = _Any()
    monkeypatch.setitem(sys.modules, "modal", fake)

    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("_modal_app_under_test", root / "modal_app.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Kind:
    """Stands in for ``modal.types.FileEntryType``.

    Mirrors the real thing's *shape* -- an enum whose members have ``.name`` -- which
    matters more than it looks. The first version of this fake used the plain string
    ``"file"``, matching the production code's wrong assumption that ``entry.type`` was a
    string. The tests therefore passed while the real sync selected zero files and
    silently copied nothing, because ``FileEntryType.FILE != "file"``. A fake that encodes
    the same mistake as the code cannot catch it.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<FileEntryType.{self.name}>"


FILE = _Kind("FILE")
DIRECTORY = _Kind("DIRECTORY")


class _Entry:
    """Mirrors modal.types.FileEntry: path, type, mtime, size."""

    def __init__(self, path: str, size: int, kind: _Kind = FILE) -> None:
        self.path = path
        self.type = kind
        self.mtime = 0
        self.size = size


class _FakeVolume:
    """A volume whose files are held in memory, so no account is touched."""

    def __init__(self, files: dict[str, bytes], *, fail: set[str] | None = None) -> None:
        self.files = files
        self.fail = fail or set()
        self.reads: list[str] = []

    def listdir(self, path: str, recursive: bool = False) -> list[_Entry]:
        assert path == "/"
        return [_Entry(f"/{name}", len(body)) for name, body in self.files.items()]

    def read_file_into_fileobj(self, path: str, fileobj) -> int:  # noqa: ANN001
        name = path.lstrip("/")
        self.reads.append(name)
        if name in self.fail:
            raise RuntimeError("simulated read failure")
        body = self.files[name]
        fileobj.write(body)
        return len(body)


@pytest.fixture()
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_modal_app(monkeypatch)
    monkeypatch.setattr(module, "LOCAL_OUTPUT_DIR", tmp_path)
    return module


def _volume(module, files: dict[str, bytes], **kwargs) -> _FakeVolume:
    volume = _FakeVolume(files, **kwargs)
    module.__dict__["outputs_volume"] = volume
    return volume


def test_first_sync_fetches_everything(app, tmp_path: Path) -> None:
    _volume(app, {"modal/data/a.csv": b"a" * 10, "modal/data/b.json": b"{}"})

    report = app.sync_outputs()

    assert sorted(report["fetched"]) == ["modal/data/a.csv", "modal/data/b.json"]
    assert report["skipped"] == []
    assert (tmp_path / "modal/data/a.csv").read_bytes() == b"a" * 10


def test_second_sync_fetches_nothing(app, tmp_path: Path) -> None:
    """The reported bug: the second fetch used to fail outright rather than no-op."""
    _volume(app, {"modal/data/a.csv": b"a" * 10})
    app.sync_outputs()

    volume = _volume(app, {"modal/data/a.csv": b"a" * 10})
    report = app.sync_outputs()

    assert report["fetched"] == []
    assert report["skipped"] == ["modal/data/a.csv"]
    assert volume.reads == [], "an up-to-date file must not be transferred again"
    assert (tmp_path / "modal/data/a.csv").read_bytes() == b"a" * 10


def test_only_changed_files_are_refetched(app, tmp_path: Path) -> None:
    """Adding one output must cost one file, not the whole tree."""
    _volume(app, {"modal/data/a.csv": b"a" * 10, "modal/data/b.json": b"{}"})
    app.sync_outputs()

    # A new file appears remotely, everything else is unchanged.
    volume = _volume(
        app,
        {
            "modal/data/a.csv": b"a" * 10,
            "modal/data/b.json": b"{}",
            "modal/data/c.json": b'{"new": true}',
        },
    )
    report = app.sync_outputs()

    assert report["fetched"] == ["modal/data/c.json"]
    assert sorted(report["skipped"]) == ["modal/data/a.csv", "modal/data/b.json"]
    assert volume.reads == ["modal/data/c.json"]


def test_a_truncated_local_file_is_refetched(app, tmp_path: Path) -> None:
    """Size is the comparison precisely because a truncated download looks like this."""
    partial = tmp_path / "modal/data/a.csv"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"a" * 4)  # short

    volume = _volume(app, {"modal/data/a.csv": b"a" * 100})
    report = app.sync_outputs()

    assert report["fetched"] == ["modal/data/a.csv"]
    assert volume.reads == ["modal/data/a.csv"]
    assert partial.read_bytes() == b"a" * 100


def test_path_scopes_the_sync(app, tmp_path: Path) -> None:
    """`--path modal/data` fetches the small exports and skips the gigabytes of video."""
    volume = _volume(
        app,
        {
            "modal/data/a.csv": b"a" * 10,
            "modal/step1/visualization/videos/10005.mp4": b"v" * 5000,
        },
    )

    report = app.sync_outputs(sub_path="modal/data")

    assert report["fetched"] == ["modal/data/a.csv"]
    assert volume.reads == ["modal/data/a.csv"]
    assert not (tmp_path / "modal/step1").exists()


def test_force_refetches_everything(app, tmp_path: Path) -> None:
    _volume(app, {"modal/data/a.csv": b"a" * 10})
    app.sync_outputs()

    volume = _volume(app, {"modal/data/a.csv": b"a" * 10})
    report = app.sync_outputs(force=True)

    assert report["fetched"] == ["modal/data/a.csv"]
    assert report["skipped"] == []
    assert volume.reads == ["modal/data/a.csv"]


def test_a_failed_file_does_not_stop_the_sync(app, tmp_path: Path) -> None:
    """One unreadable file should not cost the rest of the transfer."""
    volume = _volume(
        app,
        {"modal/data/a.csv": b"a" * 10, "modal/data/b.csv": b"b" * 10},
        fail={"modal/data/b.csv"},
    )

    report = app.sync_outputs()

    assert report["fetched"] == ["modal/data/a.csv"]
    assert len(report["failed"]) == 1
    assert "b.csv" in report["failed"][0]
    assert (tmp_path / "modal/data/a.csv").exists()


def test_nested_directories_are_created(app, tmp_path: Path) -> None:
    _volume(app, {"modal/dataset/sn500/SNGS-10005/img1/000001.jpg": b"jpeg"})
    app.sync_outputs()
    assert (tmp_path / "modal/dataset/sn500/SNGS-10005/img1/000001.jpg").read_bytes() == b"jpeg"


def test_directory_entries_are_ignored(app, tmp_path: Path) -> None:
    """listdir(recursive=True) returns directories too; they must not be opened."""
    volume = _FakeVolume({"modal/data/a.csv": b"a" * 10})
    volume.listdir = lambda path, recursive=False: [
        _Entry("/modal/data", 0, kind=DIRECTORY),
        _Entry("/modal/data/a.csv", 10),
    ]
    app.__dict__["outputs_volume"] = volume

    report = app.sync_outputs()

    assert report["fetched"] == ["modal/data/a.csv"]
    assert volume.reads == ["modal/data/a.csv"]


def test_an_enum_valued_type_is_treated_as_a_file(app, tmp_path: Path) -> None:
    """The regression, and the reason the fake had to stop using the string 'file'.

    ``entry.type`` is a ``FileEntryType`` enum, so ``type != "file"`` is always true and
    every file is skipped -- the sync reported "remote: 0 file(s)" and copied nothing
    while looking like it had worked.
    """
    volume = _volume(app, {"modal/data/a.csv": b"a" * 10})
    # Prove the fake really is enum-shaped, so this test cannot pass for the wrong reason.
    assert _entry_kind_of(volume) != "file"
    assert _entry_kind_of(volume) == "FILE"

    report = app.sync_outputs()

    assert report["fetched"] == ["modal/data/a.csv"], (
        "an enum-valued type must be recognised as a file"
    )


def _entry_kind_of(volume: _FakeVolume) -> str:
    """What the production helper sees for this volume's first entry."""
    entry = volume.listdir("/", recursive=True)[0]
    return str(getattr(entry.type, "name", entry.type) or "").upper()


def test_a_string_valued_type_still_works(app, tmp_path: Path) -> None:
    """Tolerate the field being a plain string, in case the API changes shape."""
    entry = _Entry("/modal/data/a.csv", 10)
    entry.type = "file"  # a bare string, as an older or different API might give
    volume = _FakeVolume({"modal/data/a.csv": b"a" * 10})
    volume.listdir = lambda path, recursive=False: [entry]
    app.__dict__["outputs_volume"] = volume

    report = app.sync_outputs()

    assert report["fetched"] == ["modal/data/a.csv"]


def test_an_empty_selection_is_reported_and_fetches_nothing(
    app, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An empty volume must not look like a successful sync of nothing."""
    _volume(app, {"modal/data/a.csv": b"a" * 10})

    report = app.sync_outputs(sub_path="modal/nonexistent")

    assert report["fetched"] == []
    assert report["transferred_mb"] == 0.0
    out = capsys.readouterr().out
    assert "nothing to fetch" in out


def test_transferred_bytes_are_reported(app, tmp_path: Path) -> None:
    _volume(app, {"modal/data/a.csv": b"a" * 2_000_000})
    report = app.sync_outputs()
    assert report["transferred_mb"] == 2.0
