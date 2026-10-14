"""The weight manifest and the downloads module.

Covers the class of mistake made twice here: claiming a weight needs a manual
download when the pipeline fetches it itself (reid, calibration), and relying on a
filename the configs do not actually load.
"""
from __future__ import annotations

import pytest

pytest.importorskip("yaml")

from soccermaster import downloads, paths  # noqa: E402

#: Everything the shipped step-1 and step-3 configs need. If one of these disappears
#: from the manifest, a fresh install silently loses the ability to run a step.
EXPECTED = {"siglip", "soccermaster", "yolo", "qwen7b", "legibility", "reid", "calibration"}


def test_manifest_loads() -> None:
    manifest = downloads.load_manifest()
    assert manifest


def test_expected_entries_present() -> None:
    missing = EXPECTED - set(downloads.load_manifest())
    assert not missing, f"manifest is missing: {sorted(missing)}"


def test_every_entry_is_complete() -> None:
    for name, entry in downloads.load_manifest().items():
        assert entry.source in ("hf", "gdrive", "runtime", "manual"), name
        assert entry.dest, name
        assert str(entry.dest).startswith(str(paths.weights_dir())), name
        if entry.source == "hf":
            assert entry.repo, f"{name}: an hf entry needs a repo"
        if entry.source == "gdrive":
            assert entry.file_id, f"{name}: a gdrive entry needs a file_id"


def test_kind_classification() -> None:
    """`kind` drives `soccermaster prepare --list`; runtime entries are not manual."""
    manifest = downloads.load_manifest()

    assert manifest["yolo"].kind == "auto"
    assert manifest["yolo"].automatic
    # The pipeline downloads these itself on first use, so prepare must not claim
    # they are a manual step.
    assert manifest["reid"].kind == "runtime"
    assert not manifest["reid"].automatic
    assert manifest["calibration"].kind == "runtime"


def test_yolo_destination_matches_the_config() -> None:
    """The step-1 config loads a specific filename; the manifest must produce it."""
    from pathlib import Path

    expected = "yolo_v8x6_person_lr_default_best.pt"
    entry = downloads.load_manifest()["yolo"]
    assert Path(entry.dest).name == "yolo"

    # The rename map must turn whatever the hub serves into the configured name.
    served = list(entry.rename)
    assert served, "yolo needs a rename entry to match the config filename"
    assert expected in entry.rename.values(), (
        f"yolo rename targets {entry.rename} but the configs load {expected}"
    )


def test_dir_lookup_is_not_automatic_for_runtime() -> None:
    """`prepare` must refuse to fetch runtime entries rather than inventing a URL."""
    entry = downloads.load_manifest()["reid"]
    assert not entry.automatic
    assert entry.repo is None and entry.file_id is None
