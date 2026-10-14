"""The exporter, exercised against a state built to the real schema.

Covers the export bugs directly: bulky array columns leaking into the CSV, pitch
coordinates not being expanded, and the detection files being skipped (which gave
callers a KeyError) when a run found nobody.
"""
from __future__ import annotations

import json
import pickle
import zipfile

import pytest

pd = pytest.importorskip("pandas")

from soccermaster import export  # noqa: E402


@pytest.fixture
def state(tmp_path):
    """A tracker state shaped like the real one: per-sequence detections + images."""
    rows = []
    for track_id, team, role, jersey in ((1, "left", "player", 7.0), (2, "right", "goalkeeper", 1.0)):
        for frame in range(1, 5):
            rows.append({
                "image_id": f"img{frame}",
                "video_id": "10001",
                "track_id": track_id,
                "bbox_ltwh": [10.0 * frame, 20.0, 50.0, 100.0],
                "bbox_pitch": {"x_feet": 1.5 * frame, "y_feet": 2.0},
                "bbox_conf": 0.9,
                "team": team,
                "role": role,
                "jersey_number": jersey,
                "embeddings": [0.1] * 8,  # bulky: must not reach the CSV
            })
    detections = pd.DataFrame(rows)
    images = pd.DataFrame([
        {"image_id": f"img{f}", "video_id": "10001", "frame": f,
         "file_path": f"10001/img1/{f:06d}.jpg", "parameters": {"x_focal_length": 1000.0}}
        for f in range(1, 5)
    ])

    path = tmp_path / "sn-gamestate.pklz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("summary.json", json.dumps({"columns": {
            "detection": list(detections.columns), "image": list(images.columns)}}))
        zf.writestr("10001.pkl", pickle.dumps(detections))
        zf.writestr("10001_image.pkl", pickle.dumps(images))
    return path


def test_read_state_roundtrip(state) -> None:
    detections, images = export.read_state(state)

    assert len(detections) == 8
    assert len(images) == 4
    assert "track_id" in detections.columns


def test_export_writes_all_expected_files(state, tmp_path) -> None:
    written = export.export_state(state, tmp_path / "out", prefix="SNGS-10001")

    for key in ("detections_csv", "detections_jsonl", "tracks_json", "frames_csv", "summary_json"):
        assert key in written, f"{key} was not written"
        assert written[key].is_file()


def test_csv_is_flat_and_drops_bulky_columns(state, tmp_path) -> None:
    written = export.export_state(state, tmp_path / "out", prefix="seq")
    csv = pd.read_csv(written["detections_csv"])

    # bbox_ltwh and bbox_pitch are expanded into one column per component...
    assert "bbox_ltwh_left" in csv.columns
    assert "bbox_pitch_x_feet" in csv.columns
    # ...and the array columns are gone, because they would be megabytes per cell.
    assert not any("embeddings" in c for c in csv.columns)


def test_jsonl_keeps_the_arrays(state, tmp_path) -> None:
    """The CSV drops bulky columns, so the JSONL must not -- nothing may vanish."""
    written = export.export_state(state, tmp_path / "out", prefix="seq")
    first = json.loads(written["detections_jsonl"].read_text().splitlines()[0])

    assert len(first["embeddings"]) == 8


def test_tracks_json_summarises_each_tracklet(state, tmp_path) -> None:
    written = export.export_state(state, tmp_path / "out", prefix="seq")
    tracks = json.loads(written["tracks_json"].read_text())

    assert len(tracks) == 2
    assert {t["team"] for t in tracks} == {"left", "right"}
    assert {t["jersey_number"] for t in tracks} == {7.0, 1.0}
    for track in tracks:
        assert track["n_detections"] == 4


def test_summary_records_what_was_omitted(state, tmp_path) -> None:
    written = export.export_state(state, tmp_path / "out", prefix="seq")
    summary = json.loads(written["summary_json"].read_text())

    assert summary["n_tracks"] == 2
    assert summary["n_frames"] == 4
    assert summary["omitted_from_csv"]


def test_empty_state_still_writes_detection_files(tmp_path) -> None:
    """A run that finds nobody is legitimate; callers must not get a KeyError."""
    empty = pd.DataFrame()
    path = tmp_path / "empty.pklz"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("summary.json", json.dumps({"columns": {"detection": [], "image": []}}))
        zf.writestr("10001.pkl", pickle.dumps(empty))

    written = export.export_state(path, tmp_path / "out", prefix="seq")

    assert "detections_csv" in written
    assert written["detections_csv"].is_file()
    assert "tracks_json" in written
    assert json.loads(written["tracks_json"].read_text()) == []


def test_missing_state_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        export.export_state(tmp_path / "nope.pklz", tmp_path)
