"""Turn a tracker state into usable data.

The pipeline's real output is structured: every tracklet, every player, with
bounding boxes, jersey numbers, roles, teams and -- once calibration has run --
pitch-space coordinates in metres. The annotated video is one view of that; this
module is the data.

A tracker state is a zip of pickles. Per sequence it holds a detections frame
(one row per person per frame) and an image frame (one row per frame, carrying
camera parameters). ``export_state`` reads those and writes:

- ``detections.csv``   flat scalars, one row per person per frame, spreadsheet-ready
- ``detections.jsonl`` one JSON object per row, nested fields intact
- ``tracks.json``      one entry per tracklet, with its consensus team/jersey/role
- ``frames.csv``       per-frame metadata and camera parameters
- ``summary.json``     column inventory and counts, plus what was deliberately dropped

Usage::

    from soccermaster import export

    export.export_state("outputs/pipeline/step3/sn-gamestate.pklz", "outputs/data")
"""

from __future__ import annotations

import json
import pickle
import zipfile
from collections import Counter
from pathlib import Path

# Columns holding arrays/blobs. Kept in the JSONL (so nothing is lost) but dropped
# from the CSV, where they would be megabytes of useless text per cell.
BULKY_COLUMNS = (
    "embeddings",
    "body_masks",
    "visibility_scores",
    "keypoints",
    "lines",
    "masks",
    "scores",
)

#: Columns whose value is a dict of pitch coordinates, expanded to ``<name>_x`` etc.
PITCH_COLUMNS = ("bbox_pitch",)


def read_state(state: str | Path) -> tuple:
    """Read a tracker state ``.pklz`` into ``(detections, images)`` DataFrames.

    The zip holds ``<video_id>.pkl`` for detections and ``<video_id>_image.pkl`` for
    per-frame metadata, one pair per sequence, plus a ``summary.json`` describing the
    columns. Empty DataFrames are returned if a part is absent.
    """
    import pandas as pd

    state = Path(state)
    if not state.is_file():
        raise FileNotFoundError(f"no such tracker state: {state}")

    det_frames: list = []
    img_frames: list = []
    with zipfile.ZipFile(state) as zf:
        for name in zf.namelist():
            if not name.endswith(".pkl"):
                continue
            with zf.open(name) as fp:
                obj = pickle.load(fp)
            if not isinstance(obj, pd.DataFrame):
                continue
            # "<id>_image.pkl" is metadata; everything else is detections.
            (img_frames if name.endswith("_image.pkl") else det_frames).append(obj)

    detections = pd.concat(det_frames, ignore_index=True) if det_frames else pd.DataFrame()
    images = pd.concat(img_frames, ignore_index=True) if img_frames else pd.DataFrame()
    return detections, images


def _as_mapping(value):
    """Return a dict-like view of a cell that may be dict, Series, or None."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "get") and hasattr(value, "keys"):
        return value
    return None


def _scalar(value):
    """Best-effort conversion to a JSON/CSV friendly scalar; None if not possible."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple, dict)):
        return None
    # numpy scalars and friends
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return None
    return None


def _expand_column(df, column: str, keys: tuple[str, ...]) -> None:
    """Expand a dict-valued column into ``<column>_<key>`` columns, in place."""
    mapping = df[column].apply(_as_mapping)
    for key in keys:
        df[f"{column}_{key}"] = mapping.apply(
            lambda m, k=key: _scalar(m.get(k)) if m is not None else None
        )


def flatten_for_csv(detections):
    """Return a DataFrame with dict columns expanded and bulky columns removed."""
    import pandas as pd

    out = detections.copy()

    # bbox_ltwh / bbox_image are 4-element vectors: give each its own column.
    for column in ("bbox_ltwh", "bbox_image", "bbox_pitch"):
        if column not in out.columns:
            continue
        first = next((v for v in out[column] if v is not None), None)
        mapping = _as_mapping(first)
        if mapping is not None:
            _expand_column(out, column, tuple(mapping.keys()))
            out = out.drop(columns=[column])
        elif isinstance(first, (list, tuple)):
            _expand_vector_column(out, column, len(first))
            out = out.drop(columns=[column])

    for column in BULKY_COLUMNS:
        if column in out.columns:
            out = out.drop(columns=[column])

    # Anything still object-typed is unrepresentable in a CSV; stringify scalars and
    # blank the rest rather than letting pandas write repr() blobs.
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].apply(_scalar)
    return out


def _expand_vector_column(df, column: str, length: int) -> None:
    labels = ("left", "top", "width", "height") if length == 4 else tuple(
        f"{i}" for i in range(length)
    )
    for index, label in enumerate(labels):
        df[f"{column}_{label}"] = df[column].apply(
            lambda v, i=index: _scalar(v[i]) if isinstance(v, (list, tuple)) and len(v) > i else None
        )


def _mode(values):
    """Most common non-null value, or None."""
    cleaned = [_scalar(v) for v in values]
    cleaned = [v for v in cleaned if v is not None]
    if not cleaned:
        return None
    try:
        return Counter(cleaned).most_common(1)[0][0]
    except TypeError:
        return None


def build_tracks(detections) -> list[dict]:
    """One summary entry per tracklet: how long it lived and what it settled on."""
    if detections.empty or "track_id" not in detections.columns:
        return []

    frame_column = next(
        (c for c in ("frame", "image_id", "file_path") if c in detections.columns), None
    )
    tracks = []
    for track_id, group in detections.groupby("track_id"):
        entry = {
            "track_id": _scalar(track_id),
            "n_detections": int(len(group)),
        }
        if frame_column:
            frames = sorted(v for v in (_scalar(x) for x in group[frame_column]) if v is not None)
            if frames:
                entry["first_frame"] = frames[0]
                entry["last_frame"] = frames[-1]
        for source, key in (
            ("team", "team"),
            ("team_cluster", "team_cluster"),
            ("jersey_number", "jersey_number"),
            ("jersey_number_detection", "jersey_number"),
            ("role", "role"),
            ("role_detection", "role"),
            ("category", "category"),
            ("person_id", "person_id"),
        ):
            if source in group.columns:
                value = _mode(group[source])
                if value is not None and key not in entry:
                    entry[key] = value
        if "bbox_conf" in group.columns:
            confs = [_scalar(v) for v in group["bbox_conf"]]
            confs = [c for c in confs if isinstance(c, (int, float))]
            if confs:
                entry["bbox_conf_mean"] = round(sum(confs) / len(confs), 4)
        tracks.append(entry)

    tracks.sort(key=lambda t: (t.get("first_frame") is None, t.get("first_frame"), t["track_id"]))
    return tracks


def _records(df) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts, preserving nested values."""
    import math

    records = []
    for row in df.to_dict(orient="records"):
        record = {}
        for key, value in row.items():
            if isinstance(value, dict):
                record[key] = {k: _scalar(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                record[key] = [_scalar(v) for v in value]
            elif isinstance(value, float) and math.isnan(value):
                record[key] = None
            else:
                record[key] = _scalar(value) if not isinstance(value, (str, bool, int, float)) else value
        records.append(record)
    return records


def export_state(
    state: str | Path,
    out_dir: str | Path,
    prefix: str | None = None,
    write_jsonl: bool = True,
) -> dict:
    """Export one tracker state to CSV/JSON. Returns a dict of paths written."""
    import pandas as pd

    state = Path(state)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or state.parent.name
    written: dict[str, Path] = {}

    detections, images = read_state(state)

    dropped_from_csv = [c for c in detections.columns if c in BULKY_COLUMNS]

    if not detections.empty:
        flat = flatten_for_csv(detections)
        csv_path = out_dir / f"{prefix}_detections.csv"
        flat.to_csv(csv_path, index=False)
        written["detections_csv"] = csv_path

        if write_jsonl:
            jsonl_path = out_dir / f"{prefix}_detections.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as fp:
                for record in _records(detections):
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            written["detections_jsonl"] = jsonl_path

        tracks = build_tracks(detections)
        tracks_path = out_dir / f"{prefix}_tracks.json"
        tracks_path.write_text(
            json.dumps(tracks, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written["tracks_json"] = tracks_path

    if not images.empty:
        flat_images = flatten_for_csv(images)
        frames_path = out_dir / f"{prefix}_frames.csv"
        flat_images.to_csv(frames_path, index=False)
        written["frames_csv"] = frames_path

    summary = {
        "source_state": str(state),
        "n_detections": int(len(detections)),
        "n_frames": int(len(images)),
        "n_tracks": int(detections["track_id"].nunique()) if "track_id" in detections.columns else 0,
        "detection_columns": list(detections.columns),
        "image_columns": list(images.columns),
        "omitted_from_csv": dropped_from_csv,
        "note": (
            "Array-valued columns are omitted from the CSV but present in the JSONL. "
            "bbox_ltwh_* is image pixels; bbox_pitch_* is pitch metres, available only "
            "after calibration."
        ),
    }
    summary_path = out_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    written["summary_json"] = summary_path

    return written
