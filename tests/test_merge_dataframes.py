"""`merge_dataframes` must fold detections without forcing them to a numeric dtype.

The original implementation seeded new columns and rows with ``np.nan`` and then called
``DataFrame.update``. Seeding with NaN makes those columns ``float64``, and ``update``
refuses to write a string into a float column:

    TypeError: Invalid value '['610004000001']' for dtype 'float64'

That is not hypothetical. It is what killed the first run against real footage: the
tracker emits string track ids alongside numeric boxes, and this function runs on every
batch of every video. Older pandas upcast silently, pandas 3 raises, and the project pins
only ``pandas>=2.0``, so any modern install hits it.

These tests exercise the real function rather than a copy, so the semantics the tracker
depends on are pinned: appended values win, NaN in the appended piece leaves the
accumulated frame alone, and new rows and columns are still created.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from soccermaster.tracklab.engine.engine import merge_dataframes  # noqa: E402


def test_string_track_ids_merge_into_a_numeric_frame() -> None:
    """The regression. A batch carrying ids must not raise, and must keep them as strings."""
    main = pd.DataFrame({"x": [1.0, 2.0]}, index=[0, 1])
    piece = pd.DataFrame({"x": [9.0], "track_id": ["610004000001"]}, index=[1])

    out = merge_dataframes(main, piece)

    assert out.loc[1, "track_id"] == "610004000001"
    assert out.loc[1, "x"] == 9.0
    assert out.loc[0, "x"] == 1.0
    # The id must not have been coerced to a float -- it would still round-trip for a
    # short id and silently corrupt longer ones.
    assert not pd.api.types.is_numeric_dtype(out["track_id"])


def test_appended_values_override() -> None:
    main = pd.DataFrame({"x": [1.0], "y": [1.0]}, index=[0])
    piece = pd.DataFrame({"x": [9.0], "y": [8.0]}, index=[0])

    out = merge_dataframes(main, piece)

    assert out.loc[0, "x"] == 9.0
    assert out.loc[0, "y"] == 8.0


def test_nan_in_the_appended_piece_does_not_clobber() -> None:
    """`update` skips NA in the other frame, and callers rely on that."""
    main = pd.DataFrame({"x": [1.0], "y": [2.0]}, index=[0])
    piece = pd.DataFrame({"x": [np.nan]}, index=[0])

    out = merge_dataframes(main, piece)

    assert out.loc[0, "x"] == 1.0, "NA in the appended piece must leave the value alone"
    assert out.loc[0, "y"] == 2.0


def test_new_rows_are_added() -> None:
    main = pd.DataFrame({"x": [1.0]}, index=[0])
    piece = pd.DataFrame({"x": [5.0, 6.0]}, index=[7, 8])

    out = merge_dataframes(main, piece)

    assert sorted(out.index) == [0, 7, 8]
    assert out.loc[7, "x"] == 5.0
    assert out.loc[8, "x"] == 6.0


def test_new_columns_are_added_and_padded_with_na() -> None:
    main = pd.DataFrame({"x": [1.0]}, index=[0])
    piece = pd.DataFrame({"x": [2.0], "extra": ["s"]}, index=[0])

    out = merge_dataframes(main, piece)

    assert "extra" in out.columns
    assert out.loc[0, "extra"] == "s"
    assert out["extra"].isna().sum() == 0


def test_column_order_is_mains_first_then_new() -> None:
    """The NaN-seeding version preserved main's order; so must this."""
    main = pd.DataFrame({"a": [1], "b": [2]})
    piece = pd.DataFrame({"c": [3], "a": [9]})

    assert list(merge_dataframes(main, piece).columns) == ["a", "b", "c"]


def test_merging_a_series_row_works() -> None:
    """Row-at-a-time callers pass a Series, which becomes a one-row frame."""
    main = pd.DataFrame({"x": [1.0], "track_id": ["a"]}, index=[0])
    piece = main.loc[0]

    out = merge_dataframes(main, piece)

    assert out.loc[0, "track_id"] == "a"
    assert out.loc[0, "x"] == 1.0


def test_merging_a_list_of_series_works() -> None:
    main = pd.DataFrame({"x": [1.0]}, index=[0])
    rows = [pd.Series({"x": 2.0}, name=1), pd.Series({"x": 3.0}, name=2)]

    out = merge_dataframes(main, rows)

    assert sorted(out.index) == [0, 1, 2]


def test_empty_appended_piece_is_a_no_op() -> None:
    main = pd.DataFrame({"x": [1.0]}, index=[0])

    out = merge_dataframes(main, pd.DataFrame())

    assert out.loc[0, "x"] == 1.0


def test_empty_appended_list_is_a_no_op() -> None:
    main = pd.DataFrame({"x": [1.0]}, index=[0])
    assert merge_dataframes(main, []).loc[0, "x"] == 1.0


def test_mixed_dtypes_survive_repeated_merges() -> None:
    """The tracker merges once per batch, so any dtype drift compounds across a video."""
    frame = pd.DataFrame(
        {"bbox_ltwh": [[1.0, 2.0, 3.0, 4.0]], "track_id": ["t0"], "score": [0.5]}, index=[0]
    )
    for i in range(1, 25):
        piece = pd.DataFrame(
            {
                "bbox_ltwh": [[float(i)] * 4],
                "track_id": [f"t{i}"],
                "score": [0.5],
            },
            index=[i],
        )
        frame = merge_dataframes(frame, piece)

    assert len(frame) == 25
    assert frame.loc[24, "track_id"] == "t24"
    assert not pd.api.types.is_numeric_dtype(frame["track_id"])
    assert pd.api.types.is_numeric_dtype(frame["score"])
