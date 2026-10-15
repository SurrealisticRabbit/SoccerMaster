"""Tracklet attributes must be assignable into their columns.

This is the third instance of the same pandas-3 tightening, after `merge_dataframes` and
the calibration metadata access. The codebase repeatedly creates a column numerically --
``detections[cols] = np.nan`` or ``[np.nan] * len(df)`` -- because NaN is the natural
"unknown", and then assigns values that are not numbers: jersey numbers, roles, team
labels, and ``None`` for a tracklet with no majority.

pandas 2 silently widened the column to ``object`` on that first assignment. pandas 3
raises::

    TypeError: Invalid value '[None, None, None, ...]' for dtype 'float64'

which is what killed step 3 at the first ``MajorityVoteTrackletFilter`` -- after all
thirteen modules had run. ``object`` is the honest dtype for these columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soccermaster.tracklab.utils.dataframe_compat import ensure_object_column


def _detections(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({"track_id": [1, 1, 2][:n], "bbox_ltwh": [[0.0, 0.0, 1.0, 1.0]] * n})


# --------------------------------------------------------------------------- #
# the helper
# --------------------------------------------------------------------------- #


def test_nan_column_becomes_object() -> None:
    detections = _detections()
    detections["role"] = np.nan
    assert detections["role"].dtype == np.float64, "precondition: NaN makes it numeric"

    ensure_object_column(detections, "role")

    assert detections["role"].dtype == object


def test_absent_column_is_created() -> None:
    detections = _detections()
    ensure_object_column(detections, "team")
    assert "team" in detections.columns
    assert detections["team"].dtype == object


def test_several_columns_at_once() -> None:
    detections = _detections()
    detections[["a", "b"]] = np.nan

    ensure_object_column(detections, ["a", "b"])

    assert detections["a"].dtype == object
    assert detections["b"].dtype == object


def test_existing_values_are_preserved() -> None:
    detections = _detections()
    detections["role"] = ["player", "referee", None]

    ensure_object_column(detections, "role")

    # pandas normalises a None written through *construction* into NaN, so missingness is
    # the property to assert rather than identity with None.
    assert detections["role"].tolist()[:2] == ["player", "referee"]
    assert pd.isna(detections["role"].iloc[2])


def test_it_is_idempotent() -> None:
    detections = _detections()
    detections["role"] = np.nan
    ensure_object_column(detections, "role")
    ensure_object_column(detections, "role")
    assert detections["role"].dtype == object


# --------------------------------------------------------------------------- #
# the failure, reproduced
# --------------------------------------------------------------------------- #


def test_all_none_assignment_raises_without_the_helper() -> None:
    """The exact step-3 failure: a tracklet with no majority yields a list of Nones."""
    detections = _detections()
    detections["jersey_number"] = np.nan  # float64

    with pytest.raises(TypeError, match="Invalid value"):
        detections.loc[[0, 1], "jersey_number"] = [None, None]


def test_all_none_assignment_works_with_the_helper() -> None:
    """The reported case: `attribute_value = None` repeated per detection."""
    detections = _detections()
    detections["jersey_number"] = np.nan
    ensure_object_column(detections, "jersey_number")

    detections.loc[[0, 1], "jersey_number"] = [None, None]

    assert detections["jersey_number"].isna().all() or all(
        v is None for v in detections["jersey_number"]
    )


def test_string_assignment_raises_without_the_helper() -> None:
    """A jersey number is a string or int, neither of which fits float64."""
    detections = _detections()
    detections["jersey_number"] = np.nan

    with pytest.raises(TypeError, match="Invalid value"):
        detections.loc[[0, 1], "jersey_number"] = ["10", "10"]


def test_string_assignment_works_with_the_helper() -> None:
    detections = _detections()
    detections["jersey_number"] = np.nan
    ensure_object_column(detections, "jersey_number")

    detections.loc[[0, 1], "jersey_number"] = ["10", "10"]

    assert detections.loc[0, "jersey_number"] == "10"
    # The untouched row keeps its original NaN, which astype(object) does not turn into
    # None -- assert missingness, not identity.
    assert pd.isna(detections.loc[2, "jersey_number"])


def test_mixed_none_and_values_work_together() -> None:
    """Real tracklets: some decided, some not, in one assignment."""
    detections = _detections()
    detections["role"] = np.nan
    ensure_object_column(detections, "role")

    detections.loc[[0, 1, 2], "role"] = ["player", "player", None]

    assert detections.loc[0, "role"] == "player"
    assert detections.loc[2, "role"] is None


# --------------------------------------------------------------------------- #
# the real modules
# --------------------------------------------------------------------------- #


def test_majority_vote_filter_module_handles_undecided_tracklets() -> None:
    """`MajorityVoteTrackletFilter` is the module that failed on real data.

    Every tracklet here is undecided (all detections are None), which is the path that
    produced `attribute_value = [None] * len(tracklet)`.

    `output_columns` must be populated exactly as `__init__` does it. An earlier version of
    this test left it empty, which meant `detections[self.output_columns] = np.nan` created
    no column at all -- pandas then created the column on first assignment with the right
    dtype and the test passed over the very bug it was written to catch.
    """
    module = pytest.importorskip(
        "soccermaster.soccernet_gamestate.tracklet_agg.majority_vote_filter_api"
    )
    cls = getattr(module, "MajorityVoteTrackletFilter", None)
    if cls is None:
        pytest.skip("MajorityVoteTrackletFilter not found")

    # Mirror __init__ without needing a cfg: it derives output_columns from attributes.
    instance = cls.__new__(cls)
    instance.attributes = ["jersey_number"]
    instance.input_columns = [
        "jersey_number_detection",
        "jersey_number_confidence",
    ]
    instance.output_columns = ["jersey_number"]

    detections = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "jersey_number_detection": [None, None, None, None],
            "jersey_number_confidence": [0.0, 0.0, 0.0, 0.0],
        }
    )

    out = instance.process(detections, pd.DataFrame())

    assert "jersey_number" in out.columns
    assert len(out) == 4, "rows must survive an undecided tracklet"


def test_the_numeric_column_is_really_what_the_fix_has_to_handle() -> None:
    """Guards the guard: proves the setup above creates the float64 column.

    Without this, `test_majority_vote_filter_module_handles_undecided_tracklets` can pass
    for the wrong reason -- if the numeric column is never created, the assignment succeeds
    whatever the helper does.
    """
    detections = pd.DataFrame({"track_id": [1, 1]})
    detections[["jersey_number"]] = np.nan

    assert detections["jersey_number"].dtype == np.float64
    with pytest.raises(TypeError, match="Invalid value"):
        detections.loc[[0, 1], "jersey_number"] = [None, None]
