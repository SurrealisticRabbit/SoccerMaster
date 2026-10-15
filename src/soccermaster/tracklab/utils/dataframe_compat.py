"""Small pandas compatibility helpers for the tracking dataframes.

pandas 3 tightened value assignment: writing a non-numeric value into a numeric column now
raises ``TypeError: Invalid value ... for dtype 'float64'`` where pandas 2 silently upcast
the column to ``object``. This codebase creates those columns numerically in several places
because ``NaN`` is the natural "unknown" value, then assigns real values that are not
numbers -- so the pattern needed changing rather than the pandas version pinning. The
project pins only ``pandas>=2.0``, and both are supported.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

__all__ = ["ensure_object_column"]


def ensure_object_column(
    detections: pd.DataFrame, columns: str | Iterable[str]
) -> None:
    """Make ``columns`` able to hold non-numeric values, creating them where absent.

    The values these columns receive are jersey numbers, roles and team labels, with
    ``None`` where a tracklet has no majority. None of those are numeric, but the columns
    are created as ``float64`` -- ``detections[cols] = np.nan`` or ``[np.nan] * len(df)``
    -- so pandas 3 refuses the assignment:

        TypeError: Invalid value '[None, None, ...]' for dtype 'float64'

    ``object`` is the honest dtype for a column of mixed or non-numeric values; the
    ``float64`` version only ever appeared to work, because older pandas quietly widened
    it on the first assignment. Doing this explicitly also makes the widening happen once
    per column rather than per assignment.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if column not in detections.columns:
            detections[column] = None
        if detections[column].dtype != object:
            detections[column] = detections[column].astype(object)
