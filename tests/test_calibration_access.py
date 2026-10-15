"""Calibration must read batch columns positionally, not by label.

Step 3 died on the first real run with::

    File ".../calibration/nbjw_calib.py", line 232, in process
        predictions = metadatas["keypoints"][0]
    KeyError: 0

``metadatas`` is the batch's image-metadata DataFrame and the index is the image ids from
the tracker state -- values like ``10007000001`` -- so ``metadatas["keypoints"]`` is a
column Series with no label ``0``. Indexing it with ``[0]`` is therefore *label* lookup
and raises. The same method already used ``metadatas.iloc[0].name`` correctly a few lines
below, so this was an inconsistency rather than an intent.

Six sites had it, across four modules: ``NBJW_Calib``, ``NBJW_Calib_Decoupled`` (the one
that failed), ``ApplyParameters`` and ``GetPitchCorners`` all read one or more columns
this way. Fixing only the failing line would have moved the wall one module along, since
step 3's pipeline runs ``ApplyParameters`` immediately afterwards.

These tests use the real classes with the camera stubbed, and a non-zero-based index, so
they fail against the original code for the right reason.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch")

nbjw = pytest.importorskip("soccermaster.soccernet_gamestate.calibration.nbjw_calib")

#: An image id as the tracker state actually produces it -- deliberately not 0.
IMAGE_ID = 10007000001


def _metadatas(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns, index=[IMAGE_ID])


class _FakeCam:
    """Records what the module asked it to do, and returns nothing usable."""

    def __init__(self, *a, **k):  # noqa: ANN002, ANN003
        self.updated_with = None

    def update(self, predictions):  # noqa: ANN001
        self.updated_with = predictions

    def get_homography_from_ground_plane(self, **kwargs):  # noqa: ANN003, ARG002
        return None

    def heuristic_voting(self):
        return None


@pytest.fixture()
def stub_cam(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nbjw, "FramebyFrameCalib", _FakeCam)
    return _FakeCam


# --------------------------------------------------------------------------- #
# the helper
# --------------------------------------------------------------------------- #


def test_row_value_takes_the_first_row_positionally() -> None:
    metadatas = _metadatas(keypoints=[{"1": {"x": 0.1, "y": 0.2}}])
    assert nbjw._row_value(metadatas, "keypoints") == {"1": {"x": 0.1, "y": 0.2}}


def test_label_lookup_is_what_used_to_raise() -> None:
    """Documents the pandas semantics behind the bug, so a revert is caught here."""
    column = _metadatas(keypoints=[{"a": 1}])["keypoints"]
    assert list(column.index) == [IMAGE_ID]

    with pytest.raises(KeyError):
        column[0]  # noqa: B018 - this is the idiom that failed

    assert column.iloc[0] == {"a": 1}


def test_row_value_works_for_any_column() -> None:
    # Built with dtype=object: a column holding a list of lists is otherwise raveled by
    # pandas into a 2-D array rather than kept as one object, which is a property of the
    # frame construction rather than of the code under test.
    metadatas = pd.DataFrame(
        {
            "h": pd.Series([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]], index=[IMAGE_ID], dtype=object),
            "parameters": pd.Series([{"a": 1}], index=[IMAGE_ID], dtype=object),
        }
    )
    assert nbjw._row_value(metadatas, "h") == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert nbjw._row_value(metadatas, "parameters") == {"a": 1}


# --------------------------------------------------------------------------- #
# the modules themselves
# --------------------------------------------------------------------------- #


def test_decoupled_calibration_reads_keypoints(stub_cam) -> None:
    """The module that failed: line 232 in the traceback."""
    module = nbjw.NBJW_Calib_Decoupled(image_width=1920, image_height=1080)
    keypoints = {"1": {"x": 0.5, "y": 0.5}}

    _detections, image_pred = module.process(
        batch=None,
        detections=pd.DataFrame(),
        metadatas=_metadatas(keypoints=[keypoints]),
    )

    assert module.cam.updated_with == keypoints, "the keypoints never reached the camera"
    assert "parameters" in image_pred.columns
    assert "h" in image_pred.columns
    assert image_pred.index.tolist() == [IMAGE_ID], "the output row keeps the image id"


def test_decoupled_calibration_does_not_read_the_wrong_row(stub_cam) -> None:
    """With several rows the first is used -- the modules are batch_size=1."""
    module = nbjw.NBJW_Calib_Decoupled(image_width=1920, image_height=1080)
    metadatas = pd.DataFrame(
        {"keypoints": [{"first": True}, {"second": True}]},
        index=[10007000001, 10007000002],
    )

    module.process(batch=None, detections=pd.DataFrame(), metadatas=metadatas)

    assert module.cam.updated_with == {"first": True}


def test_calibration_reads_keypoints(stub_cam, monkeypatch: pytest.MonkeyPatch) -> None:
    """NBJW_Calib had the same idiom."""
    monkeypatch.setattr(nbjw, "FramebyFrameCalib", _FakeCam)
    module = nbjw.NBJW_Calib(
        image_width=1920, image_height=1080, batch_size=1, use_prev_homography=False
    )
    keypoints = {"2": {"x": 0.1, "y": 0.1}}

    module.process(
        batch=None,
        detections=pd.DataFrame(columns=["bbox_ltwh"]),
        metadatas=_metadatas(keypoints=[keypoints]),
    )

    assert module.cam.updated_with == keypoints


def test_apply_parameters_reads_h_and_parameters(stub_cam) -> None:
    """The next module the pipeline runs, which had the same fault twice over."""
    module = nbjw.ApplyParameters(use_h=True, use_linalg=False, use_prev_homography=False)
    _h = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    # No valid h, so it takes the "no parameters" branch -- the point is that reading the
    # columns does not raise, which is what it did before.
    _detections, = module.process(
        batch=None,
        detections=pd.DataFrame({"bbox_ltwh": [[0.0, 0.0, 1.0, 1.0]]}),
        metadatas=_metadatas(h=[None], parameters=[{}]),
    )

    assert True


def test_get_pitch_corners_reads_h_and_parameters(stub_cam) -> None:
    module = nbjw.GetPitchCorners(
        use_h=True,
        use_linalg=False,
        use_prev_homography=False,
        image_width=1920,
        image_height=1080,
    )

    module.process(
        batch=None,
        detections=pd.DataFrame(),
        metadatas=_metadatas(h=[None], parameters=[{}]),
    )

    assert True


def test_no_metadata_column_is_indexed_by_label() -> None:
    """A source-level guard: the idiom must not reappear anywhere in the module.

    Cheap, and it covers the modules that are awkward to instantiate (they need real
    camera parameters to reach the interesting branches).
    """
    import pathlib

    source = pathlib.Path(nbjw.__file__).read_text(encoding="utf-8")
    offending = [
        f"{number}: {line.strip()}"
        for number, line in enumerate(source.splitlines(), start=1)
        if 'metadatas["' in line and "][0]" in line
    ]
    assert not offending, (
        "metadata columns must be read with .iloc[0], not [0] (label lookup): "
        + "; ".join(offending)
    )
