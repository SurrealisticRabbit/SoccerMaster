"""SAM 2 based segmentation refinement (step 2 of the GSR pipeline).

The SAM 2 runtime is an optional dependency; import it lazily via
``build_sam2_video_predictor`` so that ``import soccermaster`` works without the
``sam2`` extra installed.
"""

from __future__ import annotations


def build_sam2_video_predictor(config_file: str, checkpoint: str, **kwargs):
    """Build a SAM 2 video predictor.

    Requires the optional ``sam2`` extra (``pip install -e ".[sam2]"``) plus the
    vendored SAM 2 package (or a git install of ``facebookresearch/sam2``).
    """
    try:
        from sam2.build_sam import build_sam2_video_predictor as _build
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "SAM 2 is required for the segmentation refinement step. "
            'Install it with:  pip install -e ".[sam2]"'
        ) from exc
    return _build(config_file, checkpoint, **kwargs)
