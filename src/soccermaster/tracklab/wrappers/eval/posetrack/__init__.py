"""PoseTrack evaluators.

These are optional: the soccer game-state pipeline never uses them, but
``wrappers/__init__.py`` imports this package eagerly via ``from .eval import *``, so
an unresolvable PoseTrack dependency would otherwise make the whole ``track`` module
unimportable and take the pipeline down with it.

Import failures are therefore reported and skipped rather than raised. Install
``poseval`` (a `pipeline` extra dependency) and the vendored ``posetrack21``
packages to use these evaluators.
"""
import logging

log = logging.getLogger(__name__)

try:
    from .posetrack18_evaluator import PoseTrack18Evaluator
except ImportError as exc:  # pragma: no cover - depends on optional extras
    PoseTrack18Evaluator = None
    log.warning(f"PoseTrack18Evaluator unavailable ({exc}); PoseTrack18 eval disabled")

try:
    from .posetrack21_evaluator import PoseTrack21Evaluator
except ImportError as exc:  # pragma: no cover - depends on optional extras
    PoseTrack21Evaluator = None
    log.warning(f"PoseTrack21Evaluator unavailable ({exc}); PoseTrack21 eval disabled")

__all__ = ["PoseTrack18Evaluator", "PoseTrack21Evaluator"]
