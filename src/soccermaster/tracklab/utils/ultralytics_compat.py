"""Compatibility shims for running ultralytics on modern PyTorch."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_applied = False


def allow_ultralytics_globals() -> None:
    """Let ultralytics 8.0.x checkpoints load under torch >= 2.6.

    torch 2.6 changed ``torch.load``'s ``weights_only`` default to True. ultralytics
    8.0.100 -- the version this project pins -- calls ``torch.load`` without passing
    that argument, so loading a YOLO checkpoint fails with

        _pickle.UnpicklingError: Weights only load failed ...
        Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel

    PyTorch's documented remedy is to allowlist the classes you trust. The
    checkpoints come from this project's own Hugging Face repo, so allowlisting the
    ultralytics model classes is appropriate and is far narrower than switching
    ``weights_only`` off globally.

    Safe to call repeatedly; a no-op on torch < 2.6, where the API is absent.
    """
    global _applied
    if _applied:
        return

    add_safe_globals = getattr(__import__("torch").serialization, "add_safe_globals", None)
    if add_safe_globals is None:
        _applied = True
        return

    classes = []
    for name in ("DetectionModel", "SegmentationModel", "PoseModel", "ClassificationModel"):
        try:
            module = __import__("ultralytics.nn.tasks", fromlist=[name])
            classes.append(getattr(module, name))
        except Exception:  # a given ultralytics build may not define them all
            continue

    if not classes:
        log.warning("ultralytics model classes not found; YOLO checkpoint loading may fail on torch>=2.6")
        _applied = True
        return

    try:
        add_safe_globals(classes)
    except Exception as exc:  # never block startup over this
        log.warning(f"could not allowlist ultralytics checkpoint globals: {exc}")
    _applied = True
