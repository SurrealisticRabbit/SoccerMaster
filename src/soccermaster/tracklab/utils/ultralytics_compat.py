"""Compatibility shims for running ultralytics on modern PyTorch."""
from __future__ import annotations

import logging
from contextlib import contextmanager

log = logging.getLogger(__name__)

_applied = False


def allow_ultralytics_globals() -> None:
    """Allowlist the ultralytics model classes for ``torch.load``.

    Kept as a belt-and-braces measure alongside :func:`trusted_checkpoint_load`; on
    its own it is not sufficient because the checkpoints also reference torch
    internals such as ``torch.nn.modules.container.Sequential``.
    """
    global _applied
    if _applied:
        return

    import torch

    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
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

    if classes:
        try:
            add_safe_globals(classes)
        except Exception as exc:  # never block startup over this
            log.warning(f"could not allowlist ultralytics checkpoint globals: {exc}")
    _applied = True


@contextmanager
def trusted_checkpoint_load():
    """Load checkpoints with ``weights_only=False`` for the duration of the block.

    torch 2.6 flipped ``torch.load``'s ``weights_only`` default to True. ultralytics
    8.0.100 -- the version this project pins -- calls ``torch.load`` internally
    without passing that argument, so a YOLO checkpoint fails to load on any modern
    torch::

        WeightsUnpickler error: Unsupported global: GLOBAL
        ultralytics.nn.tasks.DetectionModel

    Allowlisting the offending globals is whack-a-mole (fixing DetectionModel just
    surfaces torch.nn.modules.container.Sequential next), and ultralytics gives no way
    to pass the flag through. So scope the change narrowly instead: flip the default
    only while constructing a model, and restore it immediately afterwards.

    This is a real security trade-off and only acceptable because the checkpoint is
    fetched from this project's own Hugging Face repo by `soccermaster prepare`, not
    from an untrusted source. Do not reuse this for arbitrary user-supplied weights.
    """
    import torch

    original = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = _load
    try:
        yield
    finally:
        torch.load = original

