"""Tracker wrappers.

``wrappers/__init__.py`` imports this package eagerly (``from .track import *``), so
importing any single tracker imports all of them. Several vendored trackers are
incomplete in upstream -- ``strong_sort/deep/models.py`` has never existed in the
repository, which breaks both ``bot_sort`` and ``deep_oc_sort`` -- so one unusable
tracker would otherwise make the whole pipeline unimportable.

Each tracker is therefore imported defensively: a failure is logged and skipped, and
the rest stay available. The soccer game-state pipeline uses ``bpbreid_strong_sort``.
"""
import importlib
import logging

log = logging.getLogger(__name__)

_MODULES = {
    "StrongSORT": ("strong_sort_api", "StrongSORT"),
    "BotSORT": ("bot_sort_api", "BotSORT"),
    "ByteTrack": ("byte_track_api", "ByteTrack"),
    "OCSORT": ("oc_sort_api", "OCSORT"),
    "DeepOCSORT": ("deep_oc_sort_api", "DeepOCSORT"),
    "BPBReIDStrongSORT": ("bpbreid_strong_sort_api", "BPBReIDStrongSORT"),
    "BPBReIDDeepSORT": ("bpbreid_deep_sort_api", "BPBReIDDeepSORT"),
    "BPBReIDPitchDeepSORT": ("bpbreid_pitch_deep_sort_api", "BPBReIDPitchDeepSORT"),
}

__all__ = list(_MODULES)

for _public, (_module, _attr) in _MODULES.items():
    try:
        globals()[_public] = getattr(importlib.import_module(f".{_module}", __name__), _attr)
    except Exception as _exc:  # pragma: no cover - depends on vendored completeness
        globals()[_public] = None
        log.warning(f"{_public} unavailable ({type(_exc).__name__}: {_exc})")

del _public, _module, _attr
