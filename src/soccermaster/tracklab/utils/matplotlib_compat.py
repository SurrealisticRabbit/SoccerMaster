"""Put `matplotlib.cm.get_cmap` back, because torchreid needs it at import time.

torchreid -- and the prtreid/bpbreid forks of it that this project installs from git
-- call `matplotlib.cm.get_cmap` at *module import* time rather than inside a function:

    # prtreid/utils/visualization/visualize_query_gallery_rankings.py
    cmap = matplotlib.cm.get_cmap('hsv')

`get_cmap` was deprecated in matplotlib 3.7 and removed in 3.9, so on any current
matplotlib the import dies with:

    AttributeError: module 'matplotlib.cm' has no attribute 'get_cmap'

which takes down `soccermaster.tracklab.wrappers` and therefore the whole of step 1.

Pinning `matplotlib<3.9` is not an acceptable fix here: the 3.8.x line has no cp313
wheels, so that pin reintroduces exactly the "uninstallable on Python 3.13" problem the
open upper bound avoids. The removed function is four lines of delegation to the
registry that replaced it, so we restore it instead, immediately before the first
torchreid import.

Call `patch_legacy_cmap()` before importing torchreid/prtreid. It is idempotent, cheap
and safe to call from several entry points.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

__all__ = ["patch_legacy_cmap"]


def patch_legacy_cmap() -> bool:
    """Re-add `matplotlib.cm.get_cmap` if the installed matplotlib removed it.

    Returns True when the shim was installed, False when the real function is already
    present or matplotlib is not importable.
    """
    try:
        import matplotlib
        import matplotlib.cm as cm
    except Exception:  # noqa: BLE001 - a missing matplotlib is not this module's problem
        return False

    if hasattr(cm, "get_cmap"):
        return False

    def get_cmap(name=None, lut=None):
        """Deprecated `matplotlib.cm.get_cmap` equivalent, backed by the registry."""
        if name is None:
            name = matplotlib.rcParams["image.cmap"]
        colormap = matplotlib.colormaps[name] if isinstance(name, str) else name
        if lut is not None:
            colormap = colormap.resampled(lut)
        return colormap

    cm.get_cmap = get_cmap
    log.debug("installed the matplotlib.cm.get_cmap compatibility shim")
    return True
