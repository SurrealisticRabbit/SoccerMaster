"""SoccerMaster tracking engine (vendored TrackLab, renamed to soccermaster.tracklab)."""
# torchreid -- used by wrappers/reid and plugins/track -- calls matplotlib.cm.get_cmap at
# *import* time, and matplotlib 3.9 removed that function. A package __init__ is imported
# before any of its submodules, so patching here covers every one of those modules by
# construction. Doing it inside wrappers/__init__.py was insufficient: that file is only
# reached via the wrappers tree, and only if something imports it before torchreid.
from soccermaster.tracklab.utils.matplotlib_compat import patch_legacy_cmap

patch_legacy_cmap()
