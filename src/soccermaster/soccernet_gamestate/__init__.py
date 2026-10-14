from importlib import metadata

from soccermaster.tracklab.utils.matplotlib_compat import patch_legacy_cmap

# prtreid (a torchreid fork) calls matplotlib.cm.get_cmap at *import* time, and
# matplotlib 3.9 removed that function. This shim lives in the package __init__ rather
# than next to the imports that need it, because step 1 resolves its `reid` module to
# soccermaster.soccernet_gamestate.reid.prtreid_api.PRTReId and that module pulls
# prtreid in indirectly through .reid.prtreid_dataset before it reaches any patch of its
# own. A package __init__ always runs before its submodules, so this cannot be bypassed.
patch_legacy_cmap()

try:
    __version__ = metadata.version('sn-gamestate')
except metadata.PackageNotFoundError:
    __version__ = None