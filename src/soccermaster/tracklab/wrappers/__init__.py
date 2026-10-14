# torchreid (reached below via .reid and .track) calls matplotlib.cm.get_cmap at
# *import* time, and matplotlib 3.9 removed that function. Patch it before the imports
# below reach torchreid, otherwise importing this package raises AttributeError.
from soccermaster.tracklab.utils.matplotlib_compat import patch_legacy_cmap

patch_legacy_cmap()

from .datasets import *
from .detect_multiple import *
from .detect_single import *
from .eval import *
from .reid import *
from .track import *
from .tracklet_agg import *
