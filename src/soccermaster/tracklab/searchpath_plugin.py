"""Hydra search-path plugin for SoccerMaster tracking plugins.

Any installed package that advertises a ``soccermaster_plugin`` entry point
exposing a ``config_package`` attribute contributes its Hydra config directory
to the search path (e.g. ``soccermaster.soccernet_gamestate``).
"""

import logging
from importlib.metadata import entry_points

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "soccermaster_plugin"


class SoccerMasterSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        for plugin in entry_points(group=ENTRY_POINT_GROUP):
            module = plugin.load()
            if hasattr(module, "config_package"):
                search_path.append(provider="soccermaster", path=module.config_package)
            else:
                log.warning(f"{module} doesn't provide a config path")


# Backwards-compatible alias for the original TrackLab class name.
PbTrackSearchPathPlugin = SoccerMasterSearchPathPlugin