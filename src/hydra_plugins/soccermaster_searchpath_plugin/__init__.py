"""Hydra search-path plugin for SoccerMaster.

Hydra has to be told where ``soccermaster.soccernet_gamestate`` keeps its configs,
otherwise ``-cn gsr_step_1_example`` cannot be found: the gsr_* entry-point configs
live in a different package from the tracklab engine defaults that
``soccermaster.tracklab.main`` names in its ``@hydra.main(config_path=...)``.

The class must live inside the ``hydra_plugins`` package for Hydra to find it --
entry points alone are not enough (see ``Plugins.is_in_toplevel_plugins_module``).
"""
from __future__ import annotations

import logging
from importlib.metadata import entry_points

from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

log = logging.getLogger(__name__)

#: Extra config packages contributed by entry points, for anything that wants to
#: add configs later without editing this file.
ENTRY_POINT_GROUP = "soccermaster_plugin"

#: The configs the game-state pipeline names via ``-cn``.
DEFAULT_CONFIG_PACKAGES = ("pkg://soccermaster.soccernet_gamestate.configs",)


class SoccerMasterSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        added: set[str] = set()

        for path in DEFAULT_CONFIG_PACKAGES:
            search_path.append(provider="soccermaster", path=path)
            added.add(path)

        for plugin in entry_points(group=ENTRY_POINT_GROUP):
            try:
                module = plugin.load()
            except Exception as exc:  # never let a bad plugin break startup
                log.warning(f"could not load soccermaster plugin {plugin.name}: {exc}")
                continue
            path = getattr(module, "config_package", None)
            if path is None:
                log.warning(f"{module} provides no config_package")
                continue
            if path not in added:
                search_path.append(provider="soccermaster", path=path)
                added.add(path)


# The name this class had when it lived at soccermaster/tracklab/searchpath_plugin.py.
PbTrackSearchPathPlugin = SoccerMasterSearchPathPlugin
