"""Hydra plugin namespace package.

Hydra only discovers third-party plugins from a top-level package literally named
``hydra_plugins`` -- see ``hydra.core.plugins.Plugins._initialize`` and
``is_in_toplevel_plugins_module``. Setuptools entry points are ignored by Hydra, so
this package is required even though the same plugin is also listed as an entry
point in pyproject.toml.

Do not rename or relocate this package: the name is load-bearing.
"""
