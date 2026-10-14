"""SoccerMaster: unified soccer vision foundation model and GSR pipeline.

The public surface is re-exported lazily so that ``import soccermaster`` stays
cheap and works without the optional/heavy dependencies installed::

    import soccermaster as sm

    sm.prepare("siglip")
    model = sm.load_model()
    outputs = sm.predict(model, "frames/SNGS-10001", task="detection")
    sm.run_pipeline("data/SoccerNetGS", out_dir="outputs/pipeline")
"""

from __future__ import annotations

# Version is supplied by setuptools-scm from the git tag. Importing the generated
# module is the fastest path; the importlib fallback covers an installed build that
# did not write the file; the last resort keeps `import soccermaster` working from a
# bare source tree with no tags (which is exactly what the CI smoke test does).
try:  # pragma: no cover - depends on how the package was built
    from soccermaster._version import version as __version__
except ImportError:  # pragma: no cover
    try:
        from importlib.metadata import version as _dist_version

        __version__ = _dist_version("soccermaster")
    except Exception:
        __version__ = "0.0.0+unknown"

__all__ = [
    "analyze_video",
    "load_config",
    "load_model",
    "predict",
    "prepare",
    "prepare_all",
    "run_pipeline",
    "paths",
]

_LAZY = {
    "analyze_video": "soccermaster.api",
    "load_config": "soccermaster.api",
    "load_model": "soccermaster.api",
    "predict": "soccermaster.api",
    "prepare": "soccermaster.api",
    "prepare_all": "soccermaster.api",
    "run_pipeline": "soccermaster.api",
}


def __getattr__(name: str):
    import importlib

    if name == "paths":
        module = importlib.import_module("soccermaster.paths")
    else:
        module_name = _LAZY.get(name)
        if module_name is None:
            raise AttributeError(f"module 'soccermaster' has no attribute '{name}'")
        module = importlib.import_module(module_name)
        globals()[name] = getattr(module, name)
        return globals()[name]
    globals()[name] = module
    return module
