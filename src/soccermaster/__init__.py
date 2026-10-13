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

__version__ = "0.1.0"

__all__ = [
    "load_config",
    "load_model",
    "predict",
    "prepare",
    "prepare_all",
    "run_pipeline",
    "paths",
]

_LAZY = {
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
