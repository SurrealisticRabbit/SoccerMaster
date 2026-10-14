"""High-level notebook-friendly API for SoccerMaster.

Typical use:

    import soccermaster as sm

    sm.prepare("siglip")
    sm.prepare("soccermaster")
    model = sm.load_model()
    out = sm.predict(model, "path/to/video_frames", task="detection")

Everything here is a thin, explicit wrapper around the code under
``soccermaster.model`` (the model) and ``soccermaster.soccernet_gamestate`` /
``soccermaster.tracklab`` (the GSR pipeline). The underlying modules keep their
original behaviour; this module only resolves configuration and paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from soccermaster import paths
from soccermaster.downloads import prepare, prepare_all  # noqa: F401  (re-exported)

CONFIG_DIR = Path(__file__).parent / "foundation" / "configs"

#: Map the public task names onto the dataset keys used by the model config.
TASK_TO_DATASET = {
    "detection": "SoccerNetGSR_Detection",
    "lines": "SoccerNetGSR_Detection",
    "keypoints": "SoccerNetGSR_Detection",
    "caption": "VideoCaption",
    "classification": "VideoCaption",
    "all": None,
}

#: Which heads to keep when a single task is requested.
TASK_TO_HEADS = {
    "detection": ["SoccerNetGSR_Detection"],
    "lines": ["LinesDetection"],
    "keypoints": ["KeypointsDetection"],
    "caption": ["VideoCaption"],
    "classification": ["CaptionClassification"],
}


def load_config(name: str = "pretrain") -> dict:
    """Load a model config, following ``SUPER_CONFIG_PATH`` and retargeting weights.

    ``name`` is a config stem under ``soccermaster/model/configs``
    (``pretrain`` or ``default``). ``CKPT_PATH`` / ``TEXT_ENCODER_CKPT_PATH``
    are rewritten to point under :func:`soccermaster.paths.weights_dir`.
    """
    from soccermaster.model.configs.util import load_super_config, yaml_to_dict

    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"no such config: {path}")

    cfg = yaml_to_dict(str(path))

    super_path = cfg.get("SUPER_CONFIG_PATH")
    if super_path:
        super_path = Path(super_path)
        if not super_path.is_absolute():
            super_path = CONFIG_DIR / super_path.name
        cfg = load_super_config(cfg, str(super_path))

    weights = paths.weights_dir()
    for key in ("CKPT_PATH", "TEXT_ENCODER_CKPT_PATH"):
        value = cfg.get(key)
        if not value:
            continue
        candidate = Path(value)
        if candidate.is_absolute():
            continue
        parts = [p for p in candidate.parts if p not in (".", "..")]
        if parts and parts[0] in ("pretrained_models", "weights"):
            parts = parts[1:]
        cfg[key] = str(weights.joinpath(*parts))
    return cfg


def load_model(
    checkpoint_dir: str | Path | None = None,
    config: str = "pretrain",
    device: str | None = None,
    logger=None,
):
    """Build a :class:`MultiTaskingModel` and load its weights.

    ``checkpoint_dir`` defaults to ``weights_dir()/SoccerMaster``. Use
    :func:`soccermaster.prepare` to fetch the weights first.
    """
    import torch

    from soccermaster.model.models.multi_task import MultiTaskingModel

    cfg = load_config(config)
    if checkpoint_dir is None:
        checkpoint_dir = paths.weights_dir() / "SoccerMaster"
    checkpoint_dir = str(checkpoint_dir)

    model = MultiTaskingModel(config=cfg, logger=logger)
    model.load_checkpoint(checkpoint_dir, logger=logger, load_heads=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    return model


def _load_frames(frames) -> Sequence:
    """Normalise the ``frames`` argument to a list of RGB uint8 HWC arrays."""
    import numpy as np
    from PIL import Image

    if isinstance(frames, (str, Path)):
        directory = Path(frames)
        candidates: Iterable[Path]
        if (directory / "img1").is_dir():
            candidates = sorted((directory / "img1").glob("*.jpg"))
        else:
            candidates = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        files = list(candidates)
        if not files:
            raise FileNotFoundError(f"no frames found under {directory}")
        return [np.asarray(Image.open(f).convert("RGB")) for f in files]

    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        return list(frames)

    return list(frames)


def _frames_to_tensor(frames, cfg: dict, device: str):
    """Stack frames into the ``[B, T, C, H, W]`` tensor the model expects."""
    import numpy as np
    import torch
    import torchvision.transforms.v2 as v2
    from torchvision.transforms.functional import to_tensor

    max_size = cfg.get("AUG_MAX_SIZE", 512)
    mean = cfg.get("AUG_MEAN", [0.5, 0.5, 0.5])
    std = cfg.get("AUG_STD", [0.5, 0.5, 0.5])

    tensors = []
    for frame in frames:
        t = to_tensor(np.ascontiguousarray(frame))  # uint8 HWC -> float CHW in [0,1]
        _, h, w = t.shape
        scale = max_size / max(h, w)
        if scale != 1.0:
            t = v2.functional.resize(t, [round(h * scale), round(w * scale)], antialias=True)
        tensors.append(v2.functional.normalize(t, mean=mean, std=std))

    return torch.stack(tensors).unsqueeze(0).to(device)


def predict(model, frames, task: str = "detection", device: str | None = None) -> dict:
    """Run one forward pass and return the raw outputs of the requested head(s).

    ``task`` is one of ``detection``, ``lines``, ``keypoints``, ``caption``,
    ``classification`` or ``all``. ``frames`` may be a directory (either the
    SoccerNet layout ``<seq>/img1/*.jpg`` or a flat folder of images), a
    ``[T, H, W, C]`` array, or a sequence of HWC arrays.

    Returns the model's raw head outputs; post-processing (e.g. NMS / bbox
    rescaling) is left to the caller, mirroring ``foundation-demo``.
    """
    import torch

    if task not in TASK_TO_DATASET:
        raise ValueError(f"unknown task '{task}'. Known: {', '.join(TASK_TO_DATASET)}")

    device = device or next(model.parameters()).device
    dataset_name = TASK_TO_DATASET[task]
    datasets = model.config["DATASETS_TO_HEADS"]

    if dataset_name is None:
        targets = list(datasets)
    else:
        targets = [dataset_name]

    frame_list = _load_frames(frames)
    images = _frames_to_tensor(frame_list, model.config, str(device))

    keep = TASK_TO_HEADS.get(task)
    result: dict = {}
    with torch.no_grad():
        for ds in targets:
            heads = datasets[ds]
            if keep is not None:
                heads = [h for h in heads if h in keep]
            if not heads:
                continue
            # Temporarily narrow the head list so only the requested heads run.
            original = model.datasets_to_heads[ds]
            model.datasets_to_heads[ds] = heads
            try:
                result.update(model(images, dataset_name=ds))
            finally:
                model.datasets_to_heads[ds] = original
    return result


def run_pipeline(
    frames_dir: str | Path,
    out_dir: str | Path | None = None,
    steps: Sequence[int] = (1, 2, 3),
    video_ids: Sequence[int] | None = None,
    device: str = "cuda",
    extra_overrides: Sequence[str] | None = None,
) -> Path:
    """Run the 3-step SoccerNet game-state pipeline on a folder of frames.

    Steps run through the same Hydra configs as the original repo, with paths
    injected instead of the hardcoded ``10001/10002`` sequence ids.

    Note: step 2 requires the optional ``sam2`` extra and a CUDA build.
    """
    from soccermaster.pipeline import run_pipeline as _run

    return _run(
        frames_dir=frames_dir,
        out_dir=out_dir,
        steps=steps,
        video_ids=video_ids,
        device=device,
        extra_overrides=extra_overrides,
    )
