"""SAM 2 based segmentation refinement (step 2 of the GSR pipeline).

The SAM 2 runtime is an optional dependency; import it lazily via
``build_sam2_video_predictor`` so that ``import soccermaster`` works without the
``sam2`` extra installed.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["build_sam2_video_predictor", "config_name_from", "resolve_config_dir"]

#: Name of the SAM 2 config the pipeline uses, relative to SAM 2's own ``configs`` dir.
DEFAULT_CONFIG_NAME = "sam2.1/sam2.1_hiera_l"

#: Where the checkpoint lands under ``weights_dir()``; must match ``run_step2``'s default
#: and the manifest's destination, because a mismatch here is invisible until a GPU run.
DEFAULT_CHECKPOINT_NAME = "sam2.1_hiera_large.pt"


def config_name_from(config_file: str | Path) -> str:
    """Turn a SAM 2 config *path* into the config *name* that Hydra's ``compose`` wants.

    Upstream's own default is ``"configs/sam2.1/sam2.1_hiera_l.yaml"`` -- a path relative
    to the SAM 2 package -- and this project's ``run_step2`` passes that value straight
    through. ``compose`` resolves a name against the search path set up by
    ``initialize_config_dir``, so the leading ``configs/`` and the ``.yaml`` suffix have to
    go, leaving ``"sam2.1/sam2.1_hiera_l"``. Passing the path form raises
    ``Cannot find primary config 'configs/sam2.1/sam2.1_hiera_l.yaml'``.
    """
    name = str(config_file).replace("\\", "/").strip()
    if name.startswith("./"):
        name = name[2:]
    # Strip a leading "configs/" at any depth-1 position, since that is how both upstream's
    # default and the manifest express it.
    prefix = "configs/"
    if name.startswith(prefix):
        name = name[len(prefix):]
    elif f"/{prefix}" in name:
        name = name.split(f"/{prefix}", 1)[1]
    for suffix in (".yaml", ".yml"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def _candidate_config_dirs() -> list[Path]:
    """Places SAM 2's configs might live, most authoritative first.

    The installed package is the right answer when it exists. The weights directory is a
    fallback because ``facebook/sam2.1-hiera-large`` publishes ``sam2.1_hiera_l.yaml``
    alongside the checkpoint, so ``prepare`` fetches a usable config even if the package's
    internal layout differs from the one this code was written against -- which it has done
    between SAM 2 releases.
    """
    dirs: list[Path] = []
    try:
        import sam2

        dirs.append(Path(sam2.__file__).parent / "configs")
    except ImportError:
        pass

    try:
        from soccermaster import paths

        dirs.append(Path(paths.weights_dir()) / "sam2")
    except Exception:  # noqa: BLE001 - a missing home must not stop the import
        pass

    return dirs


def resolve_config_dir(config_file: str | Path = DEFAULT_CONFIG_NAME) -> tuple[Path, str]:
    """Return ``(config_dir, config_name)`` to initialise Hydra with.

    Raises with every location tried, because the alternative -- Hydra's own
    ``Cannot find primary config`` -- does not say where it looked or which of the many
    reasons applied.
    """
    name = config_name_from(config_file)
    leaf = Path(name).name
    attempts: list[str] = []

    for directory in _candidate_config_dirs():
        # Preferred: the name as given resolves beneath this directory, which is the SAM 2
        # package layout ("<pkg>/configs/sam2.1/sam2.1_hiera_l.yaml").
        for suffix in (".yaml", ".yml"):
            if (directory / f"{name}{suffix}").is_file():
                return directory, name
        # Fallback: the leaf yaml sits directly in this directory. That is the shape of the
        # file published beside the checkpoint, so the name has to lose its subdirectory --
        # returning the original name here would look for "<dir>/sam2.1/<leaf>.yaml" and
        # fail, which is worse than not having the fallback at all.
        for suffix in (".yaml", ".yml"):
            if (directory / f"{leaf}{suffix}").is_file():
                return directory, leaf
        attempts.append(str(directory / f"{name}.yaml"))

    raise FileNotFoundError(
        "could not find the SAM 2 config directory.\n"
        f"  wanted config : {name}.yaml\n"
        "  looked in     :\n    " + "\n    ".join(attempts) + "\n"
        "Fix by installing SAM 2 (it ships its configs) or by running "
        "`soccermaster prepare sam2`, which also fetches a usable yaml next to the "
        "checkpoint."
    )


def build_sam2_video_predictor(config_file: str | Path, checkpoint: str, **kwargs):
    """Build a SAM 2 video predictor.

    Requires SAM 2 installed -- it is vendored at ``third_party/sam2`` and is *not* part of
    this distribution's wheel, so it comes from the repository subdirectory::

        SAM2_BUILD_CUDA=0 pip install "sam2 @ git+<repo>@<ref>#subdirectory=third_party/sam2"

    Wraps the upstream builder to initialise Hydra first. Upstream calls
    ``hydra.compose(config_name=...)`` with no ``initialize_config_dir`` before it, and
    ``compose`` raises ``GlobalHydra is not initialized`` unless a search path has been set
    up -- so calling it directly could never work from here.
    """
    try:
        from hydra import initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from sam2.build_sam import build_sam2_video_predictor as _build
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            f"SAM 2 is required for the segmentation refinement step (missing: {exc.name}).\n"
            "Install it with:\n"
            "  SAM2_BUILD_CUDA=0 pip install "
            '"sam2 @ git+https://github.com/SurrealisticRabbit/SoccerMaster'
            '#subdirectory=third_party/sam2"'
        ) from exc

    config_dir, name = resolve_config_dir(config_file)

    # Hydra refuses to initialise twice in one process, and this function may be called
    # once per GPU worker.
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return _build(name, checkpoint, **kwargs)
