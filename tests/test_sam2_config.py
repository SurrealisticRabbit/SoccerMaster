"""SAM 2's config must be handed to Hydra as a name, and Hydra must be initialised first.

Two separate blockers meant step 2 could never start:

1. ``build_sam2_video_predictor`` upstream calls ``hydra.compose(config_name=...)`` with no
   ``initialize_config_dir`` before it, and ``compose`` raises
   ``GlobalHydra is not initialized`` without a search path. Nothing in this project
   initialised it, so the call could not succeed regardless of what was installed.
2. Upstream's own default is ``"configs/sam2.1/sam2.1_hiera_l.yaml"`` -- a *path* relative
   to the SAM 2 package -- which ``compose`` cannot resolve as a name.

These cover the normalisation, which is the part with real edge cases, plus the directory
search and its failure message.
"""
from __future__ import annotations

from pathlib import Path

import pytest

sam2_refine = pytest.importorskip("soccermaster.sam2_refine")


# --------------------------------------------------------------------------- #
# config name normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "given,expected",
    [
        # Upstream's default, and what run_step2 passes through.
        ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1/sam2.1_hiera_l"),
        # Already a name.
        ("sam2.1/sam2.1_hiera_l", "sam2.1/sam2.1_hiera_l"),
        # No subdirectory.
        ("configs/sam2.1_hiera_l.yaml", "sam2.1_hiera_l"),
        ("sam2.1_hiera_l", "sam2.1_hiera_l"),
        # Leading ./ and windows separators, since run_step2 builds these from paths.
        ("./configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1/sam2.1_hiera_l"),
        ("configs\\sam2.1\\sam2.1_hiera_l.yaml", "sam2.1/sam2.1_hiera_l"),
        # An absolute path: only the part after configs/ matters.
        (
            "/opt/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
            "sam2.1/sam2.1_hiera_l",
        ),
        # .yml is accepted too.
        ("configs/sam2.1/sam2.1_hiera_l.yml", "sam2.1/sam2.1_hiera_l"),
    ],
)
def test_config_name_is_what_compose_expects(given: str, expected: str) -> None:
    assert sam2_refine.config_name_from(given) == expected


def test_the_default_name_has_no_extension_or_prefix() -> None:
    """`compose` resolves a name; a leading configs/ or a .yaml suffix both fail."""
    name = sam2_refine.DEFAULT_CONFIG_NAME
    assert not name.startswith("configs/")
    assert not name.endswith((".yaml", ".yml"))


def test_normalisation_is_idempotent() -> None:
    once = sam2_refine.config_name_from("configs/sam2.1/sam2.1_hiera_l.yaml")
    assert sam2_refine.config_name_from(once) == once


def test_a_path_object_is_accepted() -> None:
    """run_step2 may hand config values that came from Path objects."""
    assert sam2_refine.config_name_from(Path("configs/sam2.1/sam2.1_hiera_l.yaml")) == (
        "sam2.1/sam2.1_hiera_l"
    )


# --------------------------------------------------------------------------- #
# directory resolution
# --------------------------------------------------------------------------- #


def test_resolves_the_vendored_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shape upstream ships: <configs>/sam2.1/sam2.1_hiera_l.yaml."""
    configs = tmp_path / "configs"
    (configs / "sam2.1").mkdir(parents=True)
    (configs / "sam2.1" / "sam2.1_hiera_l.yaml").write_text("model: {}\n")

    monkeypatch.setattr(sam2_refine, "_candidate_config_dirs", lambda: [configs])

    directory, name = sam2_refine.resolve_config_dir("configs/sam2.1/sam2.1_hiera_l.yaml")

    assert directory == configs
    assert name == "sam2.1/sam2.1_hiera_l"
    # The name must actually resolve beneath the returned directory, which is the property
    # Hydra depends on.
    assert (directory / f"{name}.yaml").is_file()


def test_resolves_a_flat_directory_and_drops_the_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shape published beside the checkpoint: the leaf yaml sits directly in the dir.

    The returned name must lose its subdirectory here. Returning the original
    "sam2.1/sam2.1_hiera_l" would send Hydra looking for
    "<dir>/sam2.1/sam2.1_hiera_l.yaml" and fail -- a fallback that breaks what it was
    meant to fix.
    """
    flat = tmp_path / "sam2"
    flat.mkdir()
    (flat / "sam2.1_hiera_l.yaml").write_text("model: {}\n")

    monkeypatch.setattr(sam2_refine, "_candidate_config_dirs", lambda: [flat])

    directory, name = sam2_refine.resolve_config_dir("configs/sam2.1/sam2.1_hiera_l.yaml")

    assert directory == flat
    assert name == "sam2.1_hiera_l", "the name must match where the file was found"
    assert (directory / f"{name}.yaml").is_file()


def test_first_candidate_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordering matters: the installed package should beat the weights fallback."""
    good = tmp_path / "pkg_configs"
    (good / "sam2.1").mkdir(parents=True)
    (good / "sam2.1" / "sam2.1_hiera_l.yaml").write_text("model: {}\n")

    other = tmp_path / "weights"
    other.mkdir()
    (other / "sam2.1_hiera_l.yaml").write_text("model: {}\n")

    monkeypatch.setattr(sam2_refine, "_candidate_config_dirs", lambda: [good, other])

    directory, name = sam2_refine.resolve_config_dir()

    assert directory == good
    assert name == "sam2.1/sam2.1_hiera_l"


def test_missing_config_says_where_it_looked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hydra's own error does not say which paths were tried, and that is the useful part."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    monkeypatch.setattr(sam2_refine, "_candidate_config_dirs", lambda: [empty])

    with pytest.raises(FileNotFoundError) as excinfo:
        sam2_refine.resolve_config_dir()

    message = str(excinfo.value)
    assert "sam2.1_hiera_l.yaml" in message
    assert str(empty) in message, "the directories tried should be listed"
    assert "prepare sam2" in message, "and how to fix it"


def test_yml_is_accepted_as_well_as_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configs = tmp_path / "configs"
    (configs / "sam2.1").mkdir(parents=True)
    (configs / "sam2.1" / "sam2.1_hiera_l.yml").write_text("model: {}\n")

    monkeypatch.setattr(sam2_refine, "_candidate_config_dirs", lambda: [configs])

    directory, name = sam2_refine.resolve_config_dir()
    assert (directory / f"{name}.yml").is_file()


# --------------------------------------------------------------------------- #
# what the pipeline and the manifest must agree on
# --------------------------------------------------------------------------- #


def test_checkpoint_name_matches_the_manifest_and_pipeline() -> None:
    """A mismatch here is invisible until a GPU run.

    The yolo checkpoint was renamed by the manifest while the config kept the old name,
    which cost a run to find. Pinning the agreement in a test is cheaper.
    """
    from soccermaster.downloads import load_manifest

    entry = load_manifest().get("sam2")
    assert entry is not None, "sam2 must be in the manifest for prepare to fetch it"

    name = sam2_refine.DEFAULT_CHECKPOINT_NAME
    assert name == "sam2.1_hiera_large.pt"
    # The manifest allows *.pt from facebook/sam2.1-hiera-large, which publishes exactly
    # this filename.
    assert any(pattern == "*.pt" for pattern in (entry.allow or [])), entry.allow
    assert entry.dest.name == "sam2", entry.dest


def test_step2_defaults_to_a_checkpoint_under_weights_dir() -> None:
    from soccermaster import paths
    from soccermaster import pipeline as pipeline_mod
    import inspect

    default = inspect.signature(pipeline_mod.run_step2).parameters["sam_checkpoint"].default
    assert default is None, "the default is resolved inside, against paths.weights_dir()"
    source = inspect.getsource(pipeline_mod.run_step2)
    assert "DEFAULT_CHECKPOINT_NAME" in source, (
        "run_step2 should take the checkpoint name from sam2_refine so the two cannot drift"
    )
    assert str(paths.weights_dir()) or True  # paths must be importable here


def test_step2_exposes_processes_per_gpu() -> None:
    """Its process pool is the only real multi-GPU path in the pipeline."""
    import inspect

    from soccermaster import pipeline as pipeline_mod

    parameters = inspect.signature(pipeline_mod.run_step2).parameters
    assert "processes_per_gpu" in parameters
    assert parameters["processes_per_gpu"].default == 1
    source = inspect.getsource(pipeline_mod.run_step2)
    assert "--max_processes_per_gpu" in source, "the flag must actually be passed through"


# --------------------------------------------------------------------------- #
# step 2's visualization
# --------------------------------------------------------------------------- #


def test_step2_can_be_asked_for_a_visualization() -> None:
    """The masks ARE step 2's output, and a run that renders none is hard to judge.

    `sam2_refine` has always supported `--visualize_video`, but nothing passed it, so a
    completed step-2 run left no video to inspect -- only a 850 MB pickle.
    """
    import inspect

    from soccermaster import pipeline as pipeline_mod

    parameters = inspect.signature(pipeline_mod.run_step2).parameters
    assert "visualize_video" in parameters
    assert parameters["visualize_video"].default is False, (
        "off by default: it walks every frame, so it should be asked for"
    )
    source = inspect.getsource(pipeline_mod.run_step2)
    assert '"--visualize_video"' in source


def test_step2_visualization_follows_the_pipeline_video_switch() -> None:
    """`analyze_video` and `run_pipeline` tie it to their own video flag.

    Otherwise `--no-video` would silently leave three videos being written by a step the
    user asked to be quiet.
    """
    import inspect

    from soccermaster import pipeline as pipeline_mod

    for name in ("analyze_video", "run_pipeline"):
        source = inspect.getsource(getattr(pipeline_mod, name))
        assert "visualize_video=" in source, f"{name} does not wire step 2's visualization"


def test_the_three_step2_videos_are_named_where_we_will_look_for_them() -> None:
    """`_generate_visualization` writes them into step2/ as <id>_*_video.mp4.

    Recorded here because the fetch path depends on it and it is not obvious: steps 1 and 3
    put videos under visualization/videos/, step 2 puts them straight into step2/.
    """
    import inspect

    from soccermaster.sam2_refine import inference

    source = inspect.getsource(inference._generate_visualization)
    for suffix in ("_prompts_video.mp4", "_detections_segments_video.mp4",
                   "_all_segments_video.mp4"):
        assert suffix in source, f"expected a {suffix} output"
    assert "args.output_dir" in source, "written into step2/, not a visualization subdir"
