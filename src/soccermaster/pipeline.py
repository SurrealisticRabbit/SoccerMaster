"""Orchestration of the three-step SoccerNet game-state pipeline.

The steps are executed exactly as documented in the upstream README -- each step
is the same Hydra-driven entry point as before (``soccermaster.tracking.main``
and ``soccermaster.refine``), invoked with explicit path overrides so nothing
depends on the old repository layout or on hardcoded sequence ids.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence

from soccermaster import paths

STEP1_CONFIG = "gsr_step_1_example"
STEP3_CONFIG = "gsr_step_3_example_accelerate"

STEP1_STATE = "step1/sn-gamestate.pklz"
STEP2_DIR = "step2"
STEP3_STATE = "step3/sn-gamestate.pklz"


def _run(command: Sequence[str], cwd: Path, extra_env: dict | None = None) -> None:
    import copy
    import os

    env = copy.deepcopy(os.environ)
    if extra_env:
        env.update(extra_env)
    print("+ " + " ".join(str(c) for c in command))
    subprocess.run([str(c) for c in command], cwd=str(cwd), env=env, check=True)


def _tracking_command(config: str, run_dir: Path, overrides: Sequence[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "soccermaster.tracking.main",
        "-cn",
        config,
        f"hydra.run.dir={run_dir}",
        "hydra.output_subdir=null",
        *overrides,
    ]


def run_step1(
    dataset_path: Path,
    out_dir: Path,
    video_ids: Sequence[int] | None = None,
    eval_set: str = "sn500",
    device: str = "cuda",
    extra_overrides: Sequence[str] = (),
) -> Path:
    """Detection + tracking. Writes ``<out_dir>/step1/sn-gamestate.pklz``."""
    state_path = (out_dir / STEP1_STATE).resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"data_dir={paths.data_dir()}",
        f"model_dir={paths.weights_dir()}",
        f"dataset.dataset_path={dataset_path}",
        f"dataset.eval_set={eval_set}",
        f"state.save_file={state_path}",
        "state.load_file=null",
        f"experiment_subname=step_1_{eval_set}",
        *extra_overrides,
    ]
    if video_ids:
        listed = ",".join(str(v) for v in video_ids)
        overrides.append(f"dataset.vids_dict.{eval_set}=[{listed}]")

    _run(
        _tracking_command(STEP1_CONFIG, out_dir / "step1", overrides),
        cwd=out_dir,
        extra_env={"CUDA_VISIBLE_DEVICES": device},
    )
    return state_path


def run_step2(
    step1_state: Path,
    dataset_path: Path,
    out_dir: Path,
    video_ids: Sequence[int] | None = None,
    eval_set: str = "sn500",
    device: str = "cuda",
    sam_checkpoint: Path | None = None,
    sam_config: str | None = None,
    extra_args: Sequence[str] = (),
) -> Path:
    """SAM 2 segmentation refinement. Requires the optional ``sam2`` extra.

    Writes ``<out_dir>/step2/refined_<step1 name>`` and returns its path.
    """
    refine_dir = (out_dir / STEP2_DIR).resolve()
    refine_dir.mkdir(parents=True, exist_ok=True)
    refined = refine_dir / f"refined_{step1_state.name}"

    checkpoint = sam_checkpoint or (paths.weights_dir() / "sam2" / "sam2.1_hiera_large.pt")
    config = sam_config or "configs/sam2.1/sam2.1_hiera_l.yaml"

    command = [
        sys.executable,
        "-m",
        "soccermaster.refine.inference",
        "--input_pklz",
        str(step1_state),
        "--dataset_root",
        str(dataset_path),
        "--output_dir",
        str(refine_dir),
        "--split",
        eval_set,
        "--sam_checkpoint",
        str(checkpoint),
        "--sam_config",
        config,
        "--gpu_list",
        device,
        "--save_refined_pklz",
        "--save_pklz_path",
        str(refined),
        *extra_args,
    ]
    if video_ids:
        command += ["--video_id_list", ",".join(str(v) for v in video_ids)]

    _run(command, cwd=out_dir, extra_env={"CUDA_VISIBLE_DEVICES": device})
    return refined


def run_step3(
    step2_state: Path,
    dataset_path: Path,
    out_dir: Path,
    video_ids: Sequence[int] | None = None,
    eval_set: str = "sn500",
    device: str = "cuda",
    extra_overrides: Sequence[str] = (),
) -> Path:
    """Calibration, jersey number, role, team. Writes step-3 state."""
    state_path = (out_dir / STEP3_STATE).resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"data_dir={paths.data_dir()}",
        f"model_dir={paths.weights_dir()}",
        f"dataset.dataset_path={dataset_path}",
        f"dataset.eval_set={eval_set}",
        f"state.load_file={(step2_state).resolve()}",
        f"state.save_file={state_path}",
        f"experiment_subname=step_3_{eval_set}",
        *extra_overrides,
    ]
    if video_ids:
        listed = ",".join(str(v) for v in video_ids)
        overrides.append(f"dataset.vids_dict.{eval_set}=[{listed}]")

    _run(
        _tracking_command(STEP3_CONFIG, out_dir / "step3", overrides),
        cwd=out_dir,
        extra_env={"CUDA_VISIBLE_DEVICES": device},
    )
    return state_path


def run_pipeline(
    frames_dir: str | Path,
    out_dir: str | Path | None = None,
    steps: Sequence[int] = (1, 2, 3),
    video_ids: Sequence[int] | None = None,
    eval_set: str = "sn500",
    device: str = "cuda",
    extra_overrides: Sequence[str] | None = None,
) -> Path:
    """Run the requested pipeline steps.

    ``frames_dir`` is the dataset root in SoccerNet layout, i.e. a directory
    containing ``<eval_set>/<sequence>/img1/*.jpg`` plus ``sequences_info.json``.
    Returns the output directory.
    """
    dataset_path = Path(frames_dir).expanduser().resolve()
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"dataset path does not exist: {dataset_path}")

    out_dir = Path(out_dir).expanduser().resolve() if out_dir else (paths.outputs_dir() / "pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    extra = list(extra_overrides or ())
    steps = list(steps)

    step1_state = out_dir / STEP1_STATE
    step2_state = out_dir / STEP2_DIR / f"refined_{Path(STEP1_STATE).name}"

    if 1 in steps:
        step1_state = run_step1(dataset_path, out_dir, video_ids, eval_set, device, extra)
    if 2 in steps:
        if not step1_state.is_file():
            raise FileNotFoundError(
                f"step 2 needs the step-1 state at {step1_state}; run step 1 first"
            )
        step2_state = run_step2(step1_state, dataset_path, out_dir, video_ids, eval_set, device)
    if 3 in steps:
        if not step2_state.is_file():
            raise FileNotFoundError(
                f"step 3 needs the step-2 state at {step2_state}; run step 2 first"
            )
        run_step3(step2_state, dataset_path, out_dir, video_ids, eval_set, device, extra)

    return out_dir
