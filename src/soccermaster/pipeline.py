"""Orchestration of the three-step SoccerNet game-state pipeline.

The steps are executed exactly as documented in the upstream README -- each step
is the same Hydra-driven entry point as before (``soccermaster.tracklab.main``
and ``soccermaster.sam2_refine``), invoked with explicit path overrides so nothing
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

#: Where PitchVisualizationEngine writes, relative to the process CWD (it hardcodes
#: ``Path("visualization")``). Each step runs with CWD set to its own run dir, so the
#: two steps do not overwrite each other's video.
VIS_SUBDIR = Path("visualization") / "videos"


def sequence_name(video_id: int) -> str:
    """``10001`` -> ``'SNGS-10001'``, the directory name the dataset wrapper expects."""
    return f"SNGS-{int(video_id):05d}"


def _run(command: Sequence[str], cwd: Path, extra_env: dict | None = None) -> None:
    """Run a pipeline step, showing its output and surfacing its error.

    The step runs in a subprocess (Hydra owns that entry point). Its output is
    captured, echoed, and on failure re-raised with the tail of stderr attached --
    without this a failing step surfaces only as ``CalledProcessError`` with no
    reason, which is the worst possible thing to debug from a notebook.
    """
    import copy
    import os

    env = copy.deepcopy(os.environ)
    # torch >= 2.6 flipped torch.load's weights_only default to True, which breaks
    # every third-party checkpoint loader in this stack that does a plain
    # torch.load(path) -- ultralytics, prtreid, and anything else in the vendored
    # trackers. Some we can fix in our own code (see
    # tracklab/utils/ultralytics_compat.py); prtreid is an external package we cannot
    # edit, and it fails with:
    #
    #   _pickle.UnpicklingError: Weights only load failed. ...
    #   Unsupported global: GLOBAL numpy.core.multiarray.scalar
    #
    # So set torch's documented escape hatch for the child process. The checkpoints
    # this loads are model weights fetched by the pipeline itself -- from Zenodo with
    # md5 verification, or from Hugging Face -- i.e. the same trust level as any
    # pretrained weights, and exactly the "trusted source" case torch documents.
    env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if extra_env:
        env.update(extra_env)

    print("+ " + " ".join(str(c) for c in command), flush=True)
    result = subprocess.run(
        [str(c) for c in command], cwd=str(cwd), env=env,
        capture_output=True, text=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    if result.returncode:
        tail = "\n".join((result.stderr or result.stdout or "").strip().splitlines()[-25:])
        raise RuntimeError(
            f"step failed (exit {result.returncode}): {' '.join(str(c) for c in command)}\n"
            f"--- last output ---\n{tail}"
        )


def _tracking_command(config: str, run_dir: Path, overrides: Sequence[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "soccermaster.tracklab.main",
        "-cn",
        config,
        f"hydra.run.dir={run_dir}",
        "hydra.output_subdir=null",
        *overrides,
    ]


def _video_filter_overrides(
    video_ids: Sequence[int] | None, eval_set: str
) -> list[str]:
    """Filter the split to specific sequences.

    The dataset wrapper compares the filter against ``os.listdir()``, i.e. against
    directory names like ``SNGS-10001``. Passing bare integers silently matches
    nothing and ends in ``AssertionError: no videos left in the 'sn500' set``, so the
    names are formatted and quoted here.
    """
    if not video_ids:
        return []
    names = ",".join(f'"{sequence_name(v)}"' for v in video_ids)
    return [f"dataset.vids_dict.{eval_set}=[{names}]"]



def _visualization_overrides(save_videos: bool, save_images: bool = False) -> list[str]:
    """Turn the annotated-output callbacks on or off.

    The shipped config has both off, so without this the pipeline silently produces
    no video even though the step configs' comments claim it saves one.
    """
    return [
        f"visualization.cfg.save_videos={save_videos}",
        f"visualization.cfg.save_images={save_images}",
    ]


def recommended_workers() -> int:
    """A sane ``num_cores`` for this machine.

    Upstream's configs hardcode ``num_cores: 64`` (step 1) and ``32`` (step 3) because
    they were run on a 64-core server. That value is fed to the ReID dataloader
    (``workers: ${num_cores}``), the tracking engine (``num_workers``) and the
    visualizer, so on a normal box -- Colab gives you about 2 vCPUs -- it spawns 64
    workers onto 2 cores. The run does not crash; it thrashes and looks hung, which is
    how a handful of frames can take half an hour.

    Cap at the machine's CPU count, and leave a core free when we can.
    """
    import os

    cpus = os.cpu_count() or 4
    return max(1, min(8, cpus - 1))


def _worker_overrides(workers: int | None = None) -> list[str]:
    return [f"num_cores={workers if workers is not None else recommended_workers()}"]


def _reid_scratch_overrides(run_dir: Path) -> list[str]:
    """Keep prtreid's scratch directory out of the CWD and out of its own way.

    prtreid builds ``os.makedirs(save_path/job_id)`` with no ``exist_ok``, where
    ``save_path`` comes from the config as the relative ``reid`` and ``job_id``
    defaults to ``0``. Two consequences:

    - the directory is created relative to the working directory, which for us is the
      step's run dir, so it lands wherever the run happens to be
    - it fails with ``FileExistsError: 'reid/0'`` on the *second* run, because the
      first one left the directory behind

    Point it at an absolute per-run path and give each run its own ``job_id``, so the
    directory never pre-exists. ``job_id`` is the config's own knob for exactly this
    ("${oc.env:SLURM_JOBID,0}"); it is unused here because we do not run under Slurm.
    """
    import random

    scratch = Path(run_dir).resolve() / "reid"
    # Must be an int: prtreid merges our config over a yacs CfgNode whose
    # project.job_id came from "${oc.env:SLURM_JOBID,0}", so a string here fails
    # with "Type mismatch (<class 'int'> vs. <class 'str'>)".
    job_id = random.randint(1, 2**31 - 1)
    return [
        f"modules.reid.save_path={scratch}",
        f"modules.reid.job_id={job_id}",
    ]


def run_step1(
    dataset_path: Path,
    out_dir: Path,
    video_ids: Sequence[int] | None = None,
    eval_set: str = "sn500",
    device: str = "cuda",
    save_videos: bool = False,
    workers: int | None = None,
    extra_overrides: Sequence[str] = (),
) -> Path:
    """Detection + tracking. Writes ``<out_dir>/step1/sn-gamestate.pklz``.

    With ``save_videos`` an annotated mp4 lands in
    ``<out_dir>/step1/visualization/videos/`` (boxes + track ids only; jersey
    numbers, roles and teams come from step 3).
    """
    run_dir = (out_dir / "step1").resolve()
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
        *_worker_overrides(workers),
        *_reid_scratch_overrides(run_dir),
        *_visualization_overrides(save_videos),
        *_video_filter_overrides(video_ids, eval_set),
        *extra_overrides,
    ]

    # CWD is the run dir because the visualization callback resolves its output
    # relative to CWD; everything else the step reads is an absolute path.
    _run(
        _tracking_command(STEP1_CONFIG, run_dir, overrides),
        cwd=run_dir,
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
        "soccermaster.sam2_refine.inference",
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
    save_videos: bool = False,
    workers: int | None = None,
    extra_overrides: Sequence[str] = (),
) -> Path:
    """Calibration, jersey number, role, team. Writes step-3 state.

    With ``save_videos`` an annotated mp4 lands in
    ``<out_dir>/step3/visualization/videos/`` -- this is the one with jersey
    numbers, teams and roles drawn on it.
    """
    run_dir = (out_dir / "step3").resolve()
    state_path = (out_dir / STEP3_STATE).resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    overrides = [
        f"data_dir={paths.data_dir()}",
        f"model_dir={paths.weights_dir()}",
        f"dataset.dataset_path={dataset_path}",
        f"dataset.eval_set={eval_set}",
        f"state.load_file={Path(step2_state).resolve()}",
        f"state.save_file={state_path}",
        f"experiment_subname=step_3_{eval_set}",
        *_worker_overrides(workers),
        *_reid_scratch_overrides(run_dir),
        *_visualization_overrides(save_videos),
        *_video_filter_overrides(video_ids, eval_set),
        *extra_overrides,
    ]

    _run(
        _tracking_command(STEP3_CONFIG, run_dir, overrides),
        cwd=run_dir,
        extra_env={"CUDA_VISIBLE_DEVICES": device},
    )
    return state_path


def find_annotated_videos(out_dir: Path) -> list[Path]:
    """Every annotated mp4 under a pipeline output dir, newest step last."""
    return sorted(Path(out_dir).rglob("visualization/videos/*.mp4"))


def extract_video_frames(
    video: str | Path,
    dataset_dir: str | Path,
    eval_set: str = "sn500",
    video_id: int = 10001,
    quality: int = 95,
    fps: float | None = None,
) -> Path:
    """Decode one video into the SoccerNet layout the pipeline expects.

    Creates ``<dataset_dir>/<eval_set>/SNGS-<video_id>/img1/*.jpg`` -- the sequence
    directory is named ``SNGS-<video_id>`` regardless of the input filename, because
    the dataset wrapper matches on that pattern.

    ``fps`` decimates while decoding. Every downstream step is per-frame, so this is the
    one control that scales the cost of a match: an 11-minute video at 25 fps is ~16,500
    frames, and each frame is tracked and then queried against a 7B VLM per player.

    Requires ffmpeg on PATH.
    """
    from soccermaster.model.data.extract_frames import extract_frames_from_video

    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"no such video: {video}")

    seq_dir = Path(dataset_dir).expanduser().resolve() / eval_set / sequence_name(video_id)
    img1 = seq_dir / "img1"
    if img1.is_dir() and any(img1.iterdir()):
        print(f"frames already present, skipping decode: {img1}")
        return seq_dir

    result = extract_frames_from_video(str(video), str(seq_dir), quality=quality, fps=fps)
    if not result.get("success"):
        raise RuntimeError(f"ffmpeg failed for {video}: {result.get('reason')}")
    print(f"decoded {result['num_frames']} frames into {img1}")
    return seq_dir


def analyze_video(
    video: str | Path,
    out_dir: str | Path | None = None,
    eval_set: str = "sn500",
    video_id: int = 10001,
    steps: Sequence[int] = (1, 3),
    device: str = "cuda",
    quality: int = 95,
    write_video: bool = True,
    extra_overrides: Sequence[str] | None = None,
    fps: float | None = None,
) -> dict:
    """Video in, structured data out.

    Decodes the video into the SoccerNet layout, runs the requested steps, then
    exports everything the run observed: per-frame detections with boxes in image
    pixels and (once calibration has run) pitch metres, jersey numbers, roles and
    teams, a per-tracklet summary, and the per-frame camera parameters.

    The annotated video is written alongside that data when ``write_video`` is set.
    It is a view of the same run, not the deliverable.

    Step 2 (SAM 2 mask refinement) is off by default: it needs the ``sam2`` extra, a
    CUDA build and its own checkpoint.

    Returns ``{"dataset", "out_dir", "states", "video_id", "sequence", "exports",
    "videos"}``.
    """
    from soccermaster import export as export_mod
    from soccermaster import paths as _paths

    video = Path(video).expanduser().resolve()
    out_dir = out_dir if out_dir is not None else (_paths.outputs_dir() / Path(video).stem)
    out_dir = Path(out_dir).expanduser().resolve()
    dataset_dir = out_dir / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = list(steps)
    extra = list(extra_overrides or ())
    sequence = sequence_name(video_id)

    print(f"video    : {video}")
    print(f"output   : {out_dir}")
    print(f"sequence : {sequence}")
    print(f"steps    : {steps}\n")

    extract_video_frames(video, dataset_dir, eval_set, video_id, quality, fps=fps)

    states: dict[int, Path] = {}
    step1_state = out_dir / STEP1_STATE
    step2_state = out_dir / STEP2_DIR / f"refined_{Path(STEP1_STATE).name}"

    if 1 in steps:
        step1_state = run_step1(
            dataset_dir, out_dir, [video_id], eval_set, device,
            save_videos=write_video, extra_overrides=extra,
        )
    states[1] = step1_state

    if 2 in steps:
        if not step1_state.is_file():
            raise FileNotFoundError(f"step 2 needs {step1_state}; run step 1 first")
        step2_state = run_step2(
            step1_state, dataset_dir, out_dir, [video_id], eval_set, device
        )
        states[2] = step2_state

    if 3 in steps:
        source = step2_state if step2_state.is_file() else step1_state
        if not source.is_file():
            raise FileNotFoundError(f"step 3 needs {source}; run step 1 first")
        states[3] = run_step3(
            source, dataset_dir, out_dir, [video_id], eval_set, device,
            save_videos=write_video, extra_overrides=extra,
        )

    # Export from the richest state available: step 3 knows jersey numbers and
    # teams, step 1 knows none of that.
    exports: dict[str, Path] = {}
    richest = max(
        (s for s in steps if s in states and Path(states[s]).is_file()), default=None
    )
    if richest is not None:
        print(f"\nexporting data from the step-{richest} state: {states[richest]}")
        exports = export_mod.export_state(
            states[richest], out_dir / "data", prefix=sequence
        )
        for kind, path in exports.items():
            print(f"  {kind:<18} {path}")
    else:
        print("\nno tracker state was produced, so there is nothing to export.")

    videos = find_annotated_videos(out_dir)
    if write_video:
        if videos:
            print("\nannotated video(s):")
            for path in videos:
                print(f"  {path}  ({path.stat().st_size / 1e6:.1f} MB)")
        else:
            print(
                "\nno annotated video was written. The visualization callback enables "
                "only when soccermaster.soccernet_gamestate.visualization is imported; "
                "check the run log for a 'visualization' line."
            )

    return {
        "dataset": dataset_dir,
        "out_dir": out_dir,
        "states": states,
        "video_id": video_id,
        "sequence": sequence,
        "exports": exports,
        "videos": videos,
    }


def run_pipeline(
    frames_dir: str | Path,
    out_dir: str | Path | None = None,
    steps: Sequence[int] = (1, 2, 3),
    video_ids: Sequence[int] | None = None,
    eval_set: str = "sn500",
    device: str = "cuda",
    save_videos: bool = False,
    extra_overrides: Sequence[str] | None = None,
) -> Path:
    """Run the requested pipeline steps on a frames directory.

    ``frames_dir`` is the dataset root in SoccerNet layout, i.e. a directory
    containing ``<eval_set>/<sequence>/img1/*.jpg`` plus ``sequences_info.json``.
    Returns the output directory.

    This only runs the steps and writes state files. Use :func:`analyze_video` when
    you have a video and want the extracted data back.
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
        step1_state = run_step1(
            dataset_path, out_dir, video_ids, eval_set, device,
            save_videos=save_videos, extra_overrides=extra,
        )
    if 2 in steps:
        if not step1_state.is_file():
            raise FileNotFoundError(
                f"step 2 needs the step-1 state at {step1_state}; run step 1 first"
            )
        step2_state = run_step2(step1_state, dataset_path, out_dir, video_ids, eval_set, device)
    if 3 in steps:
        source = step2_state if step2_state.is_file() else step1_state
        if not source.is_file():
            raise FileNotFoundError(
                f"step 3 needs a tracker state ({step1_state} or {step2_state}); "
                "run step 1 first"
            )
        run_step3(
            source, dataset_path, out_dir, video_ids, eval_set, device,
            save_videos=save_videos, extra_overrides=extra,
        )

    return out_dir
