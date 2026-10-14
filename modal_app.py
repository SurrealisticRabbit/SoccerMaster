"""Run SoccerMaster on Modal.

Modal suits this project better than a notebook runtime:

- the pipeline is a CLI, not interactive exploration, so ``modal run`` is a
  one-command feedback loop with a single clean log
- weights live in a volume and are downloaded once, rather than on every session
- no notebook session to recycle mid-run
- the image is baked, so every run has byte-identical dependencies

Setup (once):

    pip install modal
    modal token new

Run:

    modal run modal_app.py::doctor                  # read-only pre-flight
    modal run modal_app.py::prepare                 # fetch weights into the volume
    modal run modal_app.py::analyze --video my.mp4  # video in, data out

The video is uploaded from ``./videos`` (see VIDEO_DIR) and the extracted data comes
back under ``./modal-outputs`` (see fetch_outputs).

NOTE: this file has not been executed -- the author of the surrounding work had no
Modal access. Treat it as a well-researched draft: the Modal API surface is correct as
of Modal 1.x, but expect to fix one or two things on first run.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REPO = "https://github.com/SurrealisticRabbit/SoccerMaster.git"
REF = "main"  # pin to a tag or commit once you care about reproducibility

#: Local directory holding the video(s) to analyse. Mounted read-only.
VIDEO_DIR = Path("videos")
VIDEO_DIR.mkdir(exist_ok=True)

#: Local directory the exported data is written back to.
LOCAL_OUTPUT_DIR = Path("modal-outputs")

#: Python version. 3.11 is the conservative choice: the package needs >=3.10, and
#: some pinned dependencies (ultralytics==8.0.100) predate 3.13.
PYTHON_VERSION = "3.11"

app = modal.App("soccermaster")


# --------------------------------------------------------------------------- #
# Image
# --------------------------------------------------------------------------- #

# apt packages, and why each is needed:
#   ffmpeg            frame extraction (soccermaster analyze decodes video)
#   git               the pipeline extra installs prtreid/torchreid/poseval from git
#   build-essential   those same packages build wheels on install
#   libgl1            cv2 (opencv) fails to import without libGL
#   libglib2.0-0      cv2 again
image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("ffmpeg", "git", "build-essential", "libgl1", "libglib2.0-0")
    # Plain PyPI torch is the CUDA build on Linux, which is what Modal's GPU hosts
    # want. Install it first so the soccermaster install does not get a chance to
    # choose a different build.
    .pip_install("torch>=2.4", "torchvision>=0.19")
    # The `pipeline` extra carries everything the game-state steps import. Omit the
    # `sam2` extra: step 2 needs the vendored third_party/sam2 tree as well, which is
    # not in the wheel.
    .pip_install(f"soccermaster[pipeline] @ git+{REPO}@{REF}")
)

#: Weights persist here. ~5.5 GB of downloads, so this matters.
weights_volume = modal.Volume.from_name("soccermaster-weights", create_if_missing=True)

#: Pipeline outputs and exported data.
outputs_volume = modal.Volume.from_name("soccermaster-outputs", create_if_missing=True)

#: SOCCERMASTER_HOME is the single root the package reads and writes. Mounting the two
#: volumes underneath it is what makes paths.weights_dir() and paths.outputs_dir()
#: resolve to persistent storage.
VOLUMES = {
    "/data/weights": weights_volume,
    "/data/outputs": outputs_volume,
}

ENV = {"SOCCERMASTER_HOME": "/data"}

#: Set explicitly rather than relying only on the package's own default. torch >= 2.6
#: flipped torch.load's weights_only to True, which breaks the third-party checkpoint
#: loaders in this stack (prtreid, ultralytics). soccermaster sets this itself for the
#: child processes it spawns; setting it in the image covers direct CLI use too.
ENV["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


# --------------------------------------------------------------------------- #
# Remote functions
# --------------------------------------------------------------------------- #


@app.function(image=image, volumes=VOLUMES, env=ENV, timeout=1800)
def doctor(steps: str = "1,3") -> str:
    """Read-only pre-flight. Cheap enough to run before every attempt.

    Answers "will the pipeline start, or am I about to hit a wall?": environment,
    imports, weights present, and whether every _target_ in the composed Hydra configs
    actually resolves.
    """
    import subprocess
    import sys

    # The dataset check is skipped: frames are created by analyze(), so there is
    # nothing to inspect yet.
    proc = subprocess.run(
        [sys.executable, "-m", "soccermaster", "doctor", "--steps", steps],
        capture_output=True,
        text=True,
    )
    return proc.stdout + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else "")


@app.function(image=image, volumes=VOLUMES, env=ENV, timeout=7200)
def prepare(names: list[str] | None = None) -> str:
    """Download weights into the volume. Run once; they persist.

    ``names`` defaults to what steps 1 and 3 need. Costs no GPU, so it is deliberately
    not a GPU function -- but it does need a lot of disk, which a Modal container has.
    """
    import subprocess
    import sys

    names = names or ["siglip", "soccermaster", "yolo", "qwen7b"]
    out = [f"fetching: {', '.join(names)}"]
    for name in names:
        proc = subprocess.run(
            [sys.executable, "-m", "soccermaster", "prepare", name],
            capture_output=True,
            text=True,
        )
        out.append(proc.stdout.strip())
        if proc.returncode:
            out.append(f"!! {name} failed:\n{proc.stderr[-1500:]}")

    # Persist before the container shuts down.
    weights_volume.commit()
    return "\n".join(out)


@app.function(
    image=image.add_local_dir(VIDEO_DIR, remote_path="/videos"),
    gpu="T4",
    volumes=VOLUMES,
    env=ENV,
    timeout=7200,
)
def analyze(
    video: str,
    video_id: int = 10001,
    steps: tuple[int, ...] = (1, 3),
    workers: int | None = None,
) -> dict:
    """Video in, structured data out.

    Decodes the video, runs the game-state steps, and exports every tracklet with its
    boxes (image pixels and pitch metres once calibration has run), jersey number, role
    and team. Returns a summary plus the exported files' contents.
    """
    import json
    import traceback
    from pathlib import Path

    import soccermaster as sm

    video_path = Path("/videos") / video
    if not video_path.is_file():
        return {"error": f"no such video in the mount: {video_path}"}

    try:
        result = sm.analyze_video(
            video_path,
            out_dir="/data/outputs/modal",
            video_id=video_id,
            steps=list(steps),
            device="0",  # a single-GPU Modal worker; 0 is the visible device
            write_video=True,
        )
    except Exception:
        return {"error": traceback.format_exc()}

    # Read the exported data back so the caller gets it without a volume round-trip.
    payload: dict = {
        "out_dir": str(result["out_dir"]),
        "exports": {k: str(v) for k, v in result["exports"].items()},
        "videos": [str(v) for v in result["videos"]],
        "files": {},
    }
    for kind, path in result["exports"].items():
        path = Path(path)
        if not path.is_file():
            continue
        if path.suffix == ".json":
            payload["files"][kind] = json.loads(path.read_text())
        elif path.suffix == ".csv":
            lines = path.read_text().splitlines()
            payload["files"][kind] = {"header": lines[:1], "n_rows": max(0, len(lines) - 1), "preview": lines[:6]}

    outputs_volume.commit()
    return payload


@app.function(image=image, volumes=VOLUMES, env=ENV, timeout=1800)
def list_outputs() -> list[str]:
    """Everything under the outputs volume, so nothing gets lost between sessions."""
    root = Path("/data/outputs")
    if not root.is_dir():
        return []
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


# --------------------------------------------------------------------------- #
# Local entry points
# --------------------------------------------------------------------------- #


@app.local_entrypoint()
def main(cmd: str = "doctor", video: str = "", video_id: int = 10001) -> None:
    """``modal run modal_app.py --cmd <doctor|prepare|list|analyze>``"""
    if cmd == "doctor":
        print(doctor.remote())
    elif cmd == "prepare":
        print(prepare.remote())
    elif cmd == "list":
        for line in list_outputs.remote():
            print(line)
    elif cmd == "analyze":
        if not video:
            available = [p.name for p in VIDEO_DIR.iterdir() if p.is_file()]
            raise SystemExit(f"--video is required. In {VIDEO_DIR}/: {available or 'nothing'}")
        result = analyze.remote(video=video, video_id=video_id)
        import json

        print(json.dumps(result, indent=2, default=str)[:8000])
    else:
        raise SystemExit(f"unknown cmd {cmd!r}; use doctor, prepare, list or analyze")
