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

    modal run modal_app.py --cmd doctor
    modal run modal_app.py --cmd prepare
    modal run modal_app.py --cmd video --url '<share link>'
    modal run modal_app.py --cmd analyze --video fb_cut.mp4
    modal run modal_app.py --cmd fetch

Videos live in the ``soccermaster-videos`` volume rather than being baked into the image,
so a large clip is downloaded once instead of re-uploaded into the image on every run.
``--cmd fetch`` pulls the exported data back into ``./modal-outputs``.

``doctor`` has been run against the real app. The rest has not, so expect to fix one or
two things on first use.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REPO = "https://github.com/SurrealisticRabbit/SoccerMaster.git"

#: A specific commit, NOT a branch name -- and that is not just about reproducibility.
#: Modal caches image layers by their definition string, and `@main` is a fixed string
#: that never changes as the branch advances, so the cache never invalidates and the
#: container keeps running whatever `main` pointed at when the image was first built.
#: That happened here: `doctor` reported `soccermaster 1.0.1a0` (the tag commit, i.e.
#: what `main` was at the time) while `main` had already moved on, so the step-3
#: attention/dtype fixes and `--fps` were silently absent. Pinning a SHA changes the
#: definition string, which forces the rebuild, and makes the installed code version
#: explicit in this file. Bump it on each push you want deployed.
#:
#: Note this governs only the *package* installed from git. modal_app.py itself is
#: uploaded from local disk on every run, so edits here take effect immediately.
#: Bump this when anything under ``src/`` changes, or the image keeps serving the old
#: package: Modal caches layers by definition string and this is what makes it specific.
#: Current value carries the live-output fix (a step's output streams instead of being
#: buffered until exit) and the bounded weight-download timeout.
REF = "26f183f244c5b0738ddc2c095c87c71a5020dd56"

#: Local directory the exported data is written back to by ``--cmd fetch``.
LOCAL_OUTPUT_DIR = Path("modal-outputs")

#: Local folder to drop clips in for ``modal volume put``. Not uploaded automatically:
#: putting a 1.6 GB video into the image would re-upload it on every run.
VIDEO_DIR = Path("videos")

#: Python version. 3.11 is the conservative choice: the package needs >=3.10, and
#: some pinned dependencies (ultralytics==8.0.100) predate 3.13.
PYTHON_VERSION = "3.11"
GPU = "L40S"

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
OUTPUTS_VOLUME_NAME = "soccermaster-outputs"

#: Input videos. Separate from weights so they can be pruned independently.
videos_volume = modal.Volume.from_name("soccermaster-videos", create_if_missing=True)

#: SOCCERMASTER_HOME is the single root the package reads and writes. Mounting the three
#: volumes underneath it is what makes paths.weights_dir() and paths.outputs_dir()
#: resolve to persistent storage.
VOLUMES = {
    "/data/weights": weights_volume,
    "/data/outputs": outputs_volume,
    "/data/videos": videos_volume,
}

ENV = {"SOCCERMASTER_HOME": "/data"}

#: albumentations performs a network update check at import time, which is pure noise
#: here and a second way for an import to block on the network. The check prints a
#: warning we saw in every doctor run; this removes both the warning and the dependency
#: on reaching PyPI before a pipeline step can even start.
ENV["NO_ALBUMENTATIONS_UPDATE"] = "1"

#: Set explicitly rather than relying only on the package's own default. torch >= 2.6
#: flipped torch.load's weights_only to True, which breaks the third-party checkpoint
#: loaders in this stack (prtreid, ultralytics). soccermaster sets this itself for the
#: child processes it spawns; setting it in the image covers direct CLI use too.
ENV["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


# --------------------------------------------------------------------------- #
# Remote functions
# --------------------------------------------------------------------------- #


@app.function(image=image, volumes=VOLUMES, env=ENV, timeout=7200)
def fetch_video(url: str = "", name: str = "") -> dict:
    """Download a video into the videos volume, once.

    Accepts a direct URL or a Google Drive share link. Drive is why this uses gdown
    rather than a plain request: a file over ~100 MB does not come back from
    ``uc?export=download`` at all -- it returns a "Virus scan warning" HTML page and
    needs the confirmation token that gdown negotiates. Verified against the app's own
    link: the confirm URL reports ``video/mp4`` and a real byte count.

    The download persists in a volume, so it is paid for once. Baking the clip into the
    image instead would re-upload it on every run.
    """
    import time
    from pathlib import Path

    import gdown

    if not url:
        return {"error": "url is required, e.g. --url 'https://drive.google.com/file/d/<id>/view'"}

    dest_dir = Path("/data/videos")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # gdown derives a name from the URL, which for a Drive link is an opaque id, so
    # accept an explicit name and otherwise let gdown choose.
    output = str(dest_dir / name) if name else str(dest_dir) + os.sep

    started = time.time()

    # No `fuzzy=` here. That argument existed in gdown 4.x and was removed in 5.0, while
    # pyproject floors gdown at >=5.0 -- so passing it raises TypeError before any
    # download starts. gdown resolves a /file/d/<id>/view link from `url` on its own.
    out = gdown.download(url=url, output=output, quiet=False)

    # The declared return type is a union, so do not assume it is a path.
    if not isinstance(out, (str, os.PathLike)):
        return {"error": f"gdown returned {type(out).__name__}, not a path: {out!r}"}

    path = Path(out)
    if not path.is_file():
        return {"error": f"gdown reported {out} but no file is there"}

    videos_volume.commit()
    size = path.stat().st_size
    return {
        "name": path.name,
        "path": str(path),
        "size_gb": round(size / 1e9, 2),
        "seconds": round(time.time() - started, 1),
    }


@app.function(image=image, volumes=VOLUMES, env=ENV, timeout=600)
def list_videos() -> list[str]:
    """Videos already in the volume, so nothing is downloaded twice."""
    root = Path("/data/videos")
    if not root.is_dir():
        return []
    return sorted(
        f"{p.name}  ({p.stat().st_size / 1e9:.2f} GB)" for p in root.iterdir() if p.is_file()
    )


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

    # Report what the INSTALLED package actually provides. The image's package and this
    # file version independently, and that mismatch cost a run: `sm.analyze_video`
    # resolved through soccermaster.api to a copy that had lost `fps`, which looked
    # exactly like a stale image and sent me bisecting the image cache instead. Printing
    # the real signature and the identity check makes that a one-command answer.
    import inspect

    import soccermaster
    from soccermaster import pipeline

    lines = [
        "",
        "Installed package (what this image actually has)",
        "-----------------------------------------------",
        f"  version              : {soccermaster.__version__}",
        f"  path                 : {soccermaster.__file__}",
        f"  analyze_video params : {list(inspect.signature(pipeline.analyze_video).parameters)}",
        "  sm.analyze_video is pipeline.analyze_video: "
        f"{soccermaster.analyze_video is pipeline.analyze_video}",
    ]

    out = proc.stdout + "\n".join(lines)
    return out + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else "")


def default_weight_names(steps: tuple[int, ...] = (1, 3)) -> list[str]:
    """The manifest weights steps 1 and 3 need, read from doctor's own table.

    Derived rather than hand-listed, because the hand-written list this replaced had
    drifted from the truth in *both* directions: it omitted ``legibility``, which step 3
    requires (doctor reports it as a hard failure before any run), and it included
    ``siglip`` and ``soccermaster``, which the three-step pipeline never loads -- several
    GB fetched for nothing. Reading STEP_REQUIREMENTS means prepare and doctor cannot
    disagree again.

    Deliberately lives here rather than in the package: modal_app.py is uploaded from
    local disk on every run, so this takes effect immediately, whereas the same change
    inside soccermaster would need a new commit, a bumped REF and another image rebuild.
    """
    from soccermaster.preflight import STEP_REQUIREMENTS

    names: list[str] = []
    for step in steps:
        names += STEP_REQUIREMENTS[step]["manifest"]
    # reid/calibration are `runtime` (the pipeline fetches them itself) and so are absent
    # by construction. dict.fromkeys keeps insertion order while de-duplicating.
    return list(dict.fromkeys(names))


@app.function(image=image, volumes=VOLUMES, env=ENV, timeout=7200)
def prepare(names: list[str] | None = None) -> str:
    """Download weights into the volume. Run once; they persist.

    ``names`` defaults to what steps 1 and 3 need. Costs no GPU, so it is deliberately
    not a GPU function -- but it does need a lot of disk, which a Modal container has.
    """
    import subprocess
    import sys

    names = names or default_weight_names()
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
    image=image,
    gpu=GPU,
    volumes=VOLUMES,
    env=ENV,
    timeout=7200,
)
def analyze(
    video: str,
    video_id: int = 10001,
    steps: tuple[int, ...] = (1, 3),
    workers: int | None = None,
    fps: float | None = None,
) -> dict:
    """Video in, structured data out.

    Decodes the video, runs the game-state steps, and exports every tracklet with its
    boxes (image pixels and pitch metres once calibration has run), jersey number, role
    and team. Returns a summary plus the exported files' contents.

    ``fps`` decimates while decoding. Every step runs per frame, so on an 11-minute match
    (~16,500 frames at 25 fps) step 3 means asking a 7B VLM about every player in every
    frame -- pass ``fps=1`` for a first smoke test rather than waiting on the full set.
    """
    import json
    import traceback
    from pathlib import Path

    import soccermaster as sm

    videos_dir = Path("/data/videos")
    video_path = Path(video) if Path(video).is_absolute() else videos_dir / video
    if not video_path.is_file():
        return {
            "error": f"no such video: {video_path}",
            "available": (
                sorted(p.name for p in videos_dir.iterdir() if p.is_file())
                if videos_dir.is_dir()
                else []
            ),
            "hint": "fetch one first: --cmd video --url '<share link>'",
        }

    # `workers` was accepted and ignored. num_cores is the override that actually
    # controls dataloader workers, and doctor warns loudly when it exceeds the CPU count.
    extra = [f"num_cores={workers}"] if workers else None

    try:
        result = sm.analyze_video(
            video_path,
            out_dir="/data/outputs/modal",
            video_id=video_id,
            steps=list(steps),
            device="0",  # a single-GPU Modal worker; 0 is the visible device
            write_video=True,
            fps=fps,
            extra_overrides=extra,
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
def main(
    cmd: str = "doctor",
    video: str = "",
    video_id: int = 10001,
    url: str = "",
    name: str = "",
    fps: float = 0.0,
    steps: str = "1,3",
) -> None:
    """``modal run modal_app.py --cmd <doctor|prepare|video|videos|analyze|list|fetch>``"""
    import json
    import subprocess

    def as_steps() -> tuple[int, ...]:
        parts = [p.strip() for p in str(steps).replace("|", ",").split(",")]
        return tuple(int(p) for p in parts if p)

    if cmd == "doctor":
        print(doctor.remote())
    elif cmd == "prepare":
        print(prepare.remote())
    elif cmd == "video":
        print(json.dumps(fetch_video.remote(url=url, name=name), indent=2))
    elif cmd == "videos":
        for line in list_videos.remote():
            print(line)
    elif cmd == "list":
        for line in list_outputs.remote():
            print(line)
    elif cmd == "analyze":
        if not video:
            known = list_videos.remote()
            raise SystemExit(
                "--video is required. In the videos volume:\n  "
                + ("\n  ".join(known) if known else "(none -- add one with --cmd video)")
            )
        result = analyze.remote(
            video=video,
            video_id=video_id,
            steps=as_steps(),
            fps=fps or None,
        )
        print(json.dumps(result, indent=2, default=str)[:8000])
    elif cmd == "fetch":
        # Shell out to the CLI rather than reaching for a volume API: `modal volume get`
        # is the documented, stable way to pull a volume down, and it handles nesting.
        LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["modal", "volume", "get", OUTPUTS_VOLUME_NAME, "/", str(LOCAL_OUTPUT_DIR)],
            text=True,
        )
        print(f"\nexit {proc.returncode}; data is under {LOCAL_OUTPUT_DIR}/")
    else:
        raise SystemExit(
            f"unknown cmd {cmd!r}; use doctor, prepare, video, videos, analyze, list or fetch"
        )
