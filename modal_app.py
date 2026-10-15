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

Getting around a slow Origin
---------------------------
Two checkpoints are fetched lazily rather than by ``prepare``: the ReID weights and the
calibration sets. They come from Zenodo, which serves a Modal container at roughly
110 KB/s -- about 20x slower than the same URL from a laptop -- so a 396 MB checkpoint is
nearly an hour and the connection tends to drop part-way. The downloader resumes and
retries, so it does eventually finish, but the fastest route is to fetch it once locally
and push it into the volume::

    curl -L -o prtreid-soccernet-baseline.pth.tar \\
      "https://zenodo.org/records/10653453/files/prtreid-soccernet-baseline.pth.tar?download=1"
    curl -L -o hrnetv2_w32_imagenet_pretrained.pth \\
      "https://zenodo.org/records/10604211/files/hrnetv2_w32_imagenet_pretrained.pth?download=1"
    modal volume put soccermaster-weights prtreid-soccernet-baseline.pth.tar /reid/
    modal volume put soccermaster-weights hrnetv2_w32_imagenet_pretrained.pth /reid/

The pipeline accepts them only if they verify: ``download_file`` checks
``prtreid-soccernet-baseline.pth.tar`` against md5 ``9633825232bc89f23a94522c5561650e`` and
``hrnetv2_w32_imagenet_pretrained.pth`` against ``58ea12b0420aa3adaa2f74114c9f9721``, and
refetches anything that does not match. Check locally first with ``md5sum``/``certutil``.

``--cmd net`` measures throughput from the container per host, which distinguishes a slow
origin from a slow container.

What costs money and what does not
----------------------------------
Only running containers cost anything, and GPU time is what costs most. With nothing
running the bill stops: ``min_containers`` defaults to 0, so nothing is kept warm.

Storage is not a reason to fetch promptly. Volumes are **$0.09/GiB/month with 1 TiB
included**, and these three volumes total roughly 23 GiB (weights ~20, outputs ~1.2,
video 1.6), so leaving results in place overnight costs nothing. Usage is measured by a
once-daily snapshot, and deleting data can still be billed for up to four days, so
tidying up is not an instant saving either.

Two things worth knowing about a run's cost:

* the container is billed for its *lifetime*, not for GPU utilisation, so a slow weight
  download or an ffmpeg decode is paid for at GPU rates while the card sits idle;
* ``scaledown_window`` defaults to 60s, so every run carries a short billed tail after it
  returns.

Pass ``write_video=False`` for a long run to skip several GB of CPU work on a GPU-priced
container.

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
#: Current value carries the calibration fix: six sites across four step-3 modules read a
#: batch column with ``[0]``, which is label lookup against an index of image ids, so
#: step 3 raised ``KeyError: 0`` before doing any calibration.
REF = "85082400ebc345740af85cfcb84f538b21b74d00"

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


@app.function(image=image, gpu=GPU, volumes=VOLUMES, env=ENV, timeout=300)
def gpu_check() -> dict:
    """Confirm the GPU is attached and actually works, without running the pipeline.

    ``doctor`` cannot answer this. A Modal function's GPU spec is fixed at definition
    time and ``doctor`` deliberately declares none, so that a pre-flight costs nothing --
    which means it always reports "no CUDA device visible", correctly, about its own
    CPU-only container. That is easy to misread as "the GPU is not working", so this is
    the GPU-attached counterpart. It costs a few seconds of GPU time.
    """
    return describe_gpu()


@app.function(image=image, volumes=VOLUMES, env=ENV, timeout=600)
def net_check(megabytes: float = 20.0) -> dict:
    """Measure download throughput from this container, per host.

    Throughput from a Modal container is not the same as throughput from a laptop, and
    "it is downloading at KB/s" has two very different explanations: the remote host is
    throttling us, or this container's route to it is poor. Downloading a fixed number of
    bytes from several hosts distinguishes them -- if Zenodo is slow here while a control
    host is fast, it is Zenodo (or our path specifically to it), not the container.

    Costs no GPU. Uses ranged requests so it stops after the requested size.
    """
    import time

    import requests

    targets = {
        # The two checkpoints the pipeline fetches on first use.
        "zenodo/prtreid": "https://zenodo.org/records/10653453/files/"
        "prtreid-soccernet-baseline.pth.tar?download=1",
        "zenodo/hrnet": "https://zenodo.org/records/10604211/files/"
        "hrnetv2_w32_imagenet_pretrained.pth?download=1",
        # Controls: the same infrastructure that already served the weights quickly.
        "huggingface": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/resolve/main/"
        "config.json",
        "pypi": "https://files.pythonhosted.org/packages/source/r/requests/"
        "requests-2.32.3.tar.gz",
    }

    wanted = int(megabytes * 1e6)
    results: dict = {}

    for name, url in targets.items():
        try:
            started = time.time()
            got = 0
            with requests.get(
                url, stream=True, timeout=(15, 60), headers={"Range": f"bytes=0-{wanted - 1}"}
            ) as response:
                if response.status_code not in (200, 206):
                    results[name] = f"HTTP {response.status_code}"
                    continue
                for chunk in response.iter_content(chunk_size=1 << 20):
                    got += len(chunk)
                    if got >= wanted:
                        break
            elapsed = time.time() - started
            results[name] = {
                "mb_downloaded": round(got / 1e6, 2),
                "seconds": round(elapsed, 2),
                "mb_per_sec": round((got / 1e6) / elapsed, 2) if elapsed else None,
            }
        except Exception as exc:  # noqa: BLE001 - a failure here is a result too
            results[name] = f"{type(exc).__name__}: {exc}"

    return {"requested_mb": megabytes, "throughput": results}


def json_safe(payload: dict) -> dict:
    """Reduce a payload to JSON primitives, and say so if anything had to be coerced.

    Modal pickles a function's return value and deserializes it in the **client**
    environment -- here the ``.venv`` that drives ``modal run``, which has no torch. A
    four-minute run therefore died at the very last step: the pipeline finished, the
    volume was committed, and the results could not be transferred back because
    something in the payload referenced torch::

        DeserializationError: Deserialization failed because the 'torch' module is not
        available in the local environment.

    Round-tripping through JSON removes that entire class of failure: what comes back is
    plain ``str``/``int``/``float``/``bool``/``None``. It is a no-op for a payload that
    was already plain, so it costs one small serialization.

    The offending key is named rather than silently stringified, because the symptom --
    a successful run reported as a failure -- is confusing enough that the reason should
    not also be a mystery.
    """
    import json

    # type(...) in, not isinstance: `type(v) is str` excludes TorchVersion and every other
    # subclass that json accepts but pickle cannot resolve without its defining module.
    primitives = (str, int, float, bool, type(None))

    def offenders(value, path=""):  # noqa: ANN001, ANN202 - small local walker
        found: list[str] = []
        if type(value) in primitives:
            return found
        if isinstance(value, dict):
            for key, item in value.items():
                found += offenders(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                found += offenders(item, f"{path}[{index}]")
        else:
            defined = f"{type(value).__module__}.{type(value).__name__}"
            found.append(f"{path} ({defined})" if path else defined)
        return found

    problems = offenders(payload)
    if problems:
        print(
            "coerced non-plain values (pickle would need these modules client-side): "
            + "; ".join(problems[:20]),
            flush=True,
        )

    # Unconditional, not guarded on json.dumps succeeding: a str/int/float SUBCLASS
    # serializes fine and then fails to unpickle, so the round trip is what removes the
    # class rather than a check that cannot see it.
    return json.loads(json.dumps(payload, default=str))


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


def describe_gpu() -> dict:
    """What GPU this container has, proven by running a kernel rather than by a flag.

    ``torch.cuda.is_available()`` only says a device is visible; the matmul below says
    it can actually execute, which is the thing worth knowing before a long run.
    """
    import torch

    report: dict = {
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        return report

    major, minor = torch.cuda.get_device_capability()
    report.update(
        device_count=torch.cuda.device_count(),
        device_name=torch.cuda.get_device_name(0),
        capability=f"{major}.{minor}",
        vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
        # FlashAttention-2 needs Ampere or newer; on Turing the import succeeds and the
        # kernels fail, so this is the capability that actually decides the backend.
        flash_attn_usable=major >= 8,
    )
    try:
        x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        report["bf16_matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
    except Exception as exc:  # noqa: BLE001 - the report is more useful than a raise
        report["bf16_matmul_ok"] = f"{type(exc).__name__}: {exc}"
    return report


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
    # A full native-rate run is ~2 h of tracking plus setup, so the old 2 h cap would
    # have killed it mid-way and thrown the result away. Bounded rather than unlimited on
    # purpose: a hung run should stop, not bill indefinitely. The heartbeat in the
    # pipeline and the plan estimate printed above make it visible long before this.
    timeout=6 * 60 * 60,
)
def analyze(
    video: str,
    video_id: int = 10001,
    steps: tuple[int, ...] = (1, 3),
    workers: int | None = None,
    fps: float | None = None,
    max_frames: int | None = None,
    write_video: bool = True,
) -> dict:
    """Video in, structured data out.

    Decodes the video, runs the game-state steps, and exports every tracklet with its
    boxes (image pixels and pitch metres once calibration has run), jersey number, role
    and team. Returns a summary plus the exported files' contents.

    ``fps`` decimates while decoding. Every step runs per frame, so on an 11-minute match
    (~16,500 frames at 25 fps) step 3 means asking a 7B VLM about every player in every
    frame -- pass ``fps=1`` for a first smoke test rather than waiting on the full set.

    ``max_frames`` instead takes a genuine *slice* at native frame rate, which is what
    ``fps`` cannot do: an ffmpeg fps filter still decodes every input frame, so
    decimation saves the downstream steps but not the decode, and it stretches the gap
    between consecutive frames so the tracker -- which assumes small inter-frame motion
    -- is tested under unrealistic conditions. A few hundred frames at native rate is
    the honest way to judge tracking quality on a long video.

    ``write_video=False`` skips the annotated mp4. Worth setting on a long run: it was
    110 MB for 680 frames, so native rate is a few GB, it is CPU work on a GPU-priced
    container, and it is a *view* of the run rather than part of the deliverable.
    """
    import json
    import time
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

    # Report the GPU before anything else, so every run states which card it got and
    # that it can execute. Without this the only place the GPU appeared was implicitly,
    # in how long the tracking took.
    gpu = describe_gpu()
    print(f"gpu      : {json.dumps(gpu)}", flush=True)

    # Say what this run is about to cost in wall-clock before spending it. A full native
    # run is roughly two hours, and finding that out only afterwards -- or not at all --
    # is the difference between a plan and a gamble.
    from soccermaster.model.data.extract_frames import probe_duration

    duration = probe_duration(str(video_path))
    if duration:
        native_frames = int(duration * 25)  # a typical broadcast rate
        if max_frames:
            planned, rate = min(max_frames, native_frames), "native"
        elif fps:
            planned, rate = int(duration * fps), f"{fps} fps"
        else:
            planned, rate = native_frames, "native (all frames)"
        # 0.373 s/frame measured on an L40S at 1 fps; setup is a few minutes on top.
        estimate = planned * 0.373 / 3600
        print(
            f"plan     : {duration / 60:,.1f} min video, {planned:,} frames at {rate}"
            f"\n           tracking is ~{estimate:,.2f} h at the measured 0.373 s/frame,"
            f" plus a few minutes of setup",
            flush=True,
        )

    # `workers` was accepted and ignored. num_cores is the override that actually
    # controls dataloader workers, and doctor warns loudly when it exceeds the CPU count.
    extra = [f"num_cores={workers}"] if workers else None

    started = time.time()
    try:
        result = sm.analyze_video(
            video_path,
            out_dir="/data/outputs/modal",
            video_id=video_id,
            steps=list(steps),
            device="0",  # a single-GPU Modal worker; 0 is the visible device
            write_video=write_video,
            fps=fps,
            max_frames=max_frames,
            extra_overrides=extra,
        )
    except Exception:
        # Commit before returning, even on failure. `analyze_video` runs its steps in
        # sequence, so a failure late -- in step 3, say -- happens *after* step 1 has
        # written a complete tracker state and its exports. Without this the container
        # exits with uncommitted writes and an hour of tracking is thrown away because a
        # later stage failed. Modal does commit in the background every few seconds, but
        # relying on that for the only copy of a long run's output is not a plan.
        outputs_volume.commit()
        print("committed partial outputs before failing", flush=True)
        return {"error": traceback.format_exc()}

    # Read the exported data back so the caller gets it without a volume round-trip.
    payload: dict = {
        "gpu": gpu,
        "minutes": round((time.time() - started) / 60, 1),
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
    # Committed first: the produced data is the valuable thing, and it must survive even
    # if transferring the summary back fails.
    return json_safe(payload)


def _entry_kind(entry) -> str:
    """FileEntry.type as an upper-case name, whatever shape it arrives in.

    ``modal.types.FileEntryType`` is an **enum** (UNSPECIFIED, FILE, DIRECTORY, SYMLINK,
    FIFO, SOCKET), not a string. Comparing it to ``"file"`` is therefore never equal, and
    an earlier version of the sync did exactly that -- so it selected zero files, printed
    a plausible "remote: 0 file(s)", and silently copied nothing. Reading ``.name`` with a
    ``str()`` fallback works whether the field is the enum or a plain string.
    """
    kind = getattr(entry, "type", None)
    return str(getattr(kind, "name", kind) or "").upper()


def sync_outputs(sub_path: str = "", force: bool = False) -> dict:
    """Download only the files that are missing or a different size.

    Replaces a call to ``modal volume get <volume> / <dir>``, which failed outright the
    second time it was run -- it refuses an existing output directory, and the documented
    escape hatch (``--force``) re-downloads everything. On a tree containing a few GB of
    annotated video that is the wrong trade either way: either nothing is fetched, or all
    of it is.

    Sizes are the comparison rather than modification times, because a size difference is
    what a truncated download actually looks like, and mtime comparisons across a volume
    and a local filesystem invite clock-skew false positives. A file whose size differs is
    re-fetched, so a previously interrupted sync heals itself.

    ``sub_path`` scopes the sync to part of the tree -- ``--path modal/data`` fetches the
    CSVs and JSON without pulling the videos.

    Note this is a *pull*, so it never deletes anything local; files removed remotely will
    linger locally.
    """
    from pathlib import Path

    prefix = sub_path.strip("/")
    local_root = LOCAL_OUTPUT_DIR
    # Paths, not display strings: the return value is data a caller may want to act on,
    # so the human-readable sizes are reported separately below rather than baked in.
    fetched: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    sizes: dict[str, int] = {}
    transferred = 0

    # listdir is a synchronous RPC and returns FileEntry(path, type, mtime, size), so no
    # per-file subprocess and no async plumbing.
    entries = outputs_volume.listdir("/", recursive=True)
    wanted = []
    for entry in entries:
        # Only DIRECTORY is skipped. Everything else is attempted, because the failure
        # modes are asymmetric: attempting to read something that turns out not to be a
        # file raises and is reported per-file below, whereas wrongly skipping files is
        # silent -- which is precisely how this function once fetched nothing at all.
        if _entry_kind(entry) == "DIRECTORY":
            continue
        relative = str(entry.path).lstrip("/")
        # A boundary-aware prefix match, not a bare startswith. `--path modal/data` must
        # not also select `modal/dataset/...`, which is what plain startswith did: it
        # matched 4010 files where 10 were intended, so the scoped sync downloaded four
        # thousand image frames and none of the exports it was asked for -- the opposite
        # of the request, and quiet about it.
        if prefix and relative != prefix and not relative.startswith(prefix + "/"):
            continue
        wanted.append((relative, entry))

    scope = f" under '/{prefix}'" if prefix else ""
    print(f"remote: {len(wanted)} file(s){scope} of {len(entries)} entries", flush=True)
    if not wanted:
        # Loud rather than silent: an empty selection is far more likely to be a bug or a
        # wrong --path than a genuinely empty volume, and it must not look like success.
        print(
            "  nothing to fetch. Check --path, or list the volume with "
            "`modal run modal_app.py --cmd list`.",
            flush=True,
        )
        return {
            "fetched": [],
            "skipped": [],
            "failed": [],
            "transferred_mb": 0.0,
            "local_root": str(local_root),
        }

    for relative, entry in wanted:
        target = local_root / relative
        if not force and target.is_file() and target.stat().st_size == entry.size:
            skipped.append(relative)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Sync and streaming, so a several-GB file does not arrive as one bytes
            # object in memory.
            with open(target, "wb") as handle:
                written = outputs_volume.read_file_into_fileobj(entry.path, handle)
            written = written or 0
            transferred += written
            sizes[relative] = written
            fetched.append(relative)
        except Exception as exc:  # noqa: BLE001 - one bad file should not stop the sync
            failed.append(f"{relative}: {type(exc).__name__}: {exc}")

    total = len(wanted)
    print(
        f"\n{len(fetched)} fetched, {len(skipped)} already up to date"
        + (f", {len(failed)} failed" if failed else "")
        + f" (of {total})",
        flush=True,
    )
    if transferred:
        print(f"transferred {transferred / 1e6:,.1f} MB", flush=True)
    for name in fetched:
        print(f"  + {name}  ({sizes.get(name, 0) / 1e6:,.1f} MB)")
    for line in failed:
        print(f"  ! {line}")

    return {
        "fetched": fetched,
        "skipped": skipped,
        "failed": failed,
        "transferred_mb": round(transferred / 1e6, 1),
        "local_root": str(local_root),
    }


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
    max_frames: int = 0,
    no_video: bool = False,
    path: str = "",
    force: bool = False,
    steps: str = "1,3",
) -> None:
    """``modal run modal_app.py --cmd <doctor|gpu|net|prepare|video|videos|analyze|list|fetch>``"""
    import json

    def as_steps() -> tuple[int, ...]:
        parts = [p.strip() for p in str(steps).replace("|", ",").split(",")]
        return tuple(int(p) for p in parts if p)

    if cmd == "doctor":
        print(doctor.remote())
    elif cmd == "gpu":
        print(json.dumps(gpu_check.remote(), indent=2))
    elif cmd == "net":
        print(json.dumps(net_check.remote(), indent=2))
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
            max_frames=max_frames or None,
            write_video=not no_video,
        )
        print(json.dumps(result, indent=2, default=str)[:8000])
    elif cmd == "fetch":
        # A size-based sync rather than `modal volume get`, which refuses an existing
        # output directory and whose --force re-downloads everything.
        sync_outputs(sub_path=path, force=force)
    else:
        raise SystemExit(
            f"unknown cmd {cmd!r}; use doctor, gpu, net, prepare, video, videos, analyze, list or fetch"
        )
