# SoccerMaster: A Vision Foundation Model for Soccer Understanding

> **A fork of SoccerMaster, refactored into a package, for easy use in notebooks.**

This repository is a **fork**. It is not the official release. Upstream is the
official implementation of **SoccerMaster: A Vision Foundation Model for Soccer
Understanding** (CVPR 2026 Oral) by
[*Haolin Yang*](https://haolinyang-hlyang.github.io/),
[*Jiayuan Rao*](https://jyrao.github.io/),
[*Haoning Wu*](https://haoningwu3639.github.io/) and
[*Weidi Xie*](https://weidixie.github.io/).

The model, the data pipeline and the game-state reconstruction logic are the
upstream authors' work. This fork only changes **how it is packaged and driven** —
see **About this fork** below.

<div style="line-height: 1;">
  <a href="https://haolinyang-hlyang.github.io/SoccerMaster/" target="_blank" style="margin: 2px;">
    <img alt="Website" src="https://img.shields.io/badge/Website🌐-SoccerMaster-536af5?color=536af5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/pdf/2512.11016" target="_blank" style="margin: 2px;">
    <img alt="Arxiv" src="https://img.shields.io/badge/Arxiv📄-SoccerMaster-red?logo=%23B31B1B" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<div align="center">
   <img src="./images/teaser.jpg">
</div>
<p><em><strong>SoccerMaster</strong> is a unified soccer-specific vision foundation model that leverages diverse soccer content, including images and videos, to support a wide range of soccer understanding tasks, such as commentary generation, detection, tracking, classification, <em>etc</em>.</em></p>
<div align="center">
   <img src="./images/arch.jpg">
</div>
<p><em><strong>SoccerMaster Architecture.</strong> (a) The architecture of SoccerMaster, which encodes both soccer videos and images through spatial and temporal attention modules to generate semantically rich representations. (b) The pretraining tasks and downstream adaptations of SoccerMaster across both spatial perception and semantic understanding tasks.</em></p>

---

## About this fork

The upstream release ships as four separate projects glued together under `codes/`
(`SoccerMaster`, `sn-gamestate`, `tracklab`, `sam2`), each with its own
`poetry`/`setup.py`, its own import root, and a `cd`-and-`ln -s` quick start that
does not survive a pip install. That is fine for reproducing a paper on one
workstation; it is painful the moment you want to open a notebook and just use
the model.

This fork makes it one installable distribution you can `pip install` straight
from GitHub — on Colab or Kaggle — and then drive from Python:

```python
!pip install "soccermaster[pipeline] @ git+https://github.com/SurrealisticRabbit/SoccerMaster.git@main"

import soccermaster as sm

sm.prepare("siglip")                        # fetch weights, no manual layout
sm.prepare("soccermaster")

model = sm.load_model()                     # -> MultiTaskingModel
outputs = sm.predict(model, "frames/SNGS-10001", task="detection")

sm.run_pipeline("data/SoccerNetGS", steps=(1, 3))   # game-state pipeline
```

What changed:

| | Upstream | This fork |
|---|---|---|
| Packaging | four separate installable projects | one distribution, `pip install` from GitHub |
| Import root | `tracklab`, `sn_gamestate`, `sam2` | `soccermaster.*`, single namespace |
| Weight setup | manual `hf download` per model + `ln -s` | `soccermaster prepare` into a known `weights/` tree |
| Entry point | `cd` into a directory and run a script | `soccermaster` CLI + `soccermaster.api` |
| Notebook use | not supported | `load_model` / `predict` / `run_pipeline` |

The model, data pipeline and game-state logic are untouched: **same module order,
same head names, same pickle schema, same loss and schedule numbers.** Nothing has
been retrained or swapped, and no vendored code was relicensed — vendored folders
are named after their upstream, with original headers intact (see
[`PROVENANCE.md`](PROVENANCE.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).

Scope notes: SAM 2 stays an optional extra and step 2 needs a CUDA build. The
`reid` and `calibration` weights are SoccerNet-hosted and still have to be placed by
hand. The rest of this README documents the refactored package; upstream commands
survive below where they are still valid, with their paths updated.

---

## Repository layout

Package folders are named after their upstream so you can tell at a glance what
came from where:

```
pyproject.toml            # one distribution: "soccermaster"
src/soccermaster/
├── api.py                # notebook-friendly API (load_model / predict / run_pipeline)
├── cli.py                # `soccermaster` console script
├── downloads.py          # weight fetching (Hugging Face / Google Drive)
├── paths.py              # data / weights / outputs locations
├── pipeline.py           # 3-step game-state pipeline runner
├── model/                # SoccerMaster foundation model        — this paper
├── tracklab/             # TrackLab tracking framework          — vendored (MIT)
├── soccernet_gamestate/  # SoccerNet game-state reconstruction  — vendored
│   └── plugins/calibration/   # TVCalib, PnLCalib, NBJW, sn_calibration_baseline
├── sam2_refine/          # SAM 2 mask refinement                — vendored glue
└── resources/            # weight manifest and small assets
third_party/sam2/         # vendored SAM 2 library (optional, not installed by default)
docs/                     # TrackLab, SoccerNet-GSR and calibration documentation
```

`model/`, `api.py`, `cli.py`, `pipeline.py` and `downloads.py` are ours. Everything
under `tracklab/`, `soccernet_gamestate/`, `sam2_refine/`, `third_party/` and the
vendored bits of `docs/` came from other projects — see
[`PROVENANCE.md`](PROVENANCE.md) for the full attribution, including which
pretrained weights come from Google, SoccerNet, Alibaba and Meta.

## Quick Start

### 1. Environment setup

Python 3.10+ is required. PyTorch must be installed first with the CUDA wheel index
so that pip does not pull a CPU-only build over an existing CUDA install:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

pip install -e .
```

Optional extras:

```bash
pip install -e ".[pipeline]"  # everything the 3-step pipeline imports (steps 1 and 3)
pip install -e ".[sam2]"      # SAM 2 runtime deps, needed for pipeline step 2
pip install -e ".[train]"     # flash-attn, only needed to train
pip install -e ".[dev]"       # ruff + pytest
```

The base install is deliberately lean: it covers `import soccermaster`,
`load_model` and `predict`. Running the game-state pipeline needs the `pipeline`
extra, whose dependencies are only reachable from the pipeline modules:

```bash
soccermaster pipeline --dataset ./data/SoccerNetGS --steps 1,3 --device 0
```

The SAM 2 *library* itself is vendored at `third_party/sam2` and is not part of the
installed wheel. Install it as well if you want to run step 2:

```bash
pip install -e third_party/sam2
```

### Installing on Colab / Kaggle

Install straight from GitHub (`@main`, or a tag once you cut one):

```python
!pip install "soccermaster[pipeline] @ git+https://github.com/SurrealisticRabbit/SoccerMaster.git@main"
```

Two caveats on hosted notebooks:

- **Do not let pip resolve `torch`.** Colab and Kaggle ship a CUDA-matched build, and
  this project pins `torch>=2.4,<2.6`, so a normal install may *downgrade* the
  working torch and break CUDA. Install the package with `--no-deps` alongside an
  explicit dependency list, or accept the downgrade knowingly.
- **`import soccermaster` needs no third-party packages at all.** The package
  re-exports its API lazily, so the import succeeds before any dependency is
  installed — useful to confirm the install landed.

### 2. Download pretrained weights

`soccermaster prepare` reads `src/soccermaster/resources/weights_manifest.yaml`:

```bash
soccermaster prepare --list            # show every entry and its destination
soccermaster prepare --dry-run siglip  # show what would be fetched
soccermaster prepare siglip soccermaster yolo   # fetch specific entries
soccermaster prepare                   # fetch every automatic entry
```

| entry | source | notes |
|---|---|---|
| `siglip` | Hugging Face | SigLIP2 backbone used by the encoder |
| `soccermaster` | Hugging Face | SoccerMaster checkpoints |
| `yolo` | Hugging Face | person detector for step 1 |
| `qwen7b` / `qwen72b` | Hugging Face | jersey number + role recognition |
| `sam2` | Hugging Face | SAM 2 video predictor checkpoint |
| `legibility` | Google Drive | ResNet34 legibility classifier |
| `reid`, `calibration` | **manual** | from the [SoccerNet GSR dev kit](https://github.com/soccernet/sn-gamestate) |

`reid` and `calibration` cannot be fetched automatically; `prepare` prints the
upstream instructions. Place those files under `weights/reid/` and
`weights/calibration/` respectively.

### 3. Directory conventions

| purpose | default location | override |
|---|---|---|
| weights | `./weights` | `SOCCERMASTER_HOME` |
| datasets | `./data` | `SOCCERMASTER_HOME` |
| outputs | `./outputs` | `SOCCERMASTER_HOME` |

In an editable install these default to the repository root; for a regular install
they default to `~/.cache/soccermaster`. Set `SOCCERMASTER_HOME=/some/dir` to put
all three somewhere else.

## Using the model from Python

```python
import soccermaster as sm

sm.prepare("siglip")
sm.prepare("soccermaster")

model = sm.load_model()                       # -> MultiTaskingModel
outputs = sm.predict(model, "frames/SNGS-10001", task="detection")
print(outputs.keys())                         # e.g. {'SoccerNetGSR_Detection': {...}}
```

`task` is one of `detection`, `lines`, `keypoints`, `caption`, `classification` or
`all`. Outputs are the raw head outputs (no NMS / bbox rescaling), matching the
original `inference_demo.py`. For a random-input smoke test of the weights:

```bash
soccermaster foundation-demo
```

## Data Pipeline

The data pipeline follows the [SoccerNet Game State Recognition (GSR)](https://github.com/soccernet/sn-gamestate) format. Organise your input frames as:

```
data/SoccerNetGS/
├── sequences_info.json
└── sn500/
    ├── SNGS-10001/
    │   └── img1/
    │       ├── 000001.jpg
    │       └── ... (750 frames)
    └── SNGS-10002/
        └── img1/
```

`sequences_info.json` registers all sequences:

```json
{
    "sn500": [
        { "id": 0, "name": "SNGS-10001", "n_frames": 750 },
        { "id": 1, "name": "SNGS-10002", "n_frames": 722 }
    ]
}
```

Run the three steps with one command (paths and state files are injected for you):

```bash
# all steps, specific sequences
soccermaster pipeline --dataset ./data/SoccerNetGS --videos 10001,10002 --device 0

# steps 1 and 3 only (no SAM 2 refinement)
soccermaster pipeline --dataset ./data/SoccerNetGS --steps 1,3 --out ./outputs/pipeline
```

| step | what it does | config |
|---|---|---|
| 1 | person detection + tracking | `gsr_step_1_example` |
| 2 | SAM 2 mask refinement (needs the `sam2` extra) | `soccermaster.sam2_refine.inference` |
| 3 | calibration, jersey number, role, team assignment | `gsr_step_3_example_accelerate` |

Outputs land in `outputs/pipeline/step1/sn-gamestate.pklz`,
`outputs/pipeline/step2/refined_sn-gamestate.pklz` and
`outputs/pipeline/step3/sn-gamestate.pklz`.

> **Model selection for step 3:** the accelerated config
> (`gsr_step_3_example_accelerate`, used by default) runs jersey-number and role
> recognition in a single Qwen2.5-VL-7B module. For higher accuracy at higher cost,
> use `gsr_step_3_example` with Qwen2.5-VL-72B for jersey numbers.

> **Platform notes:** steps 1 and 3 and model training run as plain Python and work
> on Windows and Linux. Step 2 requires a CUDA build plus the `sam2` extra.

### Running the pipeline without the CLI

Each step is still a normal Hydra entry point, so the original commands work:

```bash
python -m soccermaster.tracklab.main -cn gsr_step_1_example
python -m soccermaster.tracklab.main -cn gsr_step_3_example_accelerate
python -m soccermaster.sam2_refine.inference --help
python -m soccermaster.model.data.extract_frames --help
```

When invoking TrackLab directly, remember it uses the current working directory:
`data_dir`, `model_dir` and `dataset.dataset_path` are `${project_dir}/...` by
default, so pass `data_dir=...`, `model_dir=...` and `dataset.dataset_path=...`
overrides — that is exactly what `soccermaster pipeline` does.

## Training and evaluation

```bash
soccermaster train --config-path src/soccermaster/model/configs/pretrain.yaml --data-root ./data
soccermaster eval  --config src/soccermaster/model/configs/pretrain.yaml --checkpoint ./weights/SoccerMaster
```

Arguments are forwarded unchanged to the original `train.py` / `eval.py`, so all
their flags are available (`--resume-training`, `--outputs-dir`, `--exp-name`, ...).

## Soccer Factory Dataset

We release the **Soccer Factory** dataset containing 7,000 videos with per-frame annotations (bounding boxes, roles, jersey numbers, and camera parameters). The data is distributed as H.264-encoded videos and JSON annotations.

### Download

Download the following files from [Dropbox](https://www.dropbox.com/scl/fo/dlurbp5se1c9180v2o1zm/ADKzrHk7Cy_p1mqNUqWDOX4?rlkey=3aw7yjzhwls6rnop4c7pn7udc&st=6p1dx8g5&dl=0):

| File | Contents | Size |
|------|----------|------|
| `soccer_factory_annotations.tar.gz` | Per-frame annotations (JSON) | 2.4 GB |
| `soccer_factory_videos_part1.tar` | 1,000 H.264 videos | 18.8 GB |
| `soccer_factory_videos_part2.tar` | 1,000 H.264 videos | 18.0 GB |
| `soccer_factory_videos_part3.tar` | 1,000 H.264 videos | 19.3 GB |
| `soccer_factory_videos_part4.tar` | 1,000 H.264 videos | 19.5 GB |
| `soccer_factory_videos_part5.tar` | 1,000 H.264 videos | 18.1 GB |
| `soccer_factory_videos_part6.tar` | 1,000 H.264 videos | 16.5 GB |
| `soccer_factory_videos_part7.tar` | 1,000 H.264 videos | 16.0 GB |
| `SHA256SUMS` | SHA-256 checksums for the archives | < 1 KB |

Each video archive contains 1,000 sequences. The parts preserve the order of the released ID list, so the IDs in an individual part are not necessarily a contiguous numeric range. Use the filenames within an archive rather than inferring an ID range from its part number. Verify downloaded files against `SHA256SUMS` before extraction.

### Setup

#### Step 1: Extract annotations

```bash
mkdir -p data/SoccerNetGS/extracted_info
tar xzf soccer_factory_annotations.tar.gz
mv annotations/* data/SoccerNetGS/extracted_info/
```

#### Step 2: Extract frames from videos

The video tar files contain H.264-encoded `.mp4` files. Use the provided script to decode them back into image sequences:

```bash
# Option A: decode directly from tar files (recommended, no intermediate extraction)
python -m soccermaster.model.data.extract_frames \
    --tar_files soccer_factory_videos_part*.tar \
    --output_dir data/SoccerNetGS/sn500 \
    --quality 95

# Option B: extract tar files first, then decode
tar xf soccer_factory_videos_part1.tar
python -m soccermaster.model.data.extract_frames \
    --video_dir ./videos \
    --output_dir data/SoccerNetGS/sn500 \
    --quality 95
```

`--output_dir` defaults to `<data_dir>/SoccerNetGS/sn500`. **Dependencies:** `ffmpeg`
must be installed (`apt install ffmpeg` or `brew install ffmpeg`). The script
supports resumption — re-running skips already-extracted sequences.

#### Step 3: Verify the final directory structure

```
data/SoccerNetGS/
├── extracted_info/                  # Annotations (JSON)
│   ├── SNGS-10001.json
│   └── ... (7000 files)
└── sn500/                           # Image sequences
    ├── SNGS-10001/
    │   └── img1/
    │       ├── 000001.jpg
    │       └── ... (up to ~750 frames per sequence)
    └── ... (7000 directories)
```

#### Step 4: Configure training

In `src/soccermaster/model/configs/default.yaml` (or on the command line):

```yaml
USE_SOCCER_FACTORY_DATA: True
SOCCER_FACTORY_DATA_DIR: ./data/SoccerNetGS/extracted_info
SOCCER_FACTORY_DATA_IMAGE_DIR: ./data/SoccerNetGS/sn500
USE_SOCCER_FACTORY_DATA_AMOUNT: 7000  # -1 to use all
```

### Annotation Format

Each JSON file (e.g., `SNGS-10001.json`) contains per-frame annotations keyed by frame ID (1-indexed string):

```json
{
  "1": {
    "people": [
      {
        "id": 1,
        "bbox_ltwh": [x, y, w, h],
        "role": "player",
        "legibility_score": 0.95,
        "jersey_number": 7.0
      }
    ],
    "valid_cam_params": true,
    "K": [[...], [...], [...]],
    "R": [[...], [...], [...]]
  }
}
```

Fields:
- `people[].bbox_ltwh`: Bounding box in (left, top, width, height) format, pixel coordinates for 1920x1080 images.
- `people[].role`: One of `"player"`, `"goalkeeper"`, `"referee"`, `"ball"`, `"other"`.
- `people[].legibility_score`: Confidence of jersey number visibility (0.0 to 1.0).
- `people[].jersey_number`: Jersey number (float or null).
- `valid_cam_params`: Whether camera calibration is available for this frame.
- `K`: 3x3 camera intrinsic matrix.
- `R`: 3x3 rotation matrix.

## Licensing

This repository vendors code from TrackLab (MIT), SAM 2 (Apache-2.0), the SoccerNet
Game State Recognition and camera-calibration kits, TVCalib and others. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the full list and the
preserved license texts under `third_party/licenses/`.

## Citations

If you find our work useful, please cite:

```bibtex
@inproceedings{yang2025soccermaster,
  title={SoccerMaster: A Vision Foundation Model for Soccer Understanding},
  author={Yang, Haolin and Rao, Jiayuan and Wu, Haoning and Xie, Weidi},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
