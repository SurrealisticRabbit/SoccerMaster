# Provenance

Which code and which weights came from where. Folder names in `src/soccermaster/`
are chosen to reflect this: a folder named after an upstream project is vendored;
everything else is ours.

## Our code

| Path | What it is |
|---|---|
| `src/soccermaster/model/` | The SoccerMaster foundation model (SigLIP2 backbone wrapper, multi-task heads, deformable DETR, data pipeline, training/eval). |
| `src/soccermaster/api.py` | Notebook-friendly API. |
| `src/soccermaster/cli.py`, `__main__.py` | `soccermaster` console script. |
| `src/soccermaster/pipeline.py` | Runner for the three game-state steps. |
| `src/soccermaster/downloads.py`, `resources/weights_manifest.yaml` | Pretrained-weight fetching. |
| `src/soccermaster/paths.py` | `data/` `weights/` `outputs/` locations. |

## Vendored code

| Path | Upstream | License |
|---|---|---|
| `src/soccermaster/tracklab/` | [TrackLab](https://github.com/Vita-Vision-Lab/tracklab) — multi-object tracking framework | MIT |
| `src/soccermaster/tracklab/trackeval/` | [TrackEval](https://github.com/JonathonLuiten/TrackEval) | MIT |
| `src/soccermaster/tracklab/plugins/track/*` | BoT-SORT, ByteTrack, OC-SORT, Deep-OC-SORT, StrongSORT, BPBReID trackers | see upstream headers |
| `src/soccermaster/tracklab/plugins/eval/*` | [PoseTrack21](https://github.com/TrackingLaboratory/posetrack21) eval (unused by the soccer path) | MIT |
| `src/soccermaster/soccernet_gamestate/` | [SoccerNet Game State Recognition](https://github.com/SoccerNet/sn-gamestate) dev kit | see upstream |
| `…/plugins/calibration/sn_calibration_baseline/` | SoccerNet camera-calibration challenge kit | see upstream |
| `…/plugins/calibration/tvcalib/` | [TVCalib](https://github.com/MM4SPA/tvcalib) | MIT |
| `…/plugins/calibration/pnlcalib/`, `…/nbjw_calib/` | PnLCalib (SoccerNet) | see upstream |
| `src/soccermaster/sam2_refine/` | Glue around Meta's [SAM 2](https://github.com/facebookresearch/sam2) | glue is ours; SAM 2 is Apache-2.0 |
| `third_party/sam2/` | SAM 2 library, unmodified (not installed by default) | Apache-2.0 |
| `third_party/licenses/` | Preserved license/citation files | — |
| `docs/tracklab/`, `docs/soccernet_gamestate/`, `docs/soccernet_calibration/` | Upstream documentation | upstream |

The SoccerNet calibration kit existed twice in the original tree. It is
now kept once, at
`src/soccermaster/soccernet_gamestate/plugins/calibration/sn_calibration_baseline/`;
`model/data/utils.py` imports `mirror_labels` from there. `SoccerPitch.symetric_classes`
— the only table that function reads — is identical in both former copies, so the
consolidation does not change behaviour.

`THIRD_PARTY_NOTICES.md` lists the same set from the license-compliance angle.

## Pretrained weights

`soccermaster prepare` reads `src/soccermaster/resources/weights_manifest.yaml`.
Sources are grouped here by vendor so it is clear what is Google, what is SoccerNet,
and what is neither:

| Manifest entry | Source | Origin |
|---|---|---|
| `siglip` | Hugging Face `google/siglip2-large-patch16-512` | **Google** (SigLIP 2 backbone) |
| `soccermaster` | Hugging Face `xleprime/SoccerMaster` | this paper |
| `yolo` | Hugging Face `xleprime/SoccerMaster` | this paper (fine-tuned YOLOv8 person detector) |
| `qwen7b`, `qwen72b` | Hugging Face `Qwen/Qwen2.5-VL-*` | **Alibaba** (jersey number + role) |
| `sam2` | Hugging Face `facebook/sam2.1-hiera-large` | **Meta** |
| `legibility` | Google Drive | jersey-number-pipeline |
| `reid`, `calibration` | manual | **SoccerNet** (sn-gamestate dev kit) |

Nothing Google-authored is *code* in this repository — `google/siglip2-*` is consumed
through the `transformers` dependency as a model, and appears on disk only as weights
under `weights/google/`.

## Migration history

The tree was originally four independent projects glued together under `codes/`:
`codes/SoccerMaster`, `codes/tracklab`, `codes/sn-gamestate`, `codes/sam2`. The
consolidation into `src/soccermaster/` happened in two passes, each committed
separately so the mapping is auditable:

| Old | New |
|---|---|
| `codes/SoccerMaster/{models,data,utils,configs,sn_calibration}` | `src/soccermaster/model/*` |
| `codes/SoccerMaster/{train,eval,inference_demo,runtime_option}.py` | `src/soccermaster/model/{train,eval,demo,options}.py` |
| `codes/tracklab/tracklab` | `src/soccermaster/tracklab` |
| `codes/tracklab/{trackeval,plugins}` | `src/soccermaster/tracklab/{trackeval,plugins}` |
| `codes/sn-gamestate/sn_gamestate` | `src/soccermaster/soccernet_gamestate` |
| `codes/sn-gamestate/plugins/calibration/*` | `src/soccermaster/soccernet_gamestate/plugins/calibration/*` |
| `codes/sam2/step_2/{inference,merge_pkl,utils}.py` | `src/soccermaster/sam2_refine/*` |
| `codes/sam2` (library) | `third_party/sam2` |

Import paths follow the folder names, e.g. `soccermaster.model.models.multi_task`,
`soccermaster.tracklab.wrappers`, `soccermaster.soccernet_gamestate.calibration`.
