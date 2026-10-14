# Third-party notices

This repository bundles or adapts code from several upstream projects. The
respective licenses are reproduced where required; full texts live under
`third_party/licenses/` and next to the vendored trees.

| Component | Location in this repo | Upstream | License |
|---|---|---|---|
| TrackLab (tracking engine) | `src/soccermaster/tracklab/` | https://github.com/Vita-Vision-Lab/tracklab | MIT |
| TrackEval (vendored) | `src/soccermaster/tracklab/trackeval/` | https://github.com/JonathonLuiten/TrackEval | MIT |
| BoT-SORT / ByteTrack / OC-SORT / StrongSORT / Deep-OC-SORT | `src/soccermaster/tracklab/plugins/track/` | see individual headers | MIT |
| PoseTrack21 eval | `src/soccermaster/tracklab/plugins/eval/` | https://github.com/TrackingLaboratory/posetrack21 | MIT |
| SAM 2 | `third_party/sam2/` | https://github.com/facebookresearch/sam2 | Apache-2.0 (see `third_party/sam2/LICENSE`) |
| SAM 2 `cctorch` helper | `third_party/sam2/LICENSE_cctorch` | https://github.com/ronghanghu/cctorch | BSD 3-Clause |
| SoccerNet Game State Recognition dev kit | `src/soccermaster/soccernet_gamestate/` | https://github.com/SoccerNet/sn-gamestate | see upstream |
| SoccerNet camera-calibration kit (`sn_calibration_baseline`) | `src/soccermaster/soccernet_gamestate/plugins/calibration/sn_calibration_baseline/` | https://github.com/SoccerNet/sn-calibration | see upstream |
| TVCalib | `src/soccermaster/soccernet_gamestate/plugins/calibration/tvcalib/` | https://github.com/MM4SPA/tvcalib | MIT |
| pnlcalib_utils / nbjw_calib (PnLCalib) | `src/soccermaster/soccernet_gamestate/plugins/calibration/pnlcalib/`, `.../nbjw_calib/` | PnLCalib (SoccerNet) | see upstream |
| MOTIP config helpers (modified) | `src/soccermaster/model/configs/util.py` | https://github.com/MCG-NJU/MOTIP | see upstream |

License texts preserved during the consolidation:

- `third_party/licenses/tracklab-LICENSE` (MIT)
- `third_party/licenses/tracklab-CITATION.cff`
- `third_party/sam2/LICENSE` (Apache-2.0)
- `third_party/sam2/LICENSE_cctorch` (BSD 3-Clause)

Original file headers are retained inside the moved files; this project does not
relicense any vendored code.

See [`PROVENANCE.md`](PROVENANCE.md) for the original-to-consolidated path mapping
and for the provenance of each pretrained-weight manifest entry.

> **Note:** the upstream SoccerMaster repository does not ship a top-level
> `LICENSE` file, so no root license is asserted here. Add one only as a
> deliberate maintainer decision.
