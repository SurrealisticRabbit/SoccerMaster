"""Render step 2's segmentation decisions from its saved result, on any machine.

Why this exists: step 2 takes about an hour, and ``sam2_refine`` only writes its own
visualization videos when passed ``--visualize_video``, which the pipeline did not pass. So
a completed step-2 run produced no video to look at -- and the run cannot simply be
repeated with the flag, because ``video_results/<id>_result.pkl`` makes it skip the video
entirely.

Worse, the masks cannot be recovered from what was saved. ``_clear_masks`` sets
``detection.mask = None`` on every structure *before* the result is pickled, and
``frame_to_bbox_prompts``, ``unprocessed_detections`` and ``video_data`` -- which the three
built-in videos need -- are local variables that are never written. Checked directly: 353
detections and 4122 unmatched segments in the saved result, zero with a mask.

What IS saved is the *structure*: which detections formed which contiguous segment, which
segments matched an existing track, and which were left unmatched. This renders that, so
its decisions can be inspected without spending another hour of GPU time. The masks
themselves still require a re-run.

Usage::

    python tools/render_step2_segments.py \\
        --result modal-outputs/modal/step2/video_results/10011_result.pkl \\
        --frames modal-outputs/modal/dataset/sn500/SNGS-10011/img1 \\
        --out modal-outputs/10011_step2_segments.mp4

Colour key:

* **green, "ID: n M: m"** -- matched; the detection's track ``n`` was matched to the
  propagated track ``m`` (``m >= 100`` means the duplicate-id fixer rewrote it)
* **red, "ID: n unmatched"** -- left unmatched, i.e. propagation from this box did not find
  a match
* **blue, "L:n u:y"** -- a low-brightness label for very short segments, which are skipped
  by ``min_continuous_segment_length`` and so are not propagated at all
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", required=True, type=Path,
                        help="video_results/<id>_result.pkl from a step-2 run")
    parser.add_argument("--frames", required=True, type=Path,
                        help="directory of decoded frames (the dataset img1 dir)")
    parser.add_argument("--out", required=True, type=Path, help="output .mp4")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--frame-limit", type=int, default=None,
                        help="render only the first N frames, for a quick look")
    parser.add_argument("--segment-overlay", action="store_true",
                        help="also outline unmatched segments using their saved bbox, "
                             "since masks are not available")
    return parser.parse_args(argv)


def load_detections(result_path: Path) -> tuple[dict, list]:
    """Return (frame_idx -> [detection], unmatched_segments)."""
    with open(result_path, "rb") as fh:
        result = pickle.load(fh)

    by_frame: dict[int, list] = defaultdict(list)
    for tracklet in result["unprocessed_data"].unprocessed_tracklets:
        for segment in tracklet.segments:
            for detection in segment.detections:
                by_frame[detection.frame_idx].append(detection)
    return by_frame, list(result.get("unmatched_segments") or [])


def frame_paths(frames_dir: Path, limit: int | None) -> list[Path]:
    """Frames in the numbering the step recorded, i.e. sorted by filename.

    ``sam2_refine`` numbers frames by their position in this sorted list, which is the same
    ordering the tracker used, so ``frame_idx`` maps directly onto it.
    """
    names = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return names[:limit] if limit else names


def render(args: argparse.Namespace) -> int:
    import cv2

    if not args.result.is_file():
        print(f"no such result: {args.result}", file=sys.stderr)
        return 2
    if not args.frames.is_dir():
        print(f"no such frames directory: {args.frames}", file=sys.stderr)
        return 2

    by_frame, unmatched = load_detections(args.result)
    paths = frame_paths(args.frames, args.frame_limit)
    if not paths:
        print(f"no frames found in {args.frames}", file=sys.stderr)
        return 2

    n_det = sum(len(v) for v in by_frame.values())
    matched = sum(1 for v in by_frame.values() for d in v if d.is_matched)
    short = sum(1 for v in by_frame.values() for d in v if d.is_short_segment)
    print(f"detections        : {n_det}")
    print(f"  matched         : {matched}")
    print(f"  unmatched       : {n_det - matched}")
    print(f"  short segments  : {short}  (skipped by propagation)")
    print(f"unmatched segments: {len(unmatched)}")
    print(f"frames to render  : {len(paths)}")

    first = cv2.imread(str(paths[0]))
    if first is None:
        print(f"cannot read {paths[0]}", file=sys.stderr)
        return 2
    height, width = first.shape[:2]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (width, height)
    )
    if not writer.isOpened():
        print(f"could not open a video writer for {args.out}", file=sys.stderr)
        return 2

    seg_by_frame: dict[int, list] = defaultdict(list)
    for segment in unmatched:
        seg_by_frame[segment.frame_idx].append(segment)

    try:
        for index, path in enumerate(paths):
            frame = cv2.imread(str(path))
            if frame is None:
                continue

            label = f"frame {index + 1}/{len(paths)}"
            cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            for detection in by_frame.get(index, []):
                x1, y1, x2, y2 = (int(v) for v in detection.bbox_xyxy)
                if detection.is_matched:
                    colour = (0, 200, 0)
                    text = f"ID: {int(detection.track_id)}  M: {int(detection.matched_track_id)}"
                    if detection.matched_track_id >= 100:
                        text += " (fixed)"
                else:
                    colour = (0, 0, 220)
                    text = f"ID: {int(detection.track_id)} unmatched"

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv2.putText(frame, text, (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
                if detection.is_short_segment:
                    cv2.putText(frame, f"L:{int(detection.segment_length)}",
                                (x1, min(height - 4, y2 + 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 160, 0), 1, cv2.LINE_AA)

            if args.segment_overlay:
                for segment in seg_by_frame.get(index, []):
                    bbox = getattr(segment, "bbox", None)
                    if bbox is None:
                        continue
                    sx1, sy1, sx2, sy2 = (int(v) for v in bbox)
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 0, 255), 1)
                    cv2.putText(frame, f"seg {int(segment.obj_id)}", (sx1, max(12, sy1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)

            writer.write(frame)
            if (index + 1) % 200 == 0:
                print(f"  rendered {index + 1}/{len(paths)}")
    finally:
        writer.release()

    size_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out}  ({size_mb:,.1f} MB)")
    print("Note: this shows step 2's segment and matching decisions, not its masks -- the "
          "masks were cleared before the run saved anything. Masks need a re-run with "
          "--visualize_video.")
    return 0


if __name__ == "__main__":
    raise SystemExit(render(parse_args()))
