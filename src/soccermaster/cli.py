"""Command line interface: ``soccermaster <subcommand>``."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _split_ids(value: str) -> list[int]:
    return [int(v) for v in value.replace(" ", "").split(",") if v]


def _cmd_prepare(args: argparse.Namespace) -> int:
    from soccermaster import downloads

    if args.list:
        entries = downloads.load_manifest()
        for name, entry in entries.items():
            kind = "auto" if entry.automatic else "manual"
            print(f"{name:<14} {kind:<7} -> {entry.dest}")
        return 0

    names = args.names or list(downloads.load_manifest())
    for name in names:
        downloads.prepare(name, dry_run=args.dry_run)
    return 0


def _cmd_pipeline(args: argparse.Namespace) -> int:
    from soccermaster import pipeline

    out = pipeline.run_pipeline(
        frames_dir=args.dataset,
        out_dir=args.out,
        steps=args.steps,
        video_ids=args.videos,
        eval_set=args.eval_set,
        device=args.device,
    )
    print(f"pipeline outputs: {out}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    import torch

    from soccermaster import api

    model = api.load_model(checkpoint_dir=args.checkpoint, config=args.config, device=args.device)
    outputs = api.predict(model, args.frames, task=args.task, device=args.device)

    for head, head_output in outputs.items():
        print(f"[{head}]")
        if isinstance(head_output, dict):
            for key, value in head_output.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape={tuple(value.shape)} dtype={value.dtype}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: {type(value).__name__} of length {len(value)}")
                else:
                    print(f"  {key}: {value}")
        elif isinstance(head_output, torch.Tensor):
            print(f"  shape={tuple(head_output.shape)} dtype={head_output.dtype}")
    return 0


def _cmd_foundation_demo(args: argparse.Namespace) -> int:
    from soccermaster.foundation import demo

    demo.main()
    return 0


def _passthrough(module: str, argv: list[str]) -> int:
    return subprocess.run([sys.executable, "-m", module, *argv]).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soccermaster",
        description="SoccerMaster: unified soccer vision foundation model and GSR pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # prepare ---------------------------------------------------------------
    p = sub.add_parser("prepare", help="download pretrained weights")
    p.add_argument("names", nargs="*", help="manifest entries to fetch (default: all)")
    p.add_argument("--list", action="store_true", help="list manifest entries and exit")
    p.add_argument("--dry-run", action="store_true", help="show what would be downloaded")
    p.set_defaults(func=_cmd_prepare)

    # pipeline --------------------------------------------------------------
    p = sub.add_parser("pipeline", help="run the 3-step game-state pipeline")
    p.add_argument("--dataset", required=True, type=Path,
                   help="dataset root in SoccerNet layout (<eval_set>/<seq>/img1)")
    p.add_argument("--out", type=Path, default=None, help="output directory")
    p.add_argument("--steps", type=_split_ids, default=[1, 2, 3], help="e.g. 1 or 1,3")
    p.add_argument("--videos", type=_split_ids, default=None, help="sequence ids, e.g. 10001,10002")
    p.add_argument("--eval-set", default="sn500", help="dataset split name (default: sn500)")
    p.add_argument("--device", default="cuda", help="CUDA_VISIBLE_DEVICES value or 'cuda'")
    p.set_defaults(func=_cmd_pipeline)

    # predict ---------------------------------------------------------------
    p = sub.add_parser("predict", help="run the foundation model on frames")
    p.add_argument("--frames", required=True, help="frame directory or '<seq>/img1'")
    p.add_argument("--checkpoint", default=None, help="SoccerMaster checkpoint directory")
    p.add_argument("--config", default="pretrain", help="config stem (default: pretrain)")
    p.add_argument("--task", default="detection",
                   choices=["detection", "lines", "keypoints", "caption", "classification", "all"])
    p.add_argument("--device", default=None)
    p.set_defaults(func=_cmd_predict)

    # foundation-demo -------------------------------------------------------
    p = sub.add_parser("foundation-demo", help="smoke-test the model with random inputs")
    p.set_defaults(func=_cmd_foundation_demo)

    # train / eval ----------------------------------------------------------
    # Arguments are forwarded verbatim to the original scripts, so these are
    # intercepted in main() before argparse runs.
    sub.add_parser("train", help="train the foundation model (forwards all args to train.py)",
                   add_help=False)
    sub.add_parser("eval", help="evaluate the foundation model (forwards all args to eval.py)",
                   add_help=False)

    return parser


_PASSTHROUGH = {
    "train": "soccermaster.foundation.train",
    "eval": "soccermaster.foundation.eval",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # train/eval take the upstream scripts' own flags; forward them untouched.
    if argv and argv[0] in _PASSTHROUGH:
        return _passthrough(_PASSTHROUGH[argv[0]], argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
