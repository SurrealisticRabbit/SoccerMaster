"""Pre-flight checks for the game-state pipeline.

Answers one question: *will the pipeline start, or am I about to hit a wall?*
Every check is read-only -- nothing is downloaded, tracked or instantiated.

The checks exist because the pipeline fails late and confusingly otherwise: a
missing weight surfaces as a ``FileNotFoundError`` deep inside a module, and a bad
``_target_`` surfaces as a Hydra ``InstantiationException`` after the dataset has
already been loaded.

Usage::

    soccermaster doctor                     # all steps
    soccermaster doctor --steps 1,3         # only what those steps need
    soccermaster doctor --dataset data/SoccerNetGS
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from soccermaster import paths

# What each step needs. "manifest" entries must be present under weights/ and can
# be fetched with `soccermaster prepare`. "runtime" entries are downloaded by the
# pipeline itself on first use, so their absence is only a note. "modules" are the
# third-party imports the step's pipeline modules pull in.
STEP_REQUIREMENTS: dict[int, dict[str, list[str]]] = {
    1: {
        "manifest": ["yolo"],
        "runtime": ["reid"],
        "config": ["gsr_step_1_example"],
        "modules": ["torch", "torchreid", "ultralytics", "hydra", "omegaconf"],
    },
    2: {
        "manifest": ["sam2"],
        "runtime": [],
        "config": [],
        "modules": ["torch", "sam2", "iopath"],
    },
    3: {
        "manifest": ["qwen7b", "legibility"],
        "runtime": ["reid", "calibration"],
        "config": ["gsr_step_3_example_accelerate"],
        "modules": ["torch", "torchreid", "transformers", "hydra", "omegaconf"],
    },
}

# Hydra configs live in two packages: tracklab ships the engine defaults, the
# soccernet_gamestate plugin ships the gsr_* entry points.
TRACKLAB_CONFIG_PKG = "soccermaster.tracklab.configs"


class Report:
    """Collects check results and prints them as an indented checklist."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def header(self, text: str) -> None:
        print(f"\n{text}")
        print("-" * len(text))

    def ok(self, text: str) -> None:
        print(f"  [ ok ] {text}")

    def warn(self, text: str, fix: str = "") -> None:
        print(f"  [warn] {text}")
        self.warnings.append(text)
        if fix:
            print(f"         fix: {fix}")

    def fail(self, text: str, fix: str = "") -> None:
        print(f"  [FAIL] {text}")
        self.failures.append(text)
        if fix:
            print(f"         fix: {fix}")

    def note(self, text: str) -> None:
        print(f"  [note] {text}")


def _steps(value: str | None) -> list[int]:
    if not value:
        return sorted(STEP_REQUIREMENTS)
    try:
        return [int(s) for s in value.replace(" ", "").split(",") if s]
    except ValueError as exc:  # pragma: no cover
        raise SystemExit(f"--steps wants e.g. 1,3 (got {value!r})") from exc


def check_environment(report: Report) -> None:
    report.header("Environment")
    try:
        import soccermaster

        report.ok(f"soccermaster {soccermaster.__version__} at {Path(soccermaster.__file__).parent}")
    except Exception as exc:  # pragma: no cover
        report.fail(f"cannot import soccermaster: {exc}", 'pip install -e .')
        return

    for label, value in (
        ("home", paths.home()),
        ("weights", paths.weights_dir()),
        ("data", paths.data_dir()),
        ("outputs", paths.outputs_dir()),
    ):
        report.note(f"{label:<8} {value}")

    edit = (paths.repo_root() / "pyproject.toml").is_file()
    if edit and os.environ.get("SOCCERMASTER_HOME") is None:
        report.warn(
            "running from a source checkout without SOCCERMASTER_HOME set, so weights "
            "resolve into the repo",
            "export SOCCERMASTER_HOME=/content/soccermaster (or set it before import)",
        )

    try:
        import torch

        report.ok(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            report.ok(f"cuda available: {torch.cuda.get_device_name(0)}")
        else:
            report.warn(
                "no CUDA device visible; steps 1 and 3 can run but will be slow, "
                "step 2 needs a GPU",
                "select a GPU runtime, or run with steps=(1, 3) and expect slowness",
            )
    except ImportError:
        report.fail("torch is not importable", "pip install torch")


def check_modules(module_names: set[str], report: Report) -> None:
    report.header("Third-party imports")
    for name in sorted(module_names):
        if importlib.util.find_spec(name) is not None:
            report.ok(name)
        else:
            report.fail(
                f"{name} is not installed",
                'pip install "soccermaster[pipeline]" (and the sam2 extra for step 2)',
            )


def check_weights(steps: list[int], report: Report) -> None:
    from soccermaster.downloads import load_manifest

    report.header("Weights")
    manifest = load_manifest()

    needed_manifest: list[str] = []
    needed_runtime: list[str] = []
    for step in steps:
        needed_manifest += STEP_REQUIREMENTS[step]["manifest"]
        needed_runtime += STEP_REQUIREMENTS[step]["runtime"]

    for name in dict.fromkeys(needed_manifest):
        entry = manifest.get(name)
        if entry is None:
            report.fail(f"'{name}' is not in the manifest")
            continue
        present = entry.dest.is_dir() and any(entry.dest.iterdir())
        if present:
            report.ok(f"{name:<12} present at {entry.dest}")
        else:
            report.fail(
                f"{name:<12} missing at {entry.dest}",
                f"soccermaster prepare {name}",
            )

    for name in dict.fromkeys(needed_runtime):
        entry = manifest.get(name)
        if entry is None:
            report.note(f"{name} (not in manifest)")
            continue
        present = entry.dest.is_dir() and any(entry.dest.iterdir())
        if present:
            report.ok(f"{name:<12} present at {entry.dest}")
        else:
            report.note(
                f"{name:<12} not downloaded yet; the pipeline fetches it from Zenodo "
                "on first run (md5-verified)"
            )

    # Geometry of the paths the configs actually reference, for the common cases.
    report.note(f"step-1 detector expects {paths.weights_dir() / 'yolo' / 'yolo_v8x6_person_lr_default_best.pt'}")


def check_configs(steps: list[int], report: Report) -> None:
    """Compose each step's Hydra config and resolve every ``_target_``."""
    configs: list[str] = []
    for step in steps:
        configs += STEP_REQUIREMENTS[step]["config"]
    if not configs:
        return

    report.header("Hydra configs")
    try:
        from hydra import compose, initialize
    except ImportError:
        report.fail("hydra is not installed", 'pip install "soccermaster[pipeline]"')
        return

    for config_name in configs:
        try:
            with initialize(version_base=None, config_path=f"pkg://{TRACKLAB_CONFIG_PKG}"):
                cfg = compose(config_name=config_name)
        except Exception as exc:
            report.fail(
                f"{config_name}: cannot compose ({type(exc).__name__}: {exc})",
                "check that the package is installed so the soccermaster_plugin and "
                "hydra search-path entry points are registered (pip install -e .)",
            )
            continue

        pipeline = list(cfg.get("pipeline", []))
        report.ok(f"{config_name}: composed, {len(cfg)} keys, {len(pipeline)} pipeline modules")
        report.note("pipeline: " + ", ".join(str(p) for p in pipeline))

        check_worker_budget(cfg, config_name, report)

        # Resolve the _target_ of every module that will actually be instantiated.
        # Modules in `defaults` but not in `pipeline` are never built, so an import
        # error there is not fatal -- verify() skips them.
        missing = []
        broken = []
        for name in pipeline:
            node = cfg.get("modules", {}).get(name)
            target = node.get("_target_") if node is not None else None
            if target is None:
                missing.append(name)
                continue
            module_path, _, attr = str(target).rpartition(".")
            try:
                spec = importlib.util.find_spec(module_path)
            except (ImportError, ModuleNotFoundError, ValueError):
                spec = None
            if spec is None:
                broken.append(f"{name}: no module {module_path}")
                continue
            try:
                mod = importlib.import_module(module_path)
            except Exception as exc:
                broken.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
            if not hasattr(mod, attr):
                broken.append(f"{name}: {module_path} has no attribute {attr}")

        if missing:
            report.fail("pipeline entries with no _target_: " + ", ".join(missing))
        if broken:
            for item in broken:
                report.fail(f"unresolvable module: {item}")
        if not missing and not broken:
            report.ok(f"all {len(pipeline)} pipeline modules resolve")


def check_worker_budget(cfg, config_name: str, report: Report) -> None:
    """Warn when the config asks for far more workers than this machine has.

    Upstream hardcodes ``num_cores: 64`` (step 1) / ``32`` (step 3) from its 64-core
    server. That feeds the ReID dataloader, the tracking engine and the visualizer, so
    on a 2-vCPU notebook it spawns 64 workers onto 2 cores. Nothing crashes; the run
    thrashes and looks hung. This is a warning, not a failure, because the pipeline
    overrides it by default.
    """
    import os

    requested = cfg.get("num_cores")
    if requested is None:
        return
    cpus = os.cpu_count() or 1
    if int(requested) > cpus * 2:
        report.warn(
            f"{config_name}: num_cores={requested} but this machine has {cpus} CPUs; "
            f"that many dataloader workers will thrash and look like a hang",
            "soccermaster sets num_cores automatically; if invoking Hydra directly, "
            f"pass num_cores={max(1, min(8, cpus - 1))}",
        )
    else:
        report.note(f"{config_name}: num_cores={requested} (machine has {cpus} CPUs)")


def check_dataset(dataset: str | None, steps: list[int], report: Report) -> None:
    report.header("Dataset")
    if not dataset:
        report.note("no --dataset given; skipping layout check")
        report.note(
            "the pipeline wants <dataset>/<split>/<sequence>/img1/*.jpg plus "
            "sequences_info.json"
        )
        return

    root = Path(dataset).expanduser()
    if not root.is_dir():
        report.fail(f"dataset path does not exist: {root}")
        return

    report.ok(f"dataset root exists: {root}")
    report.note("to turn a video into frames: "
                "python -m soccermaster.model.data.extract_frames --help")

    # sequences_info.json is read by the SAM 2 step only; steps 1 and 3 discover
    # sequences by listing the split directory.
    info = root / "sequences_info.json"
    if info.is_file():
        report.ok(f"sequences_info.json present")
    elif 2 in steps:
        report.warn("sequences_info.json missing; step 2 needs it")
    else:
        report.note("sequences_info.json missing; not needed for steps 1 and 3")

    splits = [p for p in root.iterdir() if p.is_dir()]
    if not splits:
        report.warn(f"no split directories under {root} (expected e.g. sn500/)")
        return

    for split in sorted(splits)[:5]:
        seqs = [d for d in split.iterdir() if d.is_dir()]
        with_frames = [d for d in seqs if (d / "img1").is_dir()]
        report.note(f"{split.name}: {len(seqs)} sequences, {len(with_frames)} with img1/")
        if seqs and not with_frames:
            report.warn(
                f"{split.name}: no sequence has an img1/ directory",
                "extract frames with soccermaster.model.data.extract_frames",
            )
            break


def run(steps: str | None = None, dataset: str | None = None) -> int:
    """Run every check. Returns a process exit code (0 = no failures)."""
    selected = _steps(steps)
    unknown = [s for s in selected if s not in STEP_REQUIREMENTS]
    if unknown:
        raise SystemExit(f"unknown step(s) {unknown}; known: {sorted(STEP_REQUIREMENTS)}")

    report = Report()
    print("soccermaster pre-flight")
    print(f"steps checked: {', '.join(str(s) for s in selected)}")

    check_environment(report)

    modules: set[str] = set()
    for step in selected:
        modules.update(STEP_REQUIREMENTS[step]["modules"])
    check_modules(modules, report)

    check_weights(selected, report)
    check_configs(selected, report)
    check_dataset(dataset, selected, report)

    report.header("Summary")
    if report.failures:
        print(f"  {len(report.failures)} failure(s) -- the pipeline will not start as-is:")
        for item in report.failures:
            print(f"    - {item}")
    else:
        print("  no blocking failures")
    if report.warnings:
        print(f"  {len(report.warnings)} warning(s):")
        for item in report.warnings:
            print(f"    - {item}")
    return 1 if report.failures else 0
