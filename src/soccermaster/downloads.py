"""Pretrained-weight downloads driven by ``resources/weights_manifest.yaml``."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from soccermaster import paths

MANIFEST_PATH = Path(__file__).parent / "resources" / "weights_manifest.yaml"


@dataclass
class WeightEntry:
    name: str
    source: str
    dest: Path
    repo: str | None = None
    allow: list[str] = field(default_factory=list)
    file_id: str | None = None
    rename: dict[str, str] = field(default_factory=dict)
    hint: str = ""

    @property
    def automatic(self) -> bool:
        return self.source in ("hf", "gdrive")


def load_manifest() -> dict[str, WeightEntry]:
    """Parse the manifest into ``{name: WeightEntry}`` with absolute destinations."""
    raw = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    root = paths.weights_dir()
    entries: dict[str, WeightEntry] = {}
    for name, cfg in raw.items():
        cfg = cfg or {}
        entries[name] = WeightEntry(
            name=name,
            source=cfg.get("source", "manual"),
            dest=root / cfg.get("dest", name),
            repo=cfg.get("repo"),
            allow=list(cfg.get("allow") or []),
            file_id=cfg.get("file_id"),
            rename=dict(cfg.get("rename") or {}),
            hint=cfg.get("hint", ""),
        )
    return entries


def _apply_renames(entry: WeightEntry) -> None:
    """Rename fetched files in-place according to ``entry.rename``."""
    for src_name, dst_name in entry.rename.items():
        src = entry.dest / src_name
        if not src.exists():
            continue
        dst = entry.dest / dst_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def _download_hf(entry: WeightEntry) -> None:
    from huggingface_hub import snapshot_download

    entry.dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=entry.repo,
        local_dir=str(entry.dest),
        allow_patterns=entry.allow or None,
    )


def _download_gdrive(entry: WeightEntry) -> None:
    import gdown

    entry.dest.mkdir(parents=True, exist_ok=True)
    gdown.download(id=entry.file_id, output=str(entry.dest) + "/", quiet=False)


def prepare(name: str, dry_run: bool = False) -> Path:
    """Fetch one manifest entry. Returns the destination directory.

    ``dry_run`` prints what would happen without touching the network.
    """
    entries = load_manifest()
    if name not in entries:
        known = ", ".join(sorted(entries))
        raise KeyError(f"unknown weight '{name}'. Known: {known}")
    entry = entries[name]

    print(f"[{entry.name}] source={entry.source} dest={entry.dest}")
    if entry.hint:
        print(f"  {entry.hint}")

    if not entry.automatic:
        print("  manual download required; nothing to fetch automatically.")
        return entry.dest

    if dry_run:
        if entry.source == "hf":
            print(f"  would download hf://{entry.repo} (allow={entry.allow or '*'})")
        else:
            print(f"  would download gdrive id {entry.file_id}")
        return entry.dest

    if entry.source == "hf":
        _download_hf(entry)
    else:
        _download_gdrive(entry)

    if entry.rename:
        _apply_renames(entry)

    print("  done.")
    return entry.dest


def prepare_all(dry_run: bool = False) -> None:
    """Fetch every manifest entry (skipping manual ones)."""
    entries = load_manifest()
    for name in entries:
        prepare(name, dry_run=dry_run)
