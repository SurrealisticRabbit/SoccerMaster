"""Codemod: rewrite vendored TrackLab imports to the unified package namespace.

Rewrites ``tracklab``, the tracker/plugin packages and ``trackeval`` to their
new locations under ``soccermaster.tracking``. Only import statements (and
Hydra ``_target_`` / ``pkg://`` config entries) are touched.

Usage: python scripts/rewrite_tracking_imports.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "soccermaster" / "tracking"

TRACKERS = (
    "bpbreid_naive_pitch_deep_sort",
    "bpbreid_pitch_deep_sort",
    "bpbreid_strong_sort",
    "bpbreid_deep_sort",
    "deep_oc_sort",
    "byte_track",
    "strong_sort",
    "bot_sort",
    "oc_sort",
)

PY_PREFIXES = {
    "tracklab": ["soccermaster", "tracking"],
    "plugins": ["soccermaster", "tracking", "plugins"],
    "trackeval": ["soccermaster", "tracking", "trackeval"],
    "posetrack21": ["soccermaster", "tracking", "plugins", "eval", "posetrack21"],
    "posetrack21_mot": ["soccermaster", "tracking", "plugins", "eval", "posetrack21_mot"],
}
for _t in TRACKERS:
    PY_PREFIXES[_t] = ["soccermaster", "tracking", "plugins", "track", _t]

FROM_RE = re.compile(r"^(\s*from\s+)([A-Za-z_][\w.]*)(\s+import\s+.*)$")
IMPORT_RE = re.compile(r"^(\s*import\s+)([A-Za-z_][\w.]*(?:\s*,\s*[A-Za-z_][\w.]*)*)(\s+as\s+\w+)?(\s*#.*)?$")
TARGET_RE = re.compile(r"^(\s*_target_\s*:\s*)([A-Za-z_][\w.]*)(\s*)$", re.M)


def rewrite_module(dotted: str) -> str:
    parts = dotted.split(".")
    head = parts[0]
    if head in PY_PREFIXES:
        return ".".join(PY_PREFIXES[head] + parts[1:])
    return dotted


def rewrite_yaml_target(value: str) -> str:
    if value.startswith("pkg://"):
        rest = value[len("pkg://"):]
        return "pkg://" + rewrite_module(rest)
    return rewrite_module(value)


def process_python(text: str) -> str:
    out = []
    for line in text.splitlines(keepends=True):
        raw = line.rstrip("\r\n")
        eol = line[len(raw):]
        m = FROM_RE.match(raw)
        if m:
            raw = m.group(1) + rewrite_module(m.group(2)) + m.group(3)
        else:
            m = IMPORT_RE.match(raw)
            if m:
                mods = ", ".join(rewrite_module(p.strip()) for p in m.group(2).split(","))
                raw = m.group(1) + mods + (m.group(3) or "") + (m.group(4) or "")
        out.append(raw + eol)
    return "".join(out)


def process_yaml(text: str) -> str:
    return TARGET_RE.sub(lambda m: m.group(1) + rewrite_yaml_target(m.group(2)) + m.group(3), text)


changed = []
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    if path.suffix == ".py":
        fn = process_python
    elif path.suffix in (".yaml", ".yml"):
        fn = process_yaml
    else:
        continue
    old = path.read_text(encoding="utf-8")
    new = fn(old)
    if new != old:
        path.write_text(new, encoding="utf-8", newline="")
        changed.append(path.relative_to(ROOT.parents[2]).as_posix())

print(f"Rewrote {len(changed)} files:")
for c in sorted(changed):
    print(" ", c)
