"""Codemod: rewrite sn-gamestate imports to the unified soccermaster namespace.

Usage: python scripts/rewrite_gamestate_imports.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "soccermaster" / "gamestate"

CALIB = ["soccermaster", "gamestate", "plugins", "calibration"]

PREFIXES = {
    "sn_gamestate": ["soccermaster", "gamestate"],
    "tracklab": ["soccermaster", "tracking"],
    "nbjw_calib": CALIB + ["nbjw_calib"],
    "pnlcalib": CALIB + ["pnlcalib"],
    "sn_calibration_baseline": CALIB + ["sn_calibration_baseline"],
    "tvcalib": CALIB + ["tvcalib"],
}

FROM_RE = re.compile(r"^(\s*from\s+)([A-Za-z_][\w.]*)(\s+import\s+.*)$")
IMPORT_RE = re.compile(r"^(\s*import\s+)([A-Za-z_][\w.]*(?:\s*,\s*[A-Za-z_][\w.]*)*)(\s+as\s+\w+)?(\s*#.*)?$")
TARGET_RE = re.compile(r"^(\s*_target_\s*:\s*)([A-Za-z_][\w.]*)(\s*)$", re.M)
PKG_STR_RE = re.compile(r"pkg://(sn_gamestate)([\w.]*)")


def rewrite_module(dotted: str) -> str:
    parts = dotted.split(".")
    head = parts[0]
    if head in PREFIXES:
        return ".".join(PREFIXES[head] + parts[1:])
    return dotted


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
        raw = PKG_STR_RE.sub(lambda mm: "pkg://" + rewrite_module(mm.group(1) + mm.group(2)), raw)
        out.append(raw + eol)
    return "".join(out)


def process_yaml(text: str) -> str:
    return TARGET_RE.sub(lambda m: m.group(1) + rewrite_module(m.group(2)) + m.group(3), text)


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
