"""One-off codemod: rewrite SoccerMaster top-level imports to soccermaster.foundation.*.

Usage: python scripts/rewrite_foundation_imports.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "soccermaster" / "foundation"
PKGS = ("models", "data", "utils", "configs")

PATTERNS = [
    (re.compile(rf"^(\s*from\s+)({'|'.join(PKGS)})(?=[\.\s])", re.M), r"\1soccermaster.foundation.\2"),
    (re.compile(rf"^(\s*import\s+)({'|'.join(PKGS)})(?=[\.\s])", re.M), r"\1soccermaster.foundation.\2"),
]

changed = []
for py in ROOT.rglob("*.py"):
    text = py.read_text(encoding="utf-8")
    new = text
    for pat, repl in PATTERNS:
        new = pat.sub(repl, new)
    if new != text:
        py.write_text(new, encoding="utf-8", newline="")
        changed.append(py.relative_to(ROOT.parents[2]).as_posix())

print(f"Rewrote {len(changed)} files:")
for c in sorted(changed):
    print(" ", c)
