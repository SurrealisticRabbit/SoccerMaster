"""Every git or URL requirement must name its target correctly.

pip requires the name in a requirement to match the target's declared distribution name,
and rejects the install otherwise::

    Generating metadata for package sam2 produced metadata for project name sam-2
    ... has inconsistent name: expected 'sam2', but metadata has 'sam-2'
    ERROR: No matching distribution found for sam2

That failed the Modal image build, and it is easy to miss because the two names here
genuinely differ -- the vendored SAM 2 declares ``NAME = "SAM-2"`` (distribution ``sam-2``)
while its *import* name is ``sam2``, because the package directory is ``sam2/``.

Checking that a requirement merely parses is not enough: PEP 508 syntax accepts any name,
so ``Requirement("sam2 @ git+...")`` is valid and says nothing about whether the target
agrees. These tests compare canonicalised names against the actual declarations, which is
the check pip performs.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"


def _load() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _declared_name(setup_py: Path) -> str:
    """The distribution name a vendored package's setup.py declares."""
    source = setup_py.read_text(encoding="utf-8")
    match = re.search(r'^NAME\s*=\s*["\'](.+?)["\']', source, re.M)
    assert match, f"no NAME = ... found in {setup_py}"
    return match.group(1)


# --------------------------------------------------------------------------- #
# the sam2 extra specifically
# --------------------------------------------------------------------------- #


def test_sam2_extra_names_the_vendored_distribution_correctly() -> None:
    """The regression: `sam2` was rejected because the metadata says `sam-2`."""
    extra = _load()["project"]["optional-dependencies"]["sam2"]

    sam_urls = [e for e in extra if "subdirectory=third_party/sam2" in e]
    assert sam_urls, "the sam2 extra should point at the vendored tree"

    requirement = Requirement(sam_urls[0])
    declared = _declared_name(ROOT / "third_party" / "sam2" / "setup.py")

    assert canonicalize_name(requirement.name) == canonicalize_name(declared), (
        f"the extra requires {requirement.name!r} but the vendored package declares "
        f"{declared!r}; pip rejects a mismatch and fails the build"
    )


def test_sam2_extra_does_not_use_the_import_name() -> None:
    """Documents the trap: `sam2` is the import name and must not be the requirement name."""
    extra = _load()["project"]["optional-dependencies"]["sam2"]
    sam_urls = [e for e in extra if "subdirectory=third_party/sam2" in e]
    assert sam_urls

    name = Requirement(sam_urls[0]).name
    assert canonicalize_name(name) != canonicalize_name("sam2"), (
        "requiring `sam2` does not match the vendored metadata `sam-2`"
    )
    # And the package directory really is sam2/, which is why the two differ.
    assert (ROOT / "third_party" / "sam2" / "sam2" / "__init__.py").is_file()


def test_the_subdirectory_pin_is_a_commit_not_a_branch() -> None:
    """A branch is a moving ref, so an install could not be described afterwards."""
    extra = _load()["project"]["optional-dependencies"]["sam2"]
    url = Requirement([e for e in extra if "subdirectory=third_party/sam2" in e][0]).url
    assert url is not None

    ref = re.search(r"@([^#]+)#", url)
    assert ref, f"no ref found in {url}"
    assert re.fullmatch(r"[0-9a-f]{40}", ref.group(1)), (
        f"expected a full commit SHA, got {ref.group(1)!r}"
    )


# --------------------------------------------------------------------------- #
# every URL requirement, since the same rule applies to all of them
# --------------------------------------------------------------------------- #


def _all_requirements() -> list[tuple[str, str]]:
    """(where, requirement) for every declared dependency."""
    data = _load()
    found: list[tuple[str, str]] = []
    for name, spec in (data["project"].get("dependencies") and
                       {"core": data["project"]["dependencies"]} or {}).items():
        found += [(name, r) for r in spec]
    for extra, spec in data["project"].get("optional-dependencies", {}).items():
        found += [(extra, r) for r in spec]
    return found


def test_all_requirements_parse() -> None:
    for where, raw in _all_requirements():
        try:
            Requirement(raw)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"{where}: {raw!r} does not parse: {exc}")


def test_url_requirements_declare_a_name() -> None:
    """A bare git URL installs under whatever name the target claims, which is unverifiable.

    Naming it explicitly is what lets the check above catch a mismatch.
    """
    unnamed = [
        f"{where}: {raw}"
        for where, raw in _all_requirements()
        if raw.startswith(("git+", "http://", "https://"))
    ]
    assert not unnamed, "URL requirements must be written as '<name> @ <url>': " + "; ".join(
        unnamed
    )


def test_git_requirements_are_pinned_to_a_commit() -> None:
    """Unpinned git refs mean a rebuild can silently pick up different code.

    Not yet true for every dependency -- prtreid, bpbreid and poseval still track a
    branch, which is a known open item -- so this records the current state rather than
    asserting the ideal. It fails if a *new* unpinned git requirement appears.
    """
    unpinned: list[str] = []
    for where, raw in _all_requirements():
        requirement = Requirement(raw)
        url = requirement.url or ""
        if not url.startswith("git+"):
            continue
        ref = re.search(r"@([^#]+)", url.split("git+", 1)[1])
        if not ref or not re.fullmatch(r"[0-9a-f]{40}", ref.group(1)):
            unpinned.append(f"{where}: {requirement.name}")

    # The three known ones. If this list grows, a new dependency was left floating.
    assert sorted(unpinned) == ["pipeline: poseval", "pipeline: prtreid", "pipeline: torchreid"], (
        f"git requirements without a commit pin changed: {unpinned}"
    )
