"""The public API surface must be the pipeline functions, not copies of them.

``soccermaster.api`` used to declare its own ``analyze_video`` and ``run_pipeline`` as
hand-written pass-throughs that listed every argument explicitly. Both silently drifted
from the real implementations:

- ``api.analyze_video`` had lost ``quality``, and never gained ``fps``, so
  ``sm.analyze_video(..., fps=1)`` raised TypeError while the identical CLI call worked --
  the CLI goes straight to ``pipeline.analyze_video``.
- ``api.run_pipeline`` was missing ``eval_set`` and ``save_videos``.

Because ``sm.analyze_video`` resolves through ``api`` (the ``_LAZY`` map in
``__init__.py``), the broken path was the *documented* one. These tests pin the fix:
identity, not equality of signature, because the whole point is that there is now a single
definition that cannot diverge.
"""
from __future__ import annotations

import inspect

import pytest

# Re-exported by api.py, so this is light: pipeline imports only stdlib and paths.
api = pytest.importorskip("soccermaster.api")
pipeline = pytest.importorskip("soccermaster.pipeline")

import soccermaster as sm  # noqa: E402


#: Names that exist in both modules. If one of these is ever re-declared in api.py instead
#: of re-exported, the identity check below fails.
SHARED = ["analyze_video", "run_pipeline"]


@pytest.mark.parametrize("name", SHARED)
def test_api_name_is_the_pipeline_object(name: str) -> None:
    """Identity, not mere compatibility: one definition, so drift is impossible."""
    assert getattr(api, name) is getattr(pipeline, name), (
        f"soccermaster.api.{name} is not soccermaster.pipeline.{name}; it has been "
        f"re-declared instead of re-exported, which is how the signatures drifted apart"
    )


@pytest.mark.parametrize("name", SHARED)
def test_public_attribute_is_the_pipeline_object(name: str) -> None:
    """And the same must hold through the lazy ``sm.`` surface, which is what users call."""
    assert getattr(sm, name) is getattr(pipeline, name)


def test_analyze_video_accepts_the_pipeline_arguments() -> None:
    """The specific regression: `fps` and `quality` reach analyze_video via `sm.`."""
    parameters = inspect.signature(sm.analyze_video).parameters
    for expected in ("quality", "fps", "write_video", "extra_overrides", "steps"):
        assert expected in parameters, f"sm.analyze_video lost {expected!r}"


def test_run_pipeline_accepts_the_pipeline_arguments() -> None:
    parameters = inspect.signature(sm.run_pipeline).parameters
    for expected in ("eval_set", "save_videos", "steps", "video_ids"):
        assert expected in parameters, f"sm.run_pipeline lost {expected!r}"


@pytest.mark.parametrize("name", sorted(sm.__all__))
def test_every_public_name_resolves(name: str) -> None:
    """Guards the _LAZY map: a name listed in __all__ must actually be reachable."""
    assert getattr(sm, name) is not None
