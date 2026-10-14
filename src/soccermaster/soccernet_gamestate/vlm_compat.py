"""Loading settings for the Qwen2.5-VL checkpoints the step-3 modules use.

Two hardcoded values in those modules only held on the machine they were written on, and
both failed late or quietly:

**Attention backend.** All three modules passed ``attn_implementation="flash_attention_2"``.
flash-attn is optional -- it lives in the ``train`` extra -- and is installed by neither the
``pipeline`` extra nor the Modal image, so ``from_pretrained`` raised before a single weight
loaded and step 3 could not start. Installing it is not always the answer either:
FlashAttention-2 requires Ampere or newer (SM 8.0), so on a T4 (SM 7.5) the import succeeds
and the kernels fail at launch. ``sdpa`` is PyTorch's own scaled-dot-product attention:
always present, independent of GPU architecture, and close to flash-attn for inference.

**dtype keyword.** The three modules disagreed, two passing ``dtype=`` and one
``torch_dtype=``. In the transformers range this package pins (``>=4.49,<4.52``) only
``torch_dtype`` is read -- ``from_pretrained`` does ``torch_dtype = kwargs.pop("torch_dtype",
None)`` and has no ``dtype`` equivalent, and ``PretrainedConfig`` has no ``dtype`` alias to
absorb it instead. Both were checked against the installed 4.51.3 source rather than assumed.
So ``dtype=torch.bfloat16`` was silently ignored and a 7B model loaded in float32: about
28 GB of weights instead of 14 GB. transformers >= 4.56 renamed the parameter to ``dtype``,
so the spelling is chosen from the installed version rather than hardcoded.

Resolving both here keeps the three modules from drifting apart again.
"""
from __future__ import annotations

import logging
import os

import torch

log = logging.getLogger(__name__)

__all__ = ["resolve_attn_implementation", "resolve_dtype_kwarg", "vlm_load_kwargs"]

#: Force an attention backend, e.g. SOCCERMASTER_ATTN_IMPLEMENTATION=flash_attention_2
ATTN_ENV = "SOCCERMASTER_ATTN_IMPLEMENTATION"

#: transformers renamed the ``torch_dtype`` argument to ``dtype``.
_DTYPE_RENAME_VERSION = (4, 56)


def resolve_attn_implementation() -> str:
    """Return the best attention backend that can actually run here.

    ``sdpa`` unless flash-attn is importable *and* the visible GPU is Ampere or newer.
    """
    override = os.environ.get(ATTN_ENV)
    if override:
        return override

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return "sdpa"

    # Importable is not the same as usable: FlashAttention-2 needs compute capability
    # 8.0+, so on Turing the import succeeds and the kernel launch fails.
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        if major < 8:
            log.warning(
                "flash-attn is installed but this GPU reports compute capability "
                "%d.x; FlashAttention-2 needs 8.0 or newer, so falling back to sdpa.",
                major,
            )
            return "sdpa"

    return "flash_attention_2"


def resolve_dtype_kwarg(dtype: torch.dtype = torch.bfloat16) -> dict:
    """Return the dtype keyword ``from_pretrained`` reads on the installed transformers."""
    try:
        import transformers

        major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    except Exception:  # noqa: BLE001 - a broken install must not change the spelling
        return {"torch_dtype": dtype}

    key = "dtype" if (major, minor) >= _DTYPE_RENAME_VERSION else "torch_dtype"
    return {key: dtype}


def vlm_load_kwargs(dtype: torch.dtype = torch.bfloat16) -> dict:
    """Keyword arguments for ``from_pretrained`` on a step-3 Qwen checkpoint."""
    return {
        **resolve_dtype_kwarg(dtype),
        "attn_implementation": resolve_attn_implementation(),
    }
