"""Shared input-contract validation for the Sparton forward wrappers.

``SpartonHead.forward`` maps its arguments onto the wrapper signature as
``weight -> embed``, ``bias -> bias``, ``attention_mask -> mask``; the error
messages below use the wrapper argument names.

Contract notes:

- ``mask`` may be bool, integer, or floating point. Non-binary mask values
  produce weighted logits (``logits * mask``), which is defined behavior but
  not the Hugging Face attention-mask contract.
- Empty tensors (``B``, ``S``, or ``V`` equal to 0) are not validated here and
  backend behavior for them is unspecified.
- The custom ops themselves assume validated, contiguous ("canonical")
  inputs; callers that invoke the raw ops bypass these checks by design.

All checks are metadata-only (no device synchronization, no data access), so
they are safe inside ``torch.compile``-traced code.
"""

from __future__ import annotations

from typing import Optional

import torch

_FUSED_DTYPES = (torch.float16, torch.bfloat16)
# fp32 hybrid is permitted legacy behavior: it works through the compiled
# matmul + Triton reduction path but is not benchmark-covered.
_HYBRID_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def validate_forward_inputs(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
    mask: torch.Tensor,
    *,
    backend: str,
) -> None:
    """Raise ValueError/TypeError naming the argument, value, and requirement."""

    prefix = f"sparton {backend} forward:"

    if hidden.ndim != 3:
        raise ValueError(
            f"{prefix} hidden must be a 3-D [B, S, D] tensor; got ndim={hidden.ndim}"
        )
    if embed.ndim != 2:
        raise ValueError(
            f"{prefix} embed must be a 2-D [V, D] tensor; got ndim={embed.ndim}"
        )
    if mask.ndim != 2:
        raise ValueError(
            f"{prefix} mask must be a 2-D [B, S] tensor; got ndim={mask.ndim}"
        )

    batch, seq_len, dim = hidden.shape
    vocab, embed_dim = embed.shape
    if embed_dim != dim:
        raise ValueError(
            f"{prefix} embed.shape[1] must equal hidden.shape[2] ({dim}); "
            f"got {embed_dim}"
        )
    if mask.shape != (batch, seq_len):
        raise ValueError(
            f"{prefix} mask.shape must equal (B, S) = {(batch, seq_len)}; "
            f"got {tuple(mask.shape)}"
        )
    if bias is not None and bias.shape != (vocab,):
        raise ValueError(
            f"{prefix} bias.shape must equal (V,) = {(vocab,)}; "
            f"got {tuple(bias.shape)}"
        )

    if not hidden.is_cuda:
        raise ValueError(
            f"{prefix} hidden must be a CUDA tensor; got device={hidden.device}"
        )
    for name, tensor in (("embed", embed), ("bias", bias), ("mask", mask)):
        if tensor is not None and tensor.device != hidden.device:
            raise ValueError(
                f"{prefix} {name}.device must equal hidden.device "
                f"({hidden.device}); got {tensor.device}"
            )

    supported = _HYBRID_DTYPES if backend == "hybrid" else _FUSED_DTYPES
    if hidden.dtype not in supported:
        names = ", ".join(str(dtype) for dtype in supported)
        raise TypeError(
            f"{prefix} hidden.dtype must be one of ({names}); got {hidden.dtype}"
        )
    if embed.dtype != hidden.dtype:
        raise TypeError(
            f"{prefix} embed.dtype must equal hidden.dtype ({hidden.dtype}); "
            f"got {embed.dtype}"
        )
    if bias is not None and bias.dtype != hidden.dtype:
        raise TypeError(
            f"{prefix} bias.dtype must equal hidden.dtype ({hidden.dtype}); "
            f"got {bias.dtype}"
        )
    if mask.dtype.is_complex:
        raise TypeError(
            f"{prefix} mask.dtype must be bool, integer, or floating point; "
            f"got {mask.dtype}"
        )

    if backend == "optimized":
        row_bytes = dim * hidden.element_size()
        if row_bytes % 16 != 0:
            raise ValueError(
                f"{prefix} D * element_size must be a multiple of 16 bytes "
                "because TMA descriptors require 16-byte-aligned innermost "
                "strides for the hidden and embed rows (the embed rows share "
                f"the same D and dtype); got D={dim} * "
                f"{hidden.element_size()} B = {row_bytes} B"
            )


__all__ = ["validate_forward_inputs"]
