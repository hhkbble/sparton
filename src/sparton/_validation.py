"""Shared input-contract validation for the Sparton forward wrappers.

``SpartonHead.forward`` maps its arguments onto the wrapper signature as
``weight -> embed``, ``bias -> bias``, ``attention_mask -> mask``; the error
messages below use the wrapper argument names.

Contract notes:

- ``mask`` is contractually a binary {0, 1} tensor (the standard tokenizer
  ``attention_mask``), in bool, integer, or floating point. The forward
  multiplies logits by the mask, so non-binary values produce weighted
  logits — an implementation property outside the original Sparton
  contract: the shared backward does not differentiate the mask factor, so
  gradients are exact for binary masks only. Weighted-mask support would be
  an extension (backward change tested against the autograd head). Values
  are not validated because these checks are metadata-only by design (see
  below); rejecting non-binary values would need a data scan and a device
  sync.
- Empty tensors (``B``, ``S``, or ``V`` equal to 0) are not validated here and
  kernel behavior for them is unspecified.
- The custom ops themselves assume validated, contiguous ("canonical")
  inputs; callers that invoke the raw ops bypass these checks by design.

All checks are metadata-only (no device synchronization, no data access), so
they are safe inside ``torch.compile``-traced code.
"""

from __future__ import annotations

from typing import Optional

import torch

_HALF_DTYPES = (torch.float16, torch.bfloat16)
# fp32 is a retained compatibility path for the hybrid kernel: it works through
# the compiled matmul + Triton reduction path but is not benchmark-covered.
_HYBRID_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def validate_forward_inputs(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
    mask: torch.Tensor,
    *,
    kernel: str,
) -> None:
    """Raise ValueError/TypeError naming the argument, value, and requirement."""

    prefix = f"sparton {kernel} forward:"

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

    supported = _HYBRID_DTYPES if kernel == "hybrid" else _HALF_DTYPES
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

    if kernel == "optimized":
        row_bytes = dim * hidden.element_size()
        if row_bytes % 16 != 0:
            raise ValueError(
                f"{prefix} D * element_size must be a multiple of 16 bytes "
                "because TMA descriptors require 16-byte-aligned innermost "
                "strides for the hidden and embed rows (the embed rows share "
                f"the same D and dtype); got D={dim} * "
                f"{hidden.element_size()} B = {row_bytes} B"
            )


def autocast_canonicalize(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Mirror ``torch.autocast`` semantics for the Sparton forwards.

    The custom ops are not autocast-registered, so under an active CUDA
    autocast region a head with fp32 master parameters would otherwise hand
    mixed dtypes to ``validate_forward_inputs``. Cast the floating inputs to
    the autocast dtype — exactly what autocast does for ``torch.matmul`` —
    and leave everything unchanged outside autocast. ``mask`` is never cast.
    Gradients flow back to the original (e.g. fp32 master) tensors through
    the cast, as with any autocast op.
    """

    if not torch.is_autocast_enabled("cuda"):
        return hidden, embed, bias
    target = torch.get_autocast_dtype("cuda")
    if target not in (torch.float16, torch.bfloat16):
        return hidden, embed, bias

    def _cast(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None or not tensor.is_floating_point():
            return tensor
        if tensor.dtype == target:
            return tensor
        return tensor.to(target)

    return _cast(hidden), _cast(embed), _cast(bias)


def prepare_forward_inputs(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
    mask: torch.Tensor,
    *,
    kernel: str,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """Shared forward input canonicalization: autocast -> validate -> contiguous.

    The single expression of the wrapper layering rule (wrapper ->
    ``autocast_canonicalize`` -> ``validate_forward_inputs`` -> ``.contiguous()``
    -> op) that all three forward wrappers share. ``kernel`` selects the
    per-kernel validation branch (hybrid additionally accepts fp32; optimized
    adds the TMA 16-byte alignment check) and labels the error messages.
    """

    hidden, embed, bias = autocast_canonicalize(hidden, embed, bias)
    validate_forward_inputs(hidden, embed, bias, mask, kernel=kernel)
    hidden = hidden.contiguous()
    embed = embed.contiguous()
    mask = mask.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    return hidden, embed, bias, mask


__all__ = ["autocast_canonicalize", "prepare_forward_inputs", "validate_forward_inputs"]
