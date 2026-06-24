"""Sparton backward ops: ``mono`` (original) and ``optimized`` (current best).

Two custom ops with the identical saved-tensor schema, one per backward kernel:
``optimized_bwd_op`` (``sparton::optimized_bwd``) drives the optimized hidden-grad
design; ``mono_bwd_op`` (``sparton::mono_bwd``) drives the restored M2 fully-atomic
backward. Each forward registers the op matching its kernel (optimized forward ->
optimized; hybrid/naive -> mono), so the selection is per-forward, not a runtime switch.
Both accumulate gradients in float32. ``backward`` imports nothing from ``forward`` -- the
one-way edge that keeps the package acyclic.
"""

from typing import Optional, Tuple

import torch

from .mono import mono_bwd
from .optimized import optimized_bwd

_BWD_SCHEMA = (
    "(Tensor grad_out, Tensor max_scores, Tensor max_idx, Tensor hidden, "
    "Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor, Tensor?)"
)


@torch.library.custom_op("sparton::optimized_bwd", mutates_args=(), schema=_BWD_SCHEMA)
def optimized_bwd_op(
    grad_out: torch.Tensor,     # [B,V]
    max_scores: torch.Tensor,   # [B,V]
    max_idx: torch.Tensor,      # [B,V] int64
    hidden: torch.Tensor,       # [B,S,D]
    embed: torch.Tensor,        # [V,D]
    bias: Optional[torch.Tensor],         # [V] or None
    mask: torch.Tensor,         # [B,S] (unused by the backward; kept for op-signature symmetry with the forward)
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert grad_out.is_cuda, "sparton::optimized_bwd only supports CUDA"
    grad_out = grad_out.contiguous()
    hidden_grad, embed_grad, bias_grad = optimized_bwd(
        grad_out, max_scores, max_idx, hidden, embed, bias
    )
    return hidden_grad, embed_grad, bias_grad if bias is not None else None


@torch.library.custom_op("sparton::mono_bwd", mutates_args=(), schema=_BWD_SCHEMA)
def mono_bwd_op(
    grad_out: torch.Tensor,     # [B,V]
    max_scores: torch.Tensor,   # [B,V]
    max_idx: torch.Tensor,      # [B,V] int64
    hidden: torch.Tensor,       # [B,S,D]
    embed: torch.Tensor,        # [V,D]
    bias: Optional[torch.Tensor],         # [V] or None
    mask: torch.Tensor,         # [B,S] (unused by the backward; kept for op-signature symmetry with the forward)
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert grad_out.is_cuda, "sparton::mono_bwd only supports CUDA"
    grad_out = grad_out.contiguous()
    hidden_grad, embed_grad, bias_grad = mono_bwd(
        grad_out, max_scores, max_idx, hidden, embed, bias
    )
    return hidden_grad, embed_grad, bias_grad if bias is not None else None


def _bwd_fake(grad_out, max_scores, max_idx, hidden, embed, bias, mask):
    # Correct metadata only (shape/dtype/device); identical for both ops.
    return (
        torch.empty_like(hidden, dtype=torch.float32),
        torch.empty_like(embed, dtype=torch.float32),
        torch.empty_like(bias, dtype=torch.float32) if bias is not None else None,
    )


optimized_bwd_op.register_fake(_bwd_fake)
mono_bwd_op.register_fake(_bwd_fake)


__all__ = [
    "mono_bwd_op",
    "optimized_bwd_op",
    "mono_bwd",
    "optimized_bwd",
]
