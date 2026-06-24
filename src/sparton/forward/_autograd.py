"""Shared fake + per-kernel autograd registration for the forward custom ops.

All three forwards share the fake-tensor metadata rule and the saved-tensor set, but the
backward differs by kernel: the optimized forward uses the optimized backward; the hybrid
and naive forwards use the mono backward. ``register_forward(op, bwd_op)`` binds the shared
fake + setup_context and the per-kernel backward, once per forward op (registration is
still applied per op).
"""

import torch


def shared_fwd_fake(hidden, embed, bias, mask):
    # Correct metadata only (shape/dtype/device); no real compute.
    B, S, D = hidden.shape
    V, D2 = embed.shape
    out_scores = hidden.new_empty((B, V))  # same dtype/device as hidden
    out_idx = torch.empty((B, V), device=hidden.device, dtype=torch.int64)
    return out_scores, out_idx


def shared_setup_context(ctx, inputs, output):
    hidden, embed, bias, mask = inputs
    scores, idx = output
    ctx.save_for_backward(scores, idx, hidden, embed, bias, mask)


def _make_backward(bwd_op):
    """Build the autograd backward that routes through ``bwd_op`` (mono or optimized)."""

    def backward(ctx, grad_scores, grad_idx):
        # grad_idx is ignored (idx is non-differentiable).
        scores, idx, hidden, embed, bias, mask = ctx.saved_tensors
        hidden_g, embed_g, bias_g = bwd_op(
            grad_scores, scores, idx, hidden, embed, bias, mask
        )
        return hidden_g, embed_g, bias_g, None

    return backward


def register_forward(op, bwd_op) -> None:
    """Bind the shared fake + the per-kernel backward onto a forward op (once per op)."""
    op.register_fake(shared_fwd_fake)
    op.register_autograd(_make_backward(bwd_op), setup_context=shared_setup_context)
