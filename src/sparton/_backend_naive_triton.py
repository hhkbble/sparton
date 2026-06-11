from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ._backend_hybrid import fused_sparton_bwd_op
from ._validation import autocast_canonicalize, validate_forward_inputs


def _naive_config(block_s: int, block_v: int, block_d: int, warps: int, stages: int):
    return triton.Config(
        {"BLOCK_S": block_s, "BLOCK_V": block_v, "BLOCK_D": block_d},
        num_warps=warps,
        num_stages=stages,
    )


def get_naive_forward_configs():
    return [
        _naive_config(16, 32, 32, 4, 3),
        _naive_config(16, 64, 32, 4, 3),
        _naive_config(32, 32, 32, 4, 3),
        _naive_config(32, 64, 32, 8, 3),
        _naive_config(16, 32, 64, 4, 3),
        _naive_config(16, 64, 64, 4, 3),
        _naive_config(32, 32, 64, 4, 3),
        _naive_config(32, 64, 64, 8, 3),
        _naive_config(64, 16, 64, 4, 3),
        _naive_config(64, 32, 64, 8, 3),
    ]


@triton.autotune(
    configs=get_naive_forward_configs(),
    key=["S", "D", "V"],
    cache_results=True,
)
@triton.jit
def sparton_naive_forward_kernel(
    hidden_ptr,
    embed_ptr,
    bias_ptr,
    mask_ptr,
    out_scores_ptr,
    out_idx_ptr,
    B: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    V: tl.constexpr,
    stride_hb: tl.constexpr,
    stride_hs: tl.constexpr,
    stride_hd: tl.constexpr,
    stride_ev: tl.constexpr,
    stride_ed: tl.constexpr,
    stride_mb: tl.constexpr,
    stride_ms: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_ov: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_v_block = tl.program_id(1)

    offs_s = tl.arange(0, BLOCK_S)
    offs_v = pid_v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    offs_d = tl.arange(0, BLOCK_D)
    mask_v = offs_v < V

    max_vals = tl.full((BLOCK_V,), 0.0, dtype=tl.float32)
    max_idx = tl.full((BLOCK_V,), 0, dtype=tl.int64)

    for s_start in range(0, S, BLOCK_S):
        seq = s_start + offs_s
        mask_s = seq < S
        acc = tl.zeros((BLOCK_S, BLOCK_V), dtype=tl.float32)

        for d_start in range(0, D, BLOCK_D):
            dim = d_start + offs_d
            mask_d = dim < D
            hidden_tile = tl.load(
                hidden_ptr
                + pid_b * stride_hb
                + seq[:, None] * stride_hs
                + dim[None, :] * stride_hd,
                mask=mask_s[:, None] & mask_d[None, :],
                other=0.0,
            )
            embed_tile = tl.load(
                embed_ptr
                + offs_v[None, :] * stride_ev
                + dim[:, None] * stride_ed,
                mask=mask_d[:, None] & mask_v[None, :],
                other=0.0,
            )
            acc += tl.dot(hidden_tile, embed_tile, out_dtype=tl.float32)

        if HAS_BIAS:
            bias_vals = tl.load(bias_ptr + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            acc += bias_vals[None, :]

        mask_vals = tl.load(
            mask_ptr + pid_b * stride_mb + seq * stride_ms,
            mask=mask_s,
            other=0,
        ).to(tl.float32)
        acc *= mask_vals[:, None]
        acc = tl.where(mask_s[:, None] & mask_v[None, :], acc, 0.0)

        tile_max = tl.max(acc, axis=0)
        tile_arg = tl.argmax(acc, axis=0)
        better = tile_max > max_vals
        max_vals = tl.where(better, tile_max, max_vals)
        max_idx = tl.where(better, s_start + tile_arg, max_idx)

    relu_vals = tl.where(max_vals > 0.0, max_vals, 0.0)
    scores = tl.log(1.0 + relu_vals)
    out_ptrs = out_scores_ptr + pid_b * stride_ob + offs_v * stride_ov
    idx_ptrs = out_idx_ptr + pid_b * stride_ob + offs_v * stride_ov
    tl.store(out_ptrs, scores, mask=mask_v)
    tl.store(idx_ptrs, max_idx, mask=mask_v)


def _launch_naive_fwd(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hidden.ndim == 3
    assert embed.ndim == 2
    assert mask.ndim == 2
    B, S, D = hidden.shape
    V, embed_dim = embed.shape
    assert D == embed_dim
    assert mask.shape == (B, S)
    if bias is not None:
        assert bias.shape == (V,)

    scores = torch.empty((B, V), device=hidden.device, dtype=hidden.dtype)
    indices = torch.empty((B, V), device=hidden.device, dtype=torch.int64)

    def grid(meta):
        return (B, triton.cdiv(V, meta["BLOCK_V"]))

    sparton_naive_forward_kernel[grid](
        hidden,
        embed,
        bias,
        mask,
        scores,
        indices,
        B,
        S,
        D,
        V,
        hidden.stride(0),
        hidden.stride(1),
        hidden.stride(2),
        embed.stride(0),
        embed.stride(1),
        mask.stride(0),
        mask.stride(1),
        scores.stride(0),
        scores.stride(1),
        HAS_BIAS=bias is not None,
    )
    return scores, indices


@torch.library.custom_op(
    "sparton::naive_fwd",
    mutates_args=(),
    schema="(Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)",
)
def naive_fwd_op(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hidden.is_cuda, "sparton::naive_fwd only supports CUDA"
    return _launch_naive_fwd(hidden, embed, bias, mask)


@naive_fwd_op.register_fake
def _(hidden, embed, bias, mask):
    B, S, D = hidden.shape
    V, D2 = embed.shape
    out_scores = hidden.new_empty((B, V))
    out_idx = torch.empty((B, V), device=hidden.device, dtype=torch.int64)
    return out_scores, out_idx


def _setup_context(ctx, inputs, output):
    hidden, embed, bias, mask = inputs
    scores, idx = output
    ctx.save_for_backward(scores, idx, hidden, embed, bias, mask)


def _backward(ctx, grad_scores, grad_idx):
    scores, idx, hidden, embed, bias, mask = ctx.saved_tensors
    hidden_g, embed_g, bias_g = fused_sparton_bwd_op(
        grad_scores,
        scores,
        idx,
        hidden,
        embed,
        bias,
        mask,
    )
    return hidden_g, embed_g, bias_g, None


naive_fwd_op.register_autograd(_backward, setup_context=_setup_context)


def naive_forward(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden, embed, bias = autocast_canonicalize(hidden, embed, bias)
    validate_forward_inputs(hidden, embed, bias, mask, backend="naive")
    hidden = hidden.contiguous()
    embed = embed.contiguous()
    mask = mask.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    return naive_fwd_op(hidden, embed, bias, mask)


__all__ = [
    "get_naive_forward_configs",
    "naive_forward",
    "naive_fwd_op",
    "sparton_naive_forward_kernel",
]
