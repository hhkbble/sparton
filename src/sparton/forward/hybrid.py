"""Hybrid forward kernel: compiled tiled matmul + Triton max/log1p/ReLU reduction.

The compatibility forward (works below the optimized kernel's sm_90+ TMA floor,
and the only forward that also accepts fp32). Output follows the input dtype.
"""

import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

from .._validation import prepare_forward_inputs
from ..backward import mono_bwd_op
from ._autograd import register_forward


def get_hybrid_forward_configs():
    return [
        triton.Config({'BLOCK_C': 64, 'BLOCK_S': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_S': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_S': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_C': 64, 'BLOCK_S': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_S': 64}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_C': 256, 'BLOCK_S': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 256, 'BLOCK_S': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_C': 256, 'BLOCK_S': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_C': 64,  'BLOCK_S': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_C': 64,  'BLOCK_S': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_C': 64,  'BLOCK_S': 512}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_C': 128,  'BLOCK_S': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_C': 256,  'BLOCK_S': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_C': 512, 'BLOCK_S': 32},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 256, 'BLOCK_S': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_C': 256, 'BLOCK_S': 256}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_C': 128, 'BLOCK_S': 512}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_C': 512, 'BLOCK_S': 64},  num_warps=16, num_stages=3),
        triton.Config({'BLOCK_C': 512, 'BLOCK_S': 128}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_C': 1024, 'BLOCK_S': 32}, num_warps=16, num_stages=2),
    ]


@triton.autotune(
   configs=get_hybrid_forward_configs(),
   key=['S', 'C'],
   cache_results=True
)
@triton.jit
def reduce_seq_max_log1p_relu_kernel(
    logits_ptr,        # float32,float16 [B, S, C]
    mask_ptr,          # int64/bool, [B, S]
    out_vals_ptr,      # float32, float16    [B, C]
    out_idx_ptr,       # int64,      [B, C]

    B: tl.constexpr,
    S: tl.constexpr,
    C: tl.constexpr,

    stride_lb, stride_ls, stride_lc,
    stride_mb, stride_ms,
    stride_ob, stride_oc,

    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # program ids:
    #  - pid_b: which batch row
    #  - pid_c_block: which block of vocab indices
    pid_b = tl.program_id(0)
    pid_c_block = tl.program_id(1)

    # vocab indices handled by this program
    offs_c = pid_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    NEG_INF = 0.0
    max_vals = tl.full((BLOCK_C,), NEG_INF, dtype=tl.float32)
    max_idx  = tl.full((BLOCK_C,), 0,      dtype=tl.int64)

    # sweep over the sequence dimension in tiles of size BLOCK_S
    s_range = tl.arange(0, BLOCK_S)
    for s_start in range(0, S, BLOCK_S):
        offs_s = s_start + s_range
        mask_s = offs_s < S

        offs_s_b = offs_s[:, None]     # [BLOCK_S, 1]
        offs_c_b = offs_c[None, :]     # [1, BLOCK_C]

        # logits[b, s, c]
        logits_ptrs = (
            logits_ptr
            + pid_b * stride_lb
            + offs_s_b * stride_ls
            + offs_c_b * stride_lc
        )
        tile_mask = mask_s[:, None] & mask_c[None, :]
        logits_tile = tl.load(
            logits_ptrs,
            mask=tile_mask,
            other=0.0,
        )

        # mask[b, s]
        mask_ptrs = mask_ptr + pid_b * stride_mb + offs_s * stride_ms
        mask_vals = tl.load(
            mask_ptrs,
            mask=mask_s,
            other=0,
        )

        logits_tile = logits_tile * mask_vals[:, None]  # apply sequence mask
        logits_tile = tl.where(tile_mask, logits_tile, NEG_INF)

        # max over this tile’s S, keep local argmax
        tile_max_vals = tl.max(logits_tile, axis=0)     # [BLOCK_C]
        tile_argmax   = tl.argmax(logits_tile, axis=0)  # [BLOCK_C]

        better = tile_max_vals > max_vals
        max_vals = tl.where(better, tile_max_vals, max_vals)
        max_idx  = tl.where(better, s_start + tile_argmax, max_idx)

    # ReLU + log1p on max values
    zero = 0.0
    relu_vals = tl.where(max_vals > zero, max_vals, zero)
    log_vals  = tl.log(1.0 + relu_vals)

    # store results [B, C]
    out_vals_ptrs = out_vals_ptr + pid_b * stride_ob + offs_c * stride_oc
    out_idx_ptrs  = out_idx_ptr  + pid_b * stride_ob + offs_c * stride_oc

    tl.store(out_vals_ptrs, log_vals, mask=mask_c)
    tl.store(out_idx_ptrs,  max_idx, mask=mask_c)


def reduce_seq_max_log1p_relu(logits: torch.Tensor,
                                     mask: torch.Tensor):
    """
    logits: [B, S, C], float32, contiguous
    mask:   [B, S],    int32 / bool, contiguous
    returns:
        vals: [B, C] (float32)      = log1p(relu(max_s logits * mask))
        idxs: [B, C] (int64)        = argmax_s logits * mask
    """
    assert logits.ndim == 3
    assert mask.ndim == 2
    B, S, C = logits.shape
    assert mask.shape == (B, S)
    logits = logits.contiguous()
    mask   = mask.contiguous()
    vals = torch.empty((B, C), device=logits.device, dtype=logits.dtype)
    idxs = torch.empty((B, C), device=logits.device, dtype=torch.int64)

    def grid(meta):
        return (
            B,  # one program per batch row
            triton.cdiv(C, meta['BLOCK_C']),
        )

    reduce_seq_max_log1p_relu_kernel[grid](
        logits, mask,
        vals, idxs,
        B, S, C,
        logits.stride(0), logits.stride(1), logits.stride(2),
        mask.stride(0),   mask.stride(1),
        vals.stride(0),   vals.stride(1),
    )
    return vals, idxs


@torch.compile
def matmul(a, b):
    return a @ b.T

@torch.compile
def matmul_bias(a, b, c):
    return a @ b.T + c

def v_tile_from_bs(
    B, S, V,
    temp_mib=64,   #  allowed temp memory for [B,S,C]
    align=256,      # round C to multiple of 256
    min_tiles=512  # avoid too many tiny tiles
):
    # fp16 logits → 2 bytes
    temp_bytes = temp_mib * 1024 * 1024
    C = temp_bytes // (2 * B * S)

    # clamp + align
    C = max(min_tiles, min(V, (C // align) * align))
    return C


def _launch_hybrid_fwd(hidden, embed, bias, mask):
    """
    hidden: [B, S, D], float16
    embed:  [V, D],    float16
    mask:   [B, S],    int32 / bool
    """
    B, S, D = hidden.shape
    V, _ = embed.shape

    sparse_reps_tiles = []
    max_indices_tiles = []
    tile_size = v_tile_from_bs(B, S, V) #V_TILE_SIZE
    sparse_reps = torch.empty((B, V), device=hidden.device, dtype=hidden.dtype)
    max_indices = torch.empty((B, V), device=hidden.device, dtype=torch.int64)
    for i in range(0, V, tile_size):
        tile_embed = embed[i:i + tile_size]       # [C, D]
        C = tile_embed.shape[0]

        if bias is not None:
            tile_logits = matmul_bias(hidden, tile_embed, bias[i: i + tile_size])
        else:
            tile_logits = matmul(hidden, tile_embed)

        tile_vals, tile_idx = reduce_seq_max_log1p_relu(tile_logits, mask)
        sparse_reps[:, i:i+C] = tile_vals
        max_indices[:, i:i+C] = tile_idx

    return sparse_reps, max_indices


@torch.library.custom_op(
    "sparton::hybrid_fwd",
    mutates_args=(),
    schema="(Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)",
)
def hybrid_fwd_op(
    hidden: torch.Tensor,   # [B,S,D], cuda
    embed: torch.Tensor,    # [V,D],   cuda
    bias: Optional[torch.Tensor],     # [V] or None
    mask: torch.Tensor,     # [B,S],   cuda
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hidden.is_cuda, "sparton::hybrid_fwd only supports CUDA"
    return _launch_hybrid_fwd(hidden, embed, bias, mask)


register_forward(hybrid_fwd_op, mono_bwd_op)


def hybrid_forward(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Canonicalize before the op so autograd saves contiguous tensors; the
    # backward kernel computes flat offsets that assume dense [B, S, D] strides.
    hidden, embed, bias, mask = prepare_forward_inputs(
        hidden, embed, bias, mask, kernel="hybrid"
    )
    return hybrid_fwd_op(hidden, embed, bias, mask)
