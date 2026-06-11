import logging

import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

from ._validation import validate_forward_inputs

logger = logging.getLogger("sparton")

DEVICE = triton.runtime.driver.active.get_active_torch_device()
logger.debug("Sparton using device: %s", DEVICE)


def get_fast_forward_configs():
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

def get_slow_forward_configs():
    configs = []
    # H100 supports large shared memory, so we go up to 1024 on C and 512 on S
    block_c_sizes = [32, 64, 128, 256, 512]
    block_s_sizes = [32, 64, 128, 256, 512]
    num_warps_list = [4, 8, 16]
    num_stages_list = [2, 3, 4, 5]

    for block_c in block_c_sizes:
        for block_s in block_s_sizes:
            for num_warps in num_warps_list:
                for num_stages in num_stages_list:

                    num_elements = block_c * block_s
                    if num_warps == 16 and num_elements < 2048:
                        continue
                    if num_stages > 2 and num_elements < 1024:
                        continue
                    if num_elements > 16384 and num_warps < 8:
                        continue
                    if block_c == 1024 and num_warps < 8:
                        continue

                    configs.append(triton.Config(
                        {'BLOCK_C': block_c, 'BLOCK_S': block_s},
                        num_warps=num_warps,
                        num_stages=num_stages
                    ))
    return configs

@triton.autotune(
   configs=get_fast_forward_configs(),
   key=['S', 'C'],
   cache_results=True
)
@triton.jit
def reduce_seq_max_log1p_relu_kernel_with_indices(
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

def reduce_seq_max_log1p_relu_with_indices(logits: torch.Tensor,
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

    reduce_seq_max_log1p_relu_kernel_with_indices[grid](
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

def fused_sparton_fwd_with_indices(hidden, embed, bias, mask):
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

        tile_vals, tile_idx = reduce_seq_max_log1p_relu_with_indices(tile_logits, mask)
        sparse_reps[:, i:i+C] = tile_vals
        max_indices[:, i:i+C] = tile_idx

    return sparse_reps, max_indices



@triton.autotune(
   configs=get_fast_forward_configs(),
   key=['S', 'C'],
)
@triton.jit
def reduce_seq_max_log1p_relu_kernel(
    logits_ptr,        # float16/32, [B, S, C]
    mask_ptr,          # int32/bool, [B, S]
    out_vals_ptr,      # float16/32   [B, C]

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

    NEG_INF = 0
    max_vals = tl.full((BLOCK_C,), NEG_INF, dtype=tl.float32)

    s_range = tl.arange(0, BLOCK_S)
    for s_start in range(0, S, BLOCK_S):
        offs_s = s_start + s_range
        mask_s = offs_s < S

        offs_s_b = offs_s[:, None]     # [BLOCK_S, 1]
        offs_c_b = offs_c[None, :]     # [1, BLOCK_C]

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
        better = tile_max_vals > max_vals
        max_vals = tl.where(better, tile_max_vals, max_vals)

    # ReLU + log1p on max values
    zero = 0.0
    relu_vals = tl.where(max_vals > zero, max_vals, zero)
    log_vals  = tl.log(1.0 + relu_vals)

    # store results [B, C]
    out_vals_ptrs = out_vals_ptr + pid_b * stride_ob + offs_c * stride_oc
    tl.store(out_vals_ptrs, log_vals, mask=mask_c)


def reduce_seq_max_log1p_relu(logits: torch.Tensor,
                                     mask: torch.Tensor):
    """
    logits: [B, S, C], float16 or float32, contiguous
    mask:   [B, S],    int32 / bool, contiguous
    returns:
        vals: [B, C] (float16)      = log1p(relu(max_s logits * mask))
        idxs: [B, C] (int64)        = argmax_s logits * mask
    """
    assert logits.ndim == 3
    assert mask.ndim == 2
    B, S, C = logits.shape
    assert mask.shape == (B, S)

    logits = logits.contiguous()
    mask   = mask.contiguous()

    vals = torch.empty((B, C), device=logits.device, dtype=logits.dtype)

    def grid(meta):
        return (
            B,  # one program per batch row
            triton.cdiv(C, meta['BLOCK_C']),
        )

    reduce_seq_max_log1p_relu_kernel[grid](
        logits, mask,
        vals,
        B, S, C,
        logits.stride(0), logits.stride(1), logits.stride(2),
        mask.stride(0),   mask.stride(1),
        vals.stride(0),   vals.stride(1),
    )
    return vals

def fused_sparton_fwd(hidden, embed, bias, mask):
    """
    hidden: [B, S, D], float16
    embed:  [V, D],    float16
    mask:   [B, S],    int32 / bool
    """
    B, S, D = hidden.shape
    V, _ = embed.shape

    sparse_reps_tiles = []
    tile_size  = v_tile_from_bs(B, S, V) #V_CHUNK_SIZE
    for i in range(0, V, tile_size):
        tile_embed = embed[i:i + tile_size]       # [C, D]
        C = tile_embed.shape[0]
        if bias is not None:
            tile_logits = matmul_bias(hidden, tile_embed, bias[i: i + tile_size])
        else:
            tile_logits = matmul(hidden, tile_embed)

        tile_vals = reduce_seq_max_log1p_relu(tile_logits, mask)
        sparse_reps_tiles.append(tile_vals)       # [B, C]
    sparse_reps = torch.cat(sparse_reps_tiles, dim=1)  # [B, V]
    return sparse_reps

def get_fast_bwd_configs():
    return [
        triton.Config({'BLOCK_B': 16, 'BLOCK_V': 16, 'BLOCK_D': 32, 'GROUP_SIZE': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_B': 32, 'BLOCK_V': 16, 'BLOCK_D': 32, 'GROUP_SIZE': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_B': 32, 'BLOCK_V': 32, 'BLOCK_D': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_B': 64, 'BLOCK_V': 32, 'BLOCK_D': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_B': 32, 'BLOCK_V': 64, 'BLOCK_D': 64, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_B': 16, 'BLOCK_V': 32, 'BLOCK_D': 128, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_B': 32, 'BLOCK_V': 32, 'BLOCK_D': 128, 'GROUP_SIZE': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_B': 32, 'BLOCK_V': 32, 'BLOCK_D': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=16),
        triton.Config({'BLOCK_B': 16, 'BLOCK_V': 64, 'BLOCK_D': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=16),
        triton.Config({'BLOCK_B': 128, 'BLOCK_V': 16, 'BLOCK_D': 16, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    ]


def get_slow_bwd_configs():
    configs = []

    # Range definitions
    block_b_list = [16, 32, 64, 128, 256]
    block_v_list = [16, 32, 64, 128, 256]
    block_d_list = [16, 32, 64, 128, 256]
    num_warps_list = [4, 8, 16]
    num_stages_list = [2, 3, 4]

    group_size_list = [8]

    for b in block_b_list:
        for v in block_v_list:
            for d in block_d_list:
                for w in num_warps_list:
                    for s in num_stages_list:
                        for g in group_size_list:

                            total_threads = w * 32
                            total_elems = b * v * d # Approximate tile volume
                            if (total_elems) >= 1048576:
                                continue

                            if w == 16 and (b * v) < 256: continue

                            if d == 256 and w < 8: continue

                            if (b * v * d) > 4096 and w == 4: continue

                            if v < 16: continue

                            configs.append(triton.Config(
                                {'BLOCK_B': b, 'BLOCK_V': v, 'BLOCK_D': d, 'GROUP_SIZE': g},
                                num_warps=w,
                                num_stages=s
                            ))
    return configs

@triton.autotune(
    configs=get_fast_bwd_configs(),
    key=['batch_size', 'vocab_size', 'hidden_dim'],
    reset_to_zero=['hidden_grad_ptr', 'embed_grad_ptr', 'bias_grad_ptr']
)
@triton.jit
def fused_sparton_bwd_kernel_with_bias(
    grad_out_ptr, # [B, V], float32)
    max_scores_ptr, # [B, V], (float32)
    max_idx_ptr, # [B, V], (int64)
    hidden_ptr, # [B*S, D] (float32)
    embed_ptr, # [V, D] (float32)
    hidden_grad_ptr, # [B*S, D] (float32)
    embed_grad_ptr, # [V, D] (float32)
    bias_grad_ptr, # [V] (float32)
    batch_size, # B,
    seq_len, # S,
    hidden_dim: tl.constexpr, # D,
    vocab_size: tl.constexpr, # V,
    HAS_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr, # batch-block
    BLOCK_V: tl.constexpr, # seq-block
    BLOCK_D: tl.constexpr, # hidden-dim-block
    GROUP_SIZE: tl.constexpr, # group size
):
    v_block_id = tl.program_id(0)
    b_block_id = tl.program_id(1)

    num_v_blocks = tl.num_programs(0)
    num_b_blocks = tl.num_programs(1)
    # this is probabaly not needed
    b_block_id, v_block_id = tl.swizzle2d(b_block_id, v_block_id, num_b_blocks, num_v_blocks, GROUP_SIZE)

    start_b = b_block_id * BLOCK_B
    start_v = v_block_id * BLOCK_V
    offs_v = start_v + tl.arange(0, BLOCK_V).to(tl.int64)
    offs_b = start_b + tl.arange(0, BLOCK_B).to(tl.int64)

    # might not be needed
    offs_v = tl.max_contiguous(tl.multiple_of(offs_v, BLOCK_V), BLOCK_V)
    offs_b = tl.max_contiguous(tl.multiple_of(offs_b, BLOCK_B), BLOCK_B)

    offs_d = tl.arange(0, BLOCK_D)

    b_mask = (offs_b < batch_size)
    bv_mask = (offs_b[:, None] < batch_size) & (offs_v[None, :] < vocab_size)
    bv_offs = offs_b[:, None] * vocab_size + offs_v[None, :]

    # load max logits (block)
    block_max_logits = tl.load(max_scores_ptr + bv_offs, mask = bv_mask, other=0.0).to(tl.float32) # BLOCK_B x BLOCK_V

    if tl.sum(block_max_logits) == 0:
        return

    # load gradient with regard to the max logits (block)
    grad_out = tl.load(grad_out_ptr + bv_offs, mask = bv_mask, other=0.0).to(tl.float32) # BLOCK_B x BLOCK_V

    # load max indices (block)
    block_max_idx = tl.load(max_idx_ptr + bv_offs, mask = bv_mask, other=0) # BLOCK_B x BLOCK_V
    # if b_block_id ==0 and v_block_id == 0:
    #     tl.device_print("block_max_idx")

    # calculate the gradient of log(1 + relu(x)) with regard to x
    # log(x) = 1/x = 1/ exp(log(x))
    # relu_log1p_grad = (block_max_logits > 0).to(tl.float32) * grad_out * tl.exp(-block_max_logits)
    relu_log1p_grad = tl.where(block_max_logits > 0, grad_out * tl.exp(-block_max_logits), 0.0).to(tl.float32)


    mask_v = offs_v < vocab_size
    if HAS_BIAS:
        tl.atomic_add(bias_grad_ptr + offs_v, tl.sum(relu_log1p_grad, axis=0), mask = mask_v, sem = "relaxed")

    # address of the vocab emb block
    e_ptrs = embed_ptr + offs_v[:, None] * hidden_dim + offs_d[None, :]

    # address of the hidden states with selected max logit values. note: this one is non-continuous, might lead to slow data loading
    h_ptrs = hidden_ptr + offs_b[:, None, None] * seq_len * hidden_dim + block_max_idx[:, :, None] * hidden_dim + offs_d[None, None, :]

    # address of the output embedding gradient
    e_grad_ptrs = embed_grad_ptr + offs_v[:, None] * hidden_dim + offs_d[None, :]
    # address of the output hidden state gradient
    h_grad_ptrs = hidden_grad_ptr + offs_b[:, None, None] * seq_len * hidden_dim + block_max_idx[:, :, None] * hidden_dim + offs_d[None, None, :]

    valid_logit_mask = block_max_logits > 0

    for start_d in range(0, hidden_dim, BLOCK_D):
        mask_d =  start_d +  offs_d < hidden_dim
        #  b x v x hidden_size: non-coalesed reading
        hidden_state = tl.load(h_ptrs, mask= b_mask[:, None, None] & mask_d[None, None, :] & valid_logit_mask[:, :, None], other=0.0).to(tl.float32) # BLOCK_B x BLOCK_V x BLOCK_D
        embed_grad_update =  tl.sum(hidden_state * relu_log1p_grad[:, :, None], axis=0) # V x E
        # this seems to be a non-coalesed writing? output at scattered addresses
        tl.atomic_add(e_grad_ptrs, embed_grad_update, mask =  mask_v[:, None] & mask_d[None, :], sem = "relaxed")
        # loading v embedidngs, to compate gradient for hidden states
        embed = tl.load(e_ptrs, mask= mask_v[:, None] & mask_d[None, :], other=0.0, eviction_policy="evict_last").to(tl.float32)# BLOCK_V x BLOCK_D
        hidden_grad_update = (embed[None, :, :] * relu_log1p_grad[:, :, None]).to(tl.float32) # BLOCK_B x BLOCK_V x BLOCK_D
        tl.atomic_add(h_grad_ptrs, hidden_grad_update, mask  = bv_mask[:, :, None] & mask_d[None, None, :] & valid_logit_mask[:,:, None], sem = "relaxed")
        e_ptrs += BLOCK_D
        h_ptrs += BLOCK_D
        e_grad_ptrs += BLOCK_D
        h_grad_ptrs += BLOCK_D

def fused_sparton_bwd_with_bias(
    grad_output,
    max_scores,
    max_idx,
    hidden,
    embed,
    hidden_grad,
    embed_grad,
    bias_grad,
    has_bias: bool):
    B, S, D = hidden.shape
    V, D_e = embed.shape
    assert D == D_e
    assert max_scores.shape == (B, V)
    assert max_idx.shape == (B, V)
    grid = lambda meta: (triton.cdiv(V, meta['BLOCK_V']), triton.cdiv(B, meta['BLOCK_B']))
    fused_sparton_bwd_kernel_with_bias[grid](
        grad_out_ptr=grad_output,
        max_scores_ptr=max_scores,
        max_idx_ptr=max_idx,
        hidden_ptr=hidden,
        embed_ptr=embed,
        hidden_grad_ptr=hidden_grad,
        embed_grad_ptr=embed_grad,
        bias_grad_ptr=bias_grad,
        batch_size=B,
        seq_len=S,
        hidden_dim=D,
        vocab_size=V,
        HAS_BIAS=has_bias,
    )
    return hidden_grad, embed_grad, bias_grad


# register new op
@torch.library.custom_op(
    "sparton::fused_sparton_fwd",
    mutates_args=(),
    schema="(Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)",
)
def fused_sparton_fwd_op(
    hidden: torch.Tensor,   # [B,S,D], cuda
    embed: torch.Tensor,    # [V,D],   cuda
    bias: Optional[torch.Tensor],     # [V] or None
    mask: torch.Tensor,     # [B,S],   cuda
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hidden.is_cuda, "sparton::fused_sparton_fwd only supports CUDA"
    scores, idx = fused_sparton_fwd_with_indices(hidden, embed, bias, mask)
    return scores, idx

@fused_sparton_fwd_op.register_fake
def _(hidden, embed, bias, mask):
    # Must return tensors with correct metadata (shape/dtype/device) without doing real work.
    B, S, D = hidden.shape
    V, D2 = embed.shape
    out_scores = hidden.new_empty((B, V))                     # same dtype/device as hidden
    out_idx = torch.empty((B, V), device=hidden.device, dtype=torch.int64)
    return out_scores, out_idx

@torch.library.custom_op(
    "sparton::fused_sparton_bwd",
    mutates_args=(),
    schema=(
        "(Tensor grad_out, Tensor max_scores, Tensor max_idx, Tensor hidden, "
        "Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor, Tensor?)"
    ),
)
def fused_sparton_bwd_op(
    grad_out: torch.Tensor,     # [B,V]
    max_scores: torch.Tensor,   # [B,V]
    max_idx: torch.Tensor,      # [B,V] int64
    hidden: torch.Tensor,       # [B,S,D]
    embed: torch.Tensor,        # [V,D]
    bias: Optional[torch.Tensor],         # [V] or None
    mask: torch.Tensor,         # [B,S] (not used by your bwd kernel, but included for signature symmetry)
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert grad_out.is_cuda, "sparton::fused_sparton_bwd only supports CUDA"
    grad_out = grad_out.contiguous()

    hidden_grad = torch.zeros_like(hidden, dtype=torch.float32)
    embed_grad  = torch.zeros_like(embed,  dtype=torch.float32)
    bias_grad = (
        torch.zeros_like(bias, dtype=torch.float32)
        if bias is not None
        else torch.empty((), device=grad_out.device, dtype=torch.float32)
    )

    hidden_grad, embed_grad, bias_grad = fused_sparton_bwd_with_bias(
        grad_out, max_scores, max_idx,
        hidden, embed,
        hidden_grad, embed_grad, bias_grad,
        bias is not None,
    )
    return hidden_grad, embed_grad, bias_grad if bias is not None else None

@fused_sparton_bwd_op.register_fake
def _(grad_out, max_scores, max_idx, hidden, embed, bias, mask):
    # Return correct metadata only
    return (
        torch.empty_like(hidden, dtype=torch.float32),
        torch.empty_like(embed,  dtype=torch.float32),
        torch.empty_like(bias, dtype=torch.float32) if bias is not None else None,
    )

def _setup_context(ctx, inputs, output):
    hidden, embed, bias, mask = inputs
    scores, idx = output
    ctx.save_for_backward(scores, idx, hidden, embed, bias, mask)

def _backward(ctx, grad_scores, grad_idx):
    # grad_idx is ignored (idx is non-differentiable)
    scores, idx, hidden, embed, bias, mask = ctx.saved_tensors
    hidden_g, embed_g, bias_g = fused_sparton_bwd_op(grad_scores, scores, idx, hidden, embed, bias, mask)
    return hidden_g, embed_g, bias_g, None

fused_sparton_fwd_op.register_autograd(_backward, setup_context=_setup_context)


def hybrid_forward(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    validate_forward_inputs(hidden, embed, bias, mask, backend="hybrid")
    # Canonicalize before the op so autograd saves contiguous tensors; the
    # backward kernel computes flat offsets that assume dense [B, S, D] strides.
    hidden = hidden.contiguous()
    embed = embed.contiguous()
    mask = mask.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    return fused_sparton_fwd_op(hidden, embed, bias, mask)
