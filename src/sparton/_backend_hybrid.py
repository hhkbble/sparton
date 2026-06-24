import logging

import triton
import triton.language as tl
import torch
from typing import Optional, Tuple

from ._validation import autocast_canonicalize, validate_forward_inputs

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

# --- M13 split segmented backward (production path) -------------------------
#
# Mechanism and measured evidence: docs/DEVELOPMENT.md (M13 backward)
# (DEVELOPMENT.md M11 carries the segmented design this splits). Stages inside the
# unchanged custom op:
#   1. bwd_prep_kernel: g = grad_out * exp(-scores) where scores > 0 (the
#      original kernel's exact fp32 math), int32 idx, destination sort keys
#      b*S + idx (sentinel B*S for inactive entries, which sort last), and a
#      device-side active-entry count;
#   2. torch.sort by destination + one payload-gather kernel;
#   3. embed_grad_kernel (exclusive-owner plain stores, no atomics) and the
#      split hidden-grad pass: uniform_hidden_grad_kernel (branch-free
#      pipelined streaming reduction over single-destination chunks — the
#      vectorized-gather fast path, DEVELOPMENT.md M13 §5.4) plus
#      mixed_hidden_grad_kernel (the segmented scan, covering only chunks
#      with a run boundary). The two kernels' predicates are exact
#      complements at one shared CHUNK granularity; independent granularities
#      silently drop contributions (DEVELOPMENT.md M13 §5.4 item 2), which is why the
#      mixed kernel is not autotuned and runs at the uniform winner's CHUNK.


@triton.jit
def bwd_prep_kernel(
    scores_ptr,   # [B*V] input dtype
    grad_ptr,     # [B*V] fp32
    idx_ptr,      # [B*V] int64
    g_ptr,        # [B*V] fp32 out
    idx32_ptr,    # [B*V] int32 out
    keys_ptr,     # [B*V] int32 out
    n_active_ptr, # [] int32 out (zero-initialized), total active entries
    total,
    seq_len,
    vocab_size,
    num_rows,
    BLOCK: tl.constexpr,
):
    offs = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    in_range = offs < total
    scores = tl.load(scores_ptr + offs, mask=in_range, other=0.0).to(tl.float32)
    grad = tl.load(grad_ptr + offs, mask=in_range, other=0.0).to(tl.float32)
    idx = tl.load(idx_ptr + offs, mask=in_range, other=0)
    valid = scores > 0
    g = tl.where(valid, grad * tl.exp(-scores), 0.0)
    b = (offs // vocab_size).to(tl.int32)
    keys = tl.where(valid, b * seq_len + idx.to(tl.int32), num_rows)
    tl.store(g_ptr + offs, g, mask=in_range)
    tl.store(idx32_ptr + offs, idx.to(tl.int32), mask=in_range)
    tl.store(keys_ptr + offs, keys, mask=in_range)
    # Sorted keys put all active entries first, so this count bounds the
    # uniform pass's live-chunk walk on the device — no host sync.
    block_active = tl.sum((valid & in_range).to(tl.int32))
    tl.atomic_add(n_active_ptr, block_active, sem="relaxed")


@triton.jit
def bwd_gather_payload_kernel(
    perm_ptr,      # [B*V] int64 sort permutation
    g_full_ptr,    # [B*V] fp32
    g_sorted_ptr,  # [B*V] fp32 out
    v_sorted_ptr,  # [B*V] int32 out
    total,
    vocab_size,
    BLOCK: tl.constexpr,
):
    offs = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    in_range = offs < total
    perm = tl.load(perm_ptr + offs, mask=in_range, other=0)
    g = tl.load(g_full_ptr + perm, mask=in_range, other=0.0)
    tl.store(g_sorted_ptr + offs, g, mask=in_range)
    tl.store(v_sorted_ptr + offs, (perm % vocab_size).to(tl.int32), mask=in_range)


def get_embed_grad_bwd_configs():
    # Small BLOCK_B is admissible because exclusive embed-grad ownership no
    # longer multiplies atomic traffic (there are no atomics at all);
    # BLOCK_B * BLOCK_V * BLOCK_D <= 64K bounds the gathered register tile.
    return [
        triton.Config({'BLOCK_B': 8, 'BLOCK_V': 16, 'BLOCK_D': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_B': 8, 'BLOCK_V': 16, 'BLOCK_D': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_B': 8, 'BLOCK_V': 32, 'BLOCK_D': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_B': 8, 'BLOCK_V': 32, 'BLOCK_D': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_B': 8, 'BLOCK_V': 64, 'BLOCK_D': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_B': 16, 'BLOCK_V': 16, 'BLOCK_D': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_B': 16, 'BLOCK_V': 16, 'BLOCK_D': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_B': 16, 'BLOCK_V': 32, 'BLOCK_D': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_B': 16, 'BLOCK_V': 64, 'BLOCK_D': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_B': 32, 'BLOCK_V': 16, 'BLOCK_D': 32}, num_stages=2, num_warps=4),
    ]


# Exclusive-owner embed/bias gradient kernel: each CTA owns one
# embed_grad[v-tile, d-tile] patch and loops over the batch, so the stores
# need no atomics and no zero-initialized output (every element is written
# exactly once by construction — the d-axis is program_id(0) so consecutive
# CTAs share one v-tile's g/idx stream in L2). g != 0 is a correct skip
# condition regardless of why g is zero: a zero g contributes nothing.
@triton.autotune(
    configs=get_embed_grad_bwd_configs(),
    key=['batch_size', 'vocab_size', 'hidden_dim'],
)
@triton.jit
def embed_grad_kernel(
    g_ptr,            # [B, V] fp32, precomputed grad_out * exp(-scores) where active
    idx_ptr,          # [B, V] int32
    hidden_ptr,       # [B*S, D] input dtype
    embed_grad_ptr,   # [V, D] fp32, exclusive-owner plain store
    bias_grad_ptr,    # [V] fp32, exclusive-owner plain store (d-block 0)
    batch_size,
    seq_len,
    hidden_dim: tl.constexpr,
    vocab_size: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_v = tl.program_id(1)

    offs_v = (pid_v * BLOCK_V + tl.arange(0, BLOCK_V)).to(tl.int64)
    offs_d = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D)).to(tl.int64)
    mask_v = offs_v < vocab_size
    mask_d = offs_d < hidden_dim
    vd_mask = mask_v[:, None] & mask_d[None, :]

    acc_e = tl.zeros((BLOCK_V, BLOCK_D), dtype=tl.float32)
    acc_bias = tl.zeros((BLOCK_V,), dtype=tl.float32)

    for start_b in range(0, batch_size, BLOCK_B):
        offs_b = (start_b + tl.arange(0, BLOCK_B)).to(tl.int64)
        b_mask = offs_b < batch_size
        bv_mask = b_mask[:, None] & mask_v[None, :]
        bv_offs = offs_b[:, None] * vocab_size + offs_v[None, :]

        g = tl.load(g_ptr + bv_offs, mask=bv_mask, other=0.0)
        if tl.sum(tl.abs(g)) != 0:
            idx = tl.load(idx_ptr + bv_offs, mask=bv_mask, other=0).to(tl.int64)
            h_offs = (
                offs_b[:, None, None] * (seq_len * hidden_dim)
                + idx[:, :, None] * hidden_dim
                + offs_d[None, None, :]
            )
            gather_mask = (g != 0)[:, :, None] & mask_d[None, None, :]
            hidden_tile = tl.load(hidden_ptr + h_offs, mask=gather_mask, other=0.0).to(tl.float32)
            acc_e += tl.sum(hidden_tile * g[:, :, None], axis=0)
            if HAS_BIAS:
                if pid_d == 0:
                    acc_bias += tl.sum(g, axis=0)

    vd_offs = offs_v[:, None] * hidden_dim + offs_d[None, :]
    tl.store(embed_grad_ptr + vd_offs, acc_e, mask=vd_mask)
    if HAS_BIAS:
        if pid_d == 0:
            tl.store(bias_grad_ptr + offs_v, acc_bias, mask=mask_v)


def get_uniform_hidden_grad_configs():
    # Branch-free streaming kernel. CHUNK is pinned to 64 across the family:
    # the mixed fraction scales with the shared granule (m ~ runs*CHUNK/N),
    # so a larger uniform-side CHUNK silently multiplies the mixed pass's
    # coverage and forces its scan tile register-heavy (measured at
    # GRANULE=256 as a 1.30 ms mixed pass doing ~3% of the work — DEVELOPMENT.md
    # M13 §5.4 item 4). At CHUNK=64 the extra chunk-partial atomics are ~2% of
    # kernel bytes on the doc shape — the cheaper side of the trade by an
    # order of magnitude.
    return [
        triton.Config({'CHUNK': 64, 'BLOCK_D': 64}, num_stages=3, num_warps=4),
        triton.Config({'CHUNK': 64, 'BLOCK_D': 64}, num_stages=2, num_warps=8),
        triton.Config({'CHUNK': 64, 'BLOCK_D': 128}, num_stages=3, num_warps=8),
        triton.Config({'CHUNK': 64, 'BLOCK_D': 128}, num_stages=2, num_warps=4),
        triton.Config({'CHUNK': 64, 'BLOCK_D': 256}, num_stages=2, num_warps=8),
    ]


# Uniform-chunk pass of the split: a branch-free bounded-persistent
# for-loop (compiler-pipelinable, unlike a while-walk) that emits one
# chunk-partial atomic per single-destination chunk and suppresses the
# atomic otherwise. Mixed chunks (and the live/sentinel seam) are completed
# by mixed_hidden_grad_kernel; the two predicates are exact complements at
# the shared CHUNK granularity so each contribution is deposited exactly
# once. Suppression is folded into the load MASKS, not a branch: a branch
# around the tile load re-anchors its layout and de-vectorizes the gather
# (DEVELOPMENT.md M13 §5.4 items 1 and 6).
@triton.autotune(
    configs=get_uniform_hidden_grad_configs(),
    key=['batch_size', 'seq_len', 'vocab_size', 'hidden_dim'],
    reset_to_zero=['hidden_grad_ptr'],
)
@triton.jit
def uniform_hidden_grad_kernel(
    keys_ptr,         # [B*V] int32, sorted destination keys (sentinel B*S last)
    g_ptr,            # [B*V] fp32, payload g sorted to match keys
    v_ptr,            # [B*V] int32, source vocab row sorted to match keys
    embed_ptr,        # [V, D] input dtype
    hidden_grad_ptr,  # [B*S, D] fp32
    n_active_ptr,     # [] int32, active-entry count from bwd_prep_kernel
    total,            # B*V
    batch_size,
    seq_len,
    vocab_size,
    hidden_dim: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_chunk = tl.program_id(0)
    pid_d = tl.program_id(1)
    num_ctas = tl.num_programs(0)
    num_rows = batch_size * seq_len
    n_live = tl.load(n_active_ptr)
    n_chunks = tl.cdiv(n_live, CHUNK)

    offs_d = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D)).to(tl.int64)
    mask_d = offs_d < hidden_dim

    for chunk in tl.range(pid_chunk, n_chunks, num_ctas):
        offs_i = chunk * CHUNK + tl.arange(0, CHUNK)
        in_range = offs_i < total
        keys = tl.load(keys_ptr + offs_i, mask=in_range, other=num_rows)
        keys_min = tl.min(keys)
        keys_max = tl.max(keys)
        is_uniform = (keys_min == keys_max) & (keys_min < num_rows)
        row_on = in_range & is_uniform
        g = tl.load(g_ptr + offs_i, mask=row_on, other=0.0)
        v = tl.load(v_ptr + offs_i, mask=row_on, other=0).to(tl.int64)
        a_ptrs = embed_ptr + v[:, None] * hidden_dim + offs_d[None, :]
        if hidden_dim % BLOCK_D == 0:
            val = tl.load(a_ptrs, mask=row_on[:, None], other=0.0).to(tl.float32)
        else:
            val = tl.load(a_ptrs, mask=row_on[:, None] & mask_d[None, :],
                          other=0.0).to(tl.float32)
        val = val * g[:, None]
        partial = tl.sum(val, axis=0)
        tl.atomic_add(
            hidden_grad_ptr + keys_min.to(tl.int64) * hidden_dim + offs_d,
            partial, mask=is_uniform & mask_d, sem="relaxed",
        )


# Mixed-chunk pass of the split. NOT autotuned: the granule size must equal
# the uniform kernel's selected CHUNK or the complement breaks (DEVELOPMENT.md
# M13 §5.4 item 2). A mixed granule is processed as SUB-row segmented-scan
# tiles with an inner d-loop: chunk-local partials compose across tile
# boundaries (the run-boundary composition invariant), so the working tile
# need not match the granule — but SUB must divide GRANULE or the
# static_range truncates and drops lanes (host-asserted at launch). 1D
# grid: the d-dimension is the inner loop, so the granule walk's keys
# traffic is paid once, not ceil(D/BLOCK_D) times.
@triton.jit
def mixed_hidden_grad_kernel(
    keys_ptr,         # [B*V] int32, sorted destination keys (sentinel B*S last)
    g_ptr,            # [B*V] fp32, payload g sorted to match keys
    v_ptr,            # [B*V] int32, source vocab row sorted to match keys
    embed_ptr,        # [V, D] input dtype
    hidden_grad_ptr,  # [B*S, D] fp32
    total,            # B*V
    batch_size,
    seq_len,
    vocab_size,
    hidden_dim: tl.constexpr,
    GRANULE: tl.constexpr,    # the uniform kernel's selected CHUNK
    SUB: tl.constexpr,        # scan working-tile rows; divides GRANULE
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_rows = batch_size * seq_len

    granule = pid
    keep_going = tl.load(keys_ptr + granule * GRANULE) < num_rows
    while keep_going:
        offs_g = granule * GRANULE + tl.arange(0, GRANULE)
        gkeys = tl.load(keys_ptr + offs_g, mask=offs_g < total, other=num_rows)
        gmin = tl.min(gkeys)
        gmax = tl.max(gkeys)
        if gmin != gmax:
            for sub in tl.static_range(GRANULE // SUB):
                offs_i = granule * GRANULE + sub * SUB + tl.arange(0, SUB)
                in_range = offs_i < total
                keys = tl.load(keys_ptr + offs_i, mask=in_range, other=num_rows)
                valid = in_range & (keys < num_rows)
                g = tl.load(g_ptr + offs_i, mask=valid, other=0.0)
                v = tl.load(v_ptr + offs_i, mask=valid, other=0).to(tl.int64)
                prev_keys = tl.load(keys_ptr + offs_i - 1,
                                    mask=in_range & (offs_i > 0), other=-2)
                next_keys = tl.load(keys_ptr + offs_i + 1,
                                    mask=in_range & (offs_i + 1 < total), other=-3)
                lane = tl.arange(0, SUB)
                is_end = valid & ((keys != next_keys) | (lane == SUB - 1))
                is_start = valid & (keys != prev_keys)
                for d0 in tl.static_range(0, hidden_dim, BLOCK_D):
                    offs_d = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    mask_d = offs_d < hidden_dim
                    val = (
                        tl.load(
                            embed_ptr + v[:, None] * hidden_dim + offs_d[None, :],
                            mask=valid[:, None] & mask_d[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        * g[:, None]
                    )
                    csum = tl.cumsum(val, axis=0)
                    dest = keys.to(tl.int64)[:, None] * hidden_dim + offs_d[None, :]
                    end_value = tl.where(is_start[:, None], val, csum)
                    tl.atomic_add(hidden_grad_ptr + dest, end_value,
                                  mask=is_end[:, None] & mask_d[None, :],
                                  sem="relaxed")
                    tl.atomic_add(hidden_grad_ptr + dest, val - csum,
                                  mask=(is_start & ~is_end)[:, None] & mask_d[None, :],
                                  sem="relaxed")

        granule += num_ctas
        if granule * GRANULE >= total:
            keep_going = False
        else:
            keep_going = tl.load(keys_ptr + granule * GRANULE) < num_rows


def get_segmented_hidden_grad_configs():
    # CHUNK x BLOCK_D is the in-register cumsum tile (val + csum live
    # simultaneously); bounded like the embed-grad configs.
    return [
        triton.Config({'CHUNK': 32, 'BLOCK_D': 128}, num_stages=2, num_warps=4),
        triton.Config({'CHUNK': 32, 'BLOCK_D': 256}, num_stages=2, num_warps=8),
        triton.Config({'CHUNK': 64, 'BLOCK_D': 64}, num_stages=2, num_warps=4),
        triton.Config({'CHUNK': 64, 'BLOCK_D': 128}, num_stages=3, num_warps=8),
        triton.Config({'CHUNK': 64, 'BLOCK_D': 256}, num_stages=2, num_warps=8),
        triton.Config({'CHUNK': 128, 'BLOCK_D': 64}, num_stages=2, num_warps=8),
        triton.Config({'CHUNK': 128, 'BLOCK_D': 128}, num_stages=2, num_warps=8),
        triton.Config({'CHUNK': 256, 'BLOCK_D': 64}, num_stages=2, num_warps=8),
    ]


# A/B reference of record for the M13 backward swap (the M11 unified
# segmented kernel the split pass replaced). Exercised by
# test_backward_matches_legacy_kernel and `bench_backward.py --impls
# legacy`; remove when a later milestone supersedes the M13 comparison
# evidence. Reachable only through legacy_fused_sparton_bwd — production
# autograd never calls it.
#
# Sorted segmented-scan hidden-grad kernel. Contributions arrive sorted by
# destination row key b*S + idx, so a run of equal keys is consecutive and
# reduces to at most two partial-sum atomics: +cumsum at run ends (forced
# at chunk boundaries — chunk-local partials compose across chunks because
# a continued run emits its own partial with no start-correction) and
# val - cumsum at run starts; single-lane runs collapse to one exact val
# atomic. Worst case (all destinations distinct) equals one atomic per
# contribution. The persistent stride plus the scalar first-key check keeps
# sparse inputs (small nnz) at one wasted load per CTA with no host-side
# nnz sync.
@triton.autotune(
    configs=get_segmented_hidden_grad_configs(),
    # seq_len is in the key (unlike the deleted M2-era kernel's, which this
    # function's name once denoted): destination-run length scales with
    # V/S, so the optimal CHUNK for short-S inputs (long runs, fast path)
    # differs from long-S inputs (DEVELOPMENT.md M11 §5.2).
    key=['batch_size', 'seq_len', 'vocab_size', 'hidden_dim'],
    reset_to_zero=['hidden_grad_ptr'],
)
@triton.jit
def segmented_hidden_grad_kernel(
    keys_ptr,         # [B*V] int32, sorted destination keys (sentinel B*S last)
    g_ptr,            # [B*V] fp32, payload g sorted to match keys
    v_ptr,            # [B*V] int32, source vocab row sorted to match keys
    embed_ptr,        # [V, D] input dtype
    hidden_grad_ptr,  # [B*S, D] fp32
    total,            # B*V
    batch_size,
    seq_len,
    vocab_size,
    hidden_dim: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_chunk = tl.program_id(0)
    pid_d = tl.program_id(1)
    num_ctas = tl.num_programs(0)
    num_rows = batch_size * seq_len  # sentinel value and destination bound

    chunk = pid_chunk
    keep_going = tl.load(keys_ptr + chunk * CHUNK) < num_rows
    while keep_going:
        offs_i = chunk * CHUNK + tl.arange(0, CHUNK)
        in_range = offs_i < total
        offs_d = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D)).to(tl.int64)
        mask_d = offs_d < hidden_dim

        keys = tl.load(keys_ptr + offs_i, mask=in_range, other=num_rows)
        valid = in_range & (keys < num_rows)
        g = tl.load(g_ptr + offs_i, mask=valid, other=0.0)
        v = tl.load(v_ptr + offs_i, mask=valid, other=0).to(tl.int64)
        val = (
            tl.load(
                embed_ptr + v[:, None] * hidden_dim + offs_d[None, :],
                mask=valid[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            * g[:, None]
        )

        keys_min = tl.min(keys)
        keys_max = tl.max(keys)
        if keys_min == keys_max:
            # Whole chunk belongs to one destination run — the common case
            # on real index distributions (runs average V_active/S entries).
            partial = tl.sum(val, axis=0)
            tl.atomic_add(
                hidden_grad_ptr + keys_min.to(tl.int64) * hidden_dim + offs_d,
                partial, mask=mask_d, sem="relaxed",
            )
        else:
            prev_keys = tl.load(keys_ptr + offs_i - 1,
                                mask=in_range & (offs_i > 0), other=-2)
            next_keys = tl.load(keys_ptr + offs_i + 1,
                                mask=in_range & (offs_i + 1 < total), other=-3)
            csum = tl.cumsum(val, axis=0)
            lane = tl.arange(0, CHUNK)
            is_end = valid & ((keys != next_keys) | (lane == CHUNK - 1))
            is_start = valid & (keys != prev_keys)
            dest = keys.to(tl.int64)[:, None] * hidden_dim + offs_d[None, :]
            end_value = tl.where(is_start[:, None], val, csum)
            tl.atomic_add(hidden_grad_ptr + dest, end_value,
                          mask=is_end[:, None] & mask_d[None, :], sem="relaxed")
            tl.atomic_add(hidden_grad_ptr + dest, val - csum,
                          mask=(is_start & ~is_end)[:, None] & mask_d[None, :],
                          sem="relaxed")

        chunk += num_ctas
        if chunk * CHUNK >= total:
            keep_going = False
        else:
            keep_going = tl.load(keys_ptr + chunk * CHUNK) < num_rows


_BWD_PREP_BLOCK = 1024


def _bwd_shared_stages(
    grad_out: torch.Tensor,
    max_scores: torch.Tensor,
    max_idx: torch.Tensor,
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
):
    """Prep + embed/bias pass + sort + payload gather.

    Shared by the production split pass and the segmented reference path —
    the two backward designs differ only in the hidden-grad kernels.
    """

    B, S, D = hidden.shape
    V, D_e = embed.shape
    assert D == D_e
    assert max_scores.shape == (B, V)
    assert max_idx.shape == (B, V)
    # The chunk kernels index the flattened [B*V] payload streams (and the
    # embed kernel the S*D row stride) with int32-derived arithmetic; both
    # bounds are far beyond any allocatable problem, but fail loudly rather
    # than wrap silently.
    # Lane offsets reach total + CHUNK - 1 and the walk guards reach one
    # granule past total, so the bound carries a one-chunk epsilon; the
    # sentinel num_rows = B*S must itself fit int32 (the keys are int32).
    assert B * V < 2**31 - 256, "segmented backward: B*V exceeds int32 indexing"
    assert S * D < 2**31, "segmented backward: S*D exceeds int32 stride arithmetic"
    assert B * S < 2**31, "segmented backward: B*S sentinel exceeds int32"

    # hidden_grad accumulates atomically and keeps untouched rows at zero;
    # embed_grad/bias_grad are fully covered by the embed kernel's
    # unconditional exclusive-owner stores, so empty allocation is safe and
    # skips the largest fp32 fill (V x D).
    hidden_grad = torch.zeros_like(hidden, dtype=torch.float32)
    embed_grad = torch.empty_like(embed, dtype=torch.float32)
    bias_grad = (
        torch.empty_like(bias, dtype=torch.float32)
        if bias is not None
        else torch.empty((), device=grad_out.device, dtype=torch.float32)
    )

    total = B * V
    g_full = torch.empty((total,), device=grad_out.device, dtype=torch.float32)
    idx32 = torch.empty((total,), device=grad_out.device, dtype=torch.int32)
    keys_full = torch.empty((total,), device=grad_out.device, dtype=torch.int32)
    n_active = torch.zeros((), device=grad_out.device, dtype=torch.int32)
    bwd_prep_kernel[(triton.cdiv(total, _BWD_PREP_BLOCK),)](
        scores_ptr=max_scores,
        grad_ptr=grad_out,
        idx_ptr=max_idx,
        g_ptr=g_full,
        idx32_ptr=idx32,
        keys_ptr=keys_full,
        n_active_ptr=n_active,
        total=total,
        seq_len=S,
        vocab_size=V,
        num_rows=B * S,
        BLOCK=_BWD_PREP_BLOCK,
    )

    embed_grid = lambda meta: (
        triton.cdiv(D, meta['BLOCK_D']),
        triton.cdiv(V, meta['BLOCK_V']),
    )
    embed_grad_kernel[embed_grid](
        g_ptr=g_full,
        idx_ptr=idx32,
        hidden_ptr=hidden,
        embed_grad_ptr=embed_grad,
        bias_grad_ptr=bias_grad,
        batch_size=B,
        seq_len=S,
        hidden_dim=D,
        vocab_size=V,
        HAS_BIAS=bias is not None,
    )

    keys_sorted, perm = torch.sort(keys_full)
    g_sorted = torch.empty_like(g_full)
    v_sorted = torch.empty_like(idx32)
    bwd_gather_payload_kernel[(triton.cdiv(total, _BWD_PREP_BLOCK),)](
        perm_ptr=perm,
        g_full_ptr=g_full,
        g_sorted_ptr=g_sorted,
        v_sorted_ptr=v_sorted,
        total=total,
        vocab_size=V,
        BLOCK=_BWD_PREP_BLOCK,
    )
    return (hidden_grad, embed_grad, bias_grad,
            keys_sorted, g_sorted, v_sorted, n_active, total)


def split_segmented_sparton_bwd(
    grad_out: torch.Tensor,
    max_scores: torch.Tensor,
    max_idx: torch.Tensor,
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the M13 split segmented backward; returns fp32 gradient buffers."""

    B, S, D = hidden.shape
    V = embed.shape[0]
    (hidden_grad, embed_grad, bias_grad,
     keys_sorted, g_sorted, v_sorted, n_active, total) = _bwd_shared_stages(
        grad_out, max_scores, max_idx, hidden, embed, bias
    )

    # The uniform kernel autotunes; its sweep's trials pollute hidden_grad
    # but the autotuner's reset_to_zero hook zeroes the buffer after the
    # sweep, before the selected config's real launch — so launching the
    # (non-autotuned) mixed kernel strictly after keeps the accumulation
    # clean on first and cached calls alike.
    uniform_grid = lambda meta: (
        min(triton.cdiv(total, meta['CHUNK']), 4096),
        triton.cdiv(D, meta['BLOCK_D']),
    )
    uniform_hidden_grad_kernel[uniform_grid](
        keys_ptr=keys_sorted,
        g_ptr=g_sorted,
        v_ptr=v_sorted,
        embed_ptr=embed,
        hidden_grad_ptr=hidden_grad,
        n_active_ptr=n_active,
        total=total,
        batch_size=B,
        seq_len=S,
        vocab_size=V,
        hidden_dim=D,
    )

    # The mixed pass must run at the uniform kernel's selected granularity
    # (complement invariant — see the kernel comments); the selection is
    # read host-side from the autotuner, no sync. Its working tile and
    # launch shape are fixed (SUB <= 64 rows, BLOCK_D=128, 4 warps): the
    # pass covers only mixed granules, so it stays off the critical path
    # at the 168-register class instead of inheriting the uniform winner's
    # shape.
    granule = uniform_hidden_grad_kernel.best_config.kwargs['CHUNK']
    sub = min(granule, 64)
    # The complement invariant is config-family-conventional; make it
    # self-enforcing against future config edits (a non-multiple GRANULE
    # would truncate the mixed kernel's static_range and drop lanes).
    assert granule % sub == 0, "mixed pass: SUB must divide GRANULE"
    mixed_hidden_grad_kernel[(min(triton.cdiv(total, granule), 4096),)](
        keys_ptr=keys_sorted,
        g_ptr=g_sorted,
        v_ptr=v_sorted,
        embed_ptr=embed,
        hidden_grad_ptr=hidden_grad,
        total=total,
        batch_size=B,
        seq_len=S,
        vocab_size=V,
        hidden_dim=D,
        GRANULE=granule,
        SUB=sub,
        BLOCK_D=128,
        num_warps=4,
        num_stages=2,
    )
    return hidden_grad, embed_grad, bias_grad


def segmented_sparton_bwd(
    grad_out: torch.Tensor,
    max_scores: torch.Tensor,
    max_idx: torch.Tensor,
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the M11 segmented backward (A/B reference path since M13)."""

    B, S, D = hidden.shape
    V = embed.shape[0]
    (hidden_grad, embed_grad, bias_grad,
     keys_sorted, g_sorted, v_sorted, _n_active, total) = _bwd_shared_stages(
        grad_out, max_scores, max_idx, hidden, embed, bias
    )

    # Bounded persistent grid: enough CTAs to fill the device; each strides
    # through chunks and stops at the sentinel region (sorted keys).
    hidden_grid = lambda meta: (
        min(triton.cdiv(total, meta['CHUNK']), 4096),
        triton.cdiv(D, meta['BLOCK_D']),
    )
    segmented_hidden_grad_kernel[hidden_grid](
        keys_ptr=keys_sorted,
        g_ptr=g_sorted,
        v_ptr=v_sorted,
        embed_ptr=embed,
        hidden_grad_ptr=hidden_grad,
        total=total,
        batch_size=B,
        seq_len=S,
        vocab_size=V,
        hidden_dim=D,
    )
    return hidden_grad, embed_grad, bias_grad


def legacy_fused_sparton_bwd(
    grad_out: torch.Tensor,
    max_scores: torch.Tensor,
    max_idx: torch.Tensor,
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """M11 segmented backward with the production op's signature (A/B reference).

    The baton passed at M13: this wrapper re-points from the deleted M2-era
    atomic kernel to the M11 segmented design the split pass replaced.
    """

    assert grad_out.is_cuda, "legacy_fused_sparton_bwd only supports CUDA"
    grad_out = grad_out.contiguous()
    hidden_grad, embed_grad, bias_grad = segmented_sparton_bwd(
        grad_out, max_scores, max_idx, hidden, embed, bias
    )
    return hidden_grad, embed_grad, bias_grad if bias is not None else None


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

    hidden_grad, embed_grad, bias_grad = split_segmented_sparton_bwd(
        grad_out, max_scores, max_idx, hidden, embed, bias
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
    hidden, embed, bias = autocast_canonicalize(hidden, embed, bias)
    validate_forward_inputs(hidden, embed, bias, mask, backend="hybrid")
    # Canonicalize before the op so autograd saves contiguous tensors; the
    # backward kernel computes flat offsets that assume dense [B, S, D] strides.
    hidden = hidden.contiguous()
    embed = embed.contiguous()
    mask = mask.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    return fused_sparton_fwd_op(hidden, embed, bias, mask)
