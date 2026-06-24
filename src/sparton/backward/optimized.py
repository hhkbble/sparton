"""Optimized backward (the default; selected by the optimized forward).

The current best hidden-grad design: a branch-free uniform-chunk streaming pass over
single-destination chunks plus a run-boundary scan over the rest, fed by a shared prep
pipeline. Mechanism and measured evidence: docs/DEVELOPMENT.md (M13 backward).

Prep stages (before the hidden-grad pass), inside the custom op:
  1. ``bwd_prep_kernel``: ``g = grad_out * exp(-scores)`` where ``scores > 0`` (the
     forward's exact fp32 math), int32 idx, destination sort keys ``b*S + idx`` (sentinel
     ``B*S`` for inactive entries, which sort last), and a device-side active-entry count;
  2. ``torch.sort`` by destination + one payload-gather kernel;
  3. ``embed_grad_kernel`` (exclusive-owner plain stores, no atomics).
Then the two hidden-grad kernels -- ``uniform_hidden_grad_kernel`` (the streaming fast
path) and ``mixed_hidden_grad_kernel`` (the run-boundary scan) -- exact complements at one
shared CHUNK granularity so each contribution is deposited once, which is why the mixed
kernel is NOT autotuned and runs at the uniform winner's CHUNK.
"""

import triton
import triton.language as tl
import torch
from typing import Optional, Tuple


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

    The optimized backward's shared prelude, before its two hidden-grad kernels
    (the uniform streaming pass and the run-boundary scan).
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
    assert B * V < 2**31 - 256, "optimized backward: B*V exceeds int32 indexing"
    assert S * D < 2**31, "optimized backward: S*D exceeds int32 stride arithmetic"
    assert B * S < 2**31, "optimized backward: B*S sentinel exceeds int32"

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


# Uniform-chunk pass: a branch-free bounded-persistent
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


# Mixed-chunk pass. NOT autotuned: the granule size must equal
# the uniform kernel's selected CHUNK or the complement breaks (DEVELOPMENT.md
# M13 §5.4 item 2). A mixed granule is processed as SUB-row run-boundary scan
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


def optimized_bwd(
    grad_out: torch.Tensor,
    max_scores: torch.Tensor,
    max_idx: torch.Tensor,
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the optimized backward; returns fp32 gradient buffers."""

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
