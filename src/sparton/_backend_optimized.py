"""Pure-Triton persistent fused forward (the `optimized` backend, default).

A stock-`@triton.jit` implementation: a persistent grid-stride kernel over the
`(batch, vocab-tile)` output tiles, host-side TMA descriptors, `tl.dot`, and the fused
max/argmax/ReLU/log1p epilogue. The compiler owns the SMEM staging / pipelining (launch
`num_stages`) and the optional producer/consumer split (`tl.range(warp_specialize=...)`),
so there is no hand-rolled mbarrier protocol.

Design of record: docs/ARCHITECTURE.md §5.1. Two load-bearing findings shape this module:

  * **WS needs a single-result reduce.** Triton 3.7.1's auto-warp-specialization pass
    (`TritonGPUPartitionScheduling`) asserts `reduceOp.getResults().size() == 1`, so a
    2-result `tl.reduce` (value+index together) cannot be warp-specialized. The kernel
    therefore carries BOTH epilogues behind a constexpr branch: the fast 2-result combine
    for the homogeneous path, and an equivalent `tl.max` + masked `tl.min` (two
    single-result reduces) for the WS path. The branch is constexpr (trace-time), so the
    dead arm never reaches the WS pass — WS=True compiles even though the combine exists in
    source. The two arms are bit-identical (`tl.max` returns an actual element, so
    `vals == tmax` is exact, and min-of-qualifying-row-indices == the strict-`>` lowest-index
    tie-break).

  * **Tune by measurement, not derivation.** A fixed analytic tile is not optimal. The
    self-tuner measures a self-contained candidate tile set (`_CANDIDATE_TILES`, this module)
    and caches the winner keyed on `(D, V, dtype, arch)` — NOT B/S (the persistent grid fills
    the GPU regardless of trip count). WS is a tuned dimension (tried on the top-2 homogeneous
    tiles), not a manual flag. Timing is plain `triton.testing.do_bench` — run-to-run
    clock/thermal noise on this card is out of our hands and deliberately not modelled. With
    autotune off, the small `_analytic_tile` is used.

Host-side `TensorDescriptor` (not device-side `tl.make_tensor_descriptor`) means the kernel
needs no `triton.set_allocator` (no global-scratch side-effect). Forward output follows the
hidden dtype; logits/max accumulate in fp32. The op schema, saved-tensor set, and backward
delegation match the hybrid and naive backends exactly (shared `fused_sparton_bwd_op`).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from ._backend_hybrid import fused_sparton_bwd_op
from ._validation import autocast_canonicalize, validate_forward_inputs


# --- kernel -----------------------------------------------------------------


@triton.jit
def _argmax_strict_combine(v_a, i_a, v_b, i_b):
    """Strict-tie argmax combiner: higher value wins; ties -> lower index."""
    take_b = (v_b > v_a) | ((v_b == v_a) & (i_b < i_a))
    return tl.where(take_b, v_b, v_a), tl.where(take_b, i_b, i_a)


@triton.jit
def optimized_fwd_kernel(
    hidden_desc,
    embed_desc,
    bias_ptr,
    mask_ptr,
    out_scores_ptr,
    out_idx_ptr,
    B,
    S,
    NUM_CTAS,
    stride_mb,
    stride_ms,
    stride_ob,
    stride_ov,
    HAS_BIAS: tl.constexpr,
    # D and V are model constants -> constexpr: folds k_tiles/n_col/the V-tail, unrolls the
    # K-loop, and strength-reduces the persistent tile_id div/mod to a constant-divisor
    # multiply-shift. B and S stay runtime, so one compile per (model, tile) serves all (B,S).
    D: tl.constexpr,
    V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
):
    s_tiles = tl.cdiv(S, BLOCK_M)
    k_tiles = tl.cdiv(D, BLOCK_K)
    n_col = tl.cdiv(V, BLOCK_N)
    total = B * n_col
    for tile_id in tl.range(tl.program_id(0), total, NUM_CTAS, warp_specialize=WARP_SPECIALIZE):
        pid_b = tile_id // n_col
        pid_n = tile_id % n_col
        off_n = pid_n * BLOCK_N
        offs_n = off_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < V
        # Reset per tile (persistent landmine: one CTA handles many tiles). running_max
        # doubles as the 0.0 ReLU floor.
        running_max = tl.zeros([BLOCK_N], tl.float32)
        running_idx = tl.zeros([BLOCK_N], tl.int32)
        if HAS_BIAS:
            bias_vals = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        for st in range(s_tiles):
            s0 = st * BLOCK_M
            off_m = pid_b * S + s0  # batch-boundary rule: tiles never cross a batch row
            offs_s = s0 + tl.arange(0, BLOCK_M)
            row_valid = offs_s < S
            mask_vals = tl.load(
                mask_ptr + pid_b * stride_mb + offs_s * stride_ms,
                mask=row_valid,
                other=0,
            ).to(tl.float32)
            acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
            for kt in range(k_tiles):  # pipelined by the launch num_stages
                a = hidden_desc.load([off_m, kt * BLOCK_K])      # [BLOCK_M, BLOCK_K]
                b = embed_desc.load([off_n, kt * BLOCK_K])        # [BLOCK_N, BLOCK_K]
                acc = tl.dot(a, tl.trans(b), acc)                 # [BLOCK_M, BLOCK_N], fp32
            if HAS_BIAS:
                acc += bias_vals[None, :]
            vals = acc * mask_vals[:, None]
            vals = tl.where(row_valid[:, None] & mask_n[None, :], vals, 0.0)
            if WARP_SPECIALIZE:
                # Two single-result reduces (WS-compatible). Bit-identical to the combine.
                tmax = tl.max(vals, axis=0)
                row_idx = tl.arange(0, BLOCK_M).to(tl.int32)
                cand = tl.where(vals == tmax[None, :], row_idx[:, None], BLOCK_M)
                targ = tl.min(cand, axis=0)
            else:
                idx_vals = tl.arange(0, BLOCK_M).to(tl.int32)[:, None] + tl.zeros(
                    [BLOCK_M, BLOCK_N], tl.int32
                )
                tmax, targ = tl.reduce((vals, idx_vals), axis=0, combine_fn=_argmax_strict_combine)
            better = tmax > running_max
            running_max = tl.where(better, tmax, running_max)
            running_idx = tl.where(better, s0 + targ, running_idx)
        scores = tl.log(1.0 + tl.where(running_max > 0.0, running_max, 0.0))
        tl.store(out_scores_ptr + pid_b * stride_ob + offs_n * stride_ov, scores, mask=mask_n)
        tl.store(out_idx_ptr + pid_b * stride_ob + offs_n * stride_ov, running_idx.to(tl.int64), mask=mask_n)


# --- tile policy (self-contained; tailored to this pure-Triton kernel) -------
#
# A standalone tile-config space for THIS kernel. A tile here is exactly what the kernel +
# launch consume: the three block dims plus the two stock-Triton launch knobs. Warp
# specialization is a separate boolean the tuner explores; it is not part of the tile.
# `block_m` is the S-tile (the matmul M / the argmax-over-S extent), `block_n` the V-tile
# (output columns), `block_k` the D-tile (contraction).


@dataclass(frozen=True)
class OptimizedTile:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int

    @property
    def label(self) -> str:
        return f"{self.block_m}x{self.block_n}x{self.block_k} w{self.num_warps} s{self.num_stages}"


# Candidate tiles for the measured autotune sweep, spread from the 2026-06-24 re-baseline:
# the 128-family closes the large-grid gap; the 64/32 tiles serve the latency-bound
# B=1 / small-V regimes. All fit the sm_120 ~99 KB opt-in SMEM at fp16 (see _tile_smem_bytes).
_CANDIDATE_TILES = (
    OptimizedTile(128, 128, 64, 8, 3),
    OptimizedTile(128, 64, 64, 8, 4),
    OptimizedTile(128, 64, 64, 8, 3),
    OptimizedTile(64, 128, 64, 8, 3),
    OptimizedTile(128, 128, 32, 8, 4),
    OptimizedTile(128, 64, 32, 8, 4),
    OptimizedTile(64, 64, 64, 4, 4),
    OptimizedTile(64, 64, 64, 4, 3),
    OptimizedTile(64, 64, 32, 4, 4),
    OptimizedTile(64, 64, 32, 4, 3),
    OptimizedTile(32, 64, 32, 4, 4),
    OptimizedTile(64, 32, 32, 4, 4),
)

# A small, always-valid, fast-to-compile tile: the autotune-off default (correctness is
# tile-independent) and the no-valid-candidate fallback.
_FALLBACK_TILE = OptimizedTile(64, 64, 32, 4, 3)


@dataclass(frozen=True)
class _DeviceLimits:
    sm_count: int
    warp_size: int
    max_threads_per_block: int
    smem_per_block: int


def _device_limits() -> _DeviceLimits:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    smem = getattr(props, "shared_memory_per_block_optin", 0) or getattr(
        props, "shared_memory_per_block", 48 * 1024
    )
    return _DeviceLimits(
        sm_count=props.multi_processor_count,
        warp_size=getattr(props, "warp_size", 32),
        max_threads_per_block=props.max_threads_per_block,
        smem_per_block=smem,
    )


def _tile_smem_bytes(tile: OptimizedTile, elem_bytes: int) -> int:
    # The compiler stages `num_stages` copies of both operand tiles in shared memory.
    return tile.num_stages * (tile.block_m + tile.block_n) * tile.block_k * elem_bytes


def _tile_valid(tile: OptimizedTile, elem_bytes: int, limits: _DeviceLimits) -> bool:
    # Resource + tl.dot/TMA shape pre-filter (a coarse gate; the sweep's compile try/except is
    # the final arbiter). D and V do NOT constrain the tile -- TMA zero-pads partial boxes, and
    # M (=B*S) is irrelevant to a persistent grid.
    if tile.num_warps * limits.warp_size > limits.max_threads_per_block:
        return False
    if tile.block_m % 16 or tile.block_n % 16 or tile.block_k % 16:
        return False
    if (tile.block_k * elem_bytes) % 16 != 0:  # TMA inner-box 16-byte alignment
        return False
    if _tile_smem_bytes(tile, elem_bytes) > limits.smem_per_block:
        return False
    return True


def _valid_tiles(elem_bytes: int, limits: _DeviceLimits) -> list:
    return [t for t in _CANDIDATE_TILES if _tile_valid(t, elem_bytes, limits)]


def _analytic_tile(elem_bytes: int, limits: _DeviceLimits) -> OptimizedTile:
    """The autotune-off default (and no-valid-candidate fallback): a small, safe tile."""
    if _tile_valid(_FALLBACK_TILE, elem_bytes, limits):
        return _FALLBACK_TILE
    valid = _valid_tiles(elem_bytes, limits)
    return valid[0] if valid else _FALLBACK_TILE


# --- host helpers ------------------------------------------------------------


def _optimized_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "on", "true", "yes")


def _optimized_autotune_enabled() -> bool:
    # Measured autotune is the default; off -> the analytic tile + the manual WS flag
    # (used by the correctness tests/soak, where the tile choice is irrelevant).
    return os.environ.get("SPARTON_OPTIMIZED_AUTOTUNE", "on").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def _optimized_num_ctas(sm_count: int, total_tiles: int) -> int:
    """Persistent grid size. `SPARTON_OPTIMIZED_NUM_CTAS` forces an exact count (the
    NUM_CTAS=1 multi-tile correctness test sets it to 1); `SPARTON_OPTIMIZED_CTAS_PER_SM`
    (default 2) sets the per-SM multiplier. Capped at `total_tiles`."""
    forced = os.environ.get("SPARTON_OPTIMIZED_NUM_CTAS")
    if forced:
        n = int(forced)
    else:
        per_sm = int(os.environ.get("SPARTON_OPTIMIZED_CTAS_PER_SM", "2"))
        n = sm_count * max(per_sm, 1)
    return max(min(n, total_tiles), 1)


def _run_kernel(hidden, embed, bias, mask, policy, warp_specialize, limits):
    B, S, D = hidden.shape
    V, _ = embed.shape
    scores = torch.empty((B, V), device=hidden.device, dtype=hidden.dtype)
    indices = torch.empty((B, V), device=hidden.device, dtype=torch.int64)
    hidden_flat = hidden.reshape(B * S, D)
    hidden_desc = TensorDescriptor.from_tensor(hidden_flat, [policy.block_m, policy.block_k])
    embed_desc = TensorDescriptor.from_tensor(embed, [policy.block_n, policy.block_k])
    total_tiles = B * triton.cdiv(V, policy.block_n)
    num_ctas = _optimized_num_ctas(limits.sm_count, total_tiles)
    optimized_fwd_kernel[(num_ctas,)](
        hidden_desc,
        embed_desc,
        bias,
        mask,
        scores,
        indices,
        B,
        S,
        num_ctas,
        mask.stride(0),
        mask.stride(1),
        scores.stride(0),
        scores.stride(1),
        HAS_BIAS=bias is not None,
        D=D,
        V=V,
        BLOCK_M=policy.block_m,
        BLOCK_N=policy.block_n,
        BLOCK_K=policy.block_k,
        WARP_SPECIALIZE=warp_specialize,
        num_warps=policy.num_warps,
        num_stages=policy.num_stages,
    )
    return scores, indices


# --- self-implemented, measured autotune ------------------------------------
#
# Cache keyed on (D, V, dtype, arch) -- NOT B/S. Value: (policy, warp_specialize).

_TILE_CACHE: dict = {}

_ALLCLOSE_ATOL = 2e-2
_ALLCLOSE_RTOL = 2e-2


def _arch_key(hidden, embed):
    return (hidden.shape[2], embed.shape[0], hidden.dtype, torch.cuda.get_device_capability())


def _measure_best_tile(hidden, embed, bias, mask, limits):
    elem_bytes = hidden.element_size()
    analytic = _analytic_tile(elem_bytes, limits)
    policies = _valid_tiles(elem_bytes, limits)
    if not policies:
        return (analytic, False)

    def run_for(policy, ws):
        return _run_kernel(hidden, embed, bias, mask, policy, ws, limits)

    # Reference for the allclose gate: the analytic tile (homogeneous) -- a trusted same-kernel
    # baseline. Drops any miscompiled tile without an extra dependency or [B,S,V] materialization.
    ref = None
    try:
        ref = run_for(analytic, False)[0].float()
    except Exception:
        ref = None

    def passes_gate(policy, ws):
        try:
            scores, _ = run_for(policy, ws)
        except Exception:
            return False
        return ref is None or bool(torch.allclose(scores.float(), ref, atol=_ALLCLOSE_ATOL, rtol=_ALLCLOSE_RTOL))

    # Plain `triton.testing.do_bench` (the project's timer) — no custom thermal/order control;
    # run-to-run clock/thermal noise on this card is out of our hands and not modelled here.
    def bench(policy, ws):
        return triton.testing.do_bench(lambda: run_for(policy, ws))

    valid = [p for p in policies if passes_gate(p, False)]
    if not valid:
        return (analytic, False)

    # Stage 1: time the homogeneous tiles, keep the top-2.
    homog_t = {p: bench(p, False) for p in valid}
    top = sorted(valid, key=lambda p: homog_t[p])[:2]

    # Stage 2: try WS on the top-2 only (it wins the latency-bound small regime, loses large-V;
    # the measurement drops it where it loses). Pick the overall fastest (policy, ws).
    timed = {(p, False): homog_t[p] for p in top}
    for p in top:
        if passes_gate(p, True):
            timed[(p, True)] = bench(p, True)
    return min(timed, key=timed.get)


def _autotune_tile(hidden, embed, bias, mask, limits):
    key = _arch_key(hidden, embed)
    cached = _TILE_CACHE.get(key)
    if cached is not None:
        return cached
    winner = _measure_best_tile(hidden, embed, bias, mask, limits)
    _TILE_CACHE[key] = winner
    return winner


def pretune(D: int, V: int, dtype: torch.dtype, B: int = 8, S: int = 512, *, has_bias: bool = True):
    """Warm the tile cache for a model's (D, V, dtype) before serving traffic, so the
    first real call doesn't pay the one-time measurement cost. Returns (policy, ws)."""
    gen = torch.Generator(device="cuda").manual_seed(0)
    hidden = torch.randn(B, S, D, device="cuda", dtype=dtype, generator=gen) * 0.05
    embed = torch.randn(V, D, device="cuda", dtype=dtype, generator=gen) * 0.05
    bias = torch.randn(V, device="cuda", dtype=dtype, generator=gen) * 0.05 if has_bias else None
    mask = torch.ones(B, S, device="cuda", dtype=torch.int32)
    return _autotune_tile(hidden, embed, bias, mask, _device_limits())


def clear_tile_cache() -> None:
    """Drop the measured-tile cache (used by tests)."""
    _TILE_CACHE.clear()


def _launch_optimized_fwd(
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
    assert hidden.element_size() == 2, "the optimized kernel assumes 16-bit elements (fp16/bf16)"

    limits = _device_limits()
    if _optimized_autotune_enabled():
        policy, warp_specialize = _autotune_tile(hidden, embed, bias, mask, limits)
    else:
        policy = _analytic_tile(hidden.element_size(), limits)
        warp_specialize = _optimized_flag("SPARTON_OPTIMIZED_WARP_SPECIALIZE")
    return _run_kernel(hidden, embed, bias, mask, policy, warp_specialize, limits)


# --- custom op + autograd (schema/saved-tensors shared with the other backends) ---------


@torch.library.custom_op(
    "sparton::optimized_fwd",
    mutates_args=(),
    schema="(Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)",
)
def optimized_fwd_op(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert hidden.is_cuda, "sparton::optimized_fwd only supports CUDA"
    return _launch_optimized_fwd(hidden, embed, bias, mask)


@optimized_fwd_op.register_fake
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


optimized_fwd_op.register_autograd(_backward, setup_context=_setup_context)


def optimized_forward(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden, embed, bias = autocast_canonicalize(hidden, embed, bias)
    validate_forward_inputs(hidden, embed, bias, mask, backend="optimized")
    hidden = hidden.contiguous()
    embed = embed.contiguous()
    mask = mask.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    return optimized_fwd_op(hidden, embed, bias, mask)


__all__ = [
    "optimized_forward",
    "optimized_fwd_op",
    "optimized_fwd_kernel",
    "pretune",
    "clear_tile_cache",
]
