"""Backward prototype registry for bench_backward.py / ncu_backward_target.py.

Each entry maps a name to a callable with the backward-op signature
``(grad_out, max_scores, max_idx, hidden, embed, bias, mask) ->
(hidden_grad, embed_grad, bias_grad)`` so the harness can verify it against
``optimized`` and time it op-level. Nothing here is imported by the package.

Live experiment (2026-06); mechanism + landing recorded in
docs/DEVELOPMENT.md ("Post-M13 dot-reduction backward landing"). NOTE: ``dot_bs_mix`` was LANDED into
``src/sparton/backward/optimized.py`` (2026-06-26), so it now measures ≈1.00× the
production ``optimized`` and serves as its A/B faithfulness twin (a drift check for
``bench_backward``'s verify-against-optimized). Two independent levers on the
hidden-grad passes:

* uniform pass — its per-chunk reduce (tl.sum over CHUNK embed rows) serializes
  on top of the L2-bound embed gather (~21% non-overlapped). Expressing the
  reduce as a tl.dot lets the matmul software-pipeliner multi-buffer the gather
  in smem and overlap it. ``g`` is split hi/lo fp16 (COMP=1) so the fp16 MMA
  recovers fp32-grade precision (the fp16*fp16 product is exact in the fp32
  accumulator; only rounding g to fp16 loses bits).
* mixed pass — the production launch is pinned at BLOCK_D=128/SUB=64
  (register-bound). Re-tuning to BLOCK_D=64/SUB=32 (same cumsum, exact) is
  ~1.3-1.65x. A dot-segmented mixed (COMP knob) is faster still in fp16.
"""
from __future__ import annotations
import triton
import triton.language as tl
import torch

from sparton.backward import optimized as P


# ---- dot-reduction uniform pass (drop-in for uniform_hidden_grad_kernel) ----
@triton.jit
def _uniform_dot_kernel(
    keys_ptr, g_ptr, v_ptr, embed_ptr, hidden_grad_ptr, n_active_ptr,
    total, batch_size, seq_len, vocab_size,
    hidden_dim: tl.constexpr, CHUNK: tl.constexpr, BLOCK_D: tl.constexpr,
    MPAD: tl.constexpr, COMP: tl.constexpr,
):
    pid_chunk = tl.program_id(0)
    pid_d = tl.program_id(1)
    num_ctas = tl.num_programs(0)
    num_rows = batch_size * seq_len
    n_live = tl.load(n_active_ptr)
    n_chunks = tl.cdiv(n_live, CHUNK)

    offs_d = (pid_d * BLOCK_D + tl.arange(0, BLOCK_D)).to(tl.int64)
    mask_d = offs_d < hidden_dim
    m_rows = tl.arange(0, MPAD)

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
        e_tile = tl.load(a_ptrs, mask=row_on[:, None] & mask_d[None, :], other=0.0)
        if COMP == 2:
            # Block-scaled compensated: range-safe (g is fp32; casting it to the
            # model dtype would overflow under AMP loss-scaling / underflow under
            # large scores). Per-chunk s=max|g| brings g into the model dtype's
            # range, keeping embed at native precision. See research doc §4.4.
            gabs = tl.max(tl.where(row_on, tl.abs(g), 0.0))
            s = tl.where(gabs > 0.0, gabs, 1.0)
            gs = g * (1.0 / s)
            g_hi = gs.to(e_tile.dtype)
            g_lo = (gs - g_hi.to(tl.float32)).to(e_tile.dtype)
            ghi = tl.where(m_rows[:, None] == 0, g_hi[None, :], 0.0).to(e_tile.dtype)
            glo = tl.where(m_rows[:, None] == 0, g_lo[None, :], 0.0).to(e_tile.dtype)
            out = (tl.dot(ghi, e_tile, out_dtype=tl.float32)
                   + tl.dot(glo, e_tile, out_dtype=tl.float32)) * s
        elif COMP == 1:
            # Compensated, model-dtype (no scaling): fp32-grade precision BUT
            # NaNs under fp16 + AMP (g overflows fp16). bf16 models only.
            g_hi = g.to(e_tile.dtype)
            g_lo = (g - g_hi.to(tl.float32)).to(e_tile.dtype)
            ghi = tl.where(m_rows[:, None] == 0, g_hi[None, :], 0.0).to(e_tile.dtype)
            glo = tl.where(m_rows[:, None] == 0, g_lo[None, :], 0.0).to(e_tile.dtype)
            out = (tl.dot(ghi, e_tile, out_dtype=tl.float32)
                   + tl.dot(glo, e_tile, out_dtype=tl.float32))
        else:
            g_mat = tl.where(m_rows[:, None] == 0, g[None, :], 0.0).to(e_tile.dtype)
            out = tl.dot(g_mat, e_tile, out_dtype=tl.float32)
        partial = tl.sum(tl.where(m_rows[:, None] == 0, out, 0.0), axis=0)
        tl.atomic_add(
            hidden_grad_ptr + keys_min.to(tl.int64) * hidden_dim + offs_d,
            partial, mask=is_uniform & mask_d, sem="relaxed",
        )


# ---- dot-segmented mixed pass (fp16-fast alternative to the cumsum) ----
@triton.jit
def _mixed_dot_kernel(
    keys_ptr, g_ptr, v_ptr, embed_ptr, hidden_grad_ptr,
    total, batch_size, seq_len, vocab_size,
    hidden_dim: tl.constexpr, GRANULE: tl.constexpr, BLOCK_D: tl.constexpr,
    COMP: tl.constexpr,
):
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    num_rows = batch_size * seq_len
    lane = tl.arange(0, GRANULE)
    granule = pid
    keep_going = tl.load(keys_ptr + granule * GRANULE) < num_rows
    while keep_going:
        offs = granule * GRANULE + lane
        in_range = offs < total
        keys = tl.load(keys_ptr + offs, mask=in_range, other=num_rows)
        if tl.min(keys) != tl.max(keys):
            active = in_range & (keys < num_rows)
            g = tl.load(g_ptr + offs, mask=active, other=0.0)
            v = tl.load(v_ptr + offs, mask=active, other=0).to(tl.int64)
            keys_m1 = tl.load(keys_ptr + offs - 1, mask=in_range & (offs > 0),
                              other=num_rows + 1)
            is_start = active & ((lane == 0) | (keys != keys_m1))
            same = (keys[:, None] == keys[None, :]) & active[None, :]
            # The selection matrix is d-independent — build it once (hoisted out of
            # the d-loop). COMP==2 block-scales for range-safety: s=max|g| over the
            # granule brings g into the model dtype's dense range; hi/lo split keeps
            # fp32-grade precision; *s restores magnitude (fp16+bf16+AMP safe).
            s = tl.max(tl.where(active, tl.abs(g), 0.0))
            s = tl.where(s > 0.0, s, 1.0)
            gsrc = (g * (1.0 / s)) if COMP == 2 else g
            sel = tl.where(is_start[:, None] & same, gsrc[None, :], 0.0)
            for d0 in tl.static_range(0, hidden_dim, BLOCK_D):
                offs_d = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                mask_d = offs_d < hidden_dim
                e_tile = tl.load(embed_ptr + v[:, None] * hidden_dim + offs_d[None, :],
                                 mask=active[:, None] & mask_d[None, :], other=0.0)
                if COMP == 0:
                    part = tl.dot(sel.to(e_tile.dtype), e_tile, out_dtype=tl.float32)
                else:
                    sel_hi = sel.to(e_tile.dtype)
                    sel_lo = (sel - sel_hi.to(tl.float32)).to(e_tile.dtype)
                    part = (tl.dot(sel_hi, e_tile, out_dtype=tl.float32)
                            + tl.dot(sel_lo, e_tile, out_dtype=tl.float32))
                    if COMP == 2:
                        part = part * s
                dest = keys.to(tl.int64)[:, None] * hidden_dim + offs_d[None, :]
                tl.atomic_add(hidden_grad_ptr + dest, part,
                              mask=is_start[:, None] & mask_d[None, :], sem="relaxed")
        granule += num_ctas
        if granule * GRANULE >= total:
            keep_going = False
        else:
            keep_going = tl.load(keys_ptr + granule * GRANULE) < num_rows


_CHUNK = 64


def _uniform_dot(keys, g, v, embed, hidden_grad, n_active, total, B, S, V, D,
                 comp, bd=128, warps=4, stages=3):
    _uniform_dot_kernel[(min(triton.cdiv(total, _CHUNK), 4096), triton.cdiv(D, bd))](
        keys_ptr=keys, g_ptr=g, v_ptr=v, embed_ptr=embed, hidden_grad_ptr=hidden_grad,
        n_active_ptr=n_active, total=total, batch_size=B, seq_len=S, vocab_size=V,
        hidden_dim=D, CHUNK=_CHUNK, BLOCK_D=bd, MPAD=16, COMP=comp,
        num_warps=warps, num_stages=stages)


def _prod_mixed(keys, g, v, embed, hidden_grad, total, B, S, V, D, bd, sub, warps, stages):
    P.mixed_hidden_grad_kernel[(min(triton.cdiv(total, _CHUNK), 4096),)](
        keys_ptr=keys, g_ptr=g, v_ptr=v, embed_ptr=embed, hidden_grad_ptr=hidden_grad,
        total=total, batch_size=B, seq_len=S, vocab_size=V, hidden_dim=D,
        GRANULE=_CHUNK, SUB=sub, BLOCK_D=bd, num_warps=warps, num_stages=stages)


def _dot_mixed(keys, g, v, embed, hidden_grad, total, B, S, V, D, comp, bd=64, warps=4, stages=2):
    _mixed_dot_kernel[(min(triton.cdiv(total, _CHUNK), 4096),)](
        keys_ptr=keys, g_ptr=g, v_ptr=v, embed_ptr=embed, hidden_grad_ptr=hidden_grad,
        total=total, batch_size=B, seq_len=S, vocab_size=V, hidden_dim=D,
        GRANULE=_CHUNK, BLOCK_D=bd, COMP=comp, num_warps=warps, num_stages=stages)


_UNIFORM_COMP = {"bs": 2, "comp": 1, "fp16": 0}


def _make_bwd(uniform_kind, mixed_kind):
    """uniform_kind: 'bs'|'comp'|'fp16'; mixed_kind: 'prod'|'retuned'|'dotfp16'|'dotbs'|'dotcomp'.

    'bs' = block-scaled compensated (range-safe, the fp16 production default);
    'comp' = compensated model-dtype, NO block-scale (bf16-safe -- bf16's exponent
             covers fp32 range; NaNs under fp16+AMP. The 2026-06-26 study: ≤ 'bs'
             error always, 3.4-7% faster uniform pass -> the bf16 production path);
    'fp16' = non-compensated (fastest, precision-degraded, bf16-unsafe).
    mixed 'retuned' = exact cumsum (the landed default); 'dotfp16' = non-compensated
    dot-segmented (fp16-only); 'dotbs' = BLOCK-SCALED dot-segmented (range-safe but
    pays the scale tax -> lost to cumsum at M13); 'dotcomp' = NO-SCALE compensated
    dot-segmented (bf16 range-safe, no scale tax -- the dot-mixed revisit).
    """
    def bwd(grad_out, max_scores, max_idx, hidden, embed, bias, mask=None):
        B, S, D = hidden.shape
        V = embed.shape[0]
        (hidden_grad, embed_grad, bias_grad, keys, g, v, n_active, total) = (
            P._bwd_shared_stages(grad_out, max_scores, max_idx, hidden, embed, bias)
        )
        _uniform_dot(keys, g, v, embed, hidden_grad, n_active, total, B, S, V, D,
                     comp=_UNIFORM_COMP[uniform_kind])
        if mixed_kind == "prod":
            _prod_mixed(keys, g, v, embed, hidden_grad, total, B, S, V, D,
                        bd=128, sub=64, warps=4, stages=2)
        elif mixed_kind == "retuned":
            _prod_mixed(keys, g, v, embed, hidden_grad, total, B, S, V, D,
                        bd=64, sub=32, warps=4, stages=3)
        elif mixed_kind == "dotbs":  # block-scaled dot-segmented (range-safe)
            _dot_mixed(keys, g, v, embed, hidden_grad, total, B, S, V, D, comp=2)
        elif mixed_kind == "dotcomp":  # no-scale compensated dot-segmented (bf16 range-safe)
            _dot_mixed(keys, g, v, embed, hidden_grad, total, B, S, V, D, comp=1)
        else:  # dotfp16
            _dot_mixed(keys, g, v, embed, hidden_grad, total, B, S, V, D, comp=0)
        # Mirror the op wrapper's bias optionality (the shared stages return a
        # scalar placeholder when bias is None; the op returns None).
        return hidden_grad, embed_grad, (bias_grad if bias is not None else None)
    return bwd


def _make_bwd_tuned():
    """dot_bs (range-safe) uniform + REGIME-AWARE mixed SUB. TESTED -- NO RELIABLE GAIN.

    The 2026-06-26 ISOLATED sweep (probe_track_a_tune.py) appeared to show the
    retuned mixed's optimal SUB is regime-dependent (query SUB=16, doc SUB=64) for
    +1.17x/1.19x on the isolated uniform+mixed. BUT the END-TO-END bench (this
    variant benched in a different order) showed it is statistically identical to
    'dot_bs_mix' (~1.10-1.11x) and marginally WORSE on doc -- the isolated-sweep
    gains were RTX 5090 do_bench thermal/order bias (memory: do-bench-thermal-order-bias),
    not real. Kept as a documented negative: prefer 'dot_bs_mix'. SUB still must
    divide GRANULE=64 (complement invariant); 16 and 64 both do.
    """
    def bwd(grad_out, max_scores, max_idx, hidden, embed, bias, mask=None):
        B, S, D = hidden.shape
        V = embed.shape[0]
        (hidden_grad, embed_grad, bias_grad, keys, g, v, n_active, total) = (
            P._bwd_shared_stages(grad_out, max_scores, max_idx, hidden, embed, bias)
        )
        _uniform_dot(keys, g, v, embed, hidden_grad, n_active, total, B, S, V, D,
                     comp=_UNIFORM_COMP["bs"])
        sub = 16 if S <= 96 else 64
        _prod_mixed(keys, g, v, embed, hidden_grad, total, B, S, V, D,
                    bd=64, sub=sub, warps=4, stages=3)
        return hidden_grad, embed_grad, (bias_grad if bias is not None else None)
    return bwd


PROTOTYPES = {
    # range-safe + precision-safe: the production-recommended combo (§4.4 + §5).
    "dot_bs_mix":    _make_bwd("bs", "retuned"),
    # range-safe + regime-aware mixed SUB (2026-06-26 sweep refinement).
    "dot_bs_mix_t":  _make_bwd_tuned(),
    # compensated model-dtype (bf16-only; fp16+AMP NaNs) — faster, isolates the
    # block-scale cost.
    "dot_comp_mix":  _make_bwd("comp", "retuned"),
    # fastest, precision-degraded, bf16-unsafe (for reference / fp16-only).
    "dot_fp16_mix":  _make_bwd("fp16", "dotfp16"),
    # range-safe BOTH passes as tl.dot: bs-dot uniform + bs-dot mixed. TESTED 2026-06-26:
    # correct + range-safe (fp16+bf16, 0 verify failures) but 2-3% SLOWER than the landed
    # cumsum mixed on real query+doc -- the dot-mixed's speed in 'dot_fp16_mix' came purely
    # from being non-compensated single-dot (unsafe); making it range-safe (block-scale +
    # 2x hi/lo dot + the [GRANULE,GRANULE] selection-matrix dot) erases the advantage.
    # The exact cumsum mixed ('retuned') stays the landed choice. Documented negative.
    "dot_bs_bsmix":  _make_bwd("bs", "dotbs"),
    # bf16 production candidate: no-scale compensated uniform + exact cumsum mixed
    # (the uniform-only dtype-aware change). ≤ 'dot_bs_mix' error, faster uniform pass.
    # (== "dot_comp_mix" above; named for the landing's A/B clarity.)
    # bf16 dot-mixed revisit: no-scale compensated BOTH passes. Tests whether removing
    # the block-scale tax lets the range-safe dot-mixed finally beat the cumsum on bf16.
    "dot_comp_dotcomp": _make_bwd("comp", "dotcomp"),
}
