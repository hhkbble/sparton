from __future__ import annotations

from typing import Optional, Tuple

import torch

from ._backend_hybrid import fused_sparton_bwd_op
from ._gluon_runtime import autotune, gl, gluon, mbarrier, mma_v2, tma
from ._gluon_policy_runtime import (
    configs_for_policies,
    element_ty_for_dtype,
    make_descriptor_bank,
    prune_configs_for_policies,
)
from ._runtime_policy import (
    DeviceProfile,
    GluonGemmPolicy,
    ProblemSpec,
    derive_optimized_forward_policies,
    optimized_forward_policy_universe,
    torch_device_profile,
)


_OPTIMIZED_FORWARD_POLICIES = optimized_forward_policy_universe()
if len(_OPTIMIZED_FORWARD_POLICIES) != 11:
    raise RuntimeError(
        "optimized Gluon forward descriptor-bank signature expects 11 policies, "
        f"got {len(_OPTIMIZED_FORWARD_POLICIES)}"
    )


def get_optimized_forward_policies(
    problem: ProblemSpec,
    device: DeviceProfile,
) -> tuple[GluonGemmPolicy, ...]:
    return derive_optimized_forward_policies(problem, device)


def get_optimized_forward_configs():
    return configs_for_policies(_OPTIMIZED_FORWARD_POLICIES)


def _problem_from_autotune_args(named_args, kwargs) -> ProblemSpec:
    get_arg = lambda name: kwargs[name] if name in kwargs else named_args[name]
    return ProblemSpec(
        M=int(get_arg("B")) * int(get_arg("S")),
        N=int(get_arg("V")),
        K=int(get_arg("D")),
        dtype_name="fp16",
    )


def _prune_optimized_forward_configs(configs, named_args, **kwargs):
    policies = get_optimized_forward_policies(
        _problem_from_autotune_args(named_args, kwargs),
        torch_device_profile(),
    )
    return prune_configs_for_policies(configs, policies, _OPTIMIZED_FORWARD_POLICIES)


@gluon.jit
def _argmax_strict_combine(value_a, index_a, value_b, index_b):
    take_b = (value_b > value_a) | ((value_b == value_a) & (index_b < index_a))
    return gl.where(take_b, value_b, value_a), gl.where(take_b, index_b, index_a)


@autotune(
    configs=get_optimized_forward_configs(),
    key=["B", "S", "D", "V"],
    prune_configs_by={"early_config_prune": _prune_optimized_forward_configs},
    cache_results=True,
)
@gluon.jit
def sparton_optimized_forward_kernel(
    hidden_desc_0,
    embed_desc_0,
    hidden_desc_1,
    embed_desc_1,
    hidden_desc_2,
    embed_desc_2,
    hidden_desc_3,
    embed_desc_3,
    hidden_desc_4,
    embed_desc_4,
    hidden_desc_5,
    embed_desc_5,
    hidden_desc_6,
    embed_desc_6,
    hidden_desc_7,
    embed_desc_7,
    hidden_desc_8,
    embed_desc_8,
    hidden_desc_9,
    embed_desc_9,
    hidden_desc_10,
    embed_desc_10,
    bias_ptr,
    mask_ptr,
    out_scores_ptr,
    out_idx_ptr,
    B,
    S,
    D,
    V,
    stride_mb,
    stride_ms,
    stride_ob,
    stride_ov,
    HAS_BIAS: gl.constexpr,
    POLICY_ID: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    WARPS_M: gl.constexpr,
    WARPS_N: gl.constexpr,
    ELEMENT_TY: gl.constexpr,
):
    hidden_desc = hidden_desc_0
    embed_desc = embed_desc_0
    if POLICY_ID == 1:
        hidden_desc = hidden_desc_1
        embed_desc = embed_desc_1
    if POLICY_ID == 2:
        hidden_desc = hidden_desc_2
        embed_desc = embed_desc_2
    if POLICY_ID == 3:
        hidden_desc = hidden_desc_3
        embed_desc = embed_desc_3
    if POLICY_ID == 4:
        hidden_desc = hidden_desc_4
        embed_desc = embed_desc_4
    if POLICY_ID == 5:
        hidden_desc = hidden_desc_5
        embed_desc = embed_desc_5
    if POLICY_ID == 6:
        hidden_desc = hidden_desc_6
        embed_desc = embed_desc_6
    if POLICY_ID == 7:
        hidden_desc = hidden_desc_7
        embed_desc = embed_desc_7
    if POLICY_ID == 8:
        hidden_desc = hidden_desc_8
        embed_desc = embed_desc_8
    if POLICY_ID == 9:
        hidden_desc = hidden_desc_9
        embed_desc = embed_desc_9
    if POLICY_ID == 10:
        hidden_desc = hidden_desc_10
        embed_desc = embed_desc_10

    pid_b = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_n = pid_n * BLOCK_N

    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0],
        warps_per_cta=[WARPS_M, WARPS_N],
        instr_shape=[16, 8],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 2)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 2)
    row_layout: gl.constexpr = gl.SliceLayout(1, mma)
    col_layout: gl.constexpr = gl.SliceLayout(0, mma)

    hidden_smem = gl.allocate_shared_memory(
        ELEMENT_TY,
        [NUM_STAGES, BLOCK_M, BLOCK_K],
        hidden_desc.layout,
    )
    embed_smem = gl.allocate_shared_memory(
        ELEMENT_TY,
        [NUM_STAGES, BLOCK_N, BLOCK_K],
        embed_desc.layout,
    )
    bars = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())

    offs_n = off_n + gl.arange(0, BLOCK_N, col_layout)
    mask_n = offs_n < V
    running_max = gl.full([BLOCK_N], 0.0, gl.float32, col_layout)
    running_idx = gl.full([BLOCK_N], 0, gl.int32, col_layout)

    if HAS_BIAS:
        bias_vals = gl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(gl.float32)

    s_tiles = gl.cdiv(S, BLOCK_M)
    k_tiles = gl.cdiv(D, BLOCK_K)
    NBYTES: gl.constexpr = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * 2

    for st in range(s_tiles):
        s0 = st * BLOCK_M
        off_m = pid_b * S + s0

        for i in gl.static_range(NUM_STAGES):
            mbarrier.init(bars.index(i), count=1)
        gl.barrier()

        for stage in gl.static_range(NUM_STAGES - 1):
            if stage < k_tiles:
                bar = bars.index(stage)
                mbarrier.expect(bar, NBYTES)
                tma.async_copy_global_to_shared(
                    hidden_desc,
                    [off_m, stage * BLOCK_K],
                    bar,
                    hidden_smem.index(stage),
                )
                tma.async_copy_global_to_shared(
                    embed_desc,
                    [off_n, stage * BLOCK_K],
                    bar,
                    embed_smem.index(stage),
                )

        acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, mma)
        for kt in range(k_tiles):
            buf = kt % NUM_STAGES
            phase = (kt // NUM_STAGES) & 1
            mbarrier.wait(bars.index(buf), phase)
            hidden_tile = hidden_smem.index(buf).load(dot_a)
            embed_tile = embed_smem.index(buf).permute([1, 0]).load(dot_b)
            gl.barrier()
            next_kt = kt + (NUM_STAGES - 1)
            if next_kt < k_tiles:
                next_buf = next_kt % NUM_STAGES
                bar = bars.index(next_buf)
                mbarrier.expect(bar, NBYTES)
                tma.async_copy_global_to_shared(
                    hidden_desc,
                    [off_m, next_kt * BLOCK_K],
                    bar,
                    hidden_smem.index(next_buf),
                )
                tma.async_copy_global_to_shared(
                    embed_desc,
                    [off_n, next_kt * BLOCK_K],
                    bar,
                    embed_smem.index(next_buf),
                )
            acc = mma_v2(hidden_tile, embed_tile, acc)

        offs_s = s0 + gl.arange(0, BLOCK_M, row_layout)
        row_valid = offs_s < S
        mask_vals = gl.load(
            mask_ptr + pid_b * stride_mb + offs_s * stride_ms,
            mask=row_valid,
            other=0,
        ).to(gl.float32)
        if HAS_BIAS:
            acc += bias_vals[None, :]
        vals = acc * mask_vals[:, None]
        vals = gl.where(row_valid[:, None] & mask_n[None, :], vals, 0.0)

        row_idx = gl.arange(0, BLOCK_M, row_layout).to(gl.int32)
        idx_vals = row_idx[:, None] + gl.full([BLOCK_M, BLOCK_N], 0, gl.int32, mma)
        tile_max, tile_arg = gl.reduce(
            (vals, idx_vals),
            axis=0,
            combine_fn=_argmax_strict_combine,
        )
        better = tile_max > running_max
        running_max = gl.where(better, tile_max, running_max)
        running_idx = gl.where(better, s0 + tile_arg, running_idx)
        gl.barrier()

    relu_vals = gl.where(running_max > 0.0, running_max, 0.0)
    scores = gl.log(1.0 + relu_vals)
    score_ptrs = out_scores_ptr + pid_b * stride_ob + offs_n * stride_ov
    idx_ptrs = out_idx_ptr + pid_b * stride_ob + offs_n * stride_ov
    gl.store(score_ptrs, scores.to(ELEMENT_TY), mask=mask_n)
    gl.store(idx_ptrs, running_idx.to(gl.int64), mask=mask_n)


def _launch_optimized_fwd(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    import triton

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
    hidden_flat = hidden.reshape(B * S, D)
    descriptor_bank = make_descriptor_bank(
        hidden_flat,
        embed,
        _OPTIMIZED_FORWARD_POLICIES,
        max_policies=len(_OPTIMIZED_FORWARD_POLICIES),
    )

    def grid(meta):
        return (B, triton.cdiv(V, meta["BLOCK_N"]))

    sparton_optimized_forward_kernel[grid](
        *descriptor_bank,
        bias,
        mask,
        scores,
        indices,
        B,
        S,
        D,
        V,
        mask.stride(0),
        mask.stride(1),
        scores.stride(0),
        scores.stride(1),
        HAS_BIAS=bias is not None,
        ELEMENT_TY=element_ty_for_dtype(hidden.dtype),
    )
    return scores, indices


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
    hidden = hidden.contiguous()
    embed = embed.contiguous()
    mask = mask.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    return optimized_fwd_op(hidden, embed, bias, mask)


__all__ = [
    "get_optimized_forward_configs",
    "get_optimized_forward_policies",
    "optimized_forward",
    "optimized_fwd_op",
    "sparton_optimized_forward_kernel",
]
