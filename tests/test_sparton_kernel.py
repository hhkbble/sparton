from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

import pytest
import torch
from torch.testing import assert_close

from sparton._runtime_policy import (
    DeviceProfile,
    GluonGemmPolicy,
    ProblemSpec,
    derive_optimized_forward_policies,
    generate_gluon_gemm_policies,
    gluon_gemm_policy_universe,
    optimized_forward_fallback_policy,
    optimized_forward_policy_universe,
    policy_config_kwargs,
)


CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="Sparton CUDA kernel tests require a CUDA device",
)
requires_optimized_gluon = pytest.mark.usefixtures("optimized_gluon_available")

_OPTIMIZED_GLUON_AVAILABILITY: tuple[bool, str] | None = None


def _optimized_gluon_availability() -> tuple[bool, str]:
    global _OPTIMIZED_GLUON_AVAILABILITY
    if _OPTIMIZED_GLUON_AVAILABILITY is not None:
        return _OPTIMIZED_GLUON_AVAILABILITY
    if not CUDA_AVAILABLE:
        _OPTIMIZED_GLUON_AVAILABILITY = (
            False,
            "Sparton optimized Gluon tests require a CUDA device",
        )
        return _OPTIMIZED_GLUON_AVAILABILITY

    from sparton._gluon_runtime import is_gluon_backend_available

    available, reason = is_gluon_backend_available()
    _OPTIMIZED_GLUON_AVAILABILITY = (
        available,
        reason if available else f"Sparton optimized Gluon backend is unavailable: {reason}",
    )
    return _OPTIMIZED_GLUON_AVAILABILITY


@pytest.fixture
def optimized_gluon_available() -> None:
    available, reason = _optimized_gluon_availability()
    if not available:
        pytest.skip(reason)

FORWARD_CASES = [
    pytest.param(True, torch.float16, id="bias-fp16"),
    pytest.param(False, torch.float16, id="no_bias-fp16"),
    pytest.param(False, torch.bfloat16, id="no_bias-bf16"),
    pytest.param(True, torch.bfloat16, id="bias-bf16"),
]


def _reference_from_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_logits = logits * mask.to(dtype=logits.dtype)[:, :, None]
    batch_size, seq_len, vocab_size = masked_logits.shape
    running_max = torch.zeros(
        (batch_size, vocab_size),
        device=masked_logits.device,
        dtype=masked_logits.dtype,
    )
    running_idx = torch.zeros(
        (batch_size, vocab_size),
        device=masked_logits.device,
        dtype=torch.int64,
    )

    for seq_idx in range(seq_len):
        candidate = masked_logits[:, seq_idx, :]
        better = candidate > running_max
        running_max = torch.where(better, candidate, running_max)
        running_idx = torch.where(
            better,
            torch.full_like(running_idx, seq_idx),
            running_idx,
        )

    return torch.log1p(torch.relu(running_max)), running_idx


def sparton_reference(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = hidden @ embed.T
    if bias is not None:
        logits = logits + bias
    return _reference_from_logits(logits, mask)


def _make_kernel_inputs(
    *,
    device: torch.device,
    dtype: torch.dtype,
    use_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(17 + int(use_bias))
    hidden = torch.randn(
        (2, 5, 16),
        device=device,
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    embed = torch.randn(
        (19, 16),
        device=device,
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    bias = None
    if use_bias:
        bias = torch.randn(
            (19,),
            device=device,
            dtype=dtype,
            generator=generator,
            requires_grad=True,
        )
    mask = torch.tensor(
        [[1, 1, 0, 1, 0], [0, 1, 1, 0, 1]],
        device=device,
        dtype=torch.int32,
    )
    return hidden, embed, bias, mask


def _clone_leaf(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    return tensor.detach().clone().requires_grad_(tensor.requires_grad)


def _skip_if_unsupported_dtype(dtype: torch.dtype) -> None:
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")


def _score_tolerances(dtype: torch.dtype) -> dict[str, float]:
    if dtype is torch.bfloat16:
        return {"atol": 5e-2, "rtol": 5e-2}
    return {"atol": 2e-3, "rtol": 2e-3}


def _device_profile(
    *,
    sm_count: int = 170,
    max_threads_per_block: int = 1024,
    max_threads_per_sm: int = 1536,
    shared_memory_per_block_optin: int = 101376,
    shared_memory_per_sm: int = 102400,
) -> DeviceProfile:
    return DeviceProfile(
        sm_count=sm_count,
        warp_size=32,
        max_threads_per_block=max_threads_per_block,
        max_threads_per_sm=max_threads_per_sm,
        shared_memory_per_block_optin=shared_memory_per_block_optin,
        shared_memory_per_sm=shared_memory_per_sm,
        capability_major=12,
        capability_minor=0,
        device_name="test-device",
        shared_memory_per_block=49152,
        regs_per_sm=65536,
        l2_cache_size=100663296,
        memory_bus_width=512,
        total_memory=34_190_458_880,
    )


def test_reference_keeps_zero_baseline_for_all_negative_logits() -> None:
    logits = torch.full((1, 3, 2), -2.0)
    mask = torch.ones((1, 3), dtype=torch.int32)

    scores, idx = _reference_from_logits(logits, mask)

    assert_close(scores, torch.zeros_like(scores))
    assert torch.equal(idx, torch.zeros_like(idx))


def test_reference_uses_strict_improvement_tie_policy() -> None:
    logits = torch.tensor(
        [[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 2.0, 1.0], [-1.0, 0.0, 3.0]]]
    )
    mask = torch.ones((1, 4), dtype=torch.int32)

    scores, idx = _reference_from_logits(logits, mask)

    assert_close(scores, torch.log1p(torch.tensor([[0.0, 2.0, 3.0]])))
    assert torch.equal(idx, torch.tensor([[0, 2, 3]]))


def test_reference_multiplies_mask_values_before_reduction() -> None:
    logits = torch.tensor([[[5.0, 4.0], [2.0, 3.0]]])
    partial_mask = torch.tensor([[0, 1]], dtype=torch.int32)
    zero_mask = torch.zeros((1, 2), dtype=torch.int32)

    partial_scores, partial_idx = _reference_from_logits(logits, partial_mask)
    zero_scores, zero_idx = _reference_from_logits(logits, zero_mask)

    assert_close(partial_scores, torch.log1p(torch.tensor([[2.0, 3.0]])))
    assert torch.equal(partial_idx, torch.tensor([[1, 1]]))
    assert_close(zero_scores, torch.zeros_like(zero_scores))
    assert torch.equal(zero_idx, torch.zeros_like(zero_idx))


def test_gluon_runtime_import_is_lazy() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys\n"
                "import sparton._gluon_runtime as gr\n"
                "print(any(k.startswith('triton.experimental.gluon') for k in sys.modules))\n"
                "print(gr._capability_mma_family((8, 0)))\n"
                "try:\n"
                "    gr._capability_mma_family((7, 5))\n"
                "except RuntimeError as exc:\n"
                "    print(type(exc).__name__)\n"
            ),
        ],
        check=True,
        env=env,
        cwd=os.getcwd(),
        text=True,
        capture_output=True,
    )

    lines = result.stdout.strip().splitlines()[-3:]
    assert lines == ["False", "mma_v2", "RuntimeError"]


def test_gluon_policy_generator_prunes_resource_constraints() -> None:
    device = _device_profile()
    problem = ProblemSpec(M=4096, N=30522, K=768, dtype_name="fp16")

    policies = generate_gluon_gemm_policies(problem, device)

    assert GluonGemmPolicy(128, 128, 64, 3, 4, 2, 128) in policies
    assert GluonGemmPolicy(128, 128, 32, 4, 4, 2, 64) in policies
    assert all(policy.block_n <= 128 for policy in policies)
    assert all(policy.block_k * problem.element_bytes >= policy.swizzle_byte_width for policy in policies)
    assert all(policy.shared_memory_bytes(problem.element_bytes) <= device.shared_memory_per_block_optin
               for policy in policies)

    with_block_n_256 = generate_gluon_gemm_policies(
        problem,
        device,
        include_block_n_256=True,
    )
    assert GluonGemmPolicy(128, 256, 32, 3, 4, 2, 64) in with_block_n_256


def test_bench_gluon_gemm_autotune_configs_follow_policy_generator() -> None:
    from types import SimpleNamespace

    from benchmarks.bench_gluon_gemm import (
        MAX_GEMM_AUTOTUNE_POLICIES,
        gemm_autotune_configs_for_policies,
        gemm_autotune_policy_universe,
        policies_for_args,
        policy_from_gemm_autotune_config,
        prune_gemm_autotune_configs,
    )
    from sparton._gluon_policy_runtime import (
        configs_for_policies,
        policy_from_config,
        prune_configs_for_policies,
    )

    args = SimpleNamespace(
        M=4096,
        N=30522,
        K=768,
        dtype="fp16",
        device_profile="rtx5090",
        include_block_n_256=True,
    )

    universe = gemm_autotune_policy_universe()
    configs = gemm_autotune_configs_for_policies(universe)
    shared_configs = configs_for_policies(universe)
    pruned = prune_gemm_autotune_configs(configs, args)
    active_policies = policies_for_args(args)
    shared_pruned = prune_configs_for_policies(configs, active_policies, universe)

    assert universe == gluon_gemm_policy_universe(include_block_n_256=True)
    assert len(universe) == MAX_GEMM_AUTOTUNE_POLICIES
    assert len(configs) == len(universe)
    assert [
        (
            config.kwargs,
            config.num_warps,
            config.num_stages,
        )
        for config in configs
    ] == [
        (
            config.kwargs,
            config.num_warps,
            config.num_stages,
        )
        for config in shared_configs
    ]
    assert [
        config.kwargs
        for config in configs
    ] == [
        policy_config_kwargs(policy, idx)
        for idx, policy in enumerate(universe)
    ]
    assert [config.kwargs["POLICY_ID"] for config in configs] == list(range(len(configs)))
    assert [policy_from_gemm_autotune_config(config, universe) for config in configs] == list(universe)
    assert [policy_from_config(config, universe) for config in configs] == list(universe)
    assert [policy_from_gemm_autotune_config(config, universe) for config in pruned] == list(active_policies)
    assert pruned == shared_pruned
    assert GluonGemmPolicy(128, 256, 32, 3, 4, 2, 64) in active_policies

    args.include_block_n_256 = False
    pruned_without_256 = prune_gemm_autotune_configs(configs, args)
    assert all(
        policy_from_gemm_autotune_config(config, universe).block_n <= 128
        for config in pruned_without_256
    )


def test_optimized_forward_runtime_policy_derivation() -> None:
    problem = ProblemSpec(M=4096, N=30522, K=768, dtype_name="fp16")
    universe = optimized_forward_policy_universe()
    fallback = optimized_forward_fallback_policy()
    device = _device_profile()

    policies = derive_optimized_forward_policies(problem, device)

    assert len(universe) == 11
    assert universe[0] == fallback
    assert GluonGemmPolicy(128, 256, 32, 3, 4, 2, 64) not in universe
    assert set(policies).issubset(set(universe))
    assert fallback in policies

    low_shared = _device_profile(
        shared_memory_per_block_optin=32_768,
        shared_memory_per_sm=32_768,
    )
    assert all(
        policy.shared_memory_bytes(problem.element_bytes) <= low_shared.shared_memory_per_block_limit
        for policy in derive_optimized_forward_policies(problem, low_shared)
    )

    low_threads = _device_profile(
        max_threads_per_block=64,
        max_threads_per_sm=64,
    )
    assert derive_optimized_forward_policies(problem, low_threads) == tuple()

    tiny = ProblemSpec(M=10, N=19, K=16, dtype_name="fp16")
    assert derive_optimized_forward_policies(tiny, device) == (fallback,)


@requires_cuda
@pytest.mark.cuda
def test_naive_forward_autotune_configs_include_fixed_baseline(sparton_kernel) -> None:
    from sparton._backend_naive_triton import get_naive_forward_configs

    configs = get_naive_forward_configs()
    config_keys = {
        (
            config.kwargs["BLOCK_S"],
            config.kwargs["BLOCK_V"],
            config.kwargs["BLOCK_D"],
            config.num_warps,
            config.num_stages,
        )
        for config in configs
    }

    assert len(config_keys) > 1
    assert (16, 32, 32, 4, 3) in config_keys
    assert len(config_keys) == len(configs)


@requires_optimized_gluon
@pytest.mark.cuda
@pytest.mark.optimized_gluon
def test_optimized_forward_autotune_configs_follow_policy_generator(sparton_kernel) -> None:
    import sparton._backend_optimized_gluon as optimized
    from sparton._backend_optimized_gluon import (
        get_optimized_forward_configs,
        get_optimized_forward_policies,
    )

    assert not hasattr(optimized, "_POLICY_BANK_DEVICE")
    assert not hasattr(optimized, "_POLICY_BANK_PROBLEM")

    device = _device_profile()
    problem = ProblemSpec(M=4096, N=30522, K=768, dtype_name="fp16")
    fallback = optimized_forward_fallback_policy()

    policies = get_optimized_forward_policies(problem, device)
    configs = get_optimized_forward_configs()
    config_keys = {
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_N"],
            config.kwargs["BLOCK_K"],
            config.kwargs["NUM_STAGES"],
            config.kwargs["WARPS_M"],
            config.kwargs["WARPS_N"],
            config.num_warps,
        )
        for config in configs
    }

    assert fallback in policies
    assert len(configs) > 1
    assert len({config.kwargs["POLICY_ID"] for config in configs}) == len(configs)
    assert (
        fallback.block_m,
        fallback.block_n,
        fallback.block_k,
        fallback.num_stages,
        fallback.warps_m,
        fallback.warps_n,
        fallback.num_warps,
    ) in config_keys
    assert set(policies).issubset(set(optimized_forward_policy_universe()))

    tiny = ProblemSpec(M=10, N=19, K=16, dtype_name="fp16")
    assert get_optimized_forward_policies(tiny, device) == (fallback,)


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize(("use_bias", "dtype"), FORWARD_CASES)
def test_fused_forward_matches_reference(
    sparton_kernel,
    cuda_device: torch.device,
    dtype: torch.dtype,
    use_bias: bool,
) -> None:
    _skip_if_unsupported_dtype(dtype)
    hidden, embed, bias, mask = _make_kernel_inputs(
        device=cuda_device,
        dtype=dtype,
        use_bias=use_bias,
    )

    scores, idx = sparton_kernel.fused_sparton_fwd_op(hidden, embed, bias, mask)
    expected_scores, expected_idx = sparton_reference(hidden, embed, bias, mask)

    assert_close(scores.float(), expected_scores.float(), **_score_tolerances(dtype))
    assert torch.equal(idx, expected_idx)


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize(("use_bias", "dtype"), FORWARD_CASES)
def test_naive_forward_matches_reference(
    sparton_kernel,
    cuda_device: torch.device,
    dtype: torch.dtype,
    use_bias: bool,
) -> None:
    _skip_if_unsupported_dtype(dtype)
    hidden, embed, bias, mask = _make_kernel_inputs(
        device=cuda_device,
        dtype=dtype,
        use_bias=use_bias,
    )

    scores, idx = sparton_kernel.naive_forward(hidden, embed, bias, mask)
    expected_scores, expected_idx = sparton_reference(hidden, embed, bias, mask)

    assert_close(scores.float(), expected_scores.float(), **_score_tolerances(dtype))
    assert torch.equal(idx, expected_idx)


@requires_optimized_gluon
@pytest.mark.cuda
@pytest.mark.optimized_gluon
@pytest.mark.parametrize(("use_bias", "dtype"), FORWARD_CASES)
def test_optimized_forward_matches_reference(
    sparton_kernel,
    cuda_device: torch.device,
    dtype: torch.dtype,
    use_bias: bool,
) -> None:
    _skip_if_unsupported_dtype(dtype)
    hidden, embed, bias, mask = _make_kernel_inputs(
        device=cuda_device,
        dtype=dtype,
        use_bias=use_bias,
    )

    scores, idx = sparton_kernel.optimized_forward(hidden, embed, bias, mask)
    expected_scores, expected_idx = sparton_reference(hidden, embed, bias, mask)

    assert_close(scores.float(), expected_scores.float(), **_score_tolerances(dtype))
    assert torch.equal(idx, expected_idx)


@requires_cuda
@pytest.mark.cuda
def test_naive_forward_semantic_cases(
    sparton_kernel,
    cuda_device: torch.device,
) -> None:
    hidden = torch.zeros((1, 4, 16), device=cuda_device, dtype=torch.float16)
    hidden[0, :, 0] = torch.tensor([-4.0, -3.0, -2.0, -1.0], device=cuda_device)
    hidden[0, :, 1] = torch.tensor([0.0, 2.0, 2.0, 1.0], device=cuda_device)
    hidden[0, :, 2] = torch.tensor([5.0, 4.0, 3.0, 2.0], device=cuda_device)
    hidden[0, :, 3] = torch.tensor([1.0, 3.0, 2.0, 4.0], device=cuda_device)
    embed = torch.zeros((32, 16), device=cuda_device, dtype=torch.float16)
    embed[0, 0] = 1.0  # all negative: baseline zero keeps index 0
    embed[1, 1] = 1.0  # strict tie between sequence positions 1 and 2
    embed[2, 2] = 1.0  # masked position 0 would otherwise win
    embed[3, 3] = 1.0
    mask = torch.tensor([[0, 1, 1, 0]], device=cuda_device, dtype=torch.int32)

    scores, idx = sparton_kernel.naive_forward(hidden, embed, None, mask)
    expected_scores, expected_idx = sparton_reference(hidden, embed, None, mask)

    assert_close(scores.float(), expected_scores.float(), atol=2e-3, rtol=2e-3)
    assert torch.equal(idx[:, :4], expected_idx[:, :4])
    assert torch.equal(idx[:, :4], torch.tensor([[0, 1, 1, 1]], device=cuda_device))


@requires_optimized_gluon
@pytest.mark.cuda
@pytest.mark.optimized_gluon
def test_optimized_forward_semantic_cases(
    sparton_kernel,
    cuda_device: torch.device,
) -> None:
    hidden = torch.zeros((1, 4, 16), device=cuda_device, dtype=torch.float16)
    hidden[0, :, 0] = torch.tensor([-4.0, -3.0, -2.0, -1.0], device=cuda_device)
    hidden[0, :, 1] = torch.tensor([0.0, 2.0, 2.0, 1.0], device=cuda_device)
    hidden[0, :, 2] = torch.tensor([5.0, 4.0, 3.0, 2.0], device=cuda_device)
    hidden[0, :, 3] = torch.tensor([1.0, 3.0, 2.0, 4.0], device=cuda_device)
    embed = torch.zeros((32, 16), device=cuda_device, dtype=torch.float16)
    embed[0, 0] = 1.0
    embed[1, 1] = 1.0
    embed[2, 2] = 1.0
    embed[3, 3] = 1.0
    mask = torch.tensor([[0, 1, 1, 0]], device=cuda_device, dtype=torch.int32)

    scores, idx = sparton_kernel.optimized_forward(hidden, embed, None, mask)
    expected_scores, expected_idx = sparton_reference(hidden, embed, None, mask)

    assert_close(scores.float(), expected_scores.float(), atol=2e-3, rtol=2e-3)
    assert torch.equal(idx[:, :4], expected_idx[:, :4])
    assert torch.equal(idx[:, :4], torch.tensor([[0, 1, 1, 1]], device=cuda_device))


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize(("use_bias", "dtype"), FORWARD_CASES)
def test_naive_forward_handles_tails(
    sparton_kernel,
    cuda_device: torch.device,
    dtype: torch.dtype,
    use_bias: bool,
) -> None:
    _skip_if_unsupported_dtype(dtype)
    B, S, D, V = 3, 17, 48, 129
    hidden = torch.zeros((B, S, D), device=cuda_device, dtype=dtype)
    embed = torch.zeros((V, D), device=cuda_device, dtype=dtype)
    seq = torch.arange(1, S + 1, device=cuda_device, dtype=torch.float32)
    vocab = torch.arange(V, device=cuda_device)
    hidden[:, :, 0] = (seq / 16).to(dtype)
    hidden[:, :, 1] = ((S + 1 - seq) / 32).to(dtype)
    embed[:, 0] = torch.where(vocab % 5 == 0, -0.25, 0.25).to(dtype)
    embed[:, 1] = torch.where(vocab % 7 == 0, 0.125, -0.0625).to(dtype)
    bias = None
    if use_bias:
        bias = (((vocab % 11).to(torch.float32) - 5.0) / 64).to(dtype)
    mask = torch.tensor(
        [
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        ],
        device=cuda_device,
        dtype=torch.int32,
    )

    scores, idx = sparton_kernel.naive_forward(hidden, embed, bias, mask)
    expected_scores, expected_idx = sparton_reference(hidden, embed, bias, mask)

    assert_close(scores.float(), expected_scores.float(), **_score_tolerances(dtype))
    assert torch.equal(idx, expected_idx)


@requires_optimized_gluon
@pytest.mark.cuda
@pytest.mark.optimized_gluon
@pytest.mark.parametrize(("use_bias", "dtype"), FORWARD_CASES)
def test_optimized_forward_handles_tails(
    sparton_kernel,
    cuda_device: torch.device,
    dtype: torch.dtype,
    use_bias: bool,
) -> None:
    _skip_if_unsupported_dtype(dtype)
    B, S, D, V = 3, 17, 48, 129
    hidden = torch.zeros((B, S, D), device=cuda_device, dtype=dtype)
    embed = torch.zeros((V, D), device=cuda_device, dtype=dtype)
    seq = torch.arange(1, S + 1, device=cuda_device, dtype=torch.float32)
    vocab = torch.arange(V, device=cuda_device)
    hidden[:, :, 0] = (seq / 16).to(dtype)
    hidden[:, :, 1] = ((S + 1 - seq) / 32).to(dtype)
    embed[:, 0] = torch.where(vocab % 5 == 0, -0.25, 0.25).to(dtype)
    embed[:, 1] = torch.where(vocab % 7 == 0, 0.125, -0.0625).to(dtype)
    bias = None
    if use_bias:
        bias = (((vocab % 11).to(torch.float32) - 5.0) / 64).to(dtype)
    mask = torch.tensor(
        [
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        ],
        device=cuda_device,
        dtype=torch.int32,
    )

    scores, idx = sparton_kernel.optimized_forward(hidden, embed, bias, mask)
    expected_scores, expected_idx = sparton_reference(hidden, embed, bias, mask)

    assert_close(scores.float(), expected_scores.float(), **_score_tolerances(dtype))
    assert torch.equal(idx, expected_idx)


@requires_optimized_gluon
@pytest.mark.cuda
@pytest.mark.optimized_gluon
def test_optimized_forward_handles_multiple_sequence_chunks_small_d(
    sparton_kernel,
    cuda_device: torch.device,
) -> None:
    B, S, D, V = 1, 129, 16, 32
    generator = torch.Generator(device=cuda_device).manual_seed(53)
    hidden = torch.randn(
        (B, S, D),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    ) * 0.05
    embed = torch.randn(
        (V, D),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    ) * 0.05
    bias = torch.randn((V,), device=cuda_device, dtype=torch.float16, generator=generator) * 0.05
    mask = (torch.rand((B, S), device=cuda_device, generator=generator) > 0.25).to(torch.int32)

    scores, idx = sparton_kernel.optimized_forward(hidden, embed, bias, mask)
    expected_scores, expected_idx = sparton_reference(hidden, embed, bias, mask)

    assert_close(scores.float(), expected_scores.float(), atol=2e-3, rtol=2e-3)
    assert torch.equal(idx, expected_idx)


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no_bias"])
def test_fused_backward_matches_reference(
    sparton_kernel,
    cuda_device: torch.device,
    use_bias: bool,
) -> None:
    hidden, embed, bias, mask = _make_kernel_inputs(
        device=cuda_device,
        dtype=torch.float16,
        use_bias=use_bias,
    )
    ref_hidden = _clone_leaf(hidden)
    ref_embed = _clone_leaf(embed)
    ref_bias = _clone_leaf(bias)

    scores, _ = sparton_kernel.fused_sparton_fwd_op(hidden, embed, bias, mask)
    expected_scores, _ = sparton_reference(ref_hidden, ref_embed, ref_bias, mask)
    upstream = torch.randn_like(scores)

    scores.backward(upstream)
    expected_scores.backward(upstream.detach().clone())

    assert_close(hidden.grad.float(), ref_hidden.grad.float(), atol=2e-3, rtol=2e-3)
    assert_close(embed.grad.float(), ref_embed.grad.float(), atol=2e-3, rtol=2e-3)
    if use_bias:
        assert bias is not None
        assert ref_bias is not None
        assert_close(bias.grad.float(), ref_bias.grad.float(), atol=2e-3, rtol=2e-3)
    else:
        assert bias is None


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no_bias"])
def test_naive_backward_matches_reference(
    sparton_kernel,
    cuda_device: torch.device,
    use_bias: bool,
) -> None:
    hidden, embed, bias, mask = _make_kernel_inputs(
        device=cuda_device,
        dtype=torch.float16,
        use_bias=use_bias,
    )
    ref_hidden = _clone_leaf(hidden)
    ref_embed = _clone_leaf(embed)
    ref_bias = _clone_leaf(bias)

    scores, _ = sparton_kernel.naive_forward(hidden, embed, bias, mask)
    expected_scores, _ = sparton_reference(ref_hidden, ref_embed, ref_bias, mask)
    upstream = torch.randn_like(scores)

    scores.backward(upstream)
    expected_scores.backward(upstream.detach().clone())

    assert_close(hidden.grad.float(), ref_hidden.grad.float(), atol=2e-3, rtol=2e-3)
    assert_close(embed.grad.float(), ref_embed.grad.float(), atol=2e-3, rtol=2e-3)
    if use_bias:
        assert bias is not None
        assert ref_bias is not None
        assert_close(bias.grad.float(), ref_bias.grad.float(), atol=2e-3, rtol=2e-3)
    else:
        assert bias is None


@requires_optimized_gluon
@pytest.mark.cuda
@pytest.mark.optimized_gluon
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no_bias"])
def test_optimized_backward_matches_reference(
    sparton_kernel,
    cuda_device: torch.device,
    use_bias: bool,
) -> None:
    hidden, embed, bias, mask = _make_kernel_inputs(
        device=cuda_device,
        dtype=torch.float16,
        use_bias=use_bias,
    )
    ref_hidden = _clone_leaf(hidden)
    ref_embed = _clone_leaf(embed)
    ref_bias = _clone_leaf(bias)

    scores, _ = sparton_kernel.optimized_forward(hidden, embed, bias, mask)
    expected_scores, _ = sparton_reference(ref_hidden, ref_embed, ref_bias, mask)
    upstream = torch.randn_like(scores)

    scores.backward(upstream)
    expected_scores.backward(upstream.detach().clone())

    assert_close(hidden.grad.float(), ref_hidden.grad.float(), atol=2e-3, rtol=2e-3)
    assert_close(embed.grad.float(), ref_embed.grad.float(), atol=2e-3, rtol=2e-3)
    if use_bias:
        assert bias is not None
        assert ref_bias is not None
        assert_close(bias.grad.float(), ref_bias.grad.float(), atol=2e-3, rtol=2e-3)
    else:
        assert bias is None


@requires_cuda
@pytest.mark.cuda
def test_sparton_head_backend_routing(
    sparton_kernel,
    cuda_device: torch.device,
) -> None:
    hidden, embed, bias, mask = _make_kernel_inputs(
        device=cuda_device,
        dtype=torch.float16,
        use_bias=True,
    )

    default_head = sparton_kernel.SpartonHead(19, 16, use_bias=True).to(
        device=cuda_device,
        dtype=torch.float16,
    )
    hybrid_head = sparton_kernel.SpartonHead(19, 16, use_bias=True, backend="hybrid").to(
        device=cuda_device,
        dtype=torch.float16,
    )
    naive_head = sparton_kernel.SpartonHead(19, 16, use_bias=True, backend="naive").to(
        device=cuda_device,
        dtype=torch.float16,
    )
    with torch.no_grad():
        for head in (default_head, hybrid_head, naive_head):
            head.weight.copy_(embed)
            assert head.bias is not None
            head.bias.copy_(bias)

    expected_scores, _ = sparton_reference(hidden, embed, bias, mask)

    assert default_head.backend == "hybrid"
    assert hybrid_head.backend == "hybrid"
    assert naive_head.backend == "naive"
    assert_close(default_head(hidden, mask).float(), expected_scores.float(), atol=2e-3, rtol=2e-3)
    assert_close(hybrid_head(hidden, mask).float(), expected_scores.float(), atol=2e-3, rtol=2e-3)
    assert_close(naive_head(hidden, mask).float(), expected_scores.float(), atol=2e-3, rtol=2e-3)

    with pytest.raises(ValueError, match="Unknown Sparton backend"):
        sparton_kernel.SpartonHead(19, 16, backend="missing")


@requires_optimized_gluon
@pytest.mark.cuda
@pytest.mark.optimized_gluon
def test_sparton_head_optimized_backend_routing(
    sparton_kernel,
    cuda_device: torch.device,
) -> None:
    hidden, embed, bias, mask = _make_kernel_inputs(
        device=cuda_device,
        dtype=torch.float16,
        use_bias=True,
    )

    optimized_head = sparton_kernel.SpartonHead(19, 16, use_bias=True, backend="optimized").to(
        device=cuda_device,
        dtype=torch.float16,
    )
    with torch.no_grad():
        optimized_head.weight.copy_(embed)
        assert optimized_head.bias is not None
        optimized_head.bias.copy_(bias)

    expected_scores, _ = sparton_reference(hidden, embed, bias, mask)

    assert optimized_head.backend == "optimized"
    assert_close(optimized_head(hidden, mask).float(), expected_scores.float(), atol=2e-3, rtol=2e-3)


@requires_cuda
@pytest.mark.cuda
def test_sparton_backend_env_selects_default(cuda_device: torch.device) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["SPARTON_BACKEND"] = "naive"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sparton.sparton_kernel as sk; "
                "head = sk.SpartonHead(19, 16); "
                "print(head.backend)"
            ),
        ],
        check=True,
        env=env,
        cwd=os.getcwd(),
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip().splitlines()[-1] == "naive"


@requires_optimized_gluon
@pytest.mark.cuda
@pytest.mark.optimized_gluon
def test_sparton_backend_env_selects_optimized_default(cuda_device: torch.device) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    env["SPARTON_BACKEND"] = "optimized"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sparton.sparton_kernel as sk; "
                "head = sk.SpartonHead(19, 16); "
                "print(head.backend)"
            ),
        ],
        check=True,
        env=env,
        cwd=os.getcwd(),
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip().splitlines()[-1] == "optimized"


@requires_cuda
@pytest.mark.cuda
def test_sparton_head_no_bias_backward_smoke(
    sparton_kernel,
    cuda_device: torch.device,
) -> None:
    head = sparton_kernel.SpartonHead(19, 16, use_bias=False).to(
        device=cuda_device,
        dtype=torch.float16,
    )
    hidden = torch.randn(
        (2, 5, 16),
        device=cuda_device,
        dtype=torch.float16,
        requires_grad=True,
    )
    mask = torch.tensor(
        [[1, 1, 0, 1, 0], [0, 1, 1, 0, 1]],
        device=cuda_device,
        dtype=torch.int32,
    )

    head(hidden, mask).sum().backward()

    assert head.bias is None
    assert hidden.grad is not None
    assert head.weight.grad is not None


@requires_cuda
@pytest.mark.cuda
def test_naive_forward_does_not_materialize_logits(
    sparton_kernel,
    cuda_device: torch.device,
) -> None:
    B, S, D, V = 4, 64, 64, 4096
    generator = torch.Generator(device=cuda_device).manual_seed(31)
    hidden = torch.randn(
        (B, S, D),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    )
    embed = torch.randn(
        (V, D),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    )
    bias = torch.randn((V,), device=cuda_device, dtype=torch.float16, generator=generator)
    mask = (torch.rand((B, S), device=cuda_device, generator=generator) > 0.25).to(torch.int32)

    sparton_kernel.naive_forward(hidden, embed, bias, mask)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    scores, idx = sparton_kernel.naive_forward(hidden, embed, bias, mask)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    output_bytes = scores.numel() * scores.element_size() + idx.numel() * idx.element_size()
    logits_bytes = B * S * V * hidden.element_size()
    peak_extra = peak - base
    assert output_bytes <= peak_extra <= output_bytes + 1024 * 1024
    assert peak_extra < logits_bytes


@requires_optimized_gluon
@pytest.mark.cuda
@pytest.mark.optimized_gluon
def test_optimized_forward_does_not_materialize_logits(
    sparton_kernel,
    cuda_device: torch.device,
) -> None:
    B, S, D, V = 4, 64, 64, 4096
    generator = torch.Generator(device=cuda_device).manual_seed(41)
    hidden = torch.randn(
        (B, S, D),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    )
    embed = torch.randn(
        (V, D),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    )
    bias = torch.randn((V,), device=cuda_device, dtype=torch.float16, generator=generator)
    mask = (torch.rand((B, S), device=cuda_device, generator=generator) > 0.25).to(torch.int32)

    sparton_kernel.optimized_forward(hidden, embed, bias, mask)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    scores, idx = sparton_kernel.optimized_forward(hidden, embed, bias, mask)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    output_bytes = scores.numel() * scores.element_size() + idx.numel() * idx.element_size()
    logits_bytes = B * S * V * hidden.element_size()
    peak_extra = peak - base
    assert output_bytes <= peak_extra <= output_bytes * 2 + 1024 * 1024
    assert peak_extra < logits_bytes


@requires_cuda
@pytest.mark.cuda
def test_custom_op_schemas_expose_optional_bias(sparton_kernel) -> None:
    fwd_schema = str(torch.ops.sparton.fused_sparton_fwd.default._schema)
    bwd_schema = str(torch.ops.sparton.fused_sparton_bwd.default._schema)
    naive_schema = str(torch.ops.sparton.naive_fwd.default._schema)

    assert "Tensor? bias" in fwd_schema
    assert "Tensor? bias" in bwd_schema
    assert "Tensor? bias" in naive_schema
    assert "Tensor?)" in bwd_schema


@requires_optimized_gluon
@pytest.mark.cuda
@pytest.mark.optimized_gluon
def test_optimized_custom_op_schema_exposes_optional_bias(sparton_kernel) -> None:
    _optimized_op = sparton_kernel.optimized_fwd_op
    optimized_schema = str(torch.ops.sparton.optimized_fwd.default._schema)

    assert "Tensor? bias" in optimized_schema
