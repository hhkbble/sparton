from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Optional

import pytest
import torch
from torch.testing import assert_close


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_PATH = str(_REPO_ROOT / "src")

CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="Sparton CUDA kernel tests require a CUDA device",
)
requires_optimized = pytest.mark.usefixtures("optimized_available")

_OPTIMIZED_AVAILABILITY: tuple[bool, str] | None = None


def _optimized_availability() -> tuple[bool, str]:
    global _OPTIMIZED_AVAILABILITY
    if _OPTIMIZED_AVAILABILITY is not None:
        return _OPTIMIZED_AVAILABILITY
    if not CUDA_AVAILABLE:
        _OPTIMIZED_AVAILABILITY = (
            False,
            "Sparton optimized backend tests require a CUDA device",
        )
        return _OPTIMIZED_AVAILABILITY

    from sparton._backend_runtime import is_optimized_backend_available

    available, reason = is_optimized_backend_available()
    _OPTIMIZED_AVAILABILITY = (
        available,
        reason if available else f"Sparton optimized backend is unavailable: {reason}",
    )
    return _OPTIMIZED_AVAILABILITY


@pytest.fixture
def optimized_available() -> None:
    available, reason = _optimized_availability()
    if not available:
        pytest.skip(reason)


@pytest.fixture(autouse=True)
def _optimized_autotune_off(monkeypatch) -> None:
    # The optimized self-tuner's correctness is tile-independent, so default it OFF for the
    # suite (fast analytic tile). The dedicated tuner test re-enables it.
    monkeypatch.setenv("SPARTON_OPTIMIZED_AUTOTUNE", "off")


ALL_BACKENDS = ("hybrid", "naive", "optimized")


def _forward_for_backend(sparton_kernel, backend: str):
    # `optimized` (pure-Triton, host-side TMA) is gated on the sm_90+ availability probe.
    if backend == "optimized":
        available, reason = _optimized_availability()
        if not available:
            pytest.skip(reason)
    return getattr(sparton_kernel, f"{backend}_forward")

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


def reference_masked_logits(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    logits = hidden @ embed.T
    if bias is not None:
        logits = logits + bias
    return logits * mask.to(dtype=logits.dtype)[:, :, None]


def assert_index_contract(
    scores: torch.Tensor,
    idx: torch.Tensor,
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: Optional[torch.Tensor],
    mask: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    """Index contract of record (ARCHITECTURE.md §3.2).

    Wherever the returned score is positive, the chosen sequence index must
    hold a masked input-dtype logit within tolerance of the per-(b, v)
    maximum. Positions with score == 0 carry no index meaning (zero-baseline
    policy). Backends with different accumulation precision may legitimately
    pick different near-tie winners, so exact index equality is asserted only
    in deterministic constructed cases.
    """

    masked = reference_masked_logits(hidden, embed, bias, mask)
    ref_max = masked.max(dim=1).values
    chosen = masked.gather(1, idx.unsqueeze(1)).squeeze(1)
    active = scores.float() > 0
    gap = (ref_max.float() - chosen.float())[active]
    limit = atol + rtol * ref_max.float()[active].abs()
    assert bool((gap <= limit).all()), (
        "index contract violated: chosen logit trails the reference max by "
        f"{gap.max().item():.6f} (tolerance atol={atol}, rtol={rtol})"
    )


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


def _grad_tolerances(dtype: torch.dtype) -> dict[str, float]:
    # Aligned with _score_tolerances: bf16 inputs round the products feeding
    # the fp32 accumulators, so gradients carry the wider input tolerance.
    if dtype is torch.bfloat16:
        return {"atol": 5e-2, "rtol": 5e-2}
    return {"atol": 2e-3, "rtol": 2e-3}


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

    scores, idx = sparton_kernel.hybrid_forward(hidden, embed, bias, mask)
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
    expected_scores, _expected_idx = sparton_reference(hidden, embed, bias, mask)

    assert_close(scores.float(), expected_scores.float(), **_score_tolerances(dtype))
    assert_index_contract(scores, idx, hidden, embed, bias, mask, **_score_tolerances(dtype))


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
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
    expected_scores, _expected_idx = sparton_reference(hidden, embed, bias, mask)

    assert_close(scores.float(), expected_scores.float(), **_score_tolerances(dtype))
    assert_index_contract(scores, idx, hidden, embed, bias, mask, **_score_tolerances(dtype))


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


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
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


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
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


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
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
    expected_scores, _expected_idx = sparton_reference(hidden, embed, bias, mask)

    assert_close(scores.float(), expected_scores.float(), atol=2e-3, rtol=2e-3)
    assert_index_contract(scores, idx, hidden, embed, bias, mask, atol=2e-3, rtol=2e-3)


NONTINY_FORWARD_CASES = [
    pytest.param(8, 128, 768, 1283, True, torch.float16, id="8x128x768x1283-bias-fp16"),
    pytest.param(8, 128, 768, 1283, False, torch.float16, id="8x128x768x1283-no_bias-fp16"),
    pytest.param(3, 345, 768, 2048, True, torch.float16, id="3x345x768x2048-bias-fp16"),
    pytest.param(3, 345, 768, 2048, True, torch.bfloat16, id="3x345x768x2048-bias-bf16"),
    pytest.param(9, 120, 64, 1024, True, torch.float16, id="9x120x64x1024-bias-fp16"),
    pytest.param(2, 513, 1024, 1536, False, torch.float16, id="2x513x1024x1536-no_bias-fp16"),
]


def _make_nontiny_inputs(
    device: torch.device,
    B: int,
    S: int,
    D: int,
    V: int,
    dtype: torch.dtype,
    use_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(101 + V)
    hidden = torch.randn((B, S, D), device=device, dtype=dtype, generator=generator) * 0.05
    embed = torch.randn((V, D), device=device, dtype=dtype, generator=generator) * 0.05
    bias = None
    if use_bias:
        bias = torch.randn((V,), device=device, dtype=dtype, generator=generator) * 0.05
    mask = (torch.rand((B, S), device=device, generator=generator) > 0.25).to(torch.int32)
    return hidden, embed, bias, mask


@requires_cuda
@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize(("B", "S", "D", "V", "use_bias", "dtype"), NONTINY_FORWARD_CASES)
def test_naive_forward_nontiny_shapes(
    sparton_kernel,
    cuda_device: torch.device,
    B: int,
    S: int,
    D: int,
    V: int,
    use_bias: bool,
    dtype: torch.dtype,
) -> None:
    _skip_if_unsupported_dtype(dtype)
    hidden, embed, bias, mask = _make_nontiny_inputs(cuda_device, B, S, D, V, dtype, use_bias)

    scores, idx = sparton_kernel.naive_forward(hidden, embed, bias, mask)
    expected_scores, _ = sparton_reference(hidden, embed, bias, mask)

    assert_close(scores.float(), expected_scores.float(), **_score_tolerances(dtype))
    assert_index_contract(scores, idx, hidden, embed, bias, mask, **_score_tolerances(dtype))


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
def test_optimized_forward_persistent_multitile(
    sparton_kernel,
    cuda_device: torch.device,
    monkeypatch,
) -> None:
    # Landmine guard: force one persistent CTA to process MANY (b, vocab-tile) tiles
    # (NUM_CTAS=1) so a missing per-tile running-state reset would leak a prior tile's
    # max. Multiple s-tiles per tile (S > BLOCK_M) and a V-tail exercise the full loop.
    monkeypatch.setenv("SPARTON_OPTIMIZED_NUM_CTAS", "1")
    B, S, D, V = 2, 130, 64, 300  # 2 * ceil(300/64)=10 tiles, one CTA
    generator = torch.Generator(device=cuda_device).manual_seed(7)
    hidden = torch.randn((B, S, D), device=cuda_device, dtype=torch.float16, generator=generator) * 0.3
    embed = torch.randn((V, D), device=cuda_device, dtype=torch.float16, generator=generator) * 0.3
    bias = torch.randn((V,), device=cuda_device, dtype=torch.float16, generator=generator) * 0.1
    mask = (torch.rand((B, S), device=cuda_device, generator=generator) > 0.25).to(torch.int32)

    scores, idx = sparton_kernel.optimized_forward(hidden, embed, bias, mask)
    expected_scores, _ = sparton_reference(hidden, embed, bias, mask)

    assert_close(scores.float(), expected_scores.float(), **_score_tolerances(torch.float16))
    assert_index_contract(scores, idx, hidden, embed, bias, mask, **_score_tolerances(torch.float16))


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
@pytest.mark.parametrize("force_one_cta", [False, True])
def test_optimized_forward_warp_specialize_matches_homogeneous(
    sparton_kernel,
    cuda_device: torch.device,
    monkeypatch,
    force_one_cta: bool,
) -> None:
    # The warp-specialized epilogue (SPARTON_OPTIMIZED_WARP_SPECIALIZE, with autotune off)
    # must produce the SAME result as the homogeneous baseline. The WS path computes the
    # strict-tie argmax via tl.max + masked tl.min (two single-result reduces — required
    # because Triton's auto-WS pass rejects the 2-result combine); the homogeneous path uses
    # the combine. They are bit-identical by construction. Covers a partially-masked row,
    # S/V tails, bias, and the persistent NUM_CTAS=1 multi-tile reset (landmine #1).
    B, S, D, V = 2, 130, 64, 300
    generator = torch.Generator(device=cuda_device).manual_seed(5)
    hidden = torch.randn((B, S, D), device=cuda_device, dtype=torch.float16, generator=generator) * 0.3
    embed = torch.randn((V, D), device=cuda_device, dtype=torch.float16, generator=generator) * 0.3
    bias = torch.randn((V,), device=cuda_device, dtype=torch.float16, generator=generator) * 0.1
    mask = (torch.rand((B, S), device=cuda_device, generator=generator) > 0.25).to(torch.int32)
    mask[0, :50] = 0  # a partially-masked leading run

    if force_one_cta:
        monkeypatch.setenv("SPARTON_OPTIMIZED_NUM_CTAS", "1")

    monkeypatch.delenv("SPARTON_OPTIMIZED_WARP_SPECIALIZE", raising=False)
    base_scores, base_idx = sparton_kernel.optimized_forward(hidden, embed, bias, mask)
    base_scores, base_idx = base_scores.clone(), base_idx.clone()
    monkeypatch.setenv("SPARTON_OPTIMIZED_WARP_SPECIALIZE", "on")
    ws_scores, ws_idx = sparton_kernel.optimized_forward(hidden, embed, bias, mask)

    expected_scores, _ = sparton_reference(hidden, embed, bias, mask)
    assert_close(ws_scores.float(), expected_scores.float(), **_score_tolerances(torch.float16))
    assert_index_contract(ws_scores, ws_idx, hidden, embed, bias, mask, **_score_tolerances(torch.float16))
    # max + masked-min == the combine by construction -> bitwise-identical.
    assert torch.equal(ws_scores, base_scores), "warp-specialized scores differ from homogeneous"
    assert torch.equal(ws_idx, base_idx), "warp-specialized indices differ from homogeneous"


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
@pytest.mark.slow
def test_optimized_forward_autotune_selects_valid_cached_tile(
    sparton_kernel,
    cuda_device: torch.device,
    monkeypatch,
) -> None:
    # The self-tuner (SPARTON_OPTIMIZED_AUTOTUNE=on, the production default) measures the
    # candidate tiles and caches the winner keyed on (D, V, dtype, arch) — NOT B/S. Verify it
    # runs, returns a correct result, caches exactly one entry, and that a different (B, S)
    # with the same (D, V) reuses that entry (B/S excluded from the key).
    from sparton._backend_optimized import _TILE_CACHE, clear_tile_cache

    monkeypatch.setenv("SPARTON_OPTIMIZED_AUTOTUNE", "on")
    clear_tile_cache()
    B, S, D, V = 2, 64, 64, 512
    generator = torch.Generator(device=cuda_device).manual_seed(3)
    hidden = torch.randn((B, S, D), device=cuda_device, dtype=torch.float16, generator=generator) * 0.3
    embed = torch.randn((V, D), device=cuda_device, dtype=torch.float16, generator=generator) * 0.3
    bias = torch.randn((V,), device=cuda_device, dtype=torch.float16, generator=generator) * 0.1
    mask = (torch.rand((B, S), device=cuda_device, generator=generator) > 0.25).to(torch.int32)

    try:
        scores, idx = sparton_kernel.optimized_forward(hidden, embed, bias, mask)
        expected_scores, _ = sparton_reference(hidden, embed, bias, mask)
        assert_close(scores.float(), expected_scores.float(), **_score_tolerances(torch.float16))
        assert_index_contract(scores, idx, hidden, embed, bias, mask, **_score_tolerances(torch.float16))

        assert len(_TILE_CACHE) == 1, "self-tuner must cache exactly one (D,V,dtype,arch) entry"
        (key,) = list(_TILE_CACHE)
        policy, warp_specialize = _TILE_CACHE[key]
        assert key[0] == D and key[1] == V and key[2] == torch.float16
        assert policy.block_m > 0 and policy.block_n > 0 and policy.block_k > 0
        assert isinstance(warp_specialize, bool)

        # Different (B, S), same (D, V): must reuse the cached tile (key excludes B/S).
        h2 = torch.randn((1, 128, D), device=cuda_device, dtype=torch.float16, generator=generator) * 0.3
        m2 = (torch.rand((1, 128), device=cuda_device, generator=generator) > 0.25).to(torch.int32)
        sparton_kernel.optimized_forward(h2, embed, bias, m2)
        assert len(_TILE_CACHE) == 1, "a different (B,S) must not add a cache entry"
    finally:
        clear_tile_cache()


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
@pytest.mark.slow
@pytest.mark.parametrize(("B", "S", "D", "V", "use_bias", "dtype"), NONTINY_FORWARD_CASES)
def test_optimized_forward_nontiny_shapes(
    sparton_kernel,
    cuda_device: torch.device,
    B: int,
    S: int,
    D: int,
    V: int,
    use_bias: bool,
    dtype: torch.dtype,
) -> None:
    _skip_if_unsupported_dtype(dtype)
    # With autotune off (suite default) the backend uses its own self-contained analytic tile;
    # assert it resolves and that the autotune sweep has candidates, then check values + index.
    from sparton._backend_optimized import _analytic_tile, _device_limits, _valid_tiles

    limits = _device_limits()
    tile = _analytic_tile(2, limits)
    assert tile.block_m > 0 and tile.block_n > 0 and tile.block_k > 0
    assert _valid_tiles(2, limits), "the measured-autotune sweep must have at least one valid tile"

    hidden, embed, bias, mask = _make_nontiny_inputs(cuda_device, B, S, D, V, dtype, use_bias)

    scores, idx = sparton_kernel.optimized_forward(hidden, embed, bias, mask)
    expected_scores, _ = sparton_reference(hidden, embed, bias, mask)

    assert_close(scores.float(), expected_scores.float(), **_score_tolerances(dtype))
    assert_index_contract(scores, idx, hidden, embed, bias, mask, **_score_tolerances(dtype))


BACKWARD_CASES = [
    pytest.param(True, torch.float16, id="bias-fp16"),
    pytest.param(False, torch.float16, id="no_bias-fp16"),
    pytest.param(True, torch.bfloat16, id="bias-bf16"),
    pytest.param(False, torch.bfloat16, id="no_bias-bf16"),
]


def _assert_backward_matches_reference(
    sparton_kernel,
    cuda_device: torch.device,
    forward,
    use_bias: bool,
    dtype: torch.dtype,
) -> None:
    _skip_if_unsupported_dtype(dtype)
    hidden, embed, bias, mask = _make_kernel_inputs(
        device=cuda_device,
        dtype=dtype,
        use_bias=use_bias,
    )
    ref_hidden = _clone_leaf(hidden)
    ref_embed = _clone_leaf(embed)
    ref_bias = _clone_leaf(bias)

    scores, _ = forward(hidden, embed, bias, mask)
    expected_scores, _ = sparton_reference(ref_hidden, ref_embed, ref_bias, mask)
    upstream = torch.randn_like(scores)

    scores.backward(upstream)
    expected_scores.backward(upstream.detach().clone())

    tol = _grad_tolerances(dtype)
    assert_close(hidden.grad.float(), ref_hidden.grad.float(), **tol)
    assert_close(embed.grad.float(), ref_embed.grad.float(), **tol)
    if use_bias:
        assert bias is not None
        assert ref_bias is not None
        assert_close(bias.grad.float(), ref_bias.grad.float(), **tol)
    else:
        assert bias is None


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize(("use_bias", "dtype"), BACKWARD_CASES)
def test_fused_backward_matches_reference(
    sparton_kernel,
    cuda_device: torch.device,
    use_bias: bool,
    dtype: torch.dtype,
) -> None:
    _assert_backward_matches_reference(
        sparton_kernel, cuda_device, sparton_kernel.fused_sparton_fwd_op, use_bias, dtype
    )


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize(("use_bias", "dtype"), BACKWARD_CASES)
def test_naive_backward_matches_reference(
    sparton_kernel,
    cuda_device: torch.device,
    use_bias: bool,
    dtype: torch.dtype,
) -> None:
    _assert_backward_matches_reference(
        sparton_kernel, cuda_device, sparton_kernel.naive_forward, use_bias, dtype
    )


NONTINY_BACKWARD_CASES = [
    pytest.param(8, 128, 768, 1283, True, torch.float16, id="8x128x768x1283-bias-fp16"),
    pytest.param(3, 345, 768, 2048, True, torch.bfloat16, id="3x345x768x2048-bias-bf16"),
    pytest.param(2, 513, 1024, 1536, False, torch.float16, id="2x513x1024x1536-no_bias-fp16"),
]


@requires_cuda
@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize(("B", "S", "D", "V", "use_bias", "dtype"), NONTINY_BACKWARD_CASES)
def test_fused_backward_nontiny_shapes(
    sparton_kernel,
    cuda_device: torch.device,
    B: int,
    S: int,
    D: int,
    V: int,
    use_bias: bool,
    dtype: torch.dtype,
) -> None:
    """Backward gradients vs a closed-form reference at non-tiny shapes.

    Tiny shapes exercise only single-tile/single-chunk paths; these cases
    cross tile and chunk boundaries in every grid dimension. The expectation
    is computed from the kernel's own saved (scores, idx): the backward's
    contract is conditional on the forward's saved tensors, and at random
    non-tiny shapes the kernel and the PyTorch reference forward may
    legitimately pick different near-tie winners (index contract of record),
    which would re-route individual gradient elements and fail any direct
    autograd-vs-autograd comparison. Masked sequence positions can never win
    the max, so their hidden gradient must be exactly zero — asserted as a
    structural invariant, not within tolerance.
    """

    _skip_if_unsupported_dtype(dtype)
    hidden, embed, bias, mask = _make_nontiny_inputs(cuda_device, B, S, D, V, dtype, use_bias)
    hidden.requires_grad_(True)
    embed.requires_grad_(True)
    if bias is not None:
        bias.requires_grad_(True)

    scores, idx = sparton_kernel.fused_sparton_fwd_op(hidden, embed, bias, mask)

    # Pin the forward outputs the expectation conditions on: score values
    # against the reference (values are not tie-ambiguous) and indices via
    # the tie-aware contract — otherwise a forward bug at these shapes would
    # propagate identically into the closed-form expectation and pass.
    with torch.no_grad():
        ref_scores, _ = sparton_reference(
            hidden.detach(), embed.detach(),
            None if bias is None else bias.detach(), mask,
        )
        score_tol = _score_tolerances(dtype)
        assert_close(scores.float(), ref_scores.float(), **score_tol)
        assert_index_contract(
            scores, idx, hidden.detach(), embed.detach(),
            None if bias is None else bias.detach(), mask, **score_tol,
        )

    upstream = torch.randn_like(scores)
    scores.backward(upstream)

    with torch.no_grad():
        g = torch.where(
            scores.float() > 0,
            upstream.float() * torch.exp(-scores.float()),
            torch.zeros((), device=cuda_device, dtype=torch.float32),
        )
        gathered = hidden.detach().float()[
            torch.arange(B, device=cuda_device)[:, None], idx
        ]
        expected_embed_grad = torch.einsum("bv,bvd->vd", g, gathered)
        contrib = g[:, :, None] * embed.detach().float()[None, :, :]
        expected_hidden_grad = torch.zeros(
            (B, S, D), device=cuda_device, dtype=torch.float32
        )
        for b in range(B):
            expected_hidden_grad[b].index_add_(0, idx[b], contrib[b])

    tol = _grad_tolerances(dtype)
    assert_close(hidden.grad.float(), expected_hidden_grad, **tol)
    assert_close(embed.grad.float(), expected_embed_grad, **tol)
    if use_bias:
        assert_close(bias.grad.float(), g.sum(dim=0), **tol)

    masked_positions = mask == 0
    masked_grad = hidden.grad[masked_positions]
    assert torch.equal(masked_grad, torch.zeros_like(masked_grad))


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no_bias"])
def test_backward_zero_scores_produce_zero_gradients(
    sparton_kernel,
    cuda_device: torch.device,
    use_bias: bool,
) -> None:
    """All-negative logits pin the zero-score path: gradients exactly zero.

    Constructed case (deterministic, exact equality per the testing
    doctrine): every masked logit is negative, so every score is zero and
    the backward's score > 0 guard must zero every gradient bit-exactly —
    this pins the inactive/early-exit path of the backward kernels.
    """

    hidden = torch.full((2, 5, 16), -1.0, device=cuda_device, dtype=torch.float16,
                        requires_grad=True)
    embed = torch.ones((19, 16), device=cuda_device, dtype=torch.float16,
                       requires_grad=True)
    bias = None
    if use_bias:
        bias = torch.full((19,), -1.0, device=cuda_device, dtype=torch.float16,
                          requires_grad=True)
    mask = torch.ones((2, 5), device=cuda_device, dtype=torch.int32)

    scores, _ = sparton_kernel.fused_sparton_fwd_op(hidden, embed, bias, mask)
    assert torch.equal(scores, torch.zeros_like(scores))

    scores.backward(torch.randn_like(scores))

    assert torch.equal(hidden.grad, torch.zeros_like(hidden.grad))
    assert torch.equal(embed.grad, torch.zeros_like(embed.grad))
    if use_bias:
        assert torch.equal(bias.grad, torch.zeros_like(bias.grad))


@requires_cuda
@pytest.mark.cuda
def test_backward_masked_rows_yield_zero_hidden_gradient(
    sparton_kernel,
    cuda_device: torch.device,
) -> None:
    """Masked positions (including a fully masked batch row) get exact-zero
    hidden gradients; unmasked gradients still match the reference.

    Constructed case: positive logits everywhere, batch row 0 fully masked
    and row 1 partially masked. Masked positions can never win the max, so
    no backward path may touch them — exact equality, not tolerance.
    """

    hidden = torch.full((2, 5, 16), 0.1, device=cuda_device, dtype=torch.float16,
                        requires_grad=True)
    embed = torch.full((19, 16), 0.1, device=cuda_device, dtype=torch.float16,
                       requires_grad=True)
    bias = None
    mask = torch.tensor(
        [[0, 0, 0, 0, 0], [0, 1, 1, 0, 1]],
        device=cuda_device,
        dtype=torch.int32,
    )
    ref_hidden = _clone_leaf(hidden)
    ref_embed = _clone_leaf(embed)

    scores, _ = sparton_kernel.fused_sparton_fwd_op(hidden, embed, bias, mask)
    expected_scores, _ = sparton_reference(ref_hidden, ref_embed, None, mask)
    upstream = torch.randn_like(scores)

    scores.backward(upstream)
    expected_scores.backward(upstream.detach().clone())

    assert torch.equal(hidden.grad[0], torch.zeros_like(hidden.grad[0]))
    masked_grad = hidden.grad[1][mask[1] == 0]
    assert torch.equal(masked_grad, torch.zeros_like(masked_grad))
    tol = _grad_tolerances(torch.float16)
    assert_close(hidden.grad.float(), ref_hidden.grad.float(), **tol)
    assert_close(embed.grad.float(), ref_embed.grad.float(), **tol)


# The production uniform fast path deposits only when an aligned 64-entry
# sorted-key chunk is single-destination, i.e. destination runs >= the
# config family's CHUNK (src/sparton/_backend_hybrid.py,
# get_uniform_hidden_grad_configs). Every random-input backward test has
# expected run length V_active/S far below that, so without this case the
# suite never executes the uniform deposit (the DEVELOPMENT.md M9 F3 class:
# assert the activation, not just the outputs).
_UNIFORM_PATH_CHUNK = 64


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no_bias"])
def test_backward_uniform_chunk_path_matches_closed_form(
    sparton_kernel,
    cuda_device: torch.device,
    use_bias: bool,
) -> None:
    """Long destination runs activate the M13 uniform-chunk fast path.

    Constructed case: constant hidden rows make every (b, v) logit tie, so
    ties resolve to the lowest unmasked index — one destination per batch
    row, runs of length V >> CHUNK. The test first asserts the activation
    property (every aligned 64-entry chunk of the sorted destination keys
    is single-destination), then checks the gradients against the exact
    closed form (assert_close at the standard fp16 backward tolerance: the
    chunk-partial atomics legitimately reorder fp32 accumulation).
    """

    B, S, D, V = 2, 8, 64, 4096
    c_h, c_e = 0.05, 0.0625  # dyadic: D * c_h * c_e = 0.2 exactly in fp32
    hidden = torch.full((B, S, D), c_h, device=cuda_device, dtype=torch.float16)
    embed = torch.full((V, D), c_e, device=cuda_device, dtype=torch.float16)
    bias = (
        torch.zeros((V,), device=cuda_device, dtype=torch.float16)
        if use_bias
        else None
    )
    # Row 0 masks position 0 so the two batch rows win different
    # destinations (b=0 -> s=1, b=1 -> s=0).
    mask = torch.ones((B, S), device=cuda_device, dtype=torch.int32)
    mask[0, 0] = 0

    scores, idx = sparton_kernel.fused_sparton_fwd_op(hidden, embed, bias, mask)
    assert bool((scores > 0).all()), "constructed logits must all be active"
    expected_idx = torch.tensor([1, 0], device=cuda_device, dtype=idx.dtype)
    assert torch.equal(idx, expected_idx.unsqueeze(1).expand(B, V)), (
        "tie policy must pick the lowest unmasked index"
    )

    # Activation assertion: the sorted destination keys b*S + idx form B
    # runs of length V; with V a multiple of the production CHUNK, every
    # aligned chunk is single-destination — the uniform kernel, not the
    # mixed kernel, must carry every contribution.
    keys = (
        torch.arange(B, device=cuda_device).unsqueeze(1) * S + idx
    ).flatten().sort().values
    chunked = keys.view(-1, _UNIFORM_PATH_CHUNK)
    uniform_chunks = (chunked == chunked[:, :1]).all(dim=1)
    assert bool(uniform_chunks.all()), "every chunk must be single-destination"
    assert chunked.shape[0] >= 2 * (V // _UNIFORM_PATH_CHUNK)

    grad_out = torch.randn((B, V), device=cuda_device, dtype=torch.float32)
    hidden_grad, embed_grad, bias_grad = sparton_kernel.fused_sparton_bwd_op(
        grad_out, scores, idx, hidden, embed, bias, mask
    )

    # Closed form: with logit L = D*c_h*c_e (+0 bias) everywhere,
    # g = grad_out / (1 + L); the winning row s_b collects sum_v g[b, v]
    # * embed[v, :] and embed_grad[v, :] = sum_b g[b, v] * hidden[b, s_b, :].
    g = grad_out / (1.0 + D * c_h * c_e)
    expected_hidden = torch.zeros((B, S, D), device=cuda_device, dtype=torch.float32)
    for b in range(B):
        expected_hidden[b, int(expected_idx[b])] = g[b].sum() * c_e
    expected_embed = g.sum(dim=0).unsqueeze(1) * c_h * torch.ones(
        (V, D), device=cuda_device, dtype=torch.float32
    )

    assert_close(hidden_grad, expected_hidden, atol=1e-4, rtol=1e-3)
    assert_close(embed_grad, expected_embed, atol=1e-4, rtol=1e-3)
    if use_bias:
        assert_close(bias_grad, g.sum(dim=0), atol=1e-4, rtol=1e-3)
    else:
        assert bias_grad is None


@requires_cuda
@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize(("use_bias", "dtype"), BACKWARD_CASES)
def test_backward_matches_legacy_kernel(
    sparton_kernel,
    cuda_device: torch.device,
    use_bias: bool,
    dtype: torch.dtype,
) -> None:
    """A/B of record for the backward swap (M13: split vs segmented).

    Runs the production backward op (the M13 split pass) and the retained
    reference (the M11 segmented design, via legacy_fused_sparton_bwd) on
    identical non-tiny inputs (slow: autotunes both kernel families at this
    shape). Tolerance sits above the measured atomic-order self-spread of
    both designs (proxy spread <= 2.4e-5 relative legacy / <= 4.2e-6
    segmented, DEVELOPMENT.md M11 §7; the M13 §5.5 decision matrix verified 176 cells at
    rtol=atol=1e-3) and far below any real divergence.
    """

    _skip_if_unsupported_dtype(dtype)
    B, S, D, V = 3, 345, 768, 2048
    hidden, embed, bias, mask = _make_nontiny_inputs(cuda_device, B, S, D, V, dtype, use_bias)
    with torch.no_grad():
        scores, idx = sparton_kernel.fused_sparton_fwd_op(hidden, embed, bias, mask)
    grad_out = torch.randn(scores.shape, device=cuda_device, dtype=torch.float32)

    new_grads = sparton_kernel.fused_sparton_bwd_op(
        grad_out, scores, idx, hidden, embed, bias, mask
    )
    legacy_grads = sparton_kernel.legacy_fused_sparton_bwd(
        grad_out, scores, idx, hidden, embed, bias, mask
    )

    for name, new, old in zip(("hidden_grad", "embed_grad", "bias_grad"),
                              new_grads, legacy_grads):
        if not use_bias and name == "bias_grad":
            assert new is None and old is None
            continue
        assert_close(new, old, atol=1e-4, rtol=1e-3, msg=name)


AUTOCAST_DTYPES = [
    pytest.param(torch.float16, id="fp16"),
    pytest.param(torch.bfloat16, id="bf16"),
]


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
@pytest.mark.slow
def test_training_parity_smoke_autocast(
    sparton_kernel,
    cuda_device: torch.device,
) -> None:
    """Short head-only training run: hybrid and optimized stay in lockstep.

    A trimmed version of scripts/probe_training_smoke.py (the M10 tier-1
    gate); reuses its run_mode so the gate logic stays single-sourced.
    """

    from scripts.probe_training_smoke import run_mode

    failures = run_mode(
        mode="bf16",
        steps=30,
        batch=8,
        seq_len=32,
        dim=64,
        vocab=512,
        seed=3,
        lr=1e-3,
        temperature=0.05,
        lambda_flops=1e-3,
        parity_tolerance=0.1,
    )

    assert failures == []


SYNTHETIC_BWD_SOURCES = [
    pytest.param("uniform", id="uniform"),
    pytest.param("zipf", id="zipf"),
]


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize("source", SYNTHETIC_BWD_SOURCES)
def test_bench_backward_synthetic_inputs_honor_contract(
    cuda_device: torch.device,
    source: str,
) -> None:
    """Pin the synthetic input contract of scripts/bench_backward.py.

    The M11 backward harness documents its synthetic regime in its docstring;
    this test single-sources the generator (no duplicated logic) and asserts
    the contract: masks keep at least one unmasked position per row, scores
    are exactly zero off the active set, the realized active fraction tracks
    the request, and active indices point only at unmasked positions —
    mirroring the forward's zero-baseline semantics.
    """

    from scripts.bench_backward import make_synthetic_case

    active_fraction = 0.10
    case = make_synthetic_case(
        source=source,
        batch_size=4,
        seq_len=33,
        dim=16,
        vocab=2048,
        density=0.25,
        active_fraction=active_fraction,
        zipf_s=1.1,
        dtype=torch.float16,
        bias_on=True,
        seed=11,
    )
    mask = case["mask"]
    scores = case["max_scores"]
    idx = case["max_idx"]

    assert bool((mask.sum(dim=1) >= 1).all()), "row with no unmasked position"

    active = scores.float() > 0
    realized = active.float().mean(dim=1)
    assert bool(
        ((realized - active_fraction).abs() <= 0.1 * active_fraction).all()
    ), f"active fraction off target: {realized.tolist()}"
    assert bool((scores.float()[~active] == 0).all()), "nonzero score off the active set"

    chosen_mask = mask.gather(1, idx)
    assert bool(
        (chosen_mask[active] == 1).all()
    ), "active index points at a masked position"
    assert bool((idx[~active] == 0).all()), "inactive entries must carry idx 0"


@requires_cuda
@pytest.mark.cuda
def test_bench_baseline_mask_density_contract(cuda_device: torch.device) -> None:
    """Pin the --mask-density seam of scripts/bench_sparton_baseline.py.

    The M12-T4 flag must not perturb the canonical rows: at density 1.0 the
    mask is exactly the historical all-ones mask, and because the mask is the
    last generator consumer, hidden/embed/bias stay bit-identical at every
    density for the same seed.
    """

    from scripts.bench_sparton_baseline import ShapeSpec, make_inputs

    spec = ShapeSpec(batch_size=2, seq_len=64)
    common = dict(dim=32, vocab=128, dtype=torch.float16, seed=7)

    hidden_1, embed_1, bias_1, mask_1 = make_inputs(spec, **common, mask_density=1.0)
    assert torch.equal(
        mask_1, torch.ones(spec.batch_size, spec.seq_len, device="cuda", dtype=torch.int32)
    ), "density 1.0 must reproduce the all-ones mask exactly"

    hidden_h, embed_h, bias_h, mask_h = make_inputs(spec, **common, mask_density=0.5)
    assert torch.equal(hidden_1, hidden_h), "hidden draw must be density-independent"
    assert torch.equal(embed_1, embed_h), "embed draw must be density-independent"
    assert torch.equal(bias_1, bias_h), "bias draw must be density-independent"
    assert not torch.equal(mask_1, mask_h), "density 0.5 must actually mask positions"


def test_bench_host_overhead_shape_parser() -> None:
    """Pin the BxSxDxV CLI contract of scripts/bench_host_overhead.py."""

    import argparse

    from scripts.bench_host_overhead import parse_shape_list

    assert parse_shape_list("8x128x768x1280,32x128x768x30522") == (
        (8, 128, 768, 1280),
        (32, 128, 768, 30522),
    )
    for bad in ("8x128x768", "8x128x768xfoo", "8x128x0x1280", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_shape_list(bad)


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize("autocast_dtype", AUTOCAST_DTYPES)
def test_forward_backward_under_autocast(
    sparton_kernel,
    cuda_device: torch.device,
    backend: str,
    autocast_dtype: torch.dtype,
) -> None:
    _skip_if_unsupported_dtype(autocast_dtype)
    forward = _forward_for_backend(sparton_kernel, backend)
    generator = torch.Generator(device=cuda_device).manual_seed(37)
    # fp32 master tensors, as produced by AMP training setups.
    hidden = torch.randn(
        (2, 5, 16),
        device=cuda_device,
        dtype=torch.float32,
        generator=generator,
        requires_grad=True,
    )
    embed = torch.randn(
        (19, 16),
        device=cuda_device,
        dtype=torch.float32,
        generator=generator,
        requires_grad=True,
    )
    bias = torch.randn(
        (19,),
        device=cuda_device,
        dtype=torch.float32,
        generator=generator,
        requires_grad=True,
    )
    mask = torch.tensor(
        [[1, 1, 0, 1, 0], [0, 1, 1, 0, 1]],
        device=cuda_device,
        dtype=torch.int32,
    )

    with torch.autocast("cuda", dtype=autocast_dtype):
        scores, idx = forward(hidden, embed, bias, mask)

    assert scores.dtype == autocast_dtype

    cast_hidden = hidden.detach().to(autocast_dtype)
    cast_embed = embed.detach().to(autocast_dtype)
    cast_bias = bias.detach().to(autocast_dtype)
    expected_scores, _ = sparton_reference(cast_hidden, cast_embed, cast_bias, mask)
    tolerances = _score_tolerances(autocast_dtype)
    assert_close(scores.float(), expected_scores.float(), **tolerances)
    assert_index_contract(
        scores, idx, cast_hidden, cast_embed, cast_bias, mask, **tolerances
    )

    scores.float().sum().backward()
    for leaf in (hidden, embed, bias):
        assert leaf.grad is not None
        assert leaf.grad.dtype == torch.float32
        assert bool(torch.isfinite(leaf.grad).all())


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
@pytest.mark.parametrize(("use_bias", "dtype"), BACKWARD_CASES)
def test_optimized_backward_matches_reference(
    sparton_kernel,
    cuda_device: torch.device,
    use_bias: bool,
    dtype: torch.dtype,
) -> None:
    _assert_backward_matches_reference(
        sparton_kernel, cuda_device, sparton_kernel.optimized_forward, use_bias, dtype
    )


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize("backend", ["hybrid", "naive", "optimized"])
def test_forward_backward_handles_noncontiguous_inputs(
    sparton_kernel,
    cuda_device: torch.device,
    backend: str,
) -> None:
    forward = _forward_for_backend(sparton_kernel, backend)
    generator = torch.Generator(device=cuda_device).manual_seed(23)
    base_hidden = torch.randn(
        (2, 10, 16),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    )
    base_embed = torch.randn(
        (38, 16),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    )
    base_bias = torch.randn(
        (38,),
        device=cuda_device,
        dtype=torch.float16,
        generator=generator,
    )
    hidden = base_hidden[:, ::2, :].detach().requires_grad_(True)
    embed = base_embed[::2, :].detach().requires_grad_(True)
    bias = base_bias[::2].detach().requires_grad_(True)
    mask = torch.tensor(
        [[1, 1, 0, 1, 0], [0, 1, 1, 0, 1]],
        device=cuda_device,
        dtype=torch.int32,
    )
    assert not hidden.is_contiguous()
    assert not embed.is_contiguous()
    assert not bias.is_contiguous()

    ref_hidden = hidden.detach().clone().contiguous().requires_grad_(True)
    ref_embed = embed.detach().clone().contiguous().requires_grad_(True)
    ref_bias = bias.detach().clone().contiguous().requires_grad_(True)

    scores, _idx = forward(hidden, embed, bias, mask)
    expected_scores, _ = sparton_reference(ref_hidden, ref_embed, ref_bias, mask)
    upstream = torch.randn_like(scores)

    scores.backward(upstream)
    expected_scores.backward(upstream.detach().clone())

    assert_close(scores.float(), expected_scores.float(), atol=2e-3, rtol=2e-3)
    assert_close(hidden.grad.float(), ref_hidden.grad.float(), atol=2e-3, rtol=2e-3)
    assert_close(embed.grad.float(), ref_embed.grad.float(), atol=2e-3, rtol=2e-3)
    assert_close(bias.grad.float(), ref_bias.grad.float(), atol=2e-3, rtol=2e-3)


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

    # M10 promotion: the default is adaptive — optimized where the pure-Triton
    # TMA backend is available (sm_90+), hybrid (with a one-time warning) elsewhere.
    expected_default = "optimized" if _optimized_availability()[0] else "hybrid"
    assert default_head.backend == expected_default
    assert hybrid_head.backend == "hybrid"
    assert naive_head.backend == "naive"
    assert_close(default_head(hidden, mask).float(), expected_scores.float(), atol=2e-3, rtol=2e-3)
    assert_close(hybrid_head(hidden, mask).float(), expected_scores.float(), atol=2e-3, rtol=2e-3)
    assert_close(naive_head(hidden, mask).float(), expected_scores.float(), atol=2e-3, rtol=2e-3)

    with pytest.raises(ValueError, match="Unknown Sparton backend"):
        sparton_kernel.SpartonHead(19, 16, backend="missing")


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
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
def test_import_emits_no_stdout(cuda_device: torch.device) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC_PATH
    result = subprocess.run(
        [sys.executable, "-c", "import sparton"],
        check=True,
        env=env,
        cwd=str(_REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert result.stdout == ""


@requires_cuda
@pytest.mark.cuda
def test_sparton_backend_env_selects_default(cuda_device: torch.device) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC_PATH
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
        cwd=str(_REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip().splitlines()[-1] == "naive"


@requires_cuda
@pytest.mark.cuda
def test_resolve_backend_invalid_kwarg_names_argument(sparton_kernel) -> None:
    with pytest.raises(ValueError, match=r"'bogus' from backend argument"):
        sparton_kernel.resolve_backend("bogus")


@requires_cuda
@pytest.mark.cuda
def test_resolve_backend_invalid_env_names_env_var(cuda_device: torch.device) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC_PATH
    env["SPARTON_BACKEND"] = "bogus"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sparton.sparton_kernel as sk; "
                "sk.SpartonHead(19, 16)"
            ),
        ],
        check=False,
        env=env,
        cwd=str(_REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "SPARTON_BACKEND environment variable" in result.stderr


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
def test_default_backend_prefers_optimized_when_available(
    cuda_device: torch.device,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC_PATH
    env.pop("SPARTON_BACKEND", None)
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
        cwd=str(_REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip().splitlines()[-1] == "optimized"


@requires_cuda
@pytest.mark.cuda
def test_default_backend_falls_back_to_hybrid_with_one_warning(
    sparton_kernel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sparton._backend_runtime as backend_runtime

    monkeypatch.setattr(
        backend_runtime,
        "is_optimized_backend_available",
        lambda device=None: (False, "forced unavailable for test"),
    )
    monkeypatch.setattr(sparton_kernel, "_ENV_BACKEND", None)
    monkeypatch.setattr(sparton_kernel, "_DEFAULT_FALLBACK_WARNED", False)

    with pytest.warns(RuntimeWarning, match=r"falling back to 'hybrid'"):
        head = sparton_kernel.SpartonHead(19, 16)
    assert head.backend == "hybrid"

    # The fallback warning is one-time per process.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        second = sparton_kernel.SpartonHead(19, 16)
    assert second.backend == "hybrid"


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
def test_sparton_backend_env_selects_optimized_default(cuda_device: torch.device) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC_PATH
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
        cwd=str(_REPO_ROOT),
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


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
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


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
def test_optimized_custom_op_schema_exposes_optional_bias(sparton_kernel) -> None:
    _optimized_op = sparton_kernel.optimized_fwd_op
    optimized_schema = str(torch.ops.sparton.optimized_fwd.default._schema)

    assert "Tensor? bias" in optimized_schema


@requires_cuda
@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_sparton_head_torch_compile_fullgraph(
    sparton_kernel,
    cuda_device: torch.device,
    backend: str,
) -> None:
    _forward_for_backend(sparton_kernel, backend)  # skip when unavailable
    hidden, embed, bias, mask = _make_kernel_inputs(
        device=cuda_device,
        dtype=torch.float16,
        use_bias=True,
    )
    head = sparton_kernel.SpartonHead(19, 16, use_bias=True, backend=backend).to(
        device=cuda_device,
        dtype=torch.float16,
    )
    with torch.no_grad():
        head.weight.copy_(embed)
        assert head.bias is not None
        head.bias.copy_(bias)

    eager_out = head(hidden, mask)
    compiled_out = torch.compile(head, fullgraph=True)(hidden, mask)

    assert_close(compiled_out.float(), eager_out.float(), atol=2e-3, rtol=2e-3)


def _base_validation_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(29)
    return {
        "hidden": torch.randn(
            (2, 5, 16), device=device, dtype=torch.float16, generator=generator
        ),
        "embed": torch.randn(
            (19, 16), device=device, dtype=torch.float16, generator=generator
        ),
        "bias": torch.randn(
            (19,), device=device, dtype=torch.float16, generator=generator
        ),
        "mask": torch.ones((2, 5), device=device, dtype=torch.int32),
    }


VALIDATION_ERROR_CASES = [
    pytest.param(
        "hidden",
        lambda inputs: inputs["hidden"][:, 0, :],
        ValueError,
        r"hidden must be a 3-D \[B, S, D\] tensor",
        id="hidden_rank",
    ),
    pytest.param(
        "embed",
        lambda inputs: inputs["embed"][None],
        ValueError,
        r"embed must be a 2-D \[V, D\] tensor",
        id="embed_rank",
    ),
    pytest.param(
        "mask",
        lambda inputs: inputs["mask"][0],
        ValueError,
        r"mask must be a 2-D \[B, S\] tensor",
        id="mask_rank",
    ),
    pytest.param(
        "embed",
        lambda inputs: torch.randn(
            (19, 32), device=inputs["embed"].device, dtype=inputs["embed"].dtype
        ),
        ValueError,
        r"embed\.shape\[1\] must equal hidden\.shape\[2\]",
        id="embed_dim_mismatch",
    ),
    pytest.param(
        "mask",
        lambda inputs: inputs["mask"][:, :4],
        ValueError,
        r"mask\.shape must equal \(B, S\)",
        id="mask_shape_mismatch",
    ),
    pytest.param(
        "bias",
        lambda inputs: inputs["bias"][:18],
        ValueError,
        r"bias\.shape must equal \(V,\)",
        id="bias_shape_mismatch",
    ),
    pytest.param(
        "embed",
        lambda inputs: inputs["embed"].cpu(),
        ValueError,
        r"embed\.device must equal hidden\.device",
        id="embed_cross_device",
    ),
    pytest.param(
        "embed",
        lambda inputs: inputs["embed"].float(),
        TypeError,
        r"embed\.dtype must equal hidden\.dtype",
        id="embed_dtype_mismatch",
    ),
    pytest.param(
        "bias",
        lambda inputs: inputs["bias"].float(),
        TypeError,
        r"bias\.dtype must equal hidden\.dtype",
        id="bias_dtype_mismatch",
    ),
    pytest.param(
        "mask",
        lambda inputs: inputs["mask"].to(torch.complex64),
        TypeError,
        r"mask\.dtype must be bool, integer, or floating point",
        id="mask_complex",
    ),
]


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize("backend", ALL_BACKENDS)
@pytest.mark.parametrize(("field", "mutate", "exc", "match"), VALIDATION_ERROR_CASES)
def test_validation_rejects_bad_inputs(
    sparton_kernel,
    cuda_device: torch.device,
    backend: str,
    field: str,
    mutate,
    exc: type,
    match: str,
) -> None:
    forward = _forward_for_backend(sparton_kernel, backend)
    inputs = _base_validation_inputs(cuda_device)
    inputs[field] = mutate(inputs)

    with pytest.raises(exc, match=match):
        forward(inputs["hidden"], inputs["embed"], inputs["bias"], inputs["mask"])


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_validation_rejects_cpu_tensors(
    sparton_kernel,
    cuda_device: torch.device,
    backend: str,
) -> None:
    forward = _forward_for_backend(sparton_kernel, backend)
    inputs = {name: tensor.cpu() for name, tensor in _base_validation_inputs(cuda_device).items()}

    with pytest.raises(ValueError, match=r"hidden must be a CUDA tensor"):
        forward(inputs["hidden"], inputs["embed"], inputs["bias"], inputs["mask"])


@requires_cuda
@pytest.mark.cuda
@pytest.mark.parametrize("backend", ["naive", "optimized"])
def test_validation_rejects_fp32_on_fused_backends(
    sparton_kernel,
    cuda_device: torch.device,
    backend: str,
) -> None:
    forward = _forward_for_backend(sparton_kernel, backend)
    inputs = _base_validation_inputs(cuda_device)
    for name in ("hidden", "embed", "bias"):
        inputs[name] = inputs[name].float()

    with pytest.raises(TypeError, match=r"hidden\.dtype must be one of"):
        forward(inputs["hidden"], inputs["embed"], inputs["bias"], inputs["mask"])


@requires_cuda
@pytest.mark.cuda
def test_validation_allows_fp32_hybrid(
    sparton_kernel,
    cuda_device: torch.device,
) -> None:
    inputs = _base_validation_inputs(cuda_device)
    hidden = inputs["hidden"].float()
    embed = inputs["embed"].float()
    bias = inputs["bias"].float()
    mask = inputs["mask"]

    scores, _idx = sparton_kernel.hybrid_forward(hidden, embed, bias, mask)
    expected_scores, _ = sparton_reference(hidden, embed, bias, mask)

    assert_close(scores, expected_scores, atol=1e-3, rtol=1e-3)


@requires_optimized
@pytest.mark.cuda
@pytest.mark.optimized
def test_optimized_validation_rejects_unaligned_d(
    sparton_kernel,
    cuda_device: torch.device,
) -> None:
    hidden = torch.randn((2, 64, 10), device=cuda_device, dtype=torch.float16)
    embed = torch.randn((19, 10), device=cuda_device, dtype=torch.float16)
    mask = torch.ones((2, 64), device=cuda_device, dtype=torch.int32)

    with pytest.raises(ValueError, match=r"multiple of 16 bytes"):
        sparton_kernel.optimized_forward(hidden, embed, None, mask)
