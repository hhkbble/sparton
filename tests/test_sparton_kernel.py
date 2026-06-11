from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

import pytest
import torch
from torch.testing import assert_close


CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="Sparton CUDA kernel tests require a CUDA device",
)

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
    with pytest.raises(RuntimeError, match="optimized.*not available"):
        sparton_kernel.SpartonHead(19, 16, backend="optimized")


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
