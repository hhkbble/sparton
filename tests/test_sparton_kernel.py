from __future__ import annotations

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
def test_custom_op_schemas_expose_optional_bias(sparton_kernel) -> None:
    fwd_schema = str(torch.ops.sparton.fused_sparton_fwd.default._schema)
    bwd_schema = str(torch.ops.sparton.fused_sparton_bwd.default._schema)

    assert "Tensor? bias" in fwd_schema
    assert "Tensor? bias" in bwd_schema
    assert "Tensor?)" in bwd_schema
