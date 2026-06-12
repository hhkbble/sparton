"""Correctness probe for the optimized Gluon forward epilogue.

This exercises the fused O1 path that performs row masking, bias handling,
online max/argmax, S/V tails, strict cross-chunk updates, and final log1p in
Gluon.  It intentionally uses small deterministic shapes for fast bring-up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def reference(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: torch.Tensor | None,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = hidden @ embed.T
    if bias is not None:
        logits = logits + bias
    masked = logits * mask.to(dtype=logits.dtype)[:, :, None]
    B, S, V = masked.shape
    running_max = torch.zeros((B, V), device=hidden.device, dtype=hidden.dtype)
    running_idx = torch.zeros((B, V), device=hidden.device, dtype=torch.int64)
    for seq_idx in range(S):
        candidate = masked[:, seq_idx, :]
        better = candidate > running_max
        running_max = torch.where(better, candidate, running_max)
        running_idx = torch.where(
            better,
            torch.full_like(running_idx, seq_idx),
            running_idx,
        )
    return torch.log1p(torch.relu(running_max)), running_idx


def check_case(dtype: torch.dtype, use_bias: bool) -> None:
    import sparton.sparton_kernel as sk

    B, S, D, V = 3, 17, 48, 129
    hidden = torch.zeros((B, S, D), device="cuda", dtype=dtype)
    embed = torch.zeros((V, D), device="cuda", dtype=dtype)
    seq = torch.arange(1, S + 1, device="cuda", dtype=torch.float32)
    vocab = torch.arange(V, device="cuda")
    hidden[:, :, 0] = (seq / 16).to(dtype)
    hidden[:, :, 1] = ((S + 1 - seq) / 32).to(dtype)
    hidden[:, :, 2] = torch.tensor(
        [0, 2, 2, 1, 0, 2, 2, 1, 0, 2, 2, 1, 0, 2, 2, 1, 0],
        device="cuda",
        dtype=dtype,
    )
    embed[:, 0] = torch.where(vocab % 5 == 0, -0.25, 0.25).to(dtype)
    embed[:, 1] = torch.where(vocab % 7 == 0, 0.125, -0.0625).to(dtype)
    embed[:, 2] = torch.where(vocab % 11 == 0, 1.0, 0.0).to(dtype)
    bias = None
    if use_bias:
        bias = (((vocab % 13).to(torch.float32) - 6.0) / 64).to(dtype)
    mask = torch.tensor(
        [
            [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        ],
        device="cuda",
        dtype=torch.int32,
    )

    scores, idx = sk.optimized_forward(hidden, embed, bias, mask)
    expected_scores, expected_idx = reference(hidden, embed, bias, mask)
    torch.cuda.synchronize()
    tol = 5e-2 if dtype is torch.bfloat16 else 2e-3
    torch.testing.assert_close(scores.float(), expected_scores.float(), atol=tol, rtol=tol)
    assert torch.equal(idx, expected_idx)
    print(f"{dtype} bias={use_bias}: passed")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("probe_gluon_epilogue.py requires CUDA")
    from sparton._gluon_runtime import is_gluon_backend_available

    available, reason = is_gluon_backend_available()
    if not available:
        raise RuntimeError(
            "probe_gluon_epilogue.py requires the optimized Gluon backend "
            f"(CUDA sm_80+ and importable triton.experimental.gluon): {reason}"
        )
    check_case(torch.float16, False)
    check_case(torch.float16, True)
    if torch.cuda.is_bf16_supported():
        check_case(torch.bfloat16, False)
        check_case(torch.bfloat16, True)


if __name__ == "__main__":
    main()
