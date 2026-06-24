"""Shape-soak correctness gate for the optimized (pure-Triton) forward (M10 gate 5).

The `optimized` backend (`src/sparton/_backend_optimized.py`) is the pure-Triton
persistent fused forward; it is bit-close to `naive` by construction (fp32
accumulation). Routes through the public `optimized_forward`, so it tests
whichever tile the backend selects (the self-tuner is defaulted OFF here —
correctness is tile-independent; see main()).

Sweeps the M10 grid (DEVELOPMENT.md M10) — S in {1, 7, 64, 127, 128, 129, 255,
511}, B in {1, 2, 5}, D in {768, 1024}, V in {30522, 151936}, bias in {yes, no},
dtype in {fp16, bf16} — with 75%-density random masks where batch row 0 is fully
zeroed (the all-zero-row edge; for B=1 the whole batch is masked).

Each case checks scores against a vectorized input-dtype reference and the
returned indices against the tie-aware index contract of record
(ARCHITECTURE.md §3.2): wherever the score is positive, the masked logit at the
chosen index must be within score tolerance of the per-(b, v) maximum.

Honors the optimized env knobs read by the backend launcher
(``SPARTON_OPTIMIZED_AUTOTUNE`` / ``SPARTON_OPTIMIZED_WARP_SPECIALIZE`` /
``SPARTON_OPTIMIZED_NUM_CTAS`` / ``SPARTON_OPTIMIZED_CTAS_PER_SM``) — set them
in the environment to soak a specific variant.

Exits non-zero listing every failing case. Use ``--quick`` for a small smoke
subset during development; the full sweep is the promotion gate.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


FULL_S = (1, 7, 64, 127, 128, 129, 255, 511)
FULL_B = (1, 2, 5)
FULL_D = (768, 1024)
FULL_V = (30522, 151936)

QUICK_S = (1, 127, 129)
QUICK_B = (1, 5)
QUICK_D = (768,)
QUICK_V = (30522,)


def tolerances(dtype: torch.dtype) -> dict[str, float]:
    if dtype is torch.bfloat16:
        return {"atol": 5e-2, "rtol": 5e-2}
    return {"atol": 2e-3, "rtol": 2e-3}


def masked_reference_logits(
    hidden: torch.Tensor,
    embed: torch.Tensor,
    bias: torch.Tensor | None,
    mask: torch.Tensor,
) -> torch.Tensor:
    logits = hidden @ embed.T
    if bias is not None:
        logits = logits + bias
    return logits * mask.to(dtype=logits.dtype)[:, :, None]


def check_case(
    sk,
    *,
    B: int,
    S: int,
    D: int,
    V: int,
    use_bias: bool,
    dtype: torch.dtype,
    seed: int,
) -> tuple[float, float]:
    """Return (max score error, max index-contract gap); raises on failure."""

    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn((B, S, D), device="cuda", dtype=dtype, generator=generator) * 0.05
    embed = torch.randn((V, D), device="cuda", dtype=dtype, generator=generator) * 0.05
    bias = None
    if use_bias:
        bias = torch.randn((V,), device="cuda", dtype=dtype, generator=generator) * 0.05
    mask = (torch.rand((B, S), device="cuda", generator=generator) > 0.25).to(torch.int32)
    mask[0] = 0  # all-zero-row edge; for B=1 the whole batch is masked

    scores, idx = sk.optimized_forward(hidden, embed, bias, mask)

    masked = masked_reference_logits(hidden, embed, bias, mask)
    ref_scores = torch.log1p(torch.relu(masked.max(dim=1).values))
    tol = tolerances(dtype)
    torch.testing.assert_close(scores.float(), ref_scores.float(), **tol)
    score_err = (scores.float() - ref_scores.float()).abs().max().item()

    # Index contract: meaningful only where the score is positive.
    ref_max = masked.max(dim=1).values
    chosen = masked.gather(1, idx.unsqueeze(1)).squeeze(1)
    active = scores.float() > 0
    gap = (ref_max.float() - chosen.float())[active]
    max_gap = gap.max().item() if gap.numel() else 0.0
    limit = tol["atol"] + tol["rtol"] * ref_max.float()[active].abs()
    if gap.numel() and not bool((gap <= limit).all()):
        raise AssertionError(
            f"index contract violated: max gap {max_gap:.6f} exceeds tolerance"
        )
    return score_err, max_gap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="small smoke subset")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Correctness is tile-independent, so the soak runs with the self-tuner OFF (the fast
    # analytic tile) unless the caller overrides it. The measured-tile path is covered by the
    # unit tuner test and the benchmark; tuning every (D, V) here would be slow and add
    # nothing to a correctness gate.
    os.environ.setdefault("SPARTON_OPTIMIZED_AUTOTUNE", "off")

    if not torch.cuda.is_available():
        raise RuntimeError("soak_optimized_correctness.py requires CUDA")
    from sparton._backend_runtime import is_optimized_backend_available

    available, reason = is_optimized_backend_available()
    if not available:
        raise RuntimeError(
            "soak_optimized_correctness.py requires the optimized backend "
            f"(CUDA sm_90+ and importable triton.tools.tensor_descriptor): {reason}"
        )

    import sparton.sparton_kernel as sk

    s_values = QUICK_S if args.quick else FULL_S
    b_values = QUICK_B if args.quick else FULL_B
    d_values = QUICK_D if args.quick else FULL_D
    v_values = QUICK_V if args.quick else FULL_V
    dtypes = [torch.float16]
    if torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)

    cases = [
        (B, S, D, V, use_bias, dtype)
        for dtype in dtypes
        for D in d_values
        for V in v_values
        for B in b_values
        for S in s_values
        for use_bias in (True, False)
    ]
    variant = []
    for env_key in (
        "SPARTON_OPTIMIZED_AUTOTUNE",
        "SPARTON_OPTIMIZED_WARP_SPECIALIZE",
        "SPARTON_OPTIMIZED_NUM_CTAS",
        "SPARTON_OPTIMIZED_CTAS_PER_SM",
    ):
        value = os.environ.get(env_key)
        if value:
            variant.append(f"{env_key}={value}")
    variant_note = (" [" + ", ".join(variant) + "]") if variant else ""
    print(
        f"optimized shape soak: {len(cases)} cases "
        f"(S={s_values} B={b_values} D={d_values} V={v_values} "
        f"bias=y/n dtypes={[str(d) for d in dtypes]}), mask density 75%, "
        f"row 0 fully masked{variant_note}",
        flush=True,
    )

    failures: list[tuple[str, str]] = []
    worst_score_err = 0.0
    worst_gap = 0.0
    for case_index, (B, S, D, V, use_bias, dtype) in enumerate(cases):
        label = f"B={B} S={S} D={D} V={V} bias={use_bias} {str(dtype).replace('torch.', '')}"
        try:
            score_err, gap = check_case(
                sk,
                B=B,
                S=S,
                D=D,
                V=V,
                use_bias=use_bias,
                dtype=dtype,
                seed=args.seed + case_index,
            )
            worst_score_err = max(worst_score_err, score_err)
            worst_gap = max(worst_gap, gap)
        except Exception as exc:  # noqa: BLE001 — gate script reports and continues
            failures.append((label, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL {label}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        if (case_index + 1) % 64 == 0:
            torch.cuda.empty_cache()
            print(f"  ... {case_index + 1}/{len(cases)} done", flush=True)

    print(
        f"soak summary: {len(cases) - len(failures)}/{len(cases)} passed, "
        f"max score err {worst_score_err:.6f}, max index gap {worst_gap:.6f}",
        flush=True,
    )
    if failures:
        print(f"{len(failures)} FAILURES:")
        for label, message in failures:
            print(f"  {label}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
