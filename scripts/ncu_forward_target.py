"""Parameterized direct-op forward profiling target (M12 T0).

Forward analogue of ``ncu_backward_target.py``: builds synthetic inputs,
warms the optimized forward's autotune cache (the first call may launch every
pruned candidate config — those launches happen during warmup, *outside* the
NVTX range, so ``--nvtx-include`` keeps the profiler off them), then wraps
``--launches`` direct ``optimized_fwd_op`` calls in the NVTX range
``fwd_direct/`` on the main thread. The raw op bypasses validation by design
(AGENTS.md Core Kernel Invariants — profiling targets construct
contiguous-by-construction inputs themselves).

No ``--bundle`` mode: forward cost is fully determined by
``(B, S, D, V, dtype, bias)`` — the capture bundles store backward inputs
(``max_scores``/``max_idx``/``mask``), not ``hidden``/``embed``, so there is
nothing real to replay through a forward. No ``--impl``: there is one
production forward symbol. ``--mask-density`` defaults to 1.0 (unlike the
backward target's 0.75) so profiles correspond to the all-ones canonical
grid rows they gate; the kernel's work is mask-value-invariant (the mask is
an epilogue multiply, not control flow).

Intended invocation (hardened env, serial; see scripts/README.md):

  ncu --nvtx --nvtx-include "fwd_direct/" --launch-skip 1 --launch-count 1 \
      --section ... --metrics <set recorded in DEVELOPMENT.md M12> -o <report> \
      python -u scripts/ncu_forward_target.py --dtype fp16 --bias on

Prints one summary line; exits non-zero if the forward op fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def build_synthetic_case(args) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    hidden = (
        torch.randn((args.batch_size, args.seq_len, args.dim), device="cuda",
                    dtype=DTYPES[args.dtype], generator=generator) * 0.05
    )
    embed = (
        torch.randn((args.vocab, args.dim), device="cuda",
                    dtype=DTYPES[args.dtype], generator=generator) * 0.05
    )
    bias = None
    if args.bias == "on":
        bias = (
            torch.randn((args.vocab,), device="cuda",
                        dtype=DTYPES[args.dtype], generator=generator) * 0.05
        )
    mask = (
        torch.rand((args.batch_size, args.seq_len), device="cuda", generator=generator)
        < args.mask_density
    ).to(torch.int32)
    return {"hidden": hidden, "embed": embed, "bias": bias, "mask": mask}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--vocab", type=int, default=30522)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="fp16")
    parser.add_argument("--bias", choices=("on", "off"), default="on")
    parser.add_argument("--mask-density", type=float, default=1.0)
    parser.add_argument("--launches", type=int, default=3,
                        help="profiled launches inside the NVTX range")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ncu_forward_target.py requires CUDA")
    from sparton._backend_runtime import is_optimized_backend_available

    available, reason = is_optimized_backend_available()
    if not available:
        raise RuntimeError(
            "ncu_forward_target.py requires the optimized backend "
            f"(CUDA sm_90+ and importable triton.tools.tensor_descriptor): {reason}"
        )

    import sparton.sparton_kernel as sk

    case = build_synthetic_case(args)

    def fwd_call():
        return sk.optimized_fwd_op(
            case["hidden"], case["embed"], case["bias"], case["mask"]
        )

    for _ in range(3):  # warm forward autotune + compile (cached)
        fwd_call()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("fwd_direct")
    for _ in range(args.launches):
        scores, idx = fwd_call()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    B, S, D = case["hidden"].shape
    V = case["embed"].shape[0]
    active_fraction = (scores.float() > 0).float().mean().item()
    idx_in_range = bool(((idx >= 0) & (idx < S)).all().item())
    print(
        f"fwd_direct done: B={B} S={S} D={D} V={V} dtype={args.dtype} "
        f"bias={args.bias} mask_density={args.mask_density:g} "
        f"launches={args.launches} active={active_fraction:.4f} "
        f"|scores|={scores.float().norm().item():.4f} idx_in_range={idx_in_range}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
