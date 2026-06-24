"""Parameterized direct-op backward profiling target (M11 T1/T4).

Generalizes the ``hybrid_bwd_direct`` range of ``ncu_targets.py`` to arbitrary
shapes, dtypes, bias modes, and (optionally) captured index-distribution
bundles, so the same target serves the M11 before/after counter collection.

Builds forward inputs, obtains ``(max_scores, max_idx)`` through a real
``fused_sparton_fwd_op`` call (or loads them from a capture bundle), warms the
backward autotune cache, then wraps ``--launches`` direct
``fused_sparton_bwd_op`` calls in the NVTX range ``bwd_direct/`` **on the main
thread** — autograd's backward worker thread does not inherit NVTX ranges
(design v1 §2.5), so ``tensor.backward()`` must not be profiled here.

Intended invocation (hardened env, serial; see benchmarks/README.md):

  ncu --nvtx --nvtx-include "bwd_direct/" --launch-skip 1 --launch-count 1 \
      --metrics <set recorded in the M11 memo> -o <report> \
      python -u benchmarks/ncu_backward_target.py --dtype fp16 --bias on

Prints one summary line; exits non-zero if the backward op fails.
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


def build_bundle_case(args) -> dict[str, torch.Tensor | None]:
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    record = bundle["records"][args.record]
    B, S, D = record["hidden_shape"]
    V = record["max_scores"].shape[1]
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    dtype = DTYPES[args.dtype]
    case = {
        "hidden": torch.randn((B, S, D), device="cuda", dtype=dtype, generator=generator) * 0.05,
        "embed": torch.randn((V, D), device="cuda", dtype=dtype, generator=generator) * 0.05,
        "bias": (
            torch.randn((V,), device="cuda", dtype=dtype, generator=generator) * 0.05
            if args.bias == "on"
            else None
        ),
        "mask": record["mask"].to(device="cuda", dtype=torch.int32),
    }
    case["max_scores"] = record["max_scores"].to(device="cuda", dtype=dtype)
    case["max_idx"] = record["max_idx"].to(device="cuda", dtype=torch.int64)
    return case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--vocab", type=int, default=30522)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="fp16")
    parser.add_argument("--bias", choices=("on", "off"), default="on")
    parser.add_argument("--mask-density", type=float, default=0.75)
    parser.add_argument("--bundle", type=str, default=None,
                        help="capture_index_distributions.py bundle path")
    parser.add_argument("--record", type=int, default=0,
                        help="record index within --bundle")
    parser.add_argument("--launches", type=int, default=3,
                        help="profiled launches inside the NVTX range")
    parser.add_argument("--impl", type=str, default="current",
                        help="backward impl from bench_backward.build_impls "
                        "(current, legacy; prototype names only on the M11 "
                        "T3 tree, commit b5acd9c)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("ncu_backward_target.py requires CUDA")

    import sparton.sparton_kernel as sk
    from bench_backward import build_impls

    impl = build_impls([args.impl])[args.impl]

    if args.bundle is not None:
        case = build_bundle_case(args)
        scores, idx = case.pop("max_scores"), case.pop("max_idx")
    else:
        case = build_synthetic_case(args)
        scores, idx = sk.fused_sparton_fwd_op(
            case["hidden"], case["embed"], case["bias"], case["mask"]
        )
    grad_generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    grad_out = torch.randn(scores.shape, device="cuda", dtype=torch.float32,
                           generator=grad_generator)

    def bwd_call():
        return impl(
            grad_out, scores, idx, case["hidden"], case["embed"], case["bias"], case["mask"]
        )

    for _ in range(3):  # warm backward autotune + compile (cached)
        bwd_call()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("bwd_direct")
    for _ in range(args.launches):
        hidden_grad, embed_grad, bias_grad = bwd_call()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    B, S, D = case["hidden"].shape
    V = case["embed"].shape[0]
    active_fraction = (scores.float() > 0).float().mean().item()
    print(
        f"bwd_direct done: impl={args.impl} B={B} S={S} D={D} V={V} dtype={args.dtype} "
        f"bias={args.bias} source={'bundle' if args.bundle else 'synthetic'} "
        f"launches={args.launches} active={active_fraction:.4f} "
        f"|h_grad|={hidden_grad.norm().item():.4f} |e_grad|={embed_grad.norm().item():.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
