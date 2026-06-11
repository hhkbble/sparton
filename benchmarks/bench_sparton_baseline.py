"""Merged Sparton backend baseline on a fixed B/S grid.

Defaults model the projection-head dimensions of naver/splade-code-06B:
  - hidden size 1024
  - vocab size 151936
  - bf16 tensors

The benchmark runs one row per (B, S) pair with all-ones attention masks. The
naive backend uses production Triton autotune; pass ``--optimized-policy on``
to include the experimental Gluon forward with runtime GPU-derived active
autotune candidates.
Needs PYTHONPATH=src and the hardened env prefix from
docs/sparton_gluon_remaining_work_design.md section 2.4.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

import torch


DEFAULT_BATCH_SIZES = (4, 8, 16)
DEFAULT_SEQ_LENS = (256, 512, 768)
DEFAULT_DIM = 1024
DEFAULT_VOCAB = 151936


@dataclass(frozen=True)
class ShapeSpec:
    batch_size: int
    seq_len: int


def parse_int_list(value: str) -> tuple[int, ...]:
    items = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    if any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("all list values must be positive")
    return items


def parse_dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    raise argparse.ArgumentTypeError("dtype must be bf16 or fp16")


def dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float16:
        return "fp16"
    return str(dtype).replace("torch.", "")


def build_shape_grid(batch_sizes: Sequence[int], seq_lens: Sequence[int]) -> tuple[ShapeSpec, ...]:
    return tuple(ShapeSpec(batch_size, seq_len) for batch_size in batch_sizes for seq_len in seq_lens)


def bytes_to_mib(num_bytes: float) -> float:
    return num_bytes / 2**20


def tensor_element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def make_inputs(
    spec: ShapeSpec,
    *,
    dim: int,
    vocab: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden = torch.randn(
        spec.batch_size,
        spec.seq_len,
        dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    ) * 0.05
    embed = torch.randn(vocab, dim, device="cuda", dtype=dtype, generator=generator) * 0.05
    bias = torch.randn(vocab, device="cuda", dtype=dtype, generator=generator) * 0.05
    mask = torch.ones(spec.batch_size, spec.seq_len, device="cuda", dtype=torch.int32)
    return hidden, embed, bias, mask


def bench_ms(fn, *, warmup: int, rep: int) -> float:
    import triton

    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def peak_extra_memory(fn) -> int:
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def benchmark_shape(args, spec: ShapeSpec, sk) -> dict[str, object]:
    dtype = parse_dtype(args.dtype)
    hidden, embed, bias, mask = make_inputs(
        spec,
        dim=args.dim,
        vocab=args.vocab,
        dtype=dtype,
        seed=args.seed + spec.batch_size * 65537 + spec.seq_len,
    )

    def hybrid_bias():
        return sk.hybrid_forward(hidden, embed, bias, mask)

    def hybrid_nobias():
        return sk.hybrid_forward(hidden, embed, None, mask)

    def gemm_full():
        return torch.matmul(hidden.reshape(spec.batch_size * spec.seq_len, args.dim), embed.T)

    def hybrid_fwd_bwd():
        h = hidden.detach().requires_grad_(True)
        e = embed.detach().requires_grad_(True)
        b = bias.detach().requires_grad_(True)
        scores, _idx = sk.hybrid_forward(h, e, b, mask)
        scores.float().sum().backward()
        return scores

    hybrid_bias_ms = bench_ms(hybrid_bias, warmup=args.warmup, rep=args.rep)
    hybrid_nobias_ms = bench_ms(hybrid_nobias, warmup=args.warmup, rep=args.rep)
    gemm_ms = bench_ms(gemm_full, warmup=args.warmup, rep=args.rep)
    hybrid_fwd_bwd_ms = bench_ms(
        hybrid_fwd_bwd,
        warmup=args.bwd_warmup,
        rep=args.bwd_rep,
    )
    hybrid_peak = peak_extra_memory(hybrid_bias)

    naive_bias_ms = None
    naive_nobias_ms = None
    naive_peak = None
    if args.naive_policy == "on":
        def naive_bias():
            return sk.naive_forward(hidden, embed, bias, mask)

        def naive_nobias():
            return sk.naive_forward(hidden, embed, None, mask)

        naive_bias_ms = bench_ms(naive_bias, warmup=args.naive_warmup, rep=args.naive_rep)
        naive_nobias_ms = bench_ms(naive_nobias, warmup=args.naive_warmup, rep=args.naive_rep)
        naive_peak = peak_extra_memory(naive_bias)

    optimized_bias_ms = None
    optimized_nobias_ms = None
    optimized_peak = None
    if args.optimized_policy == "on":
        def optimized_bias():
            return sk.optimized_forward(hidden, embed, bias, mask)

        def optimized_nobias():
            return sk.optimized_forward(hidden, embed, None, mask)

        optimized_bias_ms = bench_ms(
            optimized_bias,
            warmup=args.optimized_warmup,
            rep=args.optimized_rep,
        )
        optimized_nobias_ms = bench_ms(
            optimized_nobias,
            warmup=args.optimized_warmup,
            rep=args.optimized_rep,
        )
        optimized_peak = peak_extra_memory(optimized_bias)

    elem_size = tensor_element_size(dtype)
    output_bytes = spec.batch_size * args.vocab * (
        elem_size + torch.empty((), dtype=torch.int64).element_size()
    )
    logits_bytes = spec.batch_size * spec.seq_len * args.vocab * elem_size
    tokens = spec.batch_size * spec.seq_len

    return {
        "B": spec.batch_size,
        "S": spec.seq_len,
        "hybrid_bias_ms": hybrid_bias_ms,
        "hybrid_nobias_ms": hybrid_nobias_ms,
        "gemm_ms": gemm_ms,
        "hybrid_overhead_pct": ((hybrid_bias_ms - gemm_ms) / gemm_ms) * 100.0,
        "hybrid_fwd_bwd_ms": hybrid_fwd_bwd_ms,
        "hybrid_peak_mib": bytes_to_mib(hybrid_peak),
        "naive_bias_ms": naive_bias_ms,
        "naive_nobias_ms": naive_nobias_ms,
        "naive_peak_mib": None if naive_peak is None else bytes_to_mib(naive_peak),
        "optimized_bias_ms": optimized_bias_ms,
        "optimized_nobias_ms": optimized_nobias_ms,
        "optimized_peak_mib": None if optimized_peak is None else bytes_to_mib(optimized_peak),
        "output_mib": bytes_to_mib(output_bytes),
        "logits_mib": bytes_to_mib(logits_bytes),
        "tokens_per_s": tokens / (hybrid_bias_ms / 1000.0),
    }


def fmt(value: object, precision: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def print_rows(rows: Sequence[dict[str, object]]) -> None:
    columns = (
        ("B", "right"),
        ("S", "right"),
        ("hyb+b ms", "right"),
        ("hyb ms", "right"),
        ("gemm ms", "right"),
        ("ovh %", "right"),
        ("hyb f+b ms", "right"),
        ("hyb MiB", "right"),
        ("naive+b ms", "right"),
        ("naive ms", "right"),
        ("naive MiB", "right"),
        ("opt+b ms", "right"),
        ("opt ms", "right"),
        ("opt MiB", "right"),
        ("out MiB", "right"),
        ("logits MiB", "right"),
        ("tok/s", "right"),
    )
    print()
    print("| " + " | ".join(name for name, _align in columns) + " |")
    print("| " + " | ".join("---:" if align == "right" else "---" for _name, align in columns) + " |")
    for row in rows:
        values = [
            row["B"],
            row["S"],
            fmt(row["hybrid_bias_ms"]),
            fmt(row["hybrid_nobias_ms"]),
            fmt(row["gemm_ms"]),
            fmt(row["hybrid_overhead_pct"], precision=1),
            fmt(row["hybrid_fwd_bwd_ms"]),
            fmt(row["hybrid_peak_mib"], precision=2),
            fmt(row["naive_bias_ms"]),
            fmt(row["naive_nobias_ms"]),
            fmt(row["naive_peak_mib"], precision=2),
            fmt(row["optimized_bias_ms"]),
            fmt(row["optimized_nobias_ms"]),
            fmt(row["optimized_peak_mib"], precision=2),
            fmt(row["output_mib"], precision=2),
            fmt(row["logits_mib"], precision=1),
            fmt(row["tokens_per_s"], precision=0),
        ]
        print("| " + " | ".join(str(value) for value in values) + " |")


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=parse_int_list, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--seq-lens", type=parse_int_list, default=DEFAULT_SEQ_LENS)
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM)
    parser.add_argument("--vocab", type=int, default=DEFAULT_VOCAB)
    parser.add_argument("--dtype", choices=("bf16", "bfloat16", "fp16", "float16"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--rep", type=int, default=16)
    parser.add_argument("--bwd-warmup", type=int, default=4)
    parser.add_argument("--bwd-rep", type=int, default=16)
    parser.add_argument("--naive-warmup", type=int, default=4)
    parser.add_argument("--naive-rep", type=int, default=16)
    parser.add_argument("--naive-policy", choices=("on", "off"), default="on")
    parser.add_argument("--optimized-warmup", type=int, default=4)
    parser.add_argument("--optimized-rep", type=int, default=16)
    parser.add_argument("--optimized-policy", choices=("on", "off"), default="off")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("bench_sparton_baseline.py requires CUDA")

    dtype = parse_dtype(args.dtype)
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("default bf16 benchmark requires CUDA BF16 support; pass --dtype fp16")
    if args.optimized_policy == "on":
        from sparton._gluon_runtime import is_gluon_backend_available

        available, reason = is_gluon_backend_available()
        if not available:
            raise RuntimeError(
                "bench_sparton_baseline.py --optimized-policy on requires the "
                "optimized Gluon backend (CUDA sm_80+ and importable "
                f"triton.experimental.gluon): {reason}"
            )

    import sparton.sparton_kernel as sk

    shapes = build_shape_grid(args.batch_sizes, args.seq_lens)
    print(
        "Sparton merged baseline: "
        f"D={args.dim} V={args.vocab} dtype={dtype_name(dtype)} "
        f"B={','.join(map(str, args.batch_sizes))} "
        f"S={','.join(map(str, args.seq_lens))} "
        f"warmup/rep={args.warmup}/{args.rep} naive={args.naive_policy} "
        f"optimized={args.optimized_policy}"
    )
    print("Times are milliseconds per fixed (B, S) row; masks are all ones.")

    rows = []
    for spec in shapes:
        rows.append(benchmark_shape(args, spec, sk))
        torch.cuda.empty_cache()
    print_rows(rows)


if __name__ == "__main__":
    main()
