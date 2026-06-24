"""Wall-minus-GPU host-overhead recorder for the forward wrappers (M12 T4).

Standing documentation of the deferred-F9 launch overhead (ARCHITECTURE.md §6.7) and
the baseline any future latency user would revive launcher v2 against. Per
(shape, backend) it reports

  gpu ms    ``triton.testing.do_bench`` (the GPU latency of record), and
  wall ms   an N-call wall-clock loop timed with ``time.perf_counter`` and
            exactly one trailing ``torch.cuda.synchronize()`` — no per-call
            sync, which would add a host<->GPU round trip to every call and
            overstate the host share,

and ``host ms = wall - gpu``. Interpretation caveat: when host launch work is
slower than the GPU (small shapes), wall is host-rate-bound and ``host ms``
is the per-call host cost; when the GPU dominates (the dev shape), ``host
ms`` approaches 0 and means "fully overlapped", not "free".

Method and historical row of record (ARCHITECTURE.md §6.3, at
``B=8 S=128 D=768 V=1280`` fp16, wall/GPU ms): hybrid 0.097/0.026, naive
0.035/0.022, optimized 0.183/0.064 — i.e. ~0.119 ms/call optimized host
overhead, ~0.051 ms of it the 22-descriptor bank rebuild.

Measures the public wrappers (``hybrid_forward``/``naive_forward``/
``optimized_forward``): F9 is about the production call path — validation,
canonicalization, and descriptor plumbing included. Record, don't threshold
(v4 M12-T4): the script gates nothing and exits 0 after printing its table;
failures to *run* a requested backend still raise.

Needs PYTHONPATH handling via its own bootstrap and the hardened env prefix;
see scripts/README.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
BACKENDS = ("hybrid", "naive", "optimized")
DEFAULT_SHAPES = "8x128x768x1280,32x128x768x30522"


def parse_shape_list(value: str) -> tuple[tuple[int, int, int, int], ...]:
    """Parse a comma-separated list of BxSxDxV shapes (e.g. 8x128x768x1280)."""
    shapes = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.lower().split("x")
        if len(parts) != 4:
            raise argparse.ArgumentTypeError(
                f"shape '{item}' must be BxSxDxV (e.g. 8x128x768x1280)"
            )
        try:
            dims = tuple(int(part) for part in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"shape '{item}' has a non-integer dimension"
            ) from exc
        if any(dim <= 0 for dim in dims):
            raise argparse.ArgumentTypeError(f"shape '{item}' dimensions must be positive")
        shapes.append(dims)
    if not shapes:
        raise argparse.ArgumentTypeError("expected a comma-separated list of BxSxDxV shapes")
    return tuple(shapes)


def parse_backend_list(value: str) -> tuple[str, ...]:
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list of backends")
    for item in items:
        if item not in BACKENDS:
            raise argparse.ArgumentTypeError(
                f"unknown backend '{item}' (choose from {', '.join(BACKENDS)})"
            )
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", type=parse_shape_list, default=parse_shape_list(DEFAULT_SHAPES),
                        help="comma-separated BxSxDxV shapes "
                        f"(default {DEFAULT_SHAPES}: the ARCHITECTURE.md §6.3 "
                        "anchor plus the dev shape)")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="fp16")
    parser.add_argument("--bias", choices=("on", "off"), default="on")
    parser.add_argument("--backends", type=parse_backend_list, default=BACKENDS)
    parser.add_argument("--calls", type=int, default=300,
                        help="wall-clock loop length (one trailing sync)")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--rep", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("bench_host_overhead.py requires CUDA")
    if "optimized" in args.backends:
        from sparton._backend_runtime import is_optimized_backend_available

        available, reason = is_optimized_backend_available()
        if not available:
            raise RuntimeError(
                "bench_host_overhead.py with the optimized backend requires the "
                "optimized backend (CUDA sm_90+ and importable "
                f"triton.tools.tensor_descriptor): {reason}"
            )

    import sparton.sparton_kernel as sk
    from bench_sparton_baseline import ShapeSpec, bench_ms, make_inputs

    wrappers = {
        "hybrid": sk.hybrid_forward,
        "naive": sk.naive_forward,
        "optimized": sk.optimized_forward,
    }
    dtype = DTYPES[args.dtype]

    rows = []
    for batch_size, seq_len, dim, vocab in args.shapes:
        hidden, embed, bias, mask = make_inputs(
            ShapeSpec(batch_size, seq_len),
            dim=dim,
            vocab=vocab,
            dtype=dtype,
            seed=args.seed + batch_size * 65537 + seq_len,
        )
        if args.bias == "off":
            bias = None
        for backend in args.backends:
            wrapper = wrappers[backend]

            def fwd_call():
                return wrapper(hidden, embed, bias, mask)

            for _ in range(3):  # compile + autotune outside both timed legs
                fwd_call()
            torch.cuda.synchronize()

            gpu_ms = bench_ms(fwd_call, warmup=args.warmup, rep=args.rep)

            torch.cuda.synchronize()  # drain before the wall window opens
            t0 = time.perf_counter()
            for _ in range(args.calls):
                fwd_call()
            torch.cuda.synchronize()  # include all queued GPU work in the window
            t1 = time.perf_counter()
            wall_ms = (t1 - t0) / args.calls * 1e3

            rows.append({
                "backend": backend,
                "B": batch_size,
                "S": seq_len,
                "D": dim,
                "V": vocab,
                "wall_ms": wall_ms,
                "gpu_ms": gpu_ms,
                "host_ms": wall_ms - gpu_ms,
            })
        torch.cuda.empty_cache()

    print(
        "Host-overhead recorder: "
        f"dtype={args.dtype} bias={args.bias} calls={args.calls} "
        f"warmup/rep={args.warmup}/{args.rep} (wall has one trailing sync)"
    )
    print()
    print("| backend | B | S | D | V | wall ms | gpu ms | host ms |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['backend']} | {row['B']} | {row['S']} | {row['D']} | {row['V']} "
            f"| {row['wall_ms']:.3f} | {row['gpu_ms']:.3f} | {row['host_ms']:.3f} |"
        )
    print(
        f"host_overhead done: shapes={len(args.shapes)} "
        f"backends={','.join(args.backends)} dtype={args.dtype} calls={args.calls}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
