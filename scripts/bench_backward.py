"""Distribution-aware backward benchmark harness (M11 T2).

Times backward implementations at the op level (each impl allocates its own
fp32 grad buffers, exactly like ``sparton::fused_sparton_bwd`` — the timing
includes the zero-fills training pays; ncu provides the kernel-level view and
the two regimes are never compared) over three index-distribution sources:

  uniform   per-row active set of ``round(active_fraction * V)`` vocab
            entries drawn uniformly; the regime of the existing tests.
  zipf      active set drawn without replacement with weight
            ``(vocab_id + 1) ** -s`` (``--zipf-s``, default 1.1). Identity
            rank-to-id mapping, so the active set concentrates at low vocab
            ids: a documented proxy for hot-token concentration in v-tile
            space (early-exit skew, load imbalance). It does NOT model
            index collisions (below) — never sufficient alone (METHODOLOGY.md §A.3).
  real      records captured by ``capture_index_distributions.py``. These
            carry the property no synthetic source reproduces: hot-row index
            collisions (measured ~20% of active vocab entries choosing the
            same sequence position), which concentrate ``hidden_grad``
            atomics onto a few rows. Real records run at their captured mask
            density; the ``--densities`` axis applies to synthetic sources
            only.

Synthetic input contract (pinned by ``test_bench_backward_synthetic_inputs_
honor_contract``): masks are Bernoulli(density) with at least one unmasked
position per row; ``max_scores`` is exactly 0 off the active set and
``U(0.05, 2.0)`` on it (only ``exp(-scores)`` magnitude matters); ``max_idx``
points uniformly at unmasked positions for active entries and 0 elsewhere —
mirroring the forward's zero-baseline semantics, under which a positive
score's winner is always unmasked. ``grad_out`` is fp32 (the dtype the
autograd wrappers pass after the standing ``scores.float().sum()`` loss
convention); hidden/embed operand values are synthesized — the backward's
access pattern depends only on scores' sparsity, ``max_idx``, and shapes.

Per cell each impl is verified against ``current`` (``assert_close``,
rtol=atol=1e-3) before timing unless ``--no-verify``; ``--determinism`` adds
the recorded (non-gate) protocol: 5 same-input repeats reporting the max
relative spread of the grad norms and of a fixed strided-sum loss proxy
(atomic-order-sensitive linear functional).

Output is one markdown row per (source, cell, dtype, bias, impl) with ms and
speedup vs ``current``; a provenance header logs versions, GPU, and seed.
Exits non-zero listing verification failures. ``--quick`` runs a small
development subset; the full matrix is the gate of record. Note that
``--quick`` pins contract-relevant axes (``bias=on``, fp16 only): it
cannot prove the axes it holds fixed — the M13 prototypes' bias_grad
optionality bug survived every quick gate (DEVELOPMENT.md M13 §8 item 2).
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


def bench_ms(fn, *, warmup: int, rep: int) -> float:
    import triton

    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def build_impls(names: list[str]) -> dict[str, callable]:
    """Map impl names to callables with the exact backward-op signature.

    ``(grad_out, max_scores, max_idx, hidden, embed, bias, mask) ->
    (hidden_grad, embed_grad, bias_grad?)`` — every impl allocates its own
    grad buffers so timings are op-level and directly comparable.
    """

    import sparton.sparton_kernel as sk

    impls: dict[str, callable] = {}
    for name in names:
        if name == "current":
            impls[name] = sk.fused_sparton_bwd_op
        elif name == "legacy":
            legacy = getattr(sk, "legacy_fused_sparton_bwd", None)
            if legacy is None:
                raise RuntimeError(
                    "impl 'legacy' requires the post-M11-T4 tree "
                    "(sparton.sparton_kernel.legacy_fused_sparton_bwd)"
                )
            impls[name] = legacy
        else:
            bench_dir = str(Path(__file__).resolve().parent)
            if bench_dir not in sys.path:
                sys.path.insert(0, bench_dir)
            try:
                from bwd_prototypes import PROTOTYPES
            except ImportError as exc:
                raise RuntimeError(
                    f"impl {name!r} requires scripts/bwd_prototypes.py "
                    f"(M11 T3 decision-probe tree): {exc}"
                ) from exc
            if name not in PROTOTYPES:
                raise RuntimeError(
                    f"unknown impl {name!r} (not current/legacy and not in "
                    f"bwd_prototypes.PROTOTYPES {sorted(PROTOTYPES)})"
                )
            impls[name] = PROTOTYPES[name]
    return impls


def make_synthetic_case(
    *,
    source: str,
    batch_size: int,
    seq_len: int,
    dim: int,
    vocab: int,
    density: float,
    active_fraction: float,
    zipf_s: float,
    dtype: torch.dtype,
    bias_on: bool,
    seed: int,
) -> dict[str, torch.Tensor | None]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    B, S, D, V = batch_size, seq_len, dim, vocab

    mask = (torch.rand((B, S), device="cuda", generator=generator) < density).to(torch.int32)
    empty_rows = mask.sum(dim=1) == 0
    mask[empty_rows, 0] = 1  # contract: at least one unmasked position per row

    k = max(1, round(active_fraction * V))
    if source == "uniform":
        keys = torch.rand((B, V), device="cuda", generator=generator)
    elif source == "zipf":
        # Gumbel-top-k = weighted sampling without replacement; identity
        # rank-to-id mapping concentrates the active set at low vocab ids.
        weights = (torch.arange(1, V + 1, device="cuda", dtype=torch.float32)) ** (-zipf_s)
        gumbel = -torch.log(
            -torch.log(torch.rand((B, V), device="cuda", generator=generator))
        )
        keys = torch.log(weights)[None, :] + gumbel
    else:
        raise ValueError(f"unknown synthetic source {source!r}")
    active_idx = keys.topk(k, dim=1).indices
    active = torch.zeros((B, V), device="cuda", dtype=torch.bool)
    active.scatter_(1, active_idx, True)

    max_scores = torch.zeros((B, V), device="cuda", dtype=dtype)
    active_values = (
        0.05 + 1.95 * torch.rand((B, V), device="cuda", generator=generator)
    ).to(dtype)
    max_scores[active] = active_values[active]

    # Active entries point uniformly at unmasked positions of their row.
    unmasked = mask.bool()
    counts = unmasked.sum(dim=1)
    positions = torch.argsort(unmasked.int(), dim=1, descending=True, stable=True)
    draw = (
        torch.rand((B, V), device="cuda", generator=generator)
        * counts[:, None].float()
    ).long().clamp_(max=(counts[:, None] - 1))
    max_idx = torch.zeros((B, V), device="cuda", dtype=torch.int64)
    sampled = positions.gather(1, draw)
    max_idx[active] = sampled[active]

    hidden = torch.randn((B, S, D), device="cuda", dtype=dtype, generator=generator) * 0.05
    embed = torch.randn((V, D), device="cuda", dtype=dtype, generator=generator) * 0.05
    bias = (
        torch.randn((V,), device="cuda", dtype=dtype, generator=generator) * 0.05
        if bias_on
        else None
    )
    grad_out = torch.randn((B, V), device="cuda", dtype=torch.float32, generator=generator)
    return {
        "max_scores": max_scores,
        "max_idx": max_idx,
        "hidden": hidden,
        "embed": embed,
        "bias": bias,
        "mask": mask,
        "grad_out": grad_out,
    }


def make_real_case(
    record: dict,
    *,
    dtype: torch.dtype,
    bias_on: bool,
    seed: int,
) -> dict[str, torch.Tensor | None]:
    B, S, D = record["hidden_shape"]
    V = record["max_scores"].shape[1]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return {
        "max_scores": record["max_scores"].to(device="cuda", dtype=dtype),
        "max_idx": record["max_idx"].to(device="cuda", dtype=torch.int64),
        "hidden": torch.randn((B, S, D), device="cuda", dtype=dtype, generator=generator) * 0.05,
        "embed": torch.randn((V, D), device="cuda", dtype=dtype, generator=generator) * 0.05,
        "bias": (
            torch.randn((V,), device="cuda", dtype=dtype, generator=generator) * 0.05
            if bias_on
            else None
        ),
        "mask": record["mask"].to(device="cuda", dtype=torch.int32),
        "grad_out": torch.randn((B, V), device="cuda", dtype=torch.float32, generator=generator),
    }


def call_impl(impl, case):
    return impl(
        case["grad_out"], case["max_scores"], case["max_idx"],
        case["hidden"], case["embed"], case["bias"], case["mask"],
    )


def loss_proxy(grads, stride: int = 4097) -> float:
    """Fixed atomic-order-sensitive linear functional of the gradients."""

    total = 0.0
    for grad in grads:
        if grad is not None and grad.dim() > 0:
            total += grad.flatten()[::stride].sum().item()
    return total


def determinism_spread(impl, case, repeats: int = 5) -> dict[str, float]:
    norms_h, norms_e, proxies = [], [], []
    for _ in range(repeats):
        hidden_grad, embed_grad, bias_grad = call_impl(impl, case)
        norms_h.append(hidden_grad.norm().item())
        norms_e.append(embed_grad.norm().item())
        proxies.append(loss_proxy((hidden_grad, embed_grad, bias_grad)))

    def rel_spread(values: list[float]) -> float:
        lo, hi = min(values), max(values)
        scale = max(abs(v) for v in values)
        return (hi - lo) / scale if scale > 0 else 0.0

    return {
        "hidden": rel_spread(norms_h),
        "embed": rel_spread(norms_e),
        "proxy": rel_spread(proxies),
    }


def verify_against_current(impl, current, case) -> None:
    got = call_impl(impl, case)
    want = call_impl(current, case)
    for name, g, w in zip(("hidden_grad", "embed_grad", "bias_grad"), got, want):
        if (g is None) != (w is None):
            raise AssertionError(f"{name}: optionality mismatch vs current")
        if g is not None and g.dim() > 0:
            torch.testing.assert_close(g, w, rtol=1e-3, atol=1e-3, msg=name)


def iter_cells(args, bundles):
    """Yield (source, label, case_builder) triples for the run matrix."""

    cell_seed = args.seed
    for source in args.sources:
        if source in ("uniform", "zipf"):
            for density in args.densities:
                cell_seed += 1
                seed = cell_seed
                label = f"{density:.0f}%"
                yield source, label, (
                    lambda dtype, bias_on, source=source, density=density, seed=seed:
                    make_synthetic_case(
                        source=source,
                        batch_size=args.batch_size,
                        seq_len=args.seq_len,
                        dim=args.dim,
                        vocab=args.vocab,
                        density=density / 100.0,
                        active_fraction=args.active_fraction,
                        zipf_s=args.zipf_s,
                        dtype=dtype,
                        bias_on=bias_on,
                        seed=seed,
                    )
                )
        elif source == "real":
            for bundle_path, bundle in bundles:
                records = bundle["records"]
                if 0 < args.max_real_records < len(records):
                    # Balance sides: records alternate query/doc, so a plain
                    # even stride would pick a single side.
                    by_side: dict[str, list] = {}
                    for record in records:
                        by_side.setdefault(record["side"], []).append(record)
                    picked = []
                    quota = max(1, args.max_real_records // max(1, len(by_side)))
                    for side_records in by_side.values():
                        stride = max(1, len(side_records) // quota)
                        picked.extend(side_records[::stride][:quota])
                    print(
                        f"NOTE: subsampled {len(picked)}/{len(records)} records of "
                        f"{bundle_path.name} (--max-real-records {args.max_real_records})",
                        flush=True,
                    )
                else:
                    picked = records
                for rec_index, record in enumerate(picked):
                    cell_seed += 1
                    seed = cell_seed
                    B, S, _D = record["hidden_shape"]
                    stats = record["stats"]
                    label = (
                        f"{bundle_path.stem}:r{rec_index}:{record['side']}:"
                        f"B{B}xS{S} act={stats['active_fraction']:.2f} "
                        f"den={stats['mask_density']:.2f} top1={stats['idx_top1_share']:.2f}"
                    )
                    yield source, label, (
                        lambda dtype, bias_on, record=record, seed=seed:
                        make_real_case(record, dtype=dtype, bias_on=bias_on, seed=seed)
                    )
        else:
            raise ValueError(f"unknown source {source!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=str, default="uniform,zipf",
                        help="comma list of uniform,zipf,real")
    parser.add_argument("--bundle", action="append", default=[],
                        help="capture bundle path (repeatable; required for real)")
    parser.add_argument("--densities", type=str, default="25,75,100",
                        help="synthetic mask densities in percent")
    parser.add_argument("--dtypes", type=str, default="fp16,bf16")
    parser.add_argument("--bias", choices=("both", "on", "off"), default="both")
    parser.add_argument("--impls", type=str, default="current")
    parser.add_argument("--active-fraction", type=float, default=None,
                        help="synthetic active fraction (default: mean of real "
                        "bundle stats if given, else 0.10)")
    parser.add_argument("--zipf-s", type=float, default=1.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--vocab", type=int, default=30522)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--rep", type=int, default=16)
    parser.add_argument("--max-real-records", type=int, default=8,
                        help="records benchmarked per bundle (0 = all)")
    parser.add_argument("--determinism", action="store_true",
                        help="record the 5-repeat spread protocol per cell")
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="small development subset")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("bench_backward.py requires CUDA")

    args.sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    args.densities = [float(d) for d in args.densities.split(",")]
    dtype_names = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    impl_names = [i.strip() for i in args.impls.split(",") if i.strip()]
    bias_modes = {"both": [True, False], "on": [True], "off": [False]}[args.bias]
    if args.quick:
        args.densities = [75.0]
        dtype_names = ["fp16"]
        bias_modes = [True]
        args.max_real_records = 2
        args.rep = 8

    bundles = []
    if "real" in args.sources:
        if not args.bundle:
            raise RuntimeError("--sources real requires at least one --bundle")
        for path_str in args.bundle:
            path = Path(path_str)
            bundle = torch.load(path, map_location="cpu", weights_only=True)
            bundles.append((path, bundle))

    if args.active_fraction is None:
        if bundles:
            all_records = [r for _p, b in bundles for r in b["records"]]
            args.active_fraction = sum(
                r["stats"]["active_fraction"] for r in all_records
            ) / len(all_records)
            fraction_source = "mean of real bundle stats"
        else:
            args.active_fraction = 0.10
            fraction_source = "default"
    else:
        fraction_source = "--active-fraction"

    import triton

    device_name = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    print(
        f"bench_backward: torch {torch.__version__} triton {triton.__version__} "
        f"{device_name} sm_{capability[0]}{capability[1]} seed={args.seed} | "
        f"impls={impl_names} sources={args.sources} dtypes={dtype_names} "
        f"bias={[('on' if b else 'off') for b in bias_modes]} "
        f"synthetic B={args.batch_size} S={args.seq_len} D={args.dim} V={args.vocab} "
        f"active_fraction={args.active_fraction:.4f} ({fraction_source}) "
        f"zipf_s={args.zipf_s}",
        flush=True,
    )

    impls = build_impls(impl_names)
    failures: list[tuple[str, str]] = []
    det_header = " det h/e/proxy |" if args.determinism else ""
    print(f"| source | cell | dtype | bias | impl | ms | x current |{det_header}", flush=True)
    print(f"|---|---|---|---|---|---|---|{'---|' if args.determinism else ''}", flush=True)

    cells = 0
    for source, label, case_builder in iter_cells(args, bundles):
        for dtype_name in dtype_names:
            for bias_on in bias_modes:
                case = None
                current_ms = None
                for impl_name in impl_names:
                    impl = impls[impl_name]
                    cell_id = f"{source}/{label}/{dtype_name}/{'bias' if bias_on else 'no_bias'}"
                    try:
                        if case is None:
                            case = case_builder(DTYPES[dtype_name], bias_on)
                        if impl_name != "current" and not args.no_verify:
                            verify_against_current(impl, impls.get("current") or
                                                   build_impls(["current"])["current"], case)
                        ms = bench_ms(lambda: call_impl(impl, case),
                                      warmup=args.warmup, rep=args.rep)
                        if impl_name == "current":
                            current_ms = ms
                        ratio = (current_ms / ms) if current_ms else float("nan")
                        det = ""
                        if args.determinism:
                            spread = determinism_spread(impl, case)
                            det = (
                                f" {spread['hidden']:.2e}/{spread['embed']:.2e}/"
                                f"{spread['proxy']:.2e} |"
                            )
                        print(
                            f"| {source} | {label} | {dtype_name} | "
                            f"{'on' if bias_on else 'off'} | {impl_name} | "
                            f"{ms:.3f} | {ratio:.3f} |{det}",
                            flush=True,
                        )
                        cells += 1
                    except Exception as exc:  # noqa: BLE001 — gate script reports and continues
                        failures.append((cell_id + f"/{impl_name}", f"{type(exc).__name__}: {exc}"))
                        print(f"FAIL {cell_id}/{impl_name}: {type(exc).__name__}: {str(exc)[:200]}",
                              flush=True)
                case = None
                torch.cuda.empty_cache()

    print(
        f"bench_backward summary: {cells} cells timed, {len(failures)} failures",
        flush=True,
    )
    if failures:
        print(f"{len(failures)} FAILURES:")
        for cell_id, message in failures:
            print(f"  {cell_id}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
