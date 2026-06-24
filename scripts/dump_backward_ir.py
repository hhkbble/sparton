"""Dump the backward kernels' autotune selections and per-config IR/SASS.

The standing config+lowering visibility tool for the optimized backward
(METHODOLOGY.md §6.1–§6.2; promoted from the M13-T0 session probe). For each
requested shape it runs one production `optimized_bwd_op` call, reads
the autotuner selections host-side (cache-hit selections print nothing
under TRITON_PRINT_AUTOTUNING — the M12 lesson), then warmup-compiles each
selected configuration and writes `ttgir`/`ptx`/SASS with per-config
filenames, plus a load/atomic instruction census per file. Use it to
confirm a lowering (gather vector widths, scan form, spills) before
benchmarking a config family, and to attribute counter surprises.

Covers: `uniform_hidden_grad_kernel` (autotuned), `mixed_hidden_grad_kernel`
(fixed config derived from the uniform winner — the complement-granularity
invariant), `embed_grad_kernel` (autotuned), and the plain-jit
`bwd_prep_kernel`/`bwd_gather_payload_kernel`. Real-record shapes need the
capture bundles (default ``tests/data/bundles/``; regenerable via
`scripts/capture_index_distributions.py`) and are skipped with a notice
when absent.

Compilation is warm-cache cheap; the only GPU work is one op call per
shape. Output defaults to ``tests/data/ir_dump/`` (gitignored). Exits
non-zero listing every failed dump.

Usage:

    env $ENV PYTHONPATH=src python -u scripts/dump_backward_ir.py \
        [--out tests/data/ir_dump] [--bundle-dir tests/data/bundles] \
        [--shapes dev,corner,query,doc]
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from bench_backward import make_synthetic_case, make_real_case  # noqa: E402

DEFAULT_NVDISASM = "/usr/local/cuda-13.2/bin/nvdisasm"

SHAPES = {
    "dev": dict(kind="synthetic", batch_size=32, seq_len=128, dim=768,
                vocab=30522, dtype=torch.float16),
    "corner": dict(kind="synthetic", batch_size=16, seq_len=512, dim=1024,
                   vocab=151936, dtype=torch.bfloat16),
    "query": dict(kind="real", bundle="swimir_de_steps0.pt", record=0,
                  dtype=torch.float16),
    "doc": dict(kind="real", bundle="swimir_de_steps150.pt", record=1,
                dtype=torch.float16),
}


def build_case(spec, bundle_dir: Path):
    if spec["kind"] == "synthetic":
        return make_synthetic_case(
            source="uniform", batch_size=spec["batch_size"],
            seq_len=spec["seq_len"], dim=spec["dim"], vocab=spec["vocab"],
            density=0.75, active_fraction=1.0, zipf_s=1.1,
            dtype=spec["dtype"], bias_on=True, seed=0)
    path = bundle_dir / spec["bundle"]
    if not path.exists():
        return None
    bundle = torch.load(path, weights_only=False)
    return make_real_case(bundle["records"][spec["record"]],
                          dtype=spec["dtype"], bias_on=True, seed=0)


def census(text: str) -> dict[str, int]:
    return dict(collections.Counter(re.findall(
        r"LDG\.E\.128|LDG\.E\.64|LDG\.E[^.\d]|LDGSTS|REDG\.E\.ADD\.F32|STL|LDL",
        text)))


def dump_compiled(tag: str, compiled, out_dir: Path, nvdisasm: str,
                  failures: list) -> None:
    for stage in ("ttgir", "ptx"):
        text = compiled.asm.get(stage)
        if text is not None:
            (out_dir / f"{tag}.{stage}").write_text(
                text if isinstance(text, str) else text.decode())
    cubin = compiled.asm.get("cubin")
    summary = {}
    if cubin is not None:
        cubin_path = out_dir / f"{tag}.cubin"
        cubin_path.write_bytes(cubin)
        proc = subprocess.run([nvdisasm, "-c", str(cubin_path)],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            failures.append((tag, f"nvdisasm failed: {proc.stderr[:120]}"))
        else:
            (out_dir / f"{tag}.sass").write_text(proc.stdout)
            summary = census(proc.stdout)
            regs = re.search(r"REG:(\d+)", proc.stdout)
            if regs:
                summary["regs"] = int(regs.group(1))
    print(f"  dumped {tag}: {summary}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str,
                        default=str(REPO_ROOT / "tests" / "data" / "ir_dump"))
    parser.add_argument("--bundle-dir", type=str,
                        default=str(REPO_ROOT / "tests" / "data" / "bundles"))
    parser.add_argument("--shapes", type=str, default="dev,corner,query,doc",
                        help="comma list from {dev,corner,query,doc}")
    parser.add_argument("--nvdisasm", type=str,
                        default=os.environ.get("NVDISASM", DEFAULT_NVDISASM))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("dump_backward_ir.py requires CUDA")

    import sparton.api as sk
    from sparton.backward.optimized import (
        bwd_gather_payload_kernel,
        bwd_prep_kernel,
        embed_grad_kernel,
        mixed_hidden_grad_kernel,
        uniform_hidden_grad_kernel,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir = Path(args.bundle_dir)
    failures: list[tuple[str, str]] = []
    torch.manual_seed(0)
    dev = torch.device("cuda")

    shape_meta = {}
    for name in [s.strip() for s in args.shapes.split(",") if s.strip()]:
        spec = SHAPES[name]
        case = build_case(spec, bundle_dir)
        if case is None:
            print(f"SKIP {name}: bundle missing under {bundle_dir} "
                  f"(see scripts/capture_index_distributions.py)", flush=True)
            continue
        uni_before = dict(uniform_hidden_grad_kernel.cache)
        emb_before = dict(embed_grad_kernel.cache)
        sk.optimized_bwd_op(
            case["grad_out"], case["max_scores"], case["max_idx"],
            case["hidden"], case["embed"], case["bias"], case["mask"])
        torch.cuda.synchronize()
        D = case["hidden"].shape[2]
        V = case["embed"].shape[0]
        dtype = case["hidden"].dtype
        print(f"== {name} (D={D} V={V} {dtype})", flush=True)
        for kernel, before, label in (
            (uniform_hidden_grad_kernel, uni_before, "uniform"),
            (embed_grad_kernel, emb_before, "embed"),
        ):
            for key, cfg in kernel.cache.items():
                if key not in before:
                    print(f"  {label} selected: {cfg.kwargs} "
                          f"warps={cfg.num_warps} stages={cfg.num_stages}",
                          flush=True)
                    shape_meta[(name, label)] = (cfg, D, V, dtype)
            if (name, label) not in shape_meta:
                print(f"  {label}: cache hit on an existing key (selection "
                      f"shared with an earlier shape)", flush=True)

    print("== dumping compiled forms", flush=True)
    f32 = torch.empty(8, device=dev, dtype=torch.float32)
    i32 = torch.empty(8, device=dev, dtype=torch.int32)
    i64 = torch.empty(8, device=dev, dtype=torch.int64)
    seen: set[str] = set()
    for (name, label), (cfg, D, V, dtype) in shape_meta.items():
        elt = torch.empty(8, device=dev, dtype=dtype)
        kw = cfg.kwargs
        if label == "uniform":
            tag = (f"uniform__{name}__CHUNK{kw['CHUNK']}_BD{kw['BLOCK_D']}"
                   f"_w{cfg.num_warps}_s{cfg.num_stages}")
            if tag in seen:
                continue
            seen.add(tag)
            compiled = uniform_hidden_grad_kernel.fn.warmup(
                keys_ptr=i32, g_ptr=f32, v_ptr=i32, embed_ptr=elt,
                hidden_grad_ptr=f32, n_active_ptr=i32, total=8, batch_size=1,
                seq_len=8, vocab_size=8, hidden_dim=D, CHUNK=kw['CHUNK'],
                BLOCK_D=kw['BLOCK_D'], num_warps=cfg.num_warps,
                num_stages=cfg.num_stages, grid=(1, 1))
            dump_compiled(tag, compiled, out_dir, args.nvdisasm, failures)
            # The mixed pass runs at the uniform winner's granularity
            # (complement invariant) with its fixed launch shape.
            granule = kw['CHUNK']
            sub = min(granule, 64)
            mtag = f"mixed__{name}__GR{granule}_SUB{sub}_BD128_w4_s2"
            if mtag not in seen:
                seen.add(mtag)
                compiled = mixed_hidden_grad_kernel.warmup(
                    keys_ptr=i32, g_ptr=f32, v_ptr=i32, embed_ptr=elt,
                    hidden_grad_ptr=f32, total=8, batch_size=1, seq_len=8,
                    vocab_size=8, hidden_dim=D, GRANULE=granule, SUB=sub,
                    BLOCK_D=128, num_warps=4, num_stages=2, grid=(1,))
                dump_compiled(mtag, compiled, out_dir, args.nvdisasm, failures)
        else:
            tag = (f"embed__{name}__BB{kw['BLOCK_B']}_BV{kw['BLOCK_V']}"
                   f"_BD{kw['BLOCK_D']}_w{cfg.num_warps}_s{cfg.num_stages}")
            if tag in seen:
                continue
            seen.add(tag)
            compiled = embed_grad_kernel.fn.warmup(
                g_ptr=f32, idx_ptr=i32, hidden_ptr=elt, embed_grad_ptr=f32,
                bias_grad_ptr=f32, batch_size=1, seq_len=8, hidden_dim=D,
                vocab_size=V, HAS_BIAS=True, BLOCK_B=kw['BLOCK_B'],
                BLOCK_V=kw['BLOCK_V'], BLOCK_D=kw['BLOCK_D'],
                num_warps=cfg.num_warps, num_stages=cfg.num_stages,
                grid=(1, 1))
            dump_compiled(tag, compiled, out_dir, args.nvdisasm, failures)

    f16 = torch.empty(8, device=dev, dtype=torch.float16)
    compiled = bwd_prep_kernel.warmup(
        scores_ptr=f16, grad_ptr=f32, idx_ptr=i64, g_ptr=f32, idx32_ptr=i32,
        keys_ptr=i32, n_active_ptr=i32, total=8, seq_len=8, vocab_size=8,
        num_rows=8, BLOCK=1024, grid=(1,))
    dump_compiled("prep__fp16_BLOCK1024", compiled, out_dir, args.nvdisasm,
                  failures)
    compiled = bwd_gather_payload_kernel.warmup(
        perm_ptr=i64, g_full_ptr=f32, g_sorted_ptr=f32, v_sorted_ptr=i32,
        total=8, vocab_size=8, BLOCK=1024, grid=(1,))
    dump_compiled("payload__BLOCK1024", compiled, out_dir, args.nvdisasm,
                  failures)

    print(f"dump_backward_ir summary: {len(seen) + 2} compiled forms -> "
          f"{out_dir}, {len(failures)} failures", flush=True)
    if failures:
        for tag, message in failures:
            print(f"  {tag}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
