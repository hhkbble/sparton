# Sparton benchmarks and probes

Validated probe/benchmark scripts for the Gluon backend refactor, promoted
from session scratch on 2026-06-11. The canonical interpretation of their
results, the profiling methodology, and the rerun command catalogue live in
[../docs/sparton_gluon_remaining_work_design.md](../docs/sparton_gluon_remaining_work_design.md)
(§2.2, §3.1, §3.2, §3.5, §11, Appendices A and B).

## Environment

All scripts assume the hardened environment prefix (see design doc §2.4 and
`AGENTS.md`):

```bash
ENV='TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas CPATH=/usr/local/cuda-13.2/include TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor'
```

Run scripts serially (no concurrent Triton/Inductor-compiling processes).

## Scripts

| Script | Purpose | Invocation |
|---|---|---|
| `probe_mma_matrix.py` | MMA availability matrix (`mma_v2` / `wgmma` / `tcgen05`), each probe in a subprocess because LLVM selection failures are fatal aborts | `env $ENV python -u probe_mma_matrix.py` |
| `bench_gluon_gemm.py` | Standalone TMA+`mma_v2` GEMM on production Sparton layouts; per-config subprocesses with timeouts (deadlock-safe); correctness + throughput vs cuBLAS | `env $ENV python -u bench_gluon_gemm.py [--config N]` |
| `bench_sparton_baseline.py` | Canonical merged hybrid/naive benchmark. Defaults model `naver/splade-code-06B` (`D=1024`, `V=151936`, bf16) on a fixed 3x3 grid: `B=4,8,16` and `S=256,512,768`, with all-ones masks | `env $ENV PYTHONPATH=../src python -u bench_sparton_baseline.py` (from this directory) |
| `bench_hybrid_baseline.py` | Compatibility wrapper for the old hybrid dev shape (`B=32,S=128,D=768,V=30522`, fp16, naive disabled) using the canonical benchmark implementation | `env $ENV PYTHONPATH=../src python -u bench_hybrid_baseline.py` (from this directory) |
| `bench_naive_baseline.py` | Compatibility wrapper for the M5 moderate naive shape (`B=4,S=64,D=64,V=4096`, fp16) using the canonical benchmark implementation | `env $ENV PYTHONPATH=../src python -u bench_naive_baseline.py` (from this directory) |
| `ncu_runner.py` | Fixed-config Gluon GEMM launcher for `ncu --launch-skip`/`--launch-count` | see design doc §11 |
| `ncu_targets.py` | NVTX-wrapped cuBLAS / hybrid-forward / direct-backward profiling targets | see design doc §11 |
| `repro_inductor_env_defects.py` | Reproducer for the two cache-cold Inductor environment defects | `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` + variants per its docstring |

`python` above is the project venv interpreter,
`/workspace/venvs/sparton/bin/python`.

`bench_sparton_baseline.py` defaults to `--warmup 4 --rep 16` for forward and
GEMM timing, `--bwd-warmup 4 --bwd-rep 16` for hybrid forward+backward, and
`--naive-warmup 4 --naive-rep 16` for naive timing. Use `--naive-policy off`
for hybrid-only sweeps. The canonical default emits exactly nine rows, one for
each `(B, S)` pair, formatted as a Markdown table.
