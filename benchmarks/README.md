# Sparton benchmarks and probes

Validated probe/benchmark scripts for the Gluon backend refactor, promoted
from session scratch on 2026-06-11. The forward plan and current rerun
command catalogue live in
[../docs/sparton_remaining_work_design_v3.md](../docs/sparton_remaining_work_design_v3.md);
the post-M8 review findings and M9/M10 provenance remain in
[../docs/sparton_remaining_work_design_v2.md](../docs/sparton_remaining_work_design_v2.md),
and the platform facts, profiling methodology, and original measured
evidence in
[../docs/sparton_gluon_remaining_work_design.md](../docs/sparton_gluon_remaining_work_design.md)
(§2.2, §3.1, §3.2, §3.5, §11, Appendices A and B).

## Environment

All scripts assume the hardened environment prefix (see design doc §2.4 and
`AGENTS.md`):

```bash
ENV='TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas CPATH=/usr/local/cuda-13.2/include TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor'
```

Run scripts serially (no concurrent Triton/Inductor-compiling processes).
Gluon-specific scripts check for CUDA sm_80+ and importable
`triton.experimental.gluon` before launching; benchmark numbers remain
hardware-specific to the validation machine named in the milestone docs.

## Scripts

| Script | Purpose | Invocation |
|---|---|---|
| `probe_mma_matrix.py` | MMA availability matrix (`mma_v2` / `wgmma` / `tcgen05`), each probe in a subprocess because LLVM selection failures are fatal aborts | `env $ENV python -u probe_mma_matrix.py` |
| `bench_gluon_gemm.py` | Standalone TMA+`mma_v2` GEMM on production Sparton layouts; uses Triton/Gluon autotune over the `_runtime_policy.py` GEMM policy universe, with runtime device/problem pruning and a ratio gate on the selected policy | `env $ENV python -u bench_gluon_gemm.py --dtype fp16 --include-block-n-256 --require-ratio 85` |
| `bench_sparton_baseline.py` | Canonical merged hybrid/autotuned-naive benchmark. Defaults model `naver/splade-code-06B` (`D=1024`, `V=151936`, bf16) on a fixed 3x3 grid: `B=4,8,16` and `S=256,512,768`, with all-ones masks. Pass `--optimized-policy on` to include the (now default) Gluon forward, its fwd+bwd timing, and its peak-memory column. | `env $ENV PYTHONPATH=../src python -u bench_sparton_baseline.py` (from this directory) |
| `bench_hybrid_baseline.py` | Compatibility wrapper for the old hybrid dev shape (`B=32,S=128,D=768,V=30522`, fp16, naive disabled) using the canonical benchmark implementation | `env $ENV PYTHONPATH=../src python -u bench_hybrid_baseline.py` (from this directory) |
| `bench_naive_baseline.py` | Compatibility wrapper for the M5 moderate naive shape (`B=4,S=64,D=64,V=4096`, fp16) using the canonical benchmark implementation | `env $ENV PYTHONPATH=../src python -u bench_naive_baseline.py` (from this directory) |
| `probe_gluon_epilogue.py` | M8 optimized-forward epilogue correctness probe for fp16/bf16, bias/no-bias, row masks, strict ties, and S/V tails | `env $ENV python -u probe_gluon_epilogue.py` |
| `soak_optimized_correctness.py` | M10 gate 5 shape soak: 384-case sweep (S/B/D/V/bias/dtype grid, random masks with a fully zeroed row) checking optimized scores vs a vectorized reference plus the tie-aware index contract; exits non-zero on failure | `env $ENV python -u soak_optimized_correctness.py [--quick]` |
| `probe_training_smoke.py` | M10 gate 6 tier-1 training smoke: 300-step head-only contrastive+FLOPS training from identical fp32 master weights, fp16 AMP (GradScaler) and bf16 autocast, hybrid-vs-optimized loss parity; exits non-zero on failure | `env $ENV python -u probe_training_smoke.py` |
| `ncu_runner.py` | Policy-derived Gluon GEMM launcher for `ncu --launch-skip`/`--launch-count` | see design doc §11 |
| `ncu_targets.py` | NVTX-wrapped cuBLAS / hybrid-forward / direct-backward profiling targets | see design doc §11 |
| `repro_inductor_env_defects.py` | Reproducer for the two cache-cold Inductor environment defects | `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` + variants per its docstring |

`python` above is the project venv interpreter,
`/workspace/venvs/sparton/bin/python`.

`bench_gluon_gemm.py` is the authoritative M7 gate path. It converts the
bounded policy universe to `triton.Config` objects, lets Triton select a policy
in-process, and reports the selected `POLICY_ID`, policy label, cache state,
correctness error, and precise ratio vs cuBLAS. It is not subprocess-isolated;
run it serially, and keep the generated candidate set bounded and previously
validated.

`bench_sparton_baseline.py` defaults to `--warmup 4 --rep 16` for forward and
GEMM timing, `--bwd-warmup 4 --bwd-rep 16` for hybrid forward+backward, and
`--naive-warmup 4 --naive-rep 16` for naive timing. The naive backend uses its
production Triton autotune search keyed by `(S, D, V)`, so cache-cold runs also
pay the compile/tune cost before timed repetitions. The optimized backend uses
a fixed production policy universe for Triton/Gluon descriptor slots, then
runtime GPU-derived active autotune candidates keyed by `(B, S, D, V)` when
`--optimized-policy on` is passed, so cache-cold optimized runs also compile
and tune before timed repetitions. Use `--naive-policy off` for hybrid-only
runs. The optimized backend is excluded unless `--optimized-policy on` is
passed. The canonical default emits exactly nine rows, one for each `(B, S)`
pair, formatted as a Markdown table.
