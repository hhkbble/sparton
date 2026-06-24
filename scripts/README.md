# Sparton scripts: benchmarks and probes

Validated probe/benchmark scripts, promoted from session scratch on 2026-06-11.
The built system — kernels, platform facts, profiling methodology, contracts,
kernel designs, and validation gates — is described in
[../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md); the per-milestone
development evidence (M2–M13) in
[../docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md); the working method and
kernel-optimization technique in
[../docs/METHODOLOGY.md](../docs/METHODOLOGY.md).

## Environment

All scripts assume the hardened environment prefix (see ARCHITECTURE.md §2.4
and `AGENTS.md`):

```bash
ENV='TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas CPATH=/usr/local/cuda-13.2/include TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor'
```

Run scripts serially (no concurrent Triton/Inductor-compiling processes). The
optimized kernel checks for CUDA sm_90+ and importable
`triton.tools.tensor_descriptor` before launching; benchmark numbers remain
hardware-specific to the validation machine named in ARCHITECTURE.md §2.1.

## Scripts

| Script | Purpose | Invocation |
|---|---|---|
| `bench_sparton_baseline.py` | Canonical merged hybrid/autotuned-naive benchmark. Defaults model `naver/splade-code-06B` (`D=1024`, `V=151936`, bf16) on a fixed 3x3 grid: `B=4,8,16` and `S=256,512,768`, with all-ones masks (`--mask-density p` switches to seeded Bernoulli masks — the M12-T4 sweep dimension; 1.0 is byte-identical to the default; `--mask-type contiguous --min-len-frac f` switches to realistic variable-length right-padding). Pass `--optimized-policy on` to include the pure-Triton `optimized` forward (persistent grid + measured self-tuner; honors the `SPARTON_OPTIMIZED_*` kernel-variant env flags). | `env $ENV PYTHONPATH=../src python -u bench_sparton_baseline.py [--optimized-policy on] [--mask-type contiguous]` (from this directory) |
| `bench_hybrid_baseline.py` | Compatibility wrapper for the old hybrid dev shape (`B=32,S=128,D=768,V=30522`, fp16, naive disabled) using the canonical benchmark implementation | `env $ENV PYTHONPATH=../src python -u bench_hybrid_baseline.py` (from this directory) |
| `bench_naive_baseline.py` | Compatibility wrapper for the M5 moderate naive shape (`B=4,S=64,D=64,V=4096`, fp16) using the canonical benchmark implementation | `env $ENV PYTHONPATH=../src python -u bench_naive_baseline.py` (from this directory) |
| `soak_optimized_correctness.py` | M10 gate 5 shape soak: sweep (S/B/D/V/bias/dtype grid, random masks with a fully zeroed row) checking optimized (pure-Triton) scores vs a vectorized reference plus the tie-aware index contract; honors the `SPARTON_OPTIMIZED_*` kernel-variant env flags (self-tuner defaulted off — correctness is tile-independent); exits non-zero on failure | `env $ENV python -u soak_optimized_correctness.py [--quick]` |
| `probe_training_smoke.py` | M10 gate 6 tier-1 training smoke: 300-step head-only contrastive+FLOPS training from identical fp32 master weights, fp16 AMP (GradScaler) and bf16 autocast, hybrid-vs-optimized loss parity; exits non-zero on failure | `env $ENV python -u probe_training_smoke.py` |
| `ncu_targets.py` | NVTX-wrapped cuBLAS / hybrid-forward / direct-backward profiling targets | see ARCHITECTURE.md §2.5 |
| `ncu_backward_target.py` | M11 parameterized direct-op backward profiling target: NVTX range `bwd_direct/` around main-thread `optimized_bwd_op` calls, arbitrary shape/dtype/bias or a capture-bundle record; the M11/M13 before/after counter source. The optimized backward is five kernels — match them by name (`--impl mono` is instead the single `mono_bwd_kernel`) and size `--launch-skip/--launch-count` in multiples of the per-call kernel count | `env $ENV PYTHONPATH=src ncu --nvtx --nvtx-include "bwd_direct/" -k "regex:bwd_prep_kernel\|embed_grad_kernel\|bwd_gather_payload_kernel\|uniform_hidden_grad_kernel\|mixed_hidden_grad_kernel" --launch-skip 5 --launch-count 5 --metrics <DEVELOPMENT.md M13 set> python -u scripts/ncu_backward_target.py --dtype fp16 --bias on` (the M13 transcripts were session artifacts; per-config IR/SASS is regenerable via `dump_backward_ir.py`) |
| `ncu_forward_target.py` | M12 parameterized direct-op forward profiling target: NVTX range `fwd_direct/` around main-thread `optimized_fwd_op` calls (autotune warms outside the range); the M12-T0 counter source. Recorded tables of record: [DEVELOPMENT.md M12](../docs/DEVELOPMENT.md) (its transcripts were session artifacts) | `env $ENV PYTHONPATH=src ncu --nvtx --nvtx-include "fwd_direct/" -k "regex:optimized_fwd_kernel" --launch-skip 1 --launch-count 1 --section <DEVELOPMENT.md M12 set> --metrics <DEVELOPMENT.md M12 set> python -u scripts/ncu_forward_target.py --dtype fp16 --bias on` |
| `bench_host_overhead.py` | M12-T4 wall-minus-GPU host-share recorder for the forward wrappers (standing documentation of the deferred-F9 launch overhead, ARCHITECTURE.md §6.7); record-don't-threshold, always exits 0. Recorded tables of record: [DEVELOPMENT.md M12](../docs/DEVELOPMENT.md) (its transcripts were session artifacts) | `env $ENV PYTHONPATH=src python -u scripts/bench_host_overhead.py` |
| `capture_index_distributions.py` | M11 real index-distribution capture: cached xlm-roberta-base via `training/model.py` (`head="sparton"`), optional 150-step tier-2 fine-tune, saves per-batch `(hidden_shape, max_scores, max_idx, mask)` + stats bundles for `bench_backward.py`. `--lambda-l1/--lambda-flops/--reg-warmup-steps` (M13-T0) override the fine-tune regularizer for sparse-regime probes; unset = M11 recipe, values recorded in bundle metadata. Bundles default to `tests/data/bundles/swimir_de_steps{N}.pt` (gitignored; see `tests/data/README.md`) and an existing file is **reused** rather than regenerated (`--force` overrides — regenerated bundles contain different records) | `env $ENV PYTHONPATH=src python -u scripts/capture_index_distributions.py --train-steps {0,150} [--quick]` |
| `bench_backward.py` | M11 distribution-aware backward harness: op-level `do_bench` timing of backward impls over {uniform, zipf, real-bundle} sources, mask densities, fp16/bf16, bias modes; per-cell verification vs `optimized` and the recorded 5-repeat determinism protocol; exits non-zero on failure | `env $ENV PYTHONPATH=src python -u scripts/bench_backward.py --sources uniform,zipf,real --bundle tests/data/bundles/swimir_de_steps0.pt --bundle tests/data/bundles/swimir_de_steps150.pt --active-fraction 0.10 --impls optimized,mono --determinism [--quick]` |
| `m13_traffic_model.py` | The M13 analytic traffic model of record (DEVELOPMENT.md M13 §4): per-buffer formulas vs the embedded M13-T0 measured counters on the four T0 shapes; exact distribution stats computed from the actual inputs (synthetic seeds + bundle records). Output of record committed at `tests/data/m13_traffic_model_out.txt` | `env $ENV PYTHONPATH=src python -u scripts/m13_traffic_model.py [--bundle-dir tests/data/bundles]` |
| `dump_backward_ir.py` | Backward autotune-selection capture + per-config TTGIR/PTX/SASS dump with a load/atomic census (METHODOLOGY.md §6.1–§6.2): the standing lowering-visibility tool for the optimized backward; warm-cache cheap, one op call per shape | `env $ENV PYTHONPATH=src python -u scripts/dump_backward_ir.py [--out tests/data/ir_dump] [--shapes dev,corner,query,doc]` |
| `repro_inductor_env_defects.py` | Reproducer for the two cache-cold Inductor environment defects | `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` + variants per its docstring |

`python` above is the project venv interpreter,
`/workspace/venvs/sparton/bin/python`.

`bench_sparton_baseline.py` defaults to `--warmup 4 --rep 16` for forward and
GEMM timing, `--bwd-warmup 4 --bwd-rep 16` for hybrid forward+backward, and
`--naive-warmup 4 --naive-rep 16` for naive timing. The naive kernel uses its
production Triton autotune search keyed by `(S, D, V)`, so cache-cold runs also
pay the compile/tune cost before timed repetitions. The optimized kernel
(pure-Triton, persistent) runs its measured self-tuner keyed on
`(D, V, dtype, arch)` on first call when `--optimized-policy on` is passed
(default off), so cache-cold optimized runs also tune before timed repetitions;
set `SPARTON_OPTIMIZED_AUTOTUNE=off` to use the fast analytic tile. Use
`--naive-policy off` for hybrid-only runs. The canonical default emits exactly
nine rows, one for each `(B, S)` pair, formatted as a Markdown table.
