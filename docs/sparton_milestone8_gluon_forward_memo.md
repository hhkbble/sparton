# Sparton Milestone 8 Gluon Forward Memo

Date: 2026-06-11.
Last updated: 2026-06-12.

## Summary

Milestones M6-M8 from
[sparton_gluon_remaining_work_design.md](sparton_gluon_remaining_work_design.md)
are complete through the experimental policy-autotuned optimized forward:

- Added `_gluon_runtime.py`, a lazy compatibility shim for Gluon imports,
  Triton-version warning, and static sm_80+ -> `mma_v2` capability dispatch.
- Added `_runtime_policy.py`, a pure-Python policy generator for Gluon GEMM
  autotune and optimized-forward runtime candidate derivation with
  shared-memory, swizzle, block-thread, and `BLOCK_N` constraints.
- Extended `bench_gluon_gemm.py` to consume policy objects, support fp16/bf16,
  include BK=32/64-byte-swizzle candidates, use Triton/Gluon autotune, and
  enforce ratio gates on the selected policy.
- Added `probe_gluon_epilogue.py` for optimized-forward epilogue correctness.
- Added `_backend_optimized_gluon.py` and `sparton::optimized_fwd`, an
  experimental non-persistent Gluon fused forward that delegates backward to
  the existing hybrid backward.
- Migrated optimized forward from a single fallback-sized launch to bounded autotune: a
  fixed production policy universe keeps descriptor slots stable, while the
  active candidate set is derived at launch from the actual CUDA device
  profile and problem shape. The `64x64x64/3/2x2` policy remains the
  small-shape fallback and one of the autotune candidates.
- Refreshed the full merged benchmark after adding bounded Triton autotune to
  the M5 `naive` forward baseline, so the M8 comparison is against the current
  autotuned naive backend rather than the original fixed-tile launch.

Hybrid remains the default backend. `optimized` is explicit opt-in and remains
experimental; it is not promoted.

## Implementation Notes

The optimized forward uses `_runtime_policy.py` for autotune candidate
selection. The validated fallback policy is exposed through
`optimized_forward_fallback_policy()` and remains:

```text
BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, NUM_STAGES=3, warps=2x2, swizzle=128
```

The kernel owns one `(batch row, vocab tile)` CTA, consumes
`hidden.reshape(B*S, D)` and row-major `embed [V, D]` through TMA descriptors,
uses `ampere.mma_v2`, keeps running `[BLOCK_N]` max values/indices in
registers, and stores only `[B, V]` scores/indices. Host-side TMA descriptors
are built for the fixed production policy universe; Triton/Gluon autotune
prunes that universe to runtime GPU-derived active candidates and selects the
matching descriptor pair by static `POLICY_ID`. Local Triton 3.6.0 has
`gluon.jit` but no `gluon.autotune`, so `_gluon_runtime.py` exposes an
`autotune` compatibility symbol that falls back to `triton.autotune`.
The optimized backend and GEMM benchmark share host-side policy/config,
descriptor-bank, dtype, and pruning helpers; their Gluon JIT kernel bodies stay
separate because the benchmark materializes `C` while optimized forward runs
the Sparton max/argmax epilogue.

The epilogue uses explicit `gl.reduce` over `(value, row_index)` because local
Gluon's `gl.max(..., return_indices=True)` lowers through a broken layout-less
`arange` path. The reduce combine chooses the lower row index on equal values;
cross-chunk state updates remain strict `>`, preserving Sparton's zero-baseline
index policy.

Custom-op surface added:

```text
sparton::optimized_fwd(Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)
```

`SpartonHead(..., backend="optimized")` and `SPARTON_BACKEND=optimized` now
select this backend. The module is imported lazily; hybrid/naive users do not
import Gluon. Optimized functional tests and probes are gated on CUDA sm_80+
and importable `triton.experimental.gluon`, not on an RTX 5090 device name;
the benchmark evidence below remains specific to the validated RTX 5090 setup.

## Validation

Commands used the hardened environment:

```bash
TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas
CPATH=/usr/local/cuda-13.2/include
TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
```

Baseline and tests:

```text
py_compile package/training/tests/benchmarks: passed
pytest -q: 47 passed, 15 warnings in 8.71s
probe_gluon_epilogue.py: fp16/bf16 bias/no-bias passed
availability-gate smokes: bench_gluon_gemm.py fp16/bf16 autotune gates passed;
ncu_runner.py passed;
tiny bench_sparton_baseline.py --optimized-policy on passed
```

M6 smoke:

```text
mma_v2: OK, max_abs_err vs fp32 matmul = 0.000006
wgmma: expected subprocess abort, LLVM cannot select wgmma.commit_group
tcgen05: expected subprocess abort, LLVM cannot select tcgen05.wait.ld
```

M7 GEMM autotune gate, dev shape `M=4096, K=768, N=30522`:

```text
fp16 selected POLICY_ID=8, 64x64x64/3/2x2, 1.038 ms, 185.0 TFLOP/s, 86.452% of cuBLAS
bf16 selected POLICY_ID=8, 64x64x64/3/2x2, 1.022 ms, 187.9 TFLOP/s, 86.343% of cuBLAS
```

Stress GEMM autotune gate, `M=4096, K=1024, N=50257`, fp16:

```text
selected POLICY_ID=6, 128x64x32/4/4x2, 2.059 ms, 204.7 TFLOP/s, 173.585% of cuBLAS
```

The stress ratio is high because cuBLAS is unusually slow on that shape in the
L2-flushed benchmark regime; quote the absolute Gluon timing (`2.059 ms`)
alongside the ratio.

M8 memory/performance on dev forward shape `B=32, S=128, D=768, V=30522`,
fp16:

```text
optimized peak extra: 9.86 MiB
outputs:              9.31 MiB
memory gate:          1.06x outputs, passed (<2x)

hybrid+b forward:     1.176 ms, 140.50 MiB peak extra
optimized+b forward:  0.900 ms,   9.86 MiB peak extra
optimized no-bias:    0.896 ms
```

Full merged benchmark update after naive autotune and optimized policy
autotune, 2026-06-12. Command:

```bash
env $ENV PYTHONPATH=src /workspace/venvs/sparton/bin/python -u \
  benchmarks/bench_sparton_baseline.py --optimized-policy on
```

Preset:

```text
D=1024
V=151936
dtype=bf16
batch sizes=4,8,16
sequence lengths=256,512,768
mask=all ones
warmup/rep=4/16
bwd_warmup/bwd_rep=4/16
naive_warmup/naive_rep=4/16
optimized_warmup/optimized_rep=4/16
naive_policy=on
optimized_policy=on
naive_autotune=10 configs, key=(S,D,V)
optimized_autotune=11-config fixed production universe, runtime GPU-derived active candidates, key=(B,S,D,V)
```

Result:

| B | S | hyb+b ms | hyb ms | gemm ms | ovh % | hyb f+b ms | hyb MiB | naive+b ms | naive ms | naive MiB | opt+b ms | opt ms | opt MiB | out MiB | logits MiB | tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 256 | 1.805 | 1.671 | 1.323 | 36.4 | 4.381 | 135.14 | 2.043 | 2.037 | 5.89 | 1.450 | 1.455 | 5.89 | 5.80 | 296.8 | 567412 |
| 4 | 512 | 3.838 | 3.386 | 2.611 | 47.0 | 6.389 | 134.51 | 4.025 | 4.084 | 5.89 | 2.931 | 2.923 | 5.89 | 5.80 | 593.5 | 533619 |
| 4 | 768 | 5.174 | 4.537 | 3.954 | 30.9 | 7.765 | 132.30 | 6.095 | 6.239 | 5.89 | 4.385 | 4.366 | 5.89 | 5.80 | 890.2 | 593717 |
| 8 | 256 | 3.850 | 3.396 | 2.638 | 46.0 | 7.256 | 140.84 | 4.030 | 4.088 | 11.59 | 2.954 | 2.918 | 11.59 | 11.59 | 593.5 | 531889 |
| 8 | 512 | 7.633 | 6.787 | 5.274 | 44.7 | 11.070 | 140.22 | 8.075 | 8.360 | 11.59 | 5.780 | 5.796 | 11.59 | 11.59 | 1187.0 | 536589 |
| 8 | 768 | 10.298 | 9.127 | 7.884 | 30.6 | 13.806 | 138.41 | 12.446 | 12.515 | 12.00 | 8.731 | 8.696 | 12.00 | 11.59 | 1780.5 | 596629 |
| 16 | 256 | 7.594 | 6.779 | 5.268 | 44.2 | 12.785 | 152.43 | 8.086 | 8.368 | 23.18 | 5.769 | 5.754 | 23.18 | 23.18 | 1187.0 | 539338 |
| 16 | 512 | 15.325 | 13.618 | 10.501 | 45.9 | 20.731 | 151.81 | 16.742 | 16.661 | 23.18 | 11.491 | 11.561 | 23.18 | 23.18 | 2374.0 | 534563 |
| 16 | 768 | 22.395 | 19.619 | 15.739 | 42.3 | 28.464 | 143.57 | 24.969 | 25.016 | 23.18 | 17.378 | 17.501 | 23.18 | 23.18 | 3561.0 | 548684 |

Interpretation: bounded autotune makes the Triton-only naive baseline roughly
twice as fast as the original fixed-tile M5 run on these realistic shapes.
Optimized forward still uses output-only memory like naive and is consistently
faster than both hybrid and autotuned naive on the measured forward-only grids.
The default remains hybrid because optimized is still experimental Gluon code,
forward-only, and not yet through the promotion process.

## Remaining Risks

- Optimized forward now beats hybrid on the measured forward-only grids, but it
  remains experimental and has not gone through the default-promotion process.
- Backward still uses the hybrid Triton kernel, so optimized forward+backward
  inherits the current backward performance profile.
- Optimized autotune is bounded by the existing policy-generator families; it
  is not an exhaustive persistent/warp-specialized search.
- The naive autotune search is bounded for practical compile time; it makes the
  baseline fairer but is not an exhaustive architecture search.
- `gl.max(..., return_indices=True)` is not usable in this local Gluon version;
  the explicit reduce combine should be rechecked on Triton upgrades.
- Persistent scheduling and warp specialization remain M9 work.
