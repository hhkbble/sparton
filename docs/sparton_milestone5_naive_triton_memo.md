# Sparton Milestone 5 Naive Triton Backend Memo

Date: 2026-06-11.

## Summary

Milestones M3-M5 from
[sparton_gluon_remaining_work_design.md](sparton_gluon_remaining_work_design.md)
are complete:

- The original hybrid implementation moved to `src/sparton/_backend_hybrid.py`.
- `src/sparton/sparton_kernel.py` is now the import-compatible public facade
  and backend router.
- `SpartonHead(..., backend=...)` supports `hybrid` and `naive`, with
  `SPARTON_BACKEND` read once at import for the default when no constructor
  backend is supplied.
- `src/sparton/_backend_naive_triton.py` adds the M5 Triton-only fused-forward
  baseline and registers `sparton::naive_fwd`.

Hybrid remains the public default. `optimized` still raises a clear
unavailable-backend error and is deferred to the Gluon milestones.

## Implementation Notes

The hybrid custom-op names are preserved:

```text
sparton::fused_sparton_fwd(Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)
sparton::fused_sparton_bwd(Tensor grad_out, Tensor max_scores, Tensor max_idx, Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor, Tensor?)
```

The naive backend adds:

```text
sparton::naive_fwd(Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)
```

`naive_forward` is the Python wrapper used by `SpartonHead(backend="naive")`.
It makes `hidden`, `embed`, `mask`, and optional `bias` contiguous before
calling the custom op so the tensors saved for autograd satisfy the current
hybrid backward kernel's contiguous-layout assumptions.

The naive forward kernel is intentionally correctness-first:

- one Triton program owns one batch row and one vocabulary block;
- fixed tiles: `BLOCK_S=16`, `BLOCK_V=32`, `BLOCK_D=32`;
- K chunks are accumulated with `tl.dot` into fp32;
- optional bias and sequence mask are applied before reduction;
- running max starts at zero and updates only on strict `>`;
- scores are stored as `log1p(relu(max))`, indices as `int64`.

Backward is not reimplemented for M5. Naive custom-op autograd saves
`scores`, `indices`, `hidden`, `embed`, `bias`, and `mask`, then calls the
existing hybrid `fused_sparton_bwd_op`.

## Validation

Commands run with the hardened environment:

```bash
TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas
CPATH=/usr/local/cuda-13.2/include
TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor
```

Focused M5 tests:

```text
15 passed, 10 deselected, 1 warning
```

Full pytest suite:

```text
25 passed, 1 warning in 7.26s
```

`benchmarks/bench_sparton_baseline.py` default realistic preset:

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
naive_policy=on
```

Output columns:

```text
B
S                         # fixed sequence length for the row
hyb+b ms                  # hybrid forward with bias, avg ms per batch
hyb ms                    # hybrid forward without bias, avg ms per batch
gemm ms                   # full-V GEMM floor, avg ms per batch
ovh %                     # hybrid+b overhead over GEMM
hyb f+b ms                # hybrid forward+backward with bias, avg ms per batch
hyb MiB                   # peak extra memory during hybrid forward
naive+b ms                # naive forward with bias, avg ms per batch
naive ms                  # naive forward without bias, avg ms per batch
naive MiB                 # peak extra memory during naive forward
out MiB                   # output scores+indices size for one batch
logits MiB                # full [B,S,V] logits tensor size
tok/s                     # tokens per second for hybrid+b
```

Full default benchmark command:

```bash
env $ENV PYTHONPATH=src /workspace/venvs/sparton/bin/python -u benchmarks/bench_sparton_baseline.py
```

Full default results:

| B | S | hyb+b ms | hyb ms | gemm ms | ovh % | hyb f+b ms | hyb MiB | naive+b ms | naive ms | naive MiB | out MiB | logits MiB | tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 256 | 1.798 | 1.669 | 1.322 | 36.0 | 4.382 | 135.14 | 3.952 | 3.948 | 5.89 | 5.80 | 296.8 | 569532 |
| 4 | 512 | 3.825 | 3.385 | 2.612 | 46.5 | 6.393 | 134.51 | 7.849 | 7.859 | 5.89 | 5.80 | 593.5 | 535404 |
| 4 | 768 | 5.310 | 4.707 | 3.920 | 35.5 | 7.917 | 132.30 | 11.850 | 11.788 | 5.89 | 5.80 | 890.2 | 578481 |
| 8 | 256 | 3.801 | 3.390 | 2.614 | 45.4 | 7.261 | 140.84 | 7.865 | 7.870 | 11.59 | 11.59 | 593.5 | 538866 |
| 8 | 512 | 7.617 | 6.800 | 5.215 | 46.1 | 11.060 | 140.22 | 16.241 | 16.263 | 11.59 | 11.59 | 1187.0 | 537758 |
| 8 | 768 | 10.628 | 9.370 | 7.884 | 34.8 | 14.131 | 138.41 | 24.576 | 24.644 | 12.00 | 11.59 | 1780.5 | 578118 |
| 16 | 256 | 7.558 | 6.757 | 5.224 | 44.7 | 12.797 | 152.43 | 16.380 | 16.226 | 23.18 | 23.18 | 1187.0 | 541964 |
| 16 | 512 | 15.340 | 13.595 | 10.503 | 46.1 | 20.746 | 151.81 | 32.621 | 32.608 | 23.18 | 23.18 | 2374.0 | 534028 |
| 16 | 768 | 22.368 | 19.719 | 15.735 | 42.2 | 28.517 | 143.57 | 49.076 | 49.070 | 23.18 | 23.18 | 3561.0 | 549362 |

The previous SentenceTransformers-style benchmark result is obsolete. The
current benchmark result of record should come from
`benchmarks/bench_sparton_baseline.py`; the compatibility wrappers still exist.

## Remaining Risks

- The naive kernel uses many small `tl.dot` operations and is not expected to
  beat hybrid on large GEMM-dominated SPLADE shapes.
- BF16 index equality is covered on deterministic tail/mask cases and the
  existing seeded small random case; arbitrary random BF16 inputs can create
  near-ties where exact indices are numerically unstable across matmul
  implementations.
- The current hybrid backward remains the only backward path and still defines
  the gradient performance profile for both `hybrid` and `naive`.
- Gluon runtime isolation and smoke tests remain deferred to M6.
