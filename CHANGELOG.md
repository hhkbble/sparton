# Changelog

All notable repository changes should be recorded here. Entries are grouped by
the date they land in the repository unless a formal release tag exists.

## 2026-06-11

### Added

- Added the M5 Triton-only `naive` backend as an experimental fused-forward
  baseline. It registers `sparton::naive_fwd`, computes scores/indices in one
  Triton kernel without materializing `[B, S, V_tile]` logits, and reuses the
  current hybrid backward through custom-op autograd.
- Added backend routing tests, naive correctness/backward tests, tail/mask/tie
  coverage, and an allocator check proving the naive forward only allocates
  outputs on the measured shape.
- Added `benchmarks/bench_naive_baseline.py` to compare M5 naive forward
  latency and peak memory against the hybrid path.
- Added `benchmarks/bench_sparton_baseline.py`, a merged hybrid/naive
  benchmark with realistic `naver/splade-code-06B` dimensions, bf16 defaults,
  a fixed `B=4,8,16` by `S=256,512,768` grid, and all-ones masks.
- Added a pytest 9 test harness configured in `pyproject.toml`, including CUDA
  kernel tests and a PyTorch reference for Sparton score/index semantics.
- Added milestone documentation for the `bias=None` backward fix in
  [docs/sparton_milestone2_bias_none_backward_memo.md](docs/sparton_milestone2_bias_none_backward_memo.md).
- Added
  [docs/sparton_milestone5_naive_triton_memo.md](docs/sparton_milestone5_naive_triton_memo.md)
  summarizing the backend split, router, naive Triton implementation,
  validation, benchmark evidence, and remaining M6+ risks.
- Added
  [docs/sparton_gluon_remaining_work_design.md](docs/sparton_gluon_remaining_work_design.md),
  the new evidence-based plan for the backend refactor. It supersedes the
  original Gluon design as the forward guide and records measured feasibility
  results on RTX 5090 (sm_120): only Ampere-style `mma_v2` works in Gluon
  (WGMMA/TCGen05 abort in LLVM), a validated TMA+`mma_v2` GEMM microbenchmark
  reaches 76% of cuBLAS on the dev shape, hybrid forward baselines, corrected
  milestones M3-M11, and backend promotion gates. After GPU performance
  counters were enabled on the host, the document gained ncu hardware-counter
  evidence: the Gluon kernel and cuBLAS use the same `mma.sync` instruction
  family at the same occupancy (gap = intra-CTA pipelining, 63.9% vs 86.6%
  tensor-pipe utilization), the hybrid forward leaves the bias add unfused and
  reads tile logits twice (583 MB vs 55 MB measured DRAM reads), and the
  backward kernel is latency/atomic-bound at ~6% utilization.
- Added `benchmarks/` with the validated probe and benchmark scripts behind
  the design document's evidence (MMA availability matrix, TMA+`mma_v2` GEMM
  microbenchmark, hybrid baselines, ncu launchers, environment-defect
  reproducer), promoted from session scratch so they survive `/tmp` cleanup;
  see `benchmarks/README.md`.
- Documented the root cause of the previously "transient" TorchInductor/Triton
  compile failure: the NVIDIA Triton wheel ships no bundled CUDA headers
  (cold `cuda_utils` builds fail without `CPATH`) and `/tmp` is mounted
  `noexec` (Inductor-redirected Triton caches cannot be `dlopen`ed). The
  hardened environment prefix in `AGENTS.md` fixes both deterministically.

### Changed

- Split the existing hybrid implementation into `src/sparton/_backend_hybrid.py`
  while keeping `src/sparton/sparton_kernel.py` as the import-compatible public
  facade.
- Added `SpartonHead(..., backend=...)` with a hybrid default and
  `SPARTON_BACKEND` import-time default selection. `backend="naive"` selects the
  M5 debug backend; `backend="optimized"` raises clearly until the Gluon
  milestones land.
- Changed `bench_hybrid_baseline.py` and `bench_naive_baseline.py` into thin
  compatibility wrappers over `bench_sparton_baseline.py`.
- Simplified the merged benchmark from SentenceTransformers-style dynamic
  batching to direct fixed `(B, S)` shape rows, and removed the obsolete
  benchmark helper tests.
- Raised the package Python requirement to `>=3.10` to align with pytest 9.
- Documented the current hybrid runtime path more precisely: tiled compiled
  PyTorch/TorchInductor matmul plus Triton reduction.
- Updated custom-op schemas so `sparton::fused_sparton_fwd` and
  `sparton::fused_sparton_bwd` declare optional bias as `Tensor?`.
- Updated `AGENTS.md` guidance to require changelog consideration after tasks,
  memo documentation for completed milestones or substantial decisions, and
  detailed git commit bodies.

### Fixed

- Fixed `SpartonHead(use_bias=False)` backward by returning `None` for the bias
  gradient and compiling out bias-gradient atomics for no-bias launches.

### Validation

- Verified M5 focused tests for naive backend, router, environment default, and
  schemas: `15 passed, 10 deselected, 1 warning`.
- Verified `py_compile` for package, training, and test files.
- Verified pytest result: `25 passed, 1 warning` on the CUDA/BF16-capable
  workspace with `TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas`.
- Verified the full merged benchmark default with `D=1024`, `V=151936`,
  bf16, batch sizes `4,8,16`, sequence lengths `256,512,768`, and
  `naive-policy=on`; it emitted the 9-row Markdown table recorded in the M5
  memo.
- Verified `benchmarks/bench_naive_baseline.py` on the default
  `B=4, S=64, D=64, V=4096` fp16 shape: hybrid forward was 0.019 ms with bias
  and 0.014 ms without bias; naive forward was 0.008 ms for both; hybrid peak
  extra memory was 2.31 MiB, while naive peak extra memory was 0.16 MiB
  (outputs only) versus a 2.00 MiB full-logits tensor.
- Verified custom-op schemas:
  - `sparton::fused_sparton_fwd(... Tensor? bias ...) -> (Tensor, Tensor)`
  - `sparton::fused_sparton_bwd(... Tensor? bias ...) -> (Tensor, Tensor, Tensor?)`
  - `sparton::naive_fwd(... Tensor? bias ...) -> (Tensor, Tensor)`

## References

- [Current platform design](docs/sparton_gluon_current_platform_design.md)
- [Design review](docs/sparton_gluon_design_review.md)
- [Milestone 2 memo](docs/sparton_milestone2_bias_none_backward_memo.md)
- [Milestone 5 memo](docs/sparton_milestone5_naive_triton_memo.md)
