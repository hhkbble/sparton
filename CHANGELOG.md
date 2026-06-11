# Changelog

All notable repository changes should be recorded here. Entries are grouped by
the date they land in the repository unless a formal release tag exists.

## 2026-06-11

### Added

- Added a pytest 9 test harness configured in `pyproject.toml`, including CUDA
  kernel tests and a PyTorch reference for Sparton score/index semantics.
- Added milestone documentation for the `bias=None` backward fix in
  [docs/sparton_milestone2_bias_none_backward_memo.md](docs/sparton_milestone2_bias_none_backward_memo.md).
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

- Verified `py_compile` for package, training, and test files.
- Verified pytest result: `11 passed, 1 warning` on the CUDA/BF16-capable
  workspace with `TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas`.
- Verified custom-op schemas:
  - `sparton::fused_sparton_fwd(... Tensor? bias ...) -> (Tensor, Tensor)`
  - `sparton::fused_sparton_bwd(... Tensor? bias ...) -> (Tensor, Tensor, Tensor?)`

## References

- [Current platform design](docs/sparton_gluon_current_platform_design.md)
- [Design review](docs/sparton_gluon_design_review.md)
- [Milestone 2 memo](docs/sparton_milestone2_bias_none_backward_memo.md)
