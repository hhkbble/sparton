# Changelog

All notable repository changes should be recorded here. Entries are grouped by
the date they land in the repository unless a formal release tag exists.

## 2026-06-12

### M9 production readiness

#### Added

- Added `hybrid_forward`, a public per-backend wrapper for the hybrid path
  that validates and canonicalizes inputs before `sparton::fused_sparton_fwd`,
  making all three backends symmetric (wrapper → validation → contiguity →
  op). `SpartonHead` now binds wrappers for every backend;
  `fused_sparton_fwd_op` remains exported for raw-op callers.
- Added `src/sparton/_validation.py` with `validate_forward_inputs`: shared
  contract checks (ranks, cross-shapes, devices, per-backend dtype sets,
  dtype equality, complex-mask rejection, and the optimized backend's TMA
  16-byte row-alignment rule) raising `ValueError`/`TypeError` that name the
  argument, the actual value, and the requirement. Mixed-dtype and
  unaligned-D inputs now fail with contract errors instead of a bare
  `AssertionError` or a mid-kernel `CompilationError`.
- Added the tie-aware `assert_index_contract` test helper implementing the
  index contract of record (design v2 §6.2) and converted the random-input
  index assertions for `naive`/`optimized` to it; deterministic constructed
  cases and the hybrid random case keep exact index equality.
- Added `@pytest.mark.slow` non-tiny forward tests (six shapes covering
  V-tail, batch-crossing S-tail, single-K-tile, and long-S cases, fp16 and
  bf16) for `naive` and `optimized`, including an assertion that the
  optimized runtime-derived candidate set is the non-fallback production
  policy bank.
- Added regression tests for non-contiguous inputs across all three backends,
  a silent-`import sparton` stdout test, backend-resolution error-source
  tests, an fp32-hybrid coverage test, validation-rule tests, and a
  `torch.compile(fullgraph=True)` capture test per backend.
- Added `SpladeModel(..., sparton_backend=...)` and a matching
  `--sparton_backend` training argument threaded to
  `SpartonHead(..., backend=...)`; the default `None` preserves existing
  behavior.

#### Fixed

- Fixed silently wrong gradients for non-contiguous inputs on the hybrid
  autograd path (design v2 finding F1): the op saved raw tensors while the
  backward kernel assumes dense strides; `hidden.grad`/`embed.grad` could be
  corrupted without any error. The hybrid wrapper now canonicalizes inputs
  before the op, matching naive/optimized.
- Fixed `training/model.py` invalid-head error text to name the accepted
  values (`'torch', 'compiled', or 'sparton'`).
- `resolve_backend` errors now say whether the invalid value came from the
  `SPARTON_BACKEND` environment variable or the `backend` argument.

#### Changed

- Changed `pyproject.toml` license metadata from MIT to Apache-2.0 to match
  the repository `LICENSE` and upstream (deliberate fix of the previously
  documented mismatch).
- Replaced the import-time `print("Sparton using device: ...")` and the
  `SpartonHead.load` `print("no bias")` with DEBUG-level records on the
  `"sparton"` logger; `import sparton` is now silent on stdout.
- Removed the import-time `torch.set_float32_matmul_precision('high')` global
  side effect. The supported fp16/bf16 paths are unaffected (the knob governs
  fp32 matmuls); fp32 hybrid users now inherit the application's own
  precision setting.
- `fused_sparton_bwd_with_bias` now returns a 3-tuple
  `(hidden_grad, embed_grad, bias_grad)` instead of a 4-tuple with a vestigial
  trailing `None` (the symbol is re-exported; its only in-repo caller was
  `fused_sparton_bwd_op`).
- `benchmarks/bench_sparton_baseline.py` hybrid columns now measure
  `hybrid_forward` (the user-visible path) instead of the raw op; A/B runs
  showed no measurable difference on the dev shape.
- `_gluon_runtime` now uses the public `gl.NVMMASharedLayout` instead of the
  private `language._layouts` module and caches the loaded Gluon namespace in
  `is_gluon_backend_available`.
- Updated `AGENTS.md` (project map with all backend modules, the
  wrapper/op layering invariant, index-contract rule, refreshed sharp edges,
  glob `py_compile` command) and `benchmarks/README.md`/`README.md` pointers
  and index-semantics documentation for the post-M8 state.

#### Removed

- Removed dead code from `_backend_hybrid.py`: unused imports (`gc`, `time`,
  `torch.amp`, `gradcheck`, `torch._dynamo`), the commented-out
  `FusedSparton` autograd class and `fused_mlm_splade` alias, a stale
  commented autotune block, and a commented debug print.

#### Validation

- Added
  [docs/sparton_milestone9_production_readiness_memo.md](docs/sparton_milestone9_production_readiness_memo.md)
  with the finding→fix mapping, the captured red→green non-contiguous
  gradient evidence, the benchmark A/B for the wrapper switch, and the
  exit-checklist transcript.
- Exit checklist (hardened env, serial): glob `py_compile` passed; full
  pytest `105 passed, 15 warnings` (quick loop `90 passed, 15 deselected`);
  dev-shape benchmark within ±5% of the recorded baselines
  (hybrid+b 1.177 ms, naive+b 1.260 ms, optimized+b 0.898 ms vs
  1.176/1.301/0.898); `import sparton` produces no stdout;
  `git diff --check` clean.
- Verified all three pytest invocation modes collect and pass:
  `python -m pytest`, the venv `pytest` console script from the repo root,
  and `pytest /workspace/sparton/tests` from a foreign working directory.
- Verified `torch.compile(fullgraph=True)` capture for all three backends
  with input validation in the traced path (suite test plus smoke).

### Added

- Added
  [docs/sparton_remaining_work_design_v2.md](docs/sparton_remaining_work_design_v2.md),
  the post-M8 forward plan produced by a line-by-line review of the M5/M8
  implementation commits against the original design. It records the verified
  current state, the design-evolution ledger, the review findings (including
  a probe-verified wrong `embed_grad` for non-contiguous hidden on the hybrid
  autograd path, the near-tie index-contract gap, and the missing non-fallback
  optimized test coverage), and the revised milestone order: M9 production
  readiness, M10 promotion decision, M11 backward track, M12 forward
  scheduling/launch overhead.
  [docs/sparton_gluon_remaining_work_design.md](docs/sparton_gluon_remaining_work_design.md)
  is now marked superseded as the forward plan while remaining authoritative
  for its platform facts and measured evidence.
- Added the M8 experimental Gluon `optimized` backend. It registers
  `sparton::optimized_fwd`, uses a non-persistent TMA + `mma_v2` fused forward
  with online max/argmax, selects from runtime GPU-derived active autotune
  candidates, keeps hybrid as the default, and delegates backward to the
  existing hybrid backward.
- Added `_gluon_runtime.py`, a lazy Gluon compatibility shim with static
  sm_80+ `mma_v2` capability dispatch, a Triton 3.6.0 validation warning, and
  a Gluon autotune compatibility export that falls back to `triton.autotune`.
- Added `_runtime_policy.py`, a resource-derived policy generator used by the
  Gluon GEMM benchmark, M7 tuning gate, and runtime GPU-derived optimized
  forward autotune pruning, including the public optimized-forward fallback
  policy helper used by tests.
- Added `benchmarks/probe_gluon_epilogue.py` for optimized-forward epilogue
  correctness across fp16/bf16 and bias/no-bias cases.
- Added optimized-backend routing, correctness, backward-smoke, schema,
  environment-default, S-tail, and allocator tests.
- Added
  [docs/sparton_milestone8_gluon_forward_memo.md](docs/sparton_milestone8_gluon_forward_memo.md)
  summarizing M6-M8 implementation, M7 GEMM gate evidence, optimized-forward
  memory/performance, validation, and remaining M9+ risks.

### Changed

- Changed `SpartonHead(..., backend="optimized")` and
  `SPARTON_BACKEND=optimized` to select the M8 experimental Gluon
  fused-forward backend.
- Changed the optimized Gluon forward from a fixed O1 policy to bounded
  autotune keyed by `(B, S, D, V)`: Triton sees a fixed production policy
  universe for stable descriptor slots, while the active candidate set is
  derived at launch from the actual CUDA device profile and problem shape.
- Changed optimized-backend tests, probes, and opt-in benchmark paths to gate
  on optimized Gluon availability (CUDA sm_80+ and importable Gluon symbols)
  instead of assuming every CUDA device or only RTX 5090 can run the backend.
- Changed the M5 `naive` Triton forward from a single fixed
  `16x32x32/4w/3s` launch to bounded Triton autotune over ten forward tile
  configs keyed by `(S, D, V)`, while keeping the original config as a
  candidate.
- Changed `benchmarks/ncu_runner.py` to profile the autotuned Gluon GEMM path
  using the shared policy/descriptor helpers instead of a removed fixed-kernel
  launcher.
- Extended `benchmarks/bench_gluon_gemm.py` to use generated policy objects,
  support fp16 and bf16, include BK=32/64-byte-swizzle candidates, enforce
  ratio gates, and use Triton/Gluon autotune as the M7 gate by converting the
  `_runtime_policy.py` GEMM policy universe to `triton.Config` objects with
  shared host-side policy/config/descriptor helpers.
- Extended `benchmarks/bench_sparton_baseline.py` with optional optimized
  forward timing and memory columns.
- Updated
  [docs/sparton_gluon_remaining_work_design.md](docs/sparton_gluon_remaining_work_design.md)
  to point to the M8 memo and note that M6-M8 are complete while `optimized`
  remains experimental.

### Validation

- Verified `py_compile` for package, training, tests, and benchmark files.
- Verified pytest result after M8, naive autotune, optimized autotune, and
  optimized-Gluon availability gates: `47 passed, 15 warnings in 8.71s` with
  the hardened CUDA/Triton environment.
- Verified M6 MMA smoke: `mma_v2` passed with max-abs error `0.000006` vs
  fp32 matmul; WGMMA and TCGen05 failed in the expected isolated subprocesses.
- Verified M7 Gluon GEMM autotune gate on dev shape
  `M=4096, K=768, N=30522`: fp16 selected `POLICY_ID=8` at `86.452%` of
  cuBLAS and bf16 selected `POLICY_ID=8` at `86.343%` of cuBLAS, both using
  the `64x64x64/3/2x2` policy.
- Verified M7 stress GEMM gate on `M=4096, K=1024, N=50257`, fp16: best ratio
  `173.585%` of cuBLAS in the L2-flushed benchmark regime.
- Verified `benchmarks/probe_gluon_epilogue.py`: fp16/bf16 bias/no-bias passed.
- Verified availability-gated benchmark/profiler smokes:
  `bench_gluon_gemm.py`, `ncu_runner.py`, and a
  tiny `bench_sparton_baseline.py --optimized-policy on` run all passed.
- Verified `benchmarks/bench_naive_baseline.py` after naive autotune:
  `B=4, S=64, D=64, V=4096`, fp16, hybrid+b `0.018 ms`, naive+b
  `0.006 ms`, naive no-bias `0.006 ms`, and naive peak extra `0.16 MiB`.
- Verified optimized dev-shape memory gate
  `B=32, S=128, D=768, V=30522`, fp16: peak extra `9.86 MiB` vs `9.31 MiB`
  outputs (`1.06x`, below the `2x` limit).
- Verified one-row optimized benchmark on the same dev shape after optimized
  autotune: hybrid+b `1.176 ms`, optimized+b `0.900 ms`, optimized no-bias
  `0.896 ms`; optimized remains experimental and is not promoted.
- Verified the full merged benchmark default with optimized enabled and
  autotuned naive/optimized:
  `D=1024`, `V=151936`, bf16, batch sizes `4,8,16`, sequence lengths
  `256,512,768`, `naive-policy=on`, and `optimized-policy=on`; it emitted the
  9-row Markdown table recorded in the M8 memo.
- Verified custom-op schema:
  `sparton::optimized_fwd(... Tensor? bias ...) -> (Tensor, Tensor)`.

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
- [Milestone 8 memo](docs/sparton_milestone8_gluon_forward_memo.md)
