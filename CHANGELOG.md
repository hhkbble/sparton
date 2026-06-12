# Changelog

All notable repository changes should be recorded here. Entries are grouped by
the date they land in the repository unless a formal release tag exists.

## 2026-06-12

### Remaining-work design v4

- Added
  [docs/sparton_remaining_work_design_v4.md](docs/sparton_remaining_work_design_v4.md),
  the post-M11 forward plan, re-grounding the remaining work in the
  measured post-M11 state (backward share down from 24–64% to ~17–52% of
  optimized fwd+bwd across the grid and ~55% on the dev shape; both
  residual costs now have named mechanisms) and restructuring it around
  the Performance-Optimization Loop. M12 (forward track) gains an
  unconditional entry-evidence task — the first-ever ncu profile of the
  *production* fused forward (only the GEMM bring-up kernel has counters,
  v1 §3.5) plus a counter-validated analytic floor model and a
  pre-registered go/no-go for the persistent/warp-specialized rewrite —
  and launcher v2's selection-parity gate is restated as deterministic
  candidate-set/ranking parity plus winner-time parity within a measured
  noise band (v3's exact-selection gate is unachievable under the ±5%
  autotune-selection jitter M11 measured). New M13 (backward residual
  track), entry-gated on an analytic ceiling for the segmented backward's
  named residual (embed-gather latency; g/idx stream re-reads), riding the
  M11 harness/registry machinery, with an explicit A/B-reference baton
  rule; its unconditional T0 absorbs the recorded M11/M10 evidence debt
  (tier-2 150-step parity rerun, training-scale determinism re-measure,
  optional sparse-regime capture probe). Cross-track ordering is decided
  by M12-T0's comparative recoverable-time table, and exit numbers are
  fixed from validated models at entry time — the M11 1.5×-clause lesson.
  All v3 rejections carry forward, joined by the M11-decided ones (B2b
  structural rejection, tensor-core one-hot accumulation, adaptive
  in-op dispatch, weighted-mask gradients) so they are not re-litigated.
  An adversarial review of the draft against the cited runs of record
  also corrected two provenance defects v4 would otherwise have inherited
  from v3: the dev-shape GEMM floor (0.880 ms) traces to the post-M8
  review run in v2 §1.2 (same-run forward 0.898 ms, 1.02×), not the M10
  memo, and the grid floor ratios are re-anchored to the preserved
  same-run M8 grid gemm column (1.094–1.123×) with v3's 1.054–1.113×
  flagged as derived from an unpreserved M10 gemm column (orients, may
  not gate; M12-T0 re-measures in one preserved run).
- Marked
  [docs/sparton_remaining_work_design_v3.md](docs/sparton_remaining_work_design_v3.md)
  superseded as the forward plan; it remains authoritative for the
  executed M11 plan of record (the spec the M11 memo's deviation ledger
  refers to) and the post-M10 snapshot and floor-ratio derivations.
  Updated the active-design-doc references in `AGENTS.md` (project map,
  orientation, exemplar table, documentation-system table, task routing)
  and `benchmarks/README.md`.

### AGENTS.md refactor: the M8→M11 operating guide

- Refactored `AGENTS.md` (maintainer-requested) to distill the M11 arc's
  engineering experience into durable method, so a cold agent can reproduce
  the milestone's level of work. New sections: **The
  Performance-Optimization Loop** (binding-resource-first altitude test;
  analytic traffic model validated against counters before any candidate is
  built; realistic-distribution capture/replay; prototype-behind-a-registry
  with per-cell verification before timing; pre-registered numeric
  early-stop rules; op-level vs kernel-level timing; autotune-key audit;
  IR-stage reading; sanitizer for ownership-semantics changes;
  named-residual-bottleneck exit) and **Milestone Review** (adversarial
  review of the diff *and* the evidence chain before a milestone closes,
  with the maintainer-ruling path for contract findings). Extended rules:
  a fourth failure-classification verdict (contract rulings), deposited
  transcripts for decision-carrying numbers, autotune-selection jitter in
  the noise band, conditional test expectations must pin their conditions,
  autograd-leaf construction, the no-editing-files-a-queued-process-imports
  hazard, and the retired-implementation rule (delete or promote to a
  test-pinned reference). Repo facts updated: suite size, M11 backward
  toolchain in the project map, House Style exemplars (`bench_backward.py`,
  capture tooling, M11 memo), task routing (M11 complete → M12 entry gate),
  `compute-sanitizer` availability, and the `torch.empty` gradient-buffer
  invariant.

### Method reference: IR-stage visibility recipe

- Added §6.1 to
  [docs/triton_gluon_kernel_optimization.md](docs/triton_gluon_kernel_optimization.md):
  the working recipe for reading Triton/Gluon intermediate stages on this
  stack (`CompiledKernel.asm` via `jit_fn.device_caches`, `nvdisasm` for
  SASS, `TRITON_KERNEL_DUMP`/`MLIR_ENABLE_DUMP`), what each stage answers,
  and where it slots into the loop (zero-GPU-cost lowering confirmation
  before benchmarking; settling ncu surprises by instruction form).
  Probed on the M11 backward kernels: the hidden-grad atomics are
  `REDG.E.ADD.F32x4` (proving the 4-wide vectorization T1 had inferred
  from counter arithmetic), the uniform-chunk guard lowers to single
  warp-level `REDUX` ops, and the `tl.cumsum` scan is visible as
  `tt.scan` + SHFL chains; dumps under `/root/profiles/m11/ir_dump/`.

### M11 sanitizer gate discharged

- After a host restart enabled `compute-sanitizer` (blocked on WSL2 during
  the milestone session — M11 memo §5.4), the deferred sanitizer gate ran
  clean on the promoted tree: **racecheck 0 hazards, memcheck 0 errors,
  initcheck 0 errors** (small shape, fp16, bias on/off,
  `current` + `legacy` impls; the autotuner sweeps every config under the
  sanitizer). initcheck additionally validates the `torch.empty`
  `embed_grad`/`bias_grad` allocation: no uninitialized read exists, so
  the exclusive-owner stores provably cover both buffers. Transcripts under
  `/root/profiles/m11/sanitizer_*.txt`; memo §5.4/§8/§9 updated.

### M11 adversarial review pass

- A 47-agent adversarial review of the M11 diff (three dimensions:
  kernel/op correctness, evidence consistency, doctrine compliance; every
  finding independently verified twice) confirmed one pre-existing kernel
  gap and a set of documentation/test corrections, all applied:
  - The review flagged that the shared backward omits the `mask[b, idx]`
    factor from all gradients. **Maintainer resolution: expected behavior
    under the original contract** — the Sparton mask is binary {0, 1} (the
    standard tokenizer `attention_mask`), under which the omission is
    exact (masked winners are eliminated by the forward/max/ReLU guard).
    `_validation.py`'s old "non-binary masks are defined behavior" note
    overstated the contract and is corrected (with AGENTS.md, README, and
    the memo §9): non-binary values weight logits in the forward as an
    implementation property outside the contract and are not
    differentiated. Weighted-mask support, if wanted, is an extension
    (backward change + autograd-head tests). A non-binary-rejection
    validation was considered and rejected: the validation layer is
    metadata-only by design (torch.compile-safe, no device sync).
  - `test_fused_backward_nontiny_shapes` now pins the forward outputs its
    closed-form expectation conditions on (reference scores + index
    contract); `test_backward_matches_legacy_kernel` is slow-marked and
    its docstring cites the actually-measured legacy self-spread.
  - Host-side `B*V < 2^31` / `S*D < 2^31` guards in
    `segmented_sparton_bwd` (fail loudly instead of wrapping int32-derived
    arithmetic; thresholds unreachable on this hardware).
  - Memo/doc numeric corrections to match the preserved runs of record
    (the `steps150`-doc deviation range is 1.35–1.43×, not 1.37–1.47×;
    per-column B2a ranges; bias-matched implied-backward grid derivation;
    real-record ncu transcripts re-deposited — the production segmented
    config measures 1.55 M red sectors vs legacy's 408.03 M on the real
    query record, 264×). The method reference
    `docs/triton_gluon_kernel_optimization.md` is now registered in
    `AGENTS.md` Reference Material.

### M11 T3–T4 — segmented backward promoted (milestone complete)

#### Changed

- The shared backward inside `sparton::fused_sparton_bwd` (all three
  backends) is now the M11 segmented design in
  `src/sparton/_backend_hybrid.py`: a fused payload-prep kernel (fp32
  `g = grad_out·exp(-scores)` where `scores > 0`, int32 idx, destination
  sort keys), `torch.sort` by destination row plus a payload-gather kernel,
  an exclusive-owner embed/bias-gradient kernel (plain stores, zero
  atomics, `torch.empty` outputs — the V×D fp32 zero-fill disappears from
  every call), and a persistent-stride segmented-scan hidden-grad kernel
  (chunk-local cumsum, at most ~two partial-sum atomics per destination
  run, sentinel quick-exit for sparse inputs, `seq_len` in the autotune
  key). Op schema, fake registration, autograd wiring, and saved tensors
  are unchanged. Mechanism and decision record:
  [docs/sparton_milestone11_backward_memo.md](docs/sparton_milestone11_backward_memo.md).
- The pre-M11 kernel is retained as `legacy_fused_sparton_bwd_kernel_with_bias`
  behind the new `legacy_fused_sparton_bwd` wrapper (A/B reference of
  record, test-pinned, reachable only explicitly); the exported
  `fused_sparton_bwd_with_bias` helper keeps its signature and launches the
  legacy kernel. `benchmarks/bwd_prototypes.py` (T3 decision probe) was
  deleted after the decision; `bench_backward.py` resolves
  `current`/`legacy`.
- Measured (run 2 of two consecutive runs, op-level `do_bench`): backward
  1.35–2.40× faster than legacy on captured-real cells (queries ≥2.15×;
  the 16 `steps150`-document cells at 1.35–1.43× sit below the design's
  1.5× exit clause — recorded deviation, memo §8), 1.09–1.29× faster on
  every synthetic cell (no regression anywhere). Dev-shape `opt f+b`
  2.325 → 1.949 ms; implied optimized backward −25% (dev) and −31…−41%
  (canonical grid). L2 reduction sectors: 96.6 M → 3.49 M (dev),
  330.6 M → 6.40 M (corner), 408 M → 6.4 M (real query record).
  `embed_grad`/`bias_grad` are now structurally deterministic; the
  `hidden_grad` proxy spread tightens ~5×.

#### Added

- Backward test coverage (suite 116 → 132 on top of T2's 114 → 116): bf16
  backward cases for all three backends (`BACKWARD_CASES`,
  `_grad_tolerances`), non-tiny backward shapes with a closed-form
  expectation from the kernel's saved `(scores, idx)` plus in-test pinning
  of those saved outputs against the reference scores and the index
  contract, exact-equality constructed cases for the zero-score and
  masked-row invariants, and the `test_backward_matches_legacy_kernel` A/B
  of record (slow-marked, like its non-tiny siblings).
- `AGENTS.md`: backward validation rule (rerun `bench_backward.py` vs
  `legacy` on captured bundles; uniform-only evidence never sufficient) and
  the updated determinism sharp edge; `README.md` backward paragraph;
  design v3 M11 status note.

#### Validation

- Hardened-env exit run: `py_compile` clean; full pytest `132 passed`
  (quick loop `109 passed, 23 deselected` after the review pass
  slow-marked the legacy A/B); shape soak `384/384`; 300-step
  AMP smoke passed both modes (bf16 parity 0.21–0.24%); backward gate
  matrix 176 cells × 2 runs, 0 verification failures; dev row run 2
  `hyb+b 1.152 / opt+b 0.880 / hyb f+b 2.262 / opt f+b 1.949 ms`; grid run
  2 all 9 rows below their M10 `opt f+b` values; `import sparton`
  stdout-silent; `git diff --check` clean. Compute Sanitizer is not
  runnable on this WSL2 host (memo §5.4).

### M11 T1–T2 — backward re-profile and distribution-aware harness

#### Added

- `benchmarks/ncu_backward_target.py`: parameterized direct-op backward
  profiling target (NVTX `bwd_direct/` on the main thread, shapes/dtype/bias
  via CLI, optional capture-bundle input) generalizing the `ncu_targets.py`
  `hybrid_bwd_direct` pattern for the M11 before/after counter collection.
- `benchmarks/capture_index_distributions.py`: captures real Sparton index
  distributions (per-batch `hidden_shape`, `max_scores` fp16, `max_idx`
  int16, `mask` uint8, plus active-fraction/density/index-collision stats)
  from the cached tier-2 stack (xlm-roberta-base, swim-ir `de`), with an
  optional 150-step fine-tune (the M10-validated recipe). Bundles of record:
  `swimir_de_steps0.pt` / `swimir_de_steps150.pt` under `/root/m11_bundles/`
  (not committed; rerun the script to regenerate). Measured stats, both
  bundles: mean active fraction 1.0000, mean mask density 0.6675, mean
  index top1-collision share 0.198 (untrained) / 0.184 (150 steps).
- `benchmarks/bench_backward.py`: distribution-aware backward harness —
  op-level `do_bench` timing over {uniform, zipf, captured-real} sources,
  synthetic mask densities {25, 75, 100}%, fp16/bf16, bias/no-bias, with
  per-cell verification against the production op and the recorded 5-repeat
  determinism protocol. Baseline of record (run 2): synthetic at active
  fraction 0.10 0.36–0.39 ms; real records 5.88–7.89 ms (V=250002).
- `test_bench_backward_synthetic_inputs_honor_contract` (uniform + zipf):
  pins the harness's documented synthetic input contract. Suite: 114 → 116.

#### Measured (T1 entry evidence, dev shape fp16 + 16×512 bf16 corner)

- Backward kernel unchanged since M2: 1.35 ms, SOL compute 6.14% / DRAM
  8.67%, 248 regs/thread, 16.5% occupancy, 96.6M L2 reduction sectors at the
  dev shape (premise check passed; full counter set in the M11 memo).

### Kernel-optimization method reference

- Added
  [docs/triton_gluon_kernel_optimization.md](docs/triton_gluon_kernel_optimization.md)
  (user-provided): methodologies, techniques, and tooling for kernel
  optimization on Triton and Gluon — bottleneck-classification-first
  profiling, autotune hygiene, occupancy/register-pressure tuning loops,
  Gluon layout/async/TMA/warp-specialization techniques, a tooling checklist
  (Compute Sanitizer, Proton, interpreter mode), and common failure modes.
  Adopted as the method reference for the M11 backward track alongside
  `AGENTS.md`.

### Remaining-work design v3

- Added
  [docs/sparton_remaining_work_design_v3.md](docs/sparton_remaining_work_design_v3.md),
  the post-M10 forward plan carrying the incomplete milestones out of design
  v2 in refined form. M11 (backward track) gains a re-profiling entry step,
  a capture-to-disk distribution harness spec that uses the now-cached M10
  tier-2 model/data for real index distributions, mask-density dimensions,
  a determinism measurement protocol (with bitwise determinism explicitly
  rejected as a gate), and ordered B2a/B2b decision criteria; M12 decouples
  launcher v2 (with a selection-parity gate before the descriptor bank is
  deleted) from the entry-gated persistent/warp-specialized rewrite, whose
  warp-specialize subprocess probe is a hard entry gate and which owns the
  D2 dedup decision point. The v2 headline "backward ≈ 62% of fwd+bwd" is
  refined to the measured shape-dependent range (24–64% across the M10
  grid, 61% on the dev shape); a numbers-provenance appendix maps every
  headline figure to the session or memo that measured it.
- Marked
  [docs/sparton_remaining_work_design_v2.md](docs/sparton_remaining_work_design_v2.md)
  superseded as the forward plan (it remains authoritative for the post-M8
  review findings, the evolution ledger, the §4 architecture rules, and the
  executed M9/M10 specifications); its M11/M12 sections are replaced by
  pointers to v3 so the future work has exactly one specification. Updated
  the active-design-doc references in `AGENTS.md` (project map, orientation,
  documentation-system table, task routing, exemplar table) and
  `benchmarks/README.md`.

### AGENTS.md method refactor

- Refactored `AGENTS.md` from a facts-and-invariants file into the full
  operating guide, distilling the engineering method that produced the
  M8→M10 arc so future agents reproduce it: a new Operating Loop section
  (probe-before-design, scope declared as will-NOT-touch lists, red→green
  bug fixes with captured failing output, numeric gates per task, efficiency
  rules); an Evidence/Measurement/Gates section (second-consecutive-run
  judging, same-config noise bands before cross-config verdicts,
  do_bench/ncu/nsys regime separation, classify-failures-before-fixing with
  the three possible verdicts); a Testing Doctrine section (tests must prove
  what they appear to prove, coverage-activation assertions,
  deterministic-vs-random assertion strength, error templates as tested API,
  single-sourced gate logic, the three pytest invocation modes); a House
  Style section with an exemplar table mapping each artifact type to its
  best in-repo instance (gate script, probe, contract checker, validation
  template, design doc, memos) plus the symmetry/one-seam/no-silent-fallback
  rules; and a Documentation System section formalizing the five-document
  role table, the supersession-chain practice, and the
  every-number-was-measured rule. Rules cite the memo or design section that
  taught them.
- Refreshed stale facts while refactoring: `transformers` 5.11.0 and
  `accelerate` 1.14.0 are now present in the venv (installed at M10 with
  torch/triton verified untouched); the training section records the
  tied-weight `torch.save` requirement, the transformers-5 smoke-only
  validation status, and the cached M10 model/dataset; Known Sharp Edges
  gains the atomic-add training non-determinism band and the expected
  GradScaler early-skip behavior; Validation routes optimized-surface
  changes to the shape soak and autograd/AMP changes to the training smoke.

### M10 promotion: optimized is the default backend

#### Changed

- **The default backend is now `optimized`** wherever the Gluon backend is
  available (CUDA sm_80+ plus importable `triton.experimental.gluon`). With
  no `backend` argument and no `SPARTON_BACKEND`, default resolution falls
  back to `hybrid` with a one-time `RuntimeWarning` when optimized is
  unavailable — the only adaptive fallback in the package; explicitly
  selected backends still raise with the reason. Rollback:
  `SPARTON_BACKEND=hybrid` or `SpartonHead(..., backend="hybrid")` (the
  hybrid path is unchanged). Promotion evidence per the design v2 §5 M10
  gates is in
  [docs/sparton_milestone10_promotion_memo.md](docs/sparton_milestone10_promotion_memo.md):
  optimized forward is 19–27% faster than hybrid on the dev shape and all
  nine canonical-grid rows, fwd+bwd is 11–20% faster, and peak extra memory
  is 1.0–1.06× outputs (hybrid: 6.2–23×).
- The forward wrappers now mirror `torch.autocast` semantics
  (`_validation.autocast_canonicalize`): under an active CUDA autocast
  region, floating inputs are cast to the autocast dtype so fp32 master
  parameters work under fp16/bf16 AMP with every backend. This fixes a
  latent gap surfaced by the M10 training gate — the M9 dtype-equality
  validation had made AMP training a hard `TypeError` on all backends
  (hybrid had only ever worked pre-M9 via TorchInductor's autocast-aware
  matmul; the fused backends never supported AMP).
- `benchmarks/bench_sparton_baseline.py` gained an `opt f+b ms` column
  (optimized forward+backward timing) alongside the existing optimized
  columns.
- `training/train.py`: `LSRTrainer.save_model` now serializes the model with
  `torch.save` — `SpladeModel` ties the head weight to the backbone word
  embeddings, and transformers 5 removed `TrainingArguments.save_safetensors`,
  so the stock `Trainer._save` can never write this model via safetensors.
- README Backend Selection, AGENTS.md invariants, and the benchmarks README
  were updated for the new default, the adaptive-fallback exception, the AMP
  behavior, and the new gate scripts.

#### Added

- Added `benchmarks/soak_optimized_correctness.py` (M10 gate 5): a 384-case
  S/B/D/V/bias/dtype sweep with random masks including fully zeroed rows,
  checking optimized scores against a vectorized reference plus the
  tie-aware index contract. Full-sweep result: `384/384 passed, max score
  err 0.001953, max index gap 0.000000`.
- Added `benchmarks/probe_training_smoke.py` (M10 gate 6 tier 1): 300-step
  head-only contrastive+FLOPS training from identical fp32 master weights
  under fp16 AMP (GradScaler) and bf16 autocast; hybrid-vs-optimized loss
  parity 0.03–0.24%, far inside the 5% gate.
- Added autocast forward/backward tests for all three backends (fp16 and
  bf16), an adaptive-default subprocess test, a one-time-fallback-warning
  test, and a slow training-parity test that reuses the smoke script's
  `run_mode`; the routing test is now availability-aware. Suite: 105 → 114.
- Added
  [docs/sparton_milestone10_promotion_memo.md](docs/sparton_milestone10_promotion_memo.md)
  with the gate-by-gate evidence, the tier-2 real-model runs (transformers
  5.11 + accelerate 1.14 installed; torch/triton untouched), the decision
  record, and residual risks.

#### Validation

- Hardened-env exit run: glob `py_compile` passed; full pytest `114 passed`
  (quick loop `98 passed, 16 deselected`); shape soak `384/384`; 300-step
  training smoke passed both modes; dev-shape benchmark run 2 `hyb+b 1.181 /
  opt+b 0.900 / hyb f+b 2.642 / opt f+b 2.325 ms`; canonical grid 9/9 rows
  optimized faster for forward and fwd+bwd; five 150-step
  `training/train.py` runs (hybrid×3, optimized×2) finite and decreasing
  with backend parity within run-to-run noise (seed-43 mean losses match to
  0.3%); `import sparton` silent; `git diff --check` clean.

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
