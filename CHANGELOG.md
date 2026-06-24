# Changelog

Short summaries of notable repository changes, grouped by the date they landed.
Full evidence (gates, profiles, deviations, known gaps) lives in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md); the built system in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); the working method in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## 2026-06-24

- **`optimized` forward is now a pure-Triton TMA kernel; all Gluon removed from the package.**
  The default `optimized` backend (`src/sparton/_backend_optimized.py`, op `sparton::optimized_fwd`)
  is a persistent, fused max/argmax/ReLU/log1p forward in stock `@triton.jit`: host-side TMA
  descriptors (`triton.tools.tensor_descriptor`; no `triton.set_allocator` side-effect), `tl.dot`,
  a grid-stride over `(batch, vocab-tile)` tiles, with `D`/`V` as `constexpr` (per-model
  loop/div-mod fold; B/S stay runtime). It picks its tile by **measurement**, not derivation: a
  self-implemented autotuner over a self-contained candidate set keyed on `(D, V, dtype, arch)` —
  **not** B/S (`SPARTON_OPTIMIZED_AUTOTUNE`, default on; off → a small analytic tile), timed with
  plain `triton.testing.do_bench`. Warp specialization is a tuned dimension: one kernel carries
  both argmax epilogues behind a `constexpr` (the 2-result `tl.reduce` combine for the homogeneous
  path; `tl.max` + masked `tl.min` for the WS path — the only form Triton 3.7.1's auto-WS accepts),
  and the tuner enables WS only where it measures faster. This kernel **replaces** the previous
  Gluon `optimized` (TMA + `mma_v2` policy bank): on the re-baseline it reaches **≈parity** at
  large/throughput shapes (the single Gluon tile was +7…16% slower) and **wins** the small/serving
  (**≈0.68×**) and small-V (**≈0.65×**) regimes.
  - **Backends are now `{hybrid, naive, optimized}`.** The interim `experiment` / `experiment_gluon`
    backend names are removed (clean break — selecting them raises like any unknown backend).
  - **Availability / default resolution:** `optimized` is the default where CUDA capability is
    **sm_90+ (Hopper)** and `triton.tools.tensor_descriptor` imports; below that floor default
    resolution falls back to `hybrid` with a one-time `RuntimeWarning`. This deliberately narrows
    the default envelope vs the old Gluon optimized (sm_80+) — host-side TMA is a Hopper-class
    feature. An explicitly selected `optimized` on an unsupported device still raises with the
    reason.
  - **No op-schema / saved-tensor / backward change** — backward is the shared M13 split segmented
    backward, unchanged.
  - **Removed:** `_backend_optimized_gluon.py`, `_backend_experiment_gluon.py`, `_gluon_runtime.py`,
    `_gluon_policy_runtime.py`, `_runtime_policy.py`, and the Gluon-only scripts (`bench_gluon_gemm`,
    `probe_gluon_epilogue`, `probe_mma_matrix`, `ncu_runner`, `probe_warp_specialize`). The
    GPU-kernel-optimization teaching blogs moved to the gitignored `docs/obsolete/`.
  - Validation: full suite **135 passed**; optimized shape soak 24/24; training-smoke
    hybrid-vs-optimized parity within tolerance (fp16 rel diff ~3e-4, bf16 ~2e-3). Evidence:
    DEVELOPMENT.md "Post-M13 — optimized promoted to a pure-Triton TMA forward"; built system in
    ARCHITECTURE.md §5.1.

  *Tried and abandoned on the way here (full arc in DEVELOPMENT.md):* a Gluon `experiment` backend
  (persistent + single hardware-derived tile + `mma_v2`); a device-side `gl.warp_specialize`
  producer/consumer variant (viable on sm_120 but not a standalone production win — superseded by
  the `tl.range(warp_specialize=)` tuned dimension); a D5 mask-aware early exit (`SKIP_MASKED` —
  1.86× on ragged padding but +1.2% full-length overhead, never promoted); and a heterogeneous
  CUDA-core offload exploration. None shipped; the pure-Triton kernel above is the convergent design.

## 2026-06-13

- **Documentation consolidated to three references.** The 16 design/review/
  platform/memo files under `docs/` were merged and deduplicated into
  [ARCHITECTURE.md](docs/ARCHITECTURE.md) (the built system),
  [DEVELOPMENT.md](docs/DEVELOPMENT.md) (the development memo of record,
  M2→M13), and [METHODOLOGY.md](docs/METHODOLOGY.md) (the working method plus
  the kernel-optimization technique reference, replacing
  `triton_gluon_kernel_optimization.md`). `AGENTS.md` slimmed to repository
  facts, invariants, environment, and validation with pointers to the three
  docs; this CHANGELOG condensed to short summaries; every cross-reference in
  code comments, `README.md`, and `scripts/README.md` repointed. No code
  behavior changed (comment/docstring text only).
- **Repository self-containment.** `benchmarks/` renamed to `scripts/`
  (`git mv`, history preserved); `tests/data/` adopted as the in-repo home for
  generated data and run artifacts (small fixtures committed, large artifacts
  gitignored and regenerable). Two session probes promoted to validated tools:
  `scripts/m13_traffic_model.py` (output of record
  `tests/data/m13_traffic_model_out.txt`) and `scripts/dump_backward_ir.py`.
  Documents must not rely on artifacts outside the repository.
- **M13 — split segmented backward promoted** (milestone complete).
  `sparton::fused_sparton_bwd` now runs the split design: a vectorized
  uniform-chunk streaming pass plus a mixed-chunk segmented scan complementing
  at one shared 64-entry granularity (schema-safe swap; op schema, fake
  registration, autograd wiring, and saved tensors untouched). Captured-real
  records improve **1.46–1.60×** over the M11 segmented design; the synthetic
  `f=0.10` short-run regime regresses 6–16% (recorded, **pending maintainer
  ratification**, with a one-commit revert). The M11 segmented design is
  retained, test-pinned, as `legacy_fused_sparton_bwd`; the M2-era atomic
  kernel was deleted. Evidence: [DEVELOPMENT.md](docs/DEVELOPMENT.md) M13.
- **M12 — forward track closed without kernel work** (milestone complete). The
  production forward kernel profiled tensor-pipe-bound at 92–94% with no
  scheduling slack, so the persistent/warp-specialized rewrite was not entered
  and the launcher-v2 deferral stands. A no-go is a recorded outcome.
  Evidence: [DEVELOPMENT.md](docs/DEVELOPMENT.md) M12. Tooling added:
  `scripts/ncu_forward_target.py`, `scripts/bench_host_overhead.py`, and a
  `--mask-density` sweep in `scripts/bench_sparton_baseline.py`.

## 2026-06-12

- **M11 — segmented backward promoted** (milestone complete). Replaced the
  shared M2-era atomic backward with a three-kernel segmented design (fused
  prep + exclusive-owner embed/bias-grad kernel + sort-then-segmented-scan
  hidden-grad kernel), cutting L2 reduction-sector traffic up to **264×** on
  captured-real query records and the implied optimized backward −32…−41%
  across the canonical grid. `embed_grad`/`bias_grad` are now exactly
  deterministic. Backward harness `scripts/bench_backward.py` and real
  index-distribution capture `scripts/capture_index_distributions.py` added.
  Evidence: [DEVELOPMENT.md](docs/DEVELOPMENT.md) M11.
- **M10 — `optimized` promoted to the default backend.** Where CUDA sm_80+ and
  `triton.experimental.gluon` are available, `SpartonHead` defaults to the
  Gluon forward (−24% forward / −12% fwd+bwd on the dev shape, ~14× less peak
  memory); otherwise it falls back to `hybrid` with a one-time `RuntimeWarning`
  (the only adaptive fallback). Explicit backend selection never falls back.
  Added autocast support (`_validation.autocast_canonicalize`) so AMP training
  with fp32 master parameters works on every backend, plus
  `scripts/soak_optimized_correctness.py` and `scripts/probe_training_smoke.py`.
  `LSRTrainer.save_model` switched to `torch.save` (tied head weight).
  Evidence: [DEVELOPMENT.md](docs/DEVELOPMENT.md) M10.
- **M9 — production readiness.** Fixed the 14 review findings (notably the
  non-contiguous-input gradient bug F1) without kernel/schema changes; added
  the `src/sparton/_validation.py` input-contract layer with named errors, the
  tie-aware index contract, a silent-import guarantee, and test-infra
  hardening. Suite 47→105. Evidence: [DEVELOPMENT.md](docs/DEVELOPMENT.md) M9.

## 2026-06-11

- **M8 — experimental Gluon `optimized` forward.** Added
  `sparton::optimized_fwd`: a single TMA + `mma_v2` Gluon kernel with bounded
  policy autotune that stores only `[B, V]` (output-only memory), plus the
  `_gluon_runtime.py` shim (the only module importing
  `triton.experimental.gluon`) and `_runtime_policy.py`. Forward-only and
  unpromoted at this point. Evidence: [DEVELOPMENT.md](docs/DEVELOPMENT.md) M8.
- **M5 — backend router + `naive` Triton baseline.** Split the hybrid
  implementation into `_backend_hybrid.py` behind the `sparton_kernel.py`
  facade/router; added `SpartonHead(..., backend=...)` (`hybrid` default,
  `SPARTON_BACKEND` import-time default) and the `tl.dot` fused-forward
  `naive` debug baseline (`sparton::naive_fwd`). Added the pytest 9 harness, a
  PyTorch reference, and the canonical benchmark scripts; raised Python to
  `>=3.10`. Evidence: [DEVELOPMENT.md](docs/DEVELOPMENT.md) M5.
- **M2 — `bias=None` backward fix.** `SpartonHead(use_bias=False)` backward now
  returns `None` for the bias gradient (bias-grad atomics compiled out); op
  schemas declare optional bias as `Tensor?`. Evidence:
  [DEVELOPMENT.md](docs/DEVELOPMENT.md) M2.
- **Gluon platform feasibility + environment hardening.** Recorded the sm_120
  MMA availability matrix (only `mma_v2` works; WGMMA/TCGen05 abort in LLVM),
  the TMA+`mma_v2` GEMM microbenchmark, hybrid baselines, and the two
  workspace defects (missing CUDA headers; `/tmp` `noexec`) with the hardened
  environment prefix. Added the `scripts/` probe/benchmark suite. Facts now in
  [ARCHITECTURE.md](docs/ARCHITECTURE.md) §2.

## References

- [Architecture](docs/ARCHITECTURE.md) — the built system.
- [Development memo](docs/DEVELOPMENT.md) — milestones M2→M13, evidence and decisions.
- [Methodology](docs/METHODOLOGY.md) — working method + kernel-optimization technique.
