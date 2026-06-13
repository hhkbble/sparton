# Changelog

Short summaries of notable repository changes, grouped by the date they landed.
Full evidence (gates, profiles, deviations, known gaps) lives in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md); the built system in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md); the working method in
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

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
