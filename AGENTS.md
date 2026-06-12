# Repository Instructions

This file applies to the entire `/workspace/sparton` repository. Treat it as
the source-grounded operating guide for future agents and contributors. It
encodes both the repository facts and the working method that produced the
M8→M11 arc; the method sections (Operating Loop, Evidence and Measurement,
Performance-Optimization Loop, Testing Doctrine, Milestone Review, House
Style, Documentation System) apply to every task, not only kernel work.
Rules cite the memo or design section that taught them so you can audit the
evidence rather than trust the rule.

## Project Map

- `src/sparton/` is the installable Python package. Its public surface is
  currently `SpartonHead`, exported from `src/sparton/__init__.py` only when
  CUDA is available.
- `src/sparton/sparton_kernel.py` is the public facade and backend router
  (`resolve_backend`, `SpartonHead`, re-exports, lazy `optimized` symbols).
  Backend implementations live beside it:
  - `_backend_hybrid.py` — compatibility backend (compiled tiled matmul +
    Triton reduction) plus the shared backward used by all backends: the
    M11 segmented backward (prep + exclusive-owner embed/bias-grad kernel +
    sorted segmented-scan hidden-grad kernel inside
    `sparton::fused_sparton_bwd`) and the retained pre-M11 kernel behind
    `legacy_fused_sparton_bwd` (A/B reference of record, test-pinned);
  - `_backend_naive_triton.py` — `tl.dot` fused-forward debug baseline with
    bounded autotune;
  - `_backend_optimized_gluon.py` — Gluon TMA + `mma_v2` fused forward with
    policy autotune; the default backend where available (M10);
  - `_gluon_runtime.py` — the only module allowed to import
    `triton.experimental.gluon`; lazy shim plus capability whitelist;
  - `_gluon_policy_runtime.py` — lazy host-side policy/config/descriptor
    helpers shared by the optimized backend and the GEMM benchmark;
  - `_runtime_policy.py` — pure-Python policy generation (no torch/triton at
    module level; imported by tests on CPU-only machines);
  - `_validation.py` — shared autocast canonicalization and input-contract
    validation used by the per-backend forward wrappers.
- `training/` is a Hugging Face training/benchmark example, not a separate
  package. `training/model.py` wraps Hugging Face MLM backbones, and
  `training/train.py` wires dataset loading, tokenization, contrastive loss,
  sparsity regularization, and `Trainer`.
- `tests/` contains the kernel/reference pytest suite (132 tests at M11
  exit; the quick loop is `-m "not slow"`). Pytest (>=9) is configured in
  `pyproject.toml`.
- `benchmarks/` contains validated probe/benchmark/gate scripts (MMA
  availability, Gluon GEMM ratio gate, merged backend baselines, shape soak,
  training smoke, profiler launchers, and the M11 backward toolchain:
  index-distribution capture, distribution-aware backward harness, backward
  profiling target); usage in `benchmarks/README.md`.
- Document map: the forward plan lives in
  `docs/sparton_remaining_work_design_v4.md`;
  `docs/sparton_remaining_work_design_v3.md` remains authoritative for the
  executed M11 plan of record and the post-M10 snapshot/ratio derivations;
  `docs/sparton_remaining_work_design_v2.md` for the post-M8 review
  findings and M9/M10 provenance;
  `docs/sparton_gluon_remaining_work_design.md` for platform facts and the
  original measured evidence; `docs/sparton_milestone11_backward_memo.md`
  for the backward's mechanism evidence and analytic traffic model.
- There is currently no lint config, typecheck config, CI config, or lockfile.
  The documented command set in Validation is the gate mechanism.

## Orientation Before Changes

- Read `README.md` for user-facing behavior, setup, examples, and documented
  project status.
- Read `CHANGELOG.md` for recent repository changes before planning or editing.
- For milestone work, read the active design doc's milestone section
  (`docs/sparton_remaining_work_design_v4.md`) before planning: it contains
  file-level task specs, gates, and recorded decisions. Do not re-derive what
  it already settles; do not silently contradict it — evolve it and record
  the evolution (see Documentation System).
- Read other design notes under `docs/` only when relevant to the requested
  work. Do not use `AGENTS.md` as a project changelog.
- Check `git status --short` and the last few commits before editing.

## The Operating Loop

The cycle that produced M9–M11; follow it for any non-trivial task.

1. **Orient** (above), then **probe before designing**: when the task starts
   from a suspicion or a review finding, write a small disposable probe that
   demonstrates the behavior before writing the plan. Never put a claim in a
   plan or doc that you have not executed (design v2 Appendix A is the model:
   every finding has a rerun recipe).
2. **Plan as ordered, independently-landable tasks**, each naming its files,
   its behavior change, the tests it adds, and a runnable gate with an
   expected result. State scope constraints up front as "will NOT touch"
   lists (M9: no kernel-body changes, no autotune-config changes, no schema
   changes, no new dependencies) — they make reviews tractable and prevent
   drive-by churn.
3. **Bug fixes are red→green**: write the regression test first, watch it
   fail against unmodified code, capture the failing output verbatim for the
   memo, then fix. Never leave a commit boundary red — the red evidence lives
   in the memo, not in history (M9 memo, F1 section).
4. **Run the gate after every task**, not only at the end. A gate is a
   command plus a number (test count, ms window, ratio, tolerance) — never an
   adjective.
5. **Document at completion**: milestone memo with evidence, CHANGELOG
   entries, status note in the design doc, and corrections to any prose
   drafted before the measurements existed.
6. **Commit once per milestone/task** with a concise subject and a detailed
   what/why/validated-how body. Commit pre-existing unrelated worktree
   changes separately first so each commit is attributable.

Efficiency rules learned the long way:

- Copy the repo's exemplar for the artifact you are creating (see House
  Style) instead of designing from scratch; the shapes are proven.
- Give long sweeps a `--quick` subset for development; run the full sweep
  only as the gate.
- Long GPU jobs can run in the background while you edit docs/tests, but
  never run two Triton/Inductor-compiling processes concurrently — and
  never edit `src/` or `benchmarks/` while a background or *queued* process
  will import them: a chained second run imports the edited tree and
  silently corrupts the A/B (M11). Docs and tests are always safe to edit.
- A pytest run launched immediately after a heavy background GPU job can
  fail transiently (subprocess-based tests); rerun and classify before
  debugging (M11 memo §8).
- Autotune and Inductor caches are persistent (`cache_results=True`,
  `TORCHINDUCTOR_CACHE_DIR`): first runs pay compile/tune cost, reruns are
  cheap. Judge timings accordingly (below) and don't fear re-running gates.
- Historical results live in memos — cite them instead of re-running history,
  but re-measure anything that gates the current decision.

## Evidence, Measurement, and Gates

- Use the hardened environment prefix (see Environment) for every CUDA/Triton
  command; run benchmarks serially.
- **Judge benchmarks on the second consecutive run** (warm caches). Recorded
  baselines of record: dev shape `B=32,S=128,D=768,V=30522` fp16 — see the
  M10/M11 memo tables; rerun the row yourself before using it as a gate.
- **Measure A-vs-A before judging A-vs-B**: repeat the same configuration and
  use that spread as the noise band. Autotune config-selection jitter between
  processes is part of the band (±5% on borderline cells — M11 memo §5.2);
  a cross-impl difference inside the same-impl band is noise, not signal.
- **Keep measurement regimes separate**: `triton.testing.do_bench`
  (L2-flushed) is the latency of record; `ncu` serializes and flushes, so its
  durations are inflated — use it for structure, counters, and ratios; `nsys`
  for kernel inventory; sanitizer-run timings are meaningless. Never compare
  numbers across regimes (v1 §2.5 records a published mistake from mixing
  them).
- **Deposit a transcript for every number that carries a decision** (tee
  ncu/bench output to a file; bundles and profiles live outside the repo,
  e.g. `/root/m11_bundles/`, `/root/profiles/`). The largest finding class
  in the M11 review was decision-carrying numbers with no preserved log;
  an unpreserved spot-run may not be quoted as a result — label it or
  re-run it (M11 memo §8 item 7).
- **Classify a failure before fixing it.** Four verdicts are possible: a
  real defect (non-contiguous `embed_grad`, M9 F1); expected behavior that
  the gate mis-asserts (GradScaler-skipped early fp16 steps are normal AMP
  scale calibration — the M10 smoke's gate was rewritten, not the code); an
  out-of-contract input (near-tie index mismatches are allowed by the index
  contract); or behavior that is in-contract but the contract itself needs a
  maintainer ruling — escalate, then record the ruling in code comments and
  docs (the M11 mask-factor finding became a contract note, not a code fix;
  memo §9). Fixing before classifying produces wrong fixes.
- Perf claims need a mechanism, not just a delta: when a result surprises
  (cuBLAS unusually slow on a stress shape), say why or flag it as
  unexplained in the memo rather than letting the ratio stand alone.
- A gate that can be expressed as a script should be one (see
  `benchmarks/soak_optimized_correctness.py`): availability check, summary
  line, non-zero exit listing every failure.

## The Performance-Optimization Loop

The M11 extension of the Operating Loop for performance work. Kernel-level
techniques live in `docs/triton_gluon_kernel_optimization.md`; this section
is the repo-proven process around them. Every rule cites the M11 memo
(`sparton_milestone11_backward_memo.md`).

1. **Name the binding resource before designing** (method ref §3.3): profile
   the current implementation (ncu, direct op calls on the main thread) and
   identify the unit that binds — M11: L2 sector pipe at 56–58% while SM sat
   below 7% (§5.1). Then ask the altitude question: is the binder an
   algorithm-level *operation count* or a lowering-level inefficiency?
   Counts that scale with the problem (sectors, atomics, bytes) are
   invariant under grid restructuring, TMA, or a Gluon port — only changing
   *which operations exist* helps. Lowering-level tools (Gluon, IR reading,
   occupancy/register tuning) pay off only when the binder is SM-side.
2. **Write the analytic traffic model first**: per-buffer formulas in the
   problem dims and block params, validated against counters before any
   candidate is built (M11 §3 matched measured sectors to four significant
   figures on two shapes). The model then predicts each candidate's ceiling
   before you invest in it, arbitrates surprises, and becomes the memo's
   expected-vs-measured spine.
3. **Benchmark on realistic data distributions.** Synthetic-uniform inputs
   missed both decisive properties of real batches (dense scores; 20–46% of
   vocab entries sharing one argmax position — §4). Capture real
   distributions once (`capture_index_distributions.py` bundles, with
   distribution stats recorded at capture time), replay them in the harness
   (`bench_backward.py`), and keep synthetic sources for regimes real data
   does not cover. The data may also overturn design assumptions — measure
   the distribution before trusting the plan's picture of it.
4. **Prototype behind a registry; production stays untouched.** Candidates
   live in `benchmarks/` with the production op's exact signature, selected
   by `--impls`; the harness numerically verifies every cell against
   production *before* timing it, so a broken prototype cannot produce a
   timing row (§5). Wire the winner into the op only at promotion; retain
   the loser of record as an explicit, test-pinned legacy reference.
5. **Pre-register numeric decision rules** for timeboxed alternatives (the
   B2b early-stop rule, §5.1) so skipping work is defensible, and record
   the decision-criteria walk in the memo.
6. **Time at the op level for the verdict; profile at the kernel level for
   structure.** The op-level closure allocates exactly what production
   allocates — buffer fills and host passes count. The host side is
   in-bounds for "kernel" optimization: M11's unlock included `torch.sort`
   plus a fused prep kernel, and fusing seven elementwise launches into one
   was worth more than any kernel tweak at small shapes (§5.2).
7. **Audit the autotune key.** A performance-relevant argument missing from
   the key silently reuses a wrong config (`seq_len`, §5.2: query-tuned
   configs served documents at ~7% cost). Log selections with
   `TRITON_PRINT_AUTOTUNING=1`; note that Triton ≥3.6 keys include argument
   dtypes automatically.
8. **Read the IR when the question is about lowering** (method ref §6.1):
   `CompiledKernel.asm` / `nvdisasm` answer instruction-form questions
   (vector widths, scan lowering, spills) at zero GPU cost. Confirm the
   lowering before benchmarking a config family; settle counter surprises
   by reading the form, not inferring it from counter arithmetic.
9. **Sanitize kernels whose ownership semantics changed** (atomics→stores,
   `torch.empty` outputs): `compute-sanitizer` racecheck/memcheck/initcheck
   on a small shape — initcheck mechanically validates empty-allocation
   claims, and one harness run sweeps every autotune config under the
   sanitizer (§5.4).
10. **Per-iteration discipline**: accept a tuning change only with a
    profiler-confirmed mechanism (method ref §4.3), re-verify correctness
    after every kernel edit (the harness does this per cell), and stop
    optimizing when the remaining gap has a named, recorded bottleneck —
    the memo's residual-bottleneck note is the entry evidence for the next
    milestone (§6).

## Testing Doctrine

- **Tests must prove what they appear to prove.** The suite was green for an
  entire milestone while only ever exercising the optimized backend's
  fallback policy, because every test shape was "tiny" (design v2 F3). When a
  code path's activation depends on input properties (shape tiers, policy
  pruning, dtype), assert the activation itself in the test —
  `test_optimized_forward_nontiny_shapes` asserts the derived candidate set
  is larger than the fallback before checking outputs.
- **Conditional expectations must pin what they condition on.** A
  closed-form expectation computed from the kernel's own saved outputs is
  the right shape for backward tests at random non-tiny shapes (autograd
  through the reference re-litigates contract-legal near-tie choices), but
  it masks forward bugs unless the same test pins those saved outputs —
  reference scores within tolerance plus the index contract
  (`test_fused_backward_nontiny_shapes`; M11 memo §8 item 6).
- **Match assertion strength to input class**: deterministic constructed
  cases (intentional ties, masked winners, dyadic-rational patterns) assert
  exact equality — they pin tie policy. Random-input cases assert the
  contract (`assert_index_contract`), because backends with different
  accumulation precision legitimately disagree at near-ties. Converting one
  into the other in either direction is a bug.
- Every validation rule has a `pytest.raises(..., match=...)` test with a
  stable message substring. Error-message templates (see `_validation.py`:
  `sparton {backend} forward: {arg}{rule}; got {actual}`) are part of the
  API — tests depend on them; change them deliberately.
- Single-source gate logic: when a gate script and a test overlap, the test
  imports the script's function (`test_training_parity_smoke_autocast` reuses
  `probe_training_smoke.run_mode`; the backward harness's synthetic-input
  contract is pinned by importing `make_synthetic_case`) instead of
  duplicating it.
- Mark expensive coverage `@pytest.mark.slow` — including tests that
  autotune kernel families at non-tiny shapes; the default `pytest -q` runs
  everything, `-m "not slow"` is the documented quick loop. Don't let the
  quick loop lose meaning by marking cheap tests slow.
- Subprocess tests use absolute paths (`_REPO_ROOT`, `_SRC_PATH` in
  `tests/test_sparton_kernel.py`), never CWD-relative ones. The suite must
  pass under `python -m pytest`, the venv `pytest` console script, and
  `pytest /workspace/sparton/tests` from a foreign working directory — run
  all three after touching test infrastructure.
- Parametrized cases live in named tables with explicit ids
  (`FORWARD_CASES`, `BACKWARD_CASES`, `VALIDATION_ERROR_CASES`,
  `NONTINY_FORWARD_CASES`); `strict_parametrization_ids` is enabled.
- Gate availability with fixtures/helpers (`_forward_for_backend`,
  `_optimized_gluon_availability`), not device-name checks — capability, not
  hardware identity.
- Tensors built for gradient tests must be autograd leaves: an arithmetic
  result like `-torch.ones(..., requires_grad=True)` is a non-leaf whose
  `.grad` stays `None`; use `torch.full`/`torch.ones` directly (M11 memo §6
  red→green note).

## Milestone Review

Before a milestone closes, run an adversarial review of the diff **and the
evidence chain**, then fix or record every finding the review cannot refute
— silence is not a verdict. The M11 shape: independent reviewers per
dimension (kernel/op correctness; every quoted number checked against its
preserved log; doctrine compliance), each finding judged by two independent
verifiers prompted to refute it. That pass caught a contract-level gap the
gates could not see, an unpreserved figure that flattered the result, and a
test-gate hole — after all functional gates were green (M11 memo §8).
Findings about the contract itself go to the maintainer for a ruling
(Evidence section, fourth verdict) rather than being "fixed". Corrections
to the milestone's own documents are themselves recorded in the memo's
deviations section — the review is part of the evidence, not a cleanup to
hide.

## House Style

Copy the repo's best instance of a pattern instead of inventing a new shape:

| Artifact | Exemplar |
|---|---|
| Gate/sweep script | `benchmarks/soak_optimized_correctness.py` (docstring contract, availability gate in `main`, `--quick`, summary line, non-zero exit with failure list) |
| Training/integration probe | `benchmarks/probe_training_smoke.py` (reusable `run_mode`, per-mode gates) |
| Perf decision harness | `benchmarks/bench_backward.py` (impl registry, per-cell verification before timing, determinism protocol, provenance header, documented synthetic-input contract) |
| Capture-to-disk distribution tooling | `benchmarks/capture_index_distributions.py` (stats recorded at capture; bundles outside the repo with rerun recipe) |
| Contract checker + test tables | `tests/test_sparton_kernel.py` (`assert_index_contract`, case tables) |
| Input validation + error template | `src/sparton/_validation.py` |
| Forward design doc | `docs/sparton_remaining_work_design_v4.md` (active plan); `docs/sparton_remaining_work_design_v2.md` for the review-born shape (findings + evolution ledger) |
| Milestone memo | `docs/sparton_milestone11_backward_memo.md` (analytic-model spine, expected-vs-measured, decision record, deviations incl. self-corrections); `docs/sparton_milestone9_production_readiness_memo.md` (finding→fix table, red→green evidence); `docs/sparton_milestone10_promotion_memo.md` (gate-by-gate decision record) |

Rules:

- **Symmetry rule**: the Nth implementation of an existing pattern mirrors
  the structure of the others byte-for-byte where semantics allow. The three
  forward wrappers are intentionally line-for-line parallel; the hybrid F1
  bug existed precisely because hybrid lacked the wrapper the others had.
- **One-seam changes**: new cross-backend behavior is one shared helper
  called at exactly one layer (`autocast_canonicalize` at the top of each
  wrapper), never N divergent copies.
- **No silent fallbacks.** The single sanctioned exception is default-backend
  resolution (one-time `RuntimeWarning`, M10). Explicit selections raise with
  the reason. Do not add a second exception — data-dependent algorithm
  dispatch inside an op would also need a host sync, which is why M11
  shipped one kernel family, not an adaptive switch.
- Diagnostics go through `logging.getLogger("sparton")` at DEBUG. Never
  `print` from library code; a regression test enforces silent import.
- Comments state constraints the code cannot show (TMA alignment origin, the
  16-bit `dtype_name` assumption, run-boundary composition invariants, D2
  cross-references) — never mechanics, never change-narration.
  Cross-reference comments about kernel twins go above the decorator stack,
  never inside `@triton.jit`/`@gluon.jit` bodies (kernel-body bytes affect
  compiled-source hashes). Above-decorator placement is NOT cache-safe
  either: Triton's JIT cache key includes the function's starting line
  number (verified on 3.6 at the M12 close — a +2-line comment edit
  re-keyed the kernel's compile and autotune caches), so never edit such
  comments while a measurement campaign is in flight; land them before the
  runs or after the last run of record.
- Retired implementations are either deleted or promoted to an explicit,
  test-pinned reference with a role comment naming its removal condition
  (`legacy_fused_sparton_bwd`, M11) — never left as silent dead code.
- Capability dispatch for fatal-failure APIs (Gluon MMA families abort the
  process at LLVM selection) uses static whitelists probed in subprocesses
  (`probe_mma_matrix.py`), never try/except fallback.
- Python: match the file's existing style; type annotations on new public
  functions; keyword-only for new optional constructor args; English
  comments.

## Core Kernel Invariants

- `SpartonHead.forward` resolves a backend at construction and calls the bound
  per-backend wrapper. The layering rule is: wrapper (`hybrid_forward`,
  `naive_forward`, `optimized_forward`) → `_validation.autocast_canonicalize`
  (mirrors `torch.autocast` semantics so fp32 master parameters work under
  AMP) → shared `_validation.py` contract checks → `.contiguous()`
  canonicalization → custom op (`sparton::fused_sparton_fwd`,
  `sparton::naive_fwd`, `sparton::optimized_fwd`).
  Wrappers are the only public callables; the ops assume validated, contiguous
  inputs, and raw-op callers (for example profiling targets) bypass validation
  by design — do not "fix" that by validating inside the ops. All three
  forwards delegate backward to `fused_sparton_bwd_op`.
- Autograd registration saves max scores, max indices, hidden states, decoder
  weights, bias, and mask. Backward uses `fused_sparton_bwd_op` and accumulates
  `hidden_grad`, `embed_grad`, and `bias_grad` in `float32`. A backward swap
  that keeps the saved-tensor set is schema-safe inside the op (M11); a
  changed saved-tensor set requires a new op name.
- Preserve CUDA-only behavior unless explicitly implementing a CPU fallback.
  `__init__.py` intentionally exposes no `SpartonHead` when CUDA is unavailable.
- Preserve tensor contracts unless the task explicitly changes them:
  `hidden` is `[B, S, D]`, decoder/embed weights are `[V, D]`, optional bias is
  `[V]`, attention mask is `[B, S]`, and output sparse reps are `[B, V]`.
- The mask semantics in both PyTorch and Triton paths are part of correctness:
  logits are masked over sequence positions before ReLU, `log1p`, and max over
  the sequence dimension. The mask contract is binary {0, 1} (the standard
  tokenizer `attention_mask`); under it the shared backward is exact — a
  masked winner forces score 0 and the `scores > 0` guard zeroes its
  gradient, so no `mask[b, idx]` factor is needed (maintainer ruling,
  M11 review; memo §9). Non-binary values weight logits in the forward as
  an implementation property, but they are outside the contract: the
  backward does not differentiate the mask factor. Supporting weighted
  masks would be an extension — backward change plus tests against the
  `head="torch"` autograd path. Values are deliberately not validated
  (`_validation.py` checks are metadata-only; a value scan needs a device
  sync).
- Numerics contract: forward output follows hidden/logit dtype; `naive`/
  `optimized` accumulate logits in fp32 (more precise than hybrid's
  input-dtype logits — intended); backward gradient buffers are `float32`.
  `hidden_grad` is zero-filled (atomic accumulation; untouched rows stay 0);
  `embed_grad`/`bias_grad` are `torch.empty` — safe only because the embed
  kernel's unconditional exclusive-owner stores cover every element
  (initcheck-validated, M11 memo §5.4); keep allocation and coverage in
  lockstep if either changes.
- Do not casually change autotune config lists, tile-size heuristics,
  `torch.library.custom_op` signatures, fake registrations, or autograd setup.
  These affect compilation, graph capture, memory behavior, and gradients.
- There are two forward-style reduction helpers: one returns max values plus
  indices for autograd, and one returns only values. Keep their intended memory
  tradeoff clear when editing.
- `optimized` is the default backend where available (M10 promotion; CUDA
  sm_80+ plus importable `triton.experimental.gluon`); `hybrid` is the
  compatibility path and must stay behaviorally stable. The ONLY adaptive
  fallback in the package is default resolution: with no `backend` argument
  and no `SPARTON_BACKEND`, an unavailable optimized backend falls back to
  hybrid with a one-time `RuntimeWarning`. An explicitly selected backend
  must keep raising with the reason when unavailable — never extend the
  fallback beyond default resolution.
- Index semantics across backends: indices are meaningful only where the score
  is positive; within a backend ties resolve to the lowest sequence index;
  across backends with different accumulation precision the near-tie winner is
  unspecified. Random-input tests must use the tie-aware
  `assert_index_contract` helper, not exact index equality.

## Documentation System

Five documents with distinct roles; keeping them in role is part of every
task's definition of done.

| Document | Role | Update trigger |
|---|---|---|
| `README.md` | User-facing setup, examples, behavior, high-level status | User-visible behavior changes |
| `AGENTS.md` (this file) | Durable repository facts, invariants, and method | A rule changes or a new durable lesson is learned; never task history |
| Active design doc (`docs/sparton_remaining_work_design_v4.md`) | The forward plan: milestones, gates, architecture rules, recorded decisions | Milestone completion gets a dated status note pointing at the memo; superseding it means a new doc plus a supersession note in the old one |
| Milestone memos (`docs/sparton_milestone*_memo.md`) | Evidence of record: finding→fix mappings, gate transcripts, red→green captures, deviations, the not-validated list | One per milestone or substantial debugging session |
| `CHANGELOG.md` | Dated user-visible changes: behavior, API/schema, packaging, validation infrastructure, fixes, milestones | Every task that changes any of those |

Authoring rules:

- **Design docs must let a cold agent execute without re-deriving**: per-task
  file lists, behavior specs, tests to add, runnable gates with expected
  results, and an explicit rejected/deferred section with rationale (so good
  ideas aren't re-litigated and bad ones aren't re-tried). Ground every claim
  in a command someone can rerun (appendix of rerun commands).
- **Memos record what actually happened**, including deviations from the plan
  and what was deliberately not validated. An honest "known gaps" section is
  mandatory. For performance memos, the spine is the analytic model and an
  expected-vs-measured table per change (M11 memo) — a reader must be able
  to see *why* each change worked, not only that it did.
- **Every number in a committed document was produced by a command run in
  that session, and quotes its run of record.** If prose was drafted before
  the measurement, correct the draft to the measured value before
  committing. Quote both numbers when two regimes disagree, with the regime
  named. An isolated unpreserved measurement may not stand as a result —
  label it as unpreserved or re-run it (M11 memo §8 item 7).
- Documents form a supersession chain, never silent replacement: the old doc
  gets a status line naming its successor and what it remains authoritative
  for.
- Commits: concise subject + detailed body covering what changed, why, and
  how it was validated (with the actual gate results).

## Reference Material

- `docs/triton_gluon_kernel_optimization.md` is the kernel-optimization
  method reference (user-provided, adopted at M11 alongside this file):
  bottleneck-classification-first profiling, autotune hygiene,
  occupancy/register tuning loops, Gluon techniques, tooling checklist
  (incl. the §6.1 IR-stage visibility recipe), failure modes. Read it
  before any Triton/Gluon performance work; update trigger mirrors this
  file's (a durable method lesson, never task history).
- Upstream project metadata in this repo points at
  `https://github.com/thongnt99/sparton`; the local remote is different.
- The README citation references:
  `https://arxiv.org/abs/2603.25011`
- Hugging Face references used by the examples:
  `https://hf.co/FacebookAI/xlm-roberta-base`,
  `https://hf.co/naver/splade-v3`, and
  `https://hf.co/datasets/nthakur/swim-ir-cross-lingual`.
- For version-sensitive Hugging Face or Triton/Gluon behavior, verify current
  installed versions and upstream docs before relying on stale local notes.

## Environment

- The package declares Python `>=3.10`.
- Use the project venv unless there is a specific reason not to:
  `/workspace/venvs/sparton/bin/python`.
- For local source imports, use either `PYTHONPATH=src` or an editable install.
  Do not assume `sparton` is installed in the venv; it was not installed during
  the source audit.
- The venv was created with `--system-site-packages`. Avoid upgrading or
  replacing system-provided `torch` or `triton`; install only missing packages
  needed for the task, and verify `torch.__version__`/`triton.__version__`
  before and after any install. As of M10, `torch`, `triton`, `datasets`,
  `transformers` (5.11.0), and `accelerate` (1.14.0) are present.
- CUDA availability alone is insufficient for kernel validation. In this
  workspace, Triton/TorchInductor probes need:
  `TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas`.
- Expect first Triton runs to include autotuning and compilation overhead.
  Avoid treating first-step timing as steady-state performance.
- Two workspace defects make cache-cold TorchInductor compiles fail unless the
  environment is hardened: the NVIDIA Triton wheel ships no bundled CUDA
  headers (`cuda.h` missing for cold `cuda_utils` builds), and `/tmp` is
  mounted `noexec` (Inductor-redirected Triton caches cannot be `dlopen`ed).
  Set `CPATH=/usr/local/cuda-13.2/include` and
  `TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor`; details in
  `docs/sparton_gluon_remaining_work_design.md`.
- Useful shell prefix for local probes:
  `TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas CPATH=/usr/local/cuda-13.2/include TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor PYTHONPATH=src /workspace/venvs/sparton/bin/python`.
- Profilers: `ncu` and `nsys` are on PATH;
  `/usr/local/cuda-13.2/bin/compute-sanitizer` works on this host since the
  post-M11 restart (under stock WSL2/WDDM it fails with "Device not
  supported" — that error means the host needs its debugger interface
  enabled, not that the code is wrong).
- Avoid running multiple Triton/Inductor-compiling processes concurrently when
  validating or benchmarking; serialize runs for attributable results.

## Task Routing

- For public API or packaging work, start from `pyproject.toml`,
  `src/sparton/__init__.py`, and `SpartonHead`.
- For kernel correctness/performance work, start from
  `src/sparton/sparton_kernel.py` and the relevant `_backend_*.py`; build a
  small PyTorch reference before changing Triton/Gluon kernels or
  custom-op/autograd wiring; probe risky APIs in subprocesses first; follow
  the Performance-Optimization Loop for perf work.
- For backward work specifically, the M11 memo carries the mechanism
  evidence and the analytic model; `benchmarks/bench_backward.py` is the
  harness and `legacy_fused_sparton_bwd` the A/B reference. The residual
  bottleneck of record is embed-gather latency in the segmented kernel (no
  unit above ~35% on real doc records — memo §6); design v4 §3 M13 is the
  entry-gated plan for chasing it.
- For training behavior, start from `training/model.py` and `training/train.py`;
  avoid importing training modules unless the optional Hugging Face
  dependencies are needed for the task.
- **M12 (forward track) is closed without kernel work** (2026-06-13): the
  production forward kernel is tensor-pipe-bound at 92–94% with the L2
  fabric at ~90% and DRAM at the compulsory floor — no scheduling bubbles
  for a persistent/warp-specialized rewrite; evidence, the one-provenance
  state table, and the terminal residual-bottleneck note (per-cycle pipe
  efficiency + L2 pressure at the autotuned 64×64×32 tile shape) live in
  `docs/sparton_milestone12_forward_memo.md`. Launcher v2 (M12-T1) stays
  deferred (v4 §6; the T3 revival trigger never fired). The next planned
  milestone is **M13 (backward residual track)**, design v4 §3: its
  unconditional T0 (analytic gather ceiling + training-scale debt) is the
  next task; kernel work stays entry-gated, with the M12 memo §3 table as
  the forward column of the cross-track comparison.

## Training and Hugging Face References

- `SpladeModel` supports `head="torch"`, `head="compiled"`, and
  `head="sparton"` (plus `sparton_backend=` for explicit backend pinning).
  Keep these modes behaviorally aligned when changing model code.
- `train.py` defaults to `FacebookAI/xlm-roberta-base` and
  `nthakur/swim-ir-cross-lingual` with languages `de,es,fr`.
- Full training downloads large Hub assets and can be expensive. Do not run it
  casually as validation; prefer small synthetic or smoke probes
  (`benchmarks/probe_training_smoke.py`) unless the user explicitly asks for
  a training run. The xlm-roberta-base weights and the swim-ir `de` split are
  cached locally since the M10 tier-2 runs.
- `SpladeModel` ties the head weight to the backbone word embeddings;
  safetensors refuses shared tensors and transformers 5 removed
  `save_safetensors`, so `LSRTrainer.save_model` serializes with `torch.save`
  — do not reintroduce safetensors saving for this model.
- `training/train.py` is smoke-validated against transformers 5.11 only
  (150-step runs); checkpointing/resume/distributed paths are unvalidated.
- The README quick-start references `naver/splade-v3`; at audit time that model
  was gated and tagged `license:cc-by-nc-sa-4.0`.
  `nthakur/swim-ir-cross-lingual` is tagged `license:cc-by-sa-4.0`.
- If touching Hugging Face integration, verify current Hub metadata and local
  package versions instead of relying only on README examples.

## Validation

- For docs-only changes, review the diff and status:
  `git diff -- README.md AGENTS.md CHANGELOG.md docs/`
  `git status --short`
- For brand-new untracked docs, use `git diff --no-index`, for example:
  `git diff --no-index -- /dev/null CHANGELOG.md`
- Syntax check for the current Python files:
  `/workspace/venvs/sparton/bin/python -m py_compile src/sparton/*.py training/*.py tests/*.py benchmarks/*.py`
- Import check from source (must print the export list with no other stdout):
  `PYTHONPATH=src /workspace/venvs/sparton/bin/python -c "import sparton; print(sparton.__all__)"`
- Pytest suite (full; append `-m "not slow"` for the quick loop):
  `TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas /workspace/venvs/sparton/bin/python -m pytest -v`
- CUDA/Triton probes in this workspace should include:
  `TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas`.
- For kernel changes, compare against a direct PyTorch reference on small CUDA
  tensors and check both forward values and backward gradients. Include masked
  sequence positions, bias and no-bias cases if relevant, multiple `S`/`V`
  shapes, and at least one nontrivial tile boundary.
- For changes that touch the optimized forward's correctness surface
  (kernel, policy generation, descriptors, validation), rerun the shape soak:
  `benchmarks/soak_optimized_correctness.py` (use `--quick` while iterating,
  the full sweep as the gate).
- For changes that touch autograd, AMP, or the training path, rerun
  `benchmarks/probe_training_smoke.py`.
- For backward-kernel changes, rerun `benchmarks/bench_backward.py` with
  `--impls current,legacy` over {uniform, zipf} and the captured-real
  bundles (regenerate via `benchmarks/capture_index_distributions.py` if
  `/root/m11_bundles/` is gone); uniform-only evidence is never sufficient
  for a backward change (M11 rule of record).
- For new or changed kernels whose ownership semantics differ from their
  predecessor (atomics→plain stores, `torch.empty` outputs), run
  `compute-sanitizer` racecheck/memcheck/initcheck on a small shape through
  the harness (commands in the M11 memo §5.4).
- For training changes, use a tiny local or sliced dataset smoke test before
  any full Hub-backed training run.

## Known Sharp Edges

- The local Git remote is `https://github.com/hhkbble/sparton.git`; the README
  and package metadata reference `https://github.com/thongnt99/sparton` and the
  citation references `https://github.com/thongnt99/lsr-kernel`.
- The `pyproject.toml` `authors` field is still the upstream placeholder; the
  Homepage URL points at the upstream repository. Changing either is an
  ownership decision, not a cleanup.
- Importing `sparton` is silent on stdout; diagnostics go through the
  `"sparton"` `logging` logger at DEBUG level. A regression test enforces the
  silent import.
- The backward is not bitwise-deterministic run-to-run. Since M11,
  `embed_grad`/`bias_grad` are exactly deterministic (exclusive-owner plain
  stores) and `hidden_grad` atomics are reduced to ~two partial sums per
  destination run, but their accumulation order still varies (measured
  per-call relative spread: gradient-norm ≤ 1.2e-7, element-sensitive
  loss-proxy ≤ 4.2e-6 — M11 memo §7; the M10 "~20% loss spread in the
  chaotic early regime" predates M11 and overstates the current band,
  which has not been re-measured at training scale). Establish same-config
  noise bands before reading meaning into cross-config training
  differences; do not promise bitwise-reproducible training.
- Under fp16 AMP, the GradScaler's default 2^16 initial scale legitimately
  overflows fp16 score-gradients in early steps; skipped steps during scale
  calibration are expected, not a bug (gate pattern in
  `probe_training_smoke.py`).
- `SpartonHead` has no CPU fallback. CPU-only environments should use PyTorch
  reference paths, not the Sparton kernel.

## Editing Expectations

- Inspect the source path you are changing before editing. Prefer actual
  runtime behavior over README claims; prefer probes over recollection.
- Keep changes surgical. Do not reformat large files, change public
  signatures, add dependencies, or alter kernel/training behavior outside the
  requested scope. Declare the scope boundary before starting (Operating
  Loop, step 2).
- Use English for comments and explain non-obvious intent, not mechanics.
- Preserve existing style unless a focused cleanup is part of the task.
- Never overwrite user changes. Check `git status --short` before editing
  when worktree state matters.
