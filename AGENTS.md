# Repository Instructions

This file applies to the entire `/workspace/sparton` repository. Treat it as
the source-grounded operating guide for future agents and contributors. It
encodes both the repository facts and the working method that produced the
M8→M13 arc; the method sections (Operating Loop, Evidence and Measurement,
Performance-Optimization Loop, Testing Doctrine, Milestone Review, House
Style, Documentation System) apply to every task, not only kernel work.
Rules cite the memo or design section that taught them so you can audit the
evidence rather than trust the rule.

How to use this file: read Project Map + Orientation always; then the
section for your task class (Task Routing names the entry points); copy the
House Style exemplar for any artifact you create; run the Validation
commands that match what you touched. The method sections are the
difference between competent output and this repo's standard — when in
doubt about process, the Operating Loop is the default and the others are
its specializations.

## Project Map

- `src/sparton/` is the installable Python package. Its public surface is
  currently `SpartonHead`, exported from `src/sparton/__init__.py` only when
  CUDA is available.
- `src/sparton/sparton_kernel.py` is the public facade and backend router
  (`resolve_backend`, `SpartonHead`, re-exports, lazy `optimized` symbols).
  Backend implementations live beside it:
  - `_backend_hybrid.py` — compatibility backend (compiled tiled matmul +
    Triton reduction) plus the shared backward used by all backends: the
    M13 split segmented backward (prep with a device-side active count +
    exclusive-owner embed/bias-grad kernel + sort + a vectorized
    uniform-chunk streaming pass plus a mixed-chunk segmented scan whose
    predicates complement at one shared CHUNK granularity, inside
    `sparton::fused_sparton_bwd`) and the retained M11 segmented kernel
    behind `legacy_fused_sparton_bwd` (A/B reference of record,
    test-pinned);
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
- `tests/` contains the kernel/reference pytest suite (136 tests at the M13
  close; the quick loop `-m "not slow"` is 113). Pytest (>=9) is configured
  in `pyproject.toml`. `tests/data/` is the repository-local home for
  *generated* data and run artifacts (never Hugging Face downloads): small
  fixtures committed, large artifacts gitignored and regenerable by a
  `scripts/` script that reuses existing files — conventions and the
  current contents of record in `tests/data/README.md`.
- `scripts/` contains validated probe/benchmark/gate scripts: MMA
  availability, Gluon GEMM ratio gate, merged backend baselines (with
  `--mask-density`), shape soak, training smoke, host-overhead recorder,
  the forward and backward direct-op profiling targets, the backward
  toolchain (index-distribution capture with regularizer pass-through and
  reuse-if-exists outputs, distribution-aware backward harness with
  per-cell verification), the M13 analytic traffic model
  (`m13_traffic_model.py`), and the backward IR/config dump tool
  (`dump_backward_ir.py`); usage in `scripts/README.md`. Only validated,
  reusable tooling lives here — disposable probes go to the gitignored
  `tests/data/runs/<label>/` until they prove durable (Operating Loop).
- Document map: the executed plan of record for M12/M13 lives in
  `docs/sparton_remaining_work_design_v4.md` (both kernel tracks closed;
  no milestone is planned after M13);
  `docs/sparton_remaining_work_design_v3.md` remains authoritative for the
  executed M11 plan of record and the post-M10 snapshot/ratio derivations;
  `docs/sparton_remaining_work_design_v2.md` for the post-M8 review
  findings and M9/M10 provenance;
  `docs/sparton_gluon_remaining_work_design.md` for platform facts and the
  original measured evidence. Mechanism evidence: the M13 memo for the
  current backward and its validated gather/traffic model, the M12 memo
  for the forward's profile and no-go decision, the M11 memo for the
  segmented design the split supersedes.
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
  the evolution (see Documentation System). A well-written milestone section
  is directly executable; the fastest sessions are the ones that act on it
  instead of re-planning it.
- Read other design notes under `docs/` only when relevant to the requested
  work. Do not use `AGENTS.md` as a project changelog.
- Check `git status --short` and the last few commits before editing.

## The Operating Loop

The cycle that produced M9–M13; follow it for any non-trivial task.

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

### Session conduct (how a fast session is actually run)

The M13 close was a single session covering entry evidence, a kernel
promotion, and the review; these are the habits that made it fast without
cutting evidence:

- **Pipeline GPU time against authoring time.** GPU jobs run in the
  background (serially — see below) while docs, tests, and the memo are
  written; the only forbidden overlap is editing `src/` or `scripts/`
  while a background or *queued* process will import them — a chained
  second run imports the edited tree and silently corrupts the A/B (M11).
  Docs and tests are always safe to edit. Sequence kernel-adjacent edits
  into campaign windows: land them before a measurement campaign starts or
  after its last run of record (the cache-key rule, House Style).
- **Write the memo incrementally, measured content only.** Create the
  skeleton early; fill each section the moment its numbers land, citing the
  transcript in the same edit; leave unmeasured sections as explicit stubs.
  Prose written ahead of results is the error class the Documentation
  System forbids — at M13 the one sentence drafted ahead of T1's outcome
  had to be retracted within minutes.
- **Measure the unit cost of an expensive planned step once before
  scheduling around it.** The M13 plan budgeted ~20 minutes per tier-2
  training run from a cold-start estimate; one measured run showed 24 s
  steady-state, which changed the campaign's shape. One probe beats a
  schedule built on an assumed cost.
- **Deposit as you go.** Every decision-carrying run is teed to a
  transcript at launch time, not reconstructed later; disposable probes,
  analysis scripts, profiles, and transcripts live under the gitignored
  `tests/data/runs/<label>/` (convention in `tests/data/README.md`) and
  the memo cites them by path. **Never deposit to `/tmp`**: background-task
  stdout buffers land there transiently, the mount is `noexec` and gets
  wiped — nothing in `/tmp` may be cited or relied on (the M13 review
  agents' scratch went there and none of it survived as evidence; the
  recorded findings did because they were in the memo). `scripts/` is only for validated, reusable
  tooling — a probe that proves durable is promoted there with
  repo-relative paths and run end-to-end at promotion (the M13 traffic
  model and IR-dump tool are the precedents). Documents must never rely
  on artifacts outside the repository: anything load-bearing either lives
  in-repo or has an in-repo regeneration script.
- Copy the repo's exemplar for the artifact you are creating (House Style)
  instead of designing from scratch; the shapes are proven.
- Give long sweeps a `--quick` subset for development; run the full sweep
  only as the gate — and know what the subset cannot prove (Testing
  Doctrine: pinned axes).
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
  M12/M13 memo tables; rerun the row yourself before using it as a gate.
- **Measure A-vs-A before judging A-vs-B**: repeat the same configuration and
  use that spread as the noise band. Autotune config-selection jitter between
  processes is part of the band (±5% on borderline cells — M11 memo §5.2);
  a cross-impl difference inside the same-impl band is noise, not signal.
  A single wild cell inside an otherwise-consistent run is classified before
  any cell of record is taken from it — a third consecutive run plus an
  interference check separates transient host noise from signal (M13 memo
  §3: two contaminated cells, classified, no record taken from that run).
- **Keep measurement regimes separate**: `triton.testing.do_bench`
  (L2-flushed) is the latency of record; `ncu` serializes and flushes, so its
  durations are inflated — use it for structure, counters, and ratios; `nsys`
  for kernel inventory and single-regime per-call shares; sanitizer-run
  timings are meaningless. Never compare numbers across regimes (v1 §2.5
  records a published mistake from mixing them). When an op-level do_bench
  number must be reconciled with a kernel-level sum, *measure* the bridge
  (M13 memo §5.2: the do_bench-vs-GPU-sum gap is itself a quoted number),
  never assume it away.
- **Deposit a transcript for every number that carries a decision** (tee
  ncu/bench output to a file under `tests/data/runs/<label>/`; generated
  bundles live in `tests/data/bundles/` — both gitignored, see
  `tests/data/README.md`). The largest finding class
  in the M11 review was decision-carrying numbers with no preserved log;
  an unpreserved spot-run may not be quoted as a result — label it or
  re-run it (M11 memo §8 item 7).
- **A profile is evidence about the exact artifact profiled.** If the code,
  its config family, or its selected config changes after the profile —
  even by an edit that "shouldn't matter" — mechanism gates must be
  re-discharged on the shipped form. The M13 review caught the E5 gate
  discharged against an intermediate variant's profile; the production
  re-profile landed at a different config and a different top unit (M13
  memo §8 item 7; the same failure class as M11 §8 item 7).
- **"Pre-registered" means committed before the measurement.** The claim is
  auditable only by the commit boundary; a rule fixed in-session before its
  runs is honest but is written as exactly that, not as pre-registration
  (M13 memo §8 item 11). Pre-register decision rules and exit numbers in a
  commit (or in the already-committed design doc) before candidate work
  starts; apply them as written; a missed clause is recorded as a deviation,
  never reinterpreted silently.
- **Classify a failure before fixing it.** Four verdicts are possible: a
  real defect (non-contiguous `embed_grad`, M9 F1); expected behavior that
  the gate mis-asserts (GradScaler-skipped early fp16 steps are normal AMP
  scale calibration — the M10 smoke's gate was rewritten, not the code); an
  out-of-contract input (near-tie index mismatches are allowed by the index
  contract); or behavior that is in-contract but the contract itself needs a
  maintainer ruling — escalate, then record the ruling in code comments and
  docs (the M11 mask-factor finding became a contract note, not a code fix;
  the M13 synthetic-sparse trade is recorded pending ratification, memo §8
  item 12). Fixing before classifying produces wrong fixes.
- Perf claims need a mechanism, not just a delta: when a result surprises
  (cuBLAS unusually slow on a stress shape), say why or flag it as
  unexplained in the memo rather than letting the ratio stand alone.
- A gate that can be expressed as a script should be one (see
  `scripts/soak_optimized_correctness.py`): availability check, summary
  line, non-zero exit listing every failure. Scripts that write files
  default their outputs under `tests/data/` (gitignored per
  `tests/data/.gitignore`; CLI-overridable) — never `/tmp` and never
  absolute paths outside the repo (`capture_index_distributions.py` →
  `tests/data/bundles/`, `dump_backward_ir.py` → `tests/data/ir_dump/`
  are the precedents).

## The Performance-Optimization Loop

The extension of the Operating Loop for performance work, proven across
M11 (backward restructure), M12 (forward no-go), and M13 (backward split).
Kernel-level techniques live in `docs/triton_gluon_kernel_optimization.md`
(read it before any Triton/Gluon performance work; §6.1–§6.2 carry the
repo-added IR and layout-attribution recipes); this section is the
repo-proven process around them.

1. **Name the binding resource before designing** (method ref §3.3): profile
   the current implementation (ncu, direct op calls on the main thread) and
   identify the unit that binds. Then ask the altitude question — what class
   of change can move that binder? The arc has a worked example of each
   answer: M11's binder was an algorithm-level *operation count* (L2
   reduction sectors — only changing which operations exist helped; TMA and
   Gluon could not); M12's presumed SM-side scheduling gap was *disproved*
   by the first profile of the production kernel (tensor pipe already
   saturated — the milestone closed without kernel work, which is a valid
   outcome); M13's binder was a *lowering coupling* (the gather's vector
   width was anchored to the scan tile's layout — neither tuning nor pure
   occupancy work could move it; a structural kernel split could). Beware
   inherited priors: M12's "SM-side" prior came from a sibling benchmark
   kernel and did not transfer to production; profile the artifact you
   intend to change.
2. **Write the analytic traffic model first, as a runnable script.**
   Per-buffer formulas in the problem dims and block params, validated
   against counters before any candidate is built (M11 §3 matched measured
   sectors to four significant figures; M13 §4 to ≤1% on the decision
   counters across four shapes). Compute distribution statistics (run
   counts, mixed fractions, active fractions) from the *actual inputs* —
   same seeds, same bundle records — never estimated. Deposit the script
   with its output (the M13 model is promoted to
   `scripts/m13_traffic_model.py`, output of record
   `tests/data/m13_traffic_model_out.txt`) so the model re-runs against
   future counters. The model then prices each candidate's ceiling
   before you invest in it, arbitrates surprises, and becomes the memo's
   expected-vs-measured spine. Pre-register the validation bar; a residual
   above it is recorded with a named mechanism (M13: the +30% query-shape
   L1 residual was intra-warp sector coalescing under index collisions),
   never absorbed.
3. **Benchmark on realistic data distributions.** Synthetic-uniform inputs
   missed both decisive properties of real batches (dense scores; 20–46% of
   vocab entries sharing one argmax position — M11 §4). Capture real
   distributions once (`capture_index_distributions.py` bundles, with
   distribution stats recorded at capture time), replay them in the harness
   (`bench_backward.py`), and keep synthetic sources for regimes real data
   does not cover. The data may also overturn design assumptions — measure
   the distribution before trusting the plan's picture of it. Know which
   regimes are synthetic-only and say so when a result depends on one (the
   `f ≪ 1` regime has no real capture; the λ-probe brackets a
   dense→collapsed transition with nothing usable inside — M13 §7).
4. **Prototype behind a registry; production stays untouched.** Candidates
   live in `scripts/` with the production op's exact signature — the
   *full* signature contract including output optionality (the M13
   prototypes' `bias_grad` placeholder-vs-None mismatch survived every
   quick gate, memo §8 item 2) — selected by `--impls`; the harness
   numerically verifies every cell against production *before* timing it,
   so a broken prototype cannot produce a timing row. Wire the winner into
   the op only at promotion; retain the loser of record as an explicit,
   test-pinned legacy reference. Commit the prototypes at the decision
   point so the matrix of record stays reproducible from history (the M11
   practice; M13 skipped it and the review recorded the gap — memo §8).
5. **Pre-register numeric decision rules and exit numbers** (see Evidence:
   committed before measurement), including an early-stop for timeboxed
   alternatives (the B2b rule, M11 §5.1; the M13 variant budget), so
   skipping work is defensible — and record the decision-criteria walk in
   the memo even for the candidates you killed: each kill is justified by a
   number, not a vibe (M13 §5.1: three of four candidates killed by
   arithmetic before any code existed).
6. **Time at the op level for the verdict; profile at the kernel level for
   structure.** The op-level closure allocates exactly what production
   allocates — buffer fills, sorts, and host passes count. The host side is
   in-bounds for "kernel" optimization: M11's unlock included `torch.sort`
   plus a fused prep kernel; fusing seven elementwise launches into one was
   worth more than any kernel tweak at small shapes (§5.2). Use one nsys
   pass for single-regime per-call shares (sort and fill kernels never
   match a Triton `-k` regex — without it the op-vs-kernel gap gets
   mis-attributed; M13 §2.1).
7. **Audit the autotune key — then price what you find.** A
   performance-relevant argument missing from the key silently reuses a
   wrong config (`seq_len`, M11 §5.2: ~7% on documents). The same audit at
   M13 found the embed kernel's identical omission costs 0.69% — inside
   the noise band, classified immaterial, no fix (memo §2.2). The audit is
   mandatory; the fix is not — the price decides. Note Triton ≥3.6 keys
   include argument dtypes automatically. Selections are read from the
   autotuner's cache/`best_config` host-side (cache-hit selections print
   nothing under `TRITON_PRINT_AUTOTUNING=1` — M12 lesson).
8. **Read the IR when the question is about lowering — and know the limits
   of static reading** (method ref §6.1–§6.2): `CompiledKernel.asm` /
   `nvdisasm` answer instruction-form questions (vector widths, scan
   lowering, spills) at zero GPU cost — `scripts/dump_backward_ir.py` is
   the standing tool for the backward family (selection capture +
   per-config TTGIR/PTX/SASS + load/atomic census into
   `tests/data/ir_dump/`); confirm the lowering before benchmarking a
   config family. When the static census and runtime
   counters disagree, stop reading listings and attribute empirically:
   compile the suspect region in isolation (differential compilation) and
   compute bytes-per-warp-instruction from sector counts ÷ executed-load
   counts — at M13 these settled in minutes what three rounds of
   instruction-counting could not (memo §5.4 items 1–3).
9. **Treat measured equivalences as mechanism evidence.** Two configs from
   different occupancy classes timing identically killed occupancy-alone as
   a lever at M13 (§2 item 1); a vectorized-but-register-heavy config tying
   a scalar-but-occupant config localized the real constraint to the
   layout coupling. When the tuner keeps flip-flopping between two configs,
   that tie is data about the binder, not noise to ignore.
10. **Single-source any parameter two cooperating kernels must agree on.**
    The M13 split's uniform/mixed predicates complement only at one shared
    chunk granularity; letting each kernel autotune its own silently
    dropped contributions (caught by the harness's verify-before-time
    gate — memo §5.4 item 2). The fix shape: one kernel owns the choice,
    the other reads it host-side (`best_config`), and an assert makes the
    compatibility self-enforcing.
11. **Sanitize kernels whose ownership semantics changed** (atomics→stores,
    `torch.empty` outputs, multiple writers with complementary predicates):
    `compute-sanitizer` racecheck/memcheck/initcheck on a small shape —
    initcheck mechanically validates empty-allocation claims, one harness
    run sweeps every autotune config under the sanitizer, and a
    non-divisible-D small shape exercises the masked-tail paths (M11 §5.4;
    M13 §5.6).
12. **Per-iteration discipline**: accept a tuning change only with a
    profiler-confirmed mechanism (method ref §4.3), re-verify correctness
    after every kernel edit (the harness does this per cell), and stop
    optimizing when the remaining gap has a named, recorded bottleneck —
    the memo's residual-bottleneck note is the entry evidence for the next
    milestone, so write it with the same provenance care as a result
    (M11 §6; M13 §9).

## Testing Doctrine

- **Tests must prove what they appear to prove.** The suite was green for an
  entire milestone while only ever exercising the optimized backend's
  fallback policy, because every test shape was "tiny" (design v2 F3). The
  class recurred at M13: the newly-promoted fast path activates only when
  destination runs reach the chunk size, and no test shape came close — the
  suite stayed green while only the scan path executed (M13 memo §8
  item 9). When a code path's activation depends on input properties (shape
  tiers, policy pruning, dtype, run structure), construct the activating
  input, **assert the activation property itself**, then assert outputs —
  `test_optimized_forward_nontiny_shapes` asserts the candidate set;
  `test_backward_uniform_chunk_path_matches_closed_form` asserts the
  sorted-key run structure before checking closed-form gradients. Budget a
  standing question at every promotion: *which inputs make the new path the
  one that executes, and does any test construct them?*
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
- **A quick subset that pins a contract-relevant axis cannot prove that
  axis.** `bench_backward.py --quick` pins `bias=on`; the M13 prototypes'
  `bias_grad`-optionality bug survived four quick gates and surfaced only
  at the full matrix (M13 memo §8 item 2). When iterating with a reduced
  matrix, enumerate which contract axes it holds fixed and run the full
  matrix before any decision that depends on one of them.
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
- A small-shape repro that passes does not clear a property that depends on
  scale-coupled choices (autotune selections, granularities): the M13
  granularity bug passed its small repro because both tuners happened to
  agree there (memo §8 item 3). Repro at the failing scale, or pin the
  coupled choice in the repro.

## Milestone Review

Before a milestone closes, run an adversarial review of the diff **and the
evidence chain**, then fix or record every finding the review cannot refute
— silence is not a verdict. The proven shape (M11, M13): independent
reviewers per dimension — kernel/op correctness, every quoted number
checked against its preserved transcript, doctrine compliance — with
findings verified or refuted independently before acting. Both passes
caught classes the green gates could not see: M11 — a contract-level gap,
an unpreserved flattering figure, a test-gate hole; M13 — a mechanism gate
discharged against an intermediate variant's profile, a pytest-activation
hole for the newly-promoted path, and double-digit number-level drifts in
a memo written the same day. Two standing conclusions: the author cannot
see their own drift, so the transcript audit is not optional; and the
number-vs-transcript dimension pays for itself every time it runs.

Findings about the contract itself go to the maintainer for a ruling
(Evidence section, fourth verdict) rather than being "fixed". When a
promotion proceeds despite a missed pre-registered clause, the protocol is:
check the cited authority's exact wording (the M13 paraphrase was stricter
than the v1 §9 criterion it cited), record the miss as a deviation in the
memo, brief the review to attack specifically that decision, name the
revert path, and request maintainer ratification if the miss touches a
regime or contract of record (M13 memo §8 items 4 and 12). Corrections to
the milestone's own documents are themselves recorded in the memo's
deviations section — the review is part of the evidence, not a cleanup to
hide.

## House Style

Copy the repo's best instance of a pattern instead of inventing a new shape:

| Artifact | Exemplar |
|---|---|
| Gate/sweep script | `scripts/soak_optimized_correctness.py` (docstring contract, availability gate in `main`, `--quick`, summary line, non-zero exit with failure list) |
| Training/integration probe | `scripts/probe_training_smoke.py` (reusable `run_mode`, per-mode gates) |
| Perf decision harness | `scripts/bench_backward.py` (impl registry, per-cell verification before timing, determinism protocol, provenance header, documented synthetic-input contract and its `--quick` limits) |
| Capture-to-disk distribution tooling | `scripts/capture_index_distributions.py` (stats recorded at capture; bundles outside the repo with rerun recipe; overrides recorded in bundle metadata) |
| Runnable analytic model | `scripts/m13_traffic_model.py` (+ its output of record `tests/data/m13_traffic_model_out.txt`): exact stats from actual inputs, expected-vs-measured table per buffer per shape, time floors with named anchors |
| Contract checker + test tables | `tests/test_sparton_kernel.py` (`assert_index_contract`, case tables) |
| Activation-pinning test | `tests/test_sparton_kernel.py::test_backward_uniform_chunk_path_matches_closed_form` (constructed activating input, the activation property asserted, closed-form outputs) |
| Input validation + error template | `src/sparton/_validation.py` |
| Design doc | `docs/sparton_remaining_work_design_v4.md` (entry-gated milestone shape: pre-registered decision rules, will-NOT-touch lists, per-task gates, comparative-table preamble, dated status notes); `docs/sparton_remaining_work_design_v2.md` for the review-born shape (findings + evolution ledger) |
| Milestone memo | `docs/sparton_milestone13_backward_memo.md` (current exemplar: model spine, variant walk with per-step mechanisms, decision ledger, review-as-evidence deviations); `docs/sparton_milestone11_backward_memo.md` (the original analytic-model shape); `docs/sparton_milestone9_production_readiness_memo.md` (finding→fix table, red→green evidence); `docs/sparton_milestone10_promotion_memo.md` (gate-by-gate decision record) |

Rules:

- **Symmetry rule**: the Nth implementation of an existing pattern mirrors
  the structure of the others byte-for-byte where semantics allow. The three
  forward wrappers are intentionally line-for-line parallel; the hybrid F1
  bug existed precisely because hybrid lacked the wrapper the others had.
- **One-seam changes**: new cross-backend behavior is one shared helper
  called at exactly one layer (`autocast_canonicalize` at the top of each
  wrapper; `_bwd_shared_stages` under both backward designs), never N
  divergent copies.
- **No silent fallbacks.** The single sanctioned exception is default-backend
  resolution (one-time `RuntimeWarning`, M10). Explicit selections raise with
  the reason. Do not add a second exception — data-dependent algorithm
  dispatch inside an op would also need a host sync, which is why M11/M13
  shipped fixed kernel families (the M13 split launches both kernels
  unconditionally with complementary device-side predicates — no dispatch).
- Diagnostics go through `logging.getLogger("sparton")` at DEBUG. Never
  `print` from library code; a regression test enforces silent import.
- Comments state constraints the code cannot show (TMA alignment origin, the
  16-bit `dtype_name` assumption, run-boundary composition invariants, the
  complement-granularity requirement) — never mechanics, never
  change-narration. Cross-reference comments about kernel twins go above
  the decorator stack, never inside `@triton.jit`/`@gluon.jit` bodies
  (kernel-body bytes affect compiled-source hashes). Above-decorator
  placement is NOT cache-safe either: Triton's JIT cache key includes the
  function's starting line number (verified on 3.6 at the M12 close — a
  +2-line comment edit re-keyed the kernel's compile and autotune caches),
  so never edit such comments while a measurement campaign is in flight;
  land them before the runs or after the last run of record.
- Asserts that guard a structural invariant carry the invariant in a
  comment and fail loudly with named bounds (`_bwd_shared_stages`' int32
  bounds; the `GRANULE % SUB` complement assert) — an invariant enforced
  only by convention on today's config list is one config edit away from a
  silent-drop bug (M13 memo §8 item 9).
- Retired implementations are either deleted or promoted to an explicit,
  test-pinned reference with a role comment naming its removal condition
  (`legacy_fused_sparton_bwd`: M11 segmented since M13) — never left as
  silent dead code, and never two legacy copies (the baton passes: the
  displaced design becomes the reference, the older one is deleted in the
  same change).
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
  that keeps the saved-tensor set is schema-safe inside the op (M11, M13); a
  changed saved-tensor set requires a new op name.
- The M13 split backward's correctness rests on three stated invariants —
  keep them in lockstep with any edit:
  1. **Complement at one granularity**: the uniform pass deposits exactly
     the single-destination chunks at the shared CHUNK granularity; the
     mixed pass deposits exactly the rest. The mixed kernel is therefore
     NOT autotuned — it runs at the uniform winner's `best_config` CHUNK,
     and `GRANULE % SUB == 0` is host-asserted (independently tuned
     granularities silently drop contributions — M13 memo §5.4 item 2).
  2. **Sorted-prefix bound**: the prep kernel's device-side active count is
     a valid loop bound only because sorted keys put every active entry
     strictly before every sentinel (`b·S + idx < B·S` for live entries).
  3. **Sub-tile composition**: the scan may process a granule in SUB-row
     tiles only because chunk-local partials compose across tile
     boundaries (forced `is_end` at the last lane; a continued run emits
     its own partial with no start-correction) — the same invariant the
     M11 chunk boundaries relied on.
- Numerics contract: forward output follows hidden/logit dtype; `naive`/
  `optimized` accumulate logits in fp32 (more precise than hybrid's
  input-dtype logits — intended); backward gradient buffers are `float32`.
  `hidden_grad` is zero-filled (atomic accumulation; untouched rows stay 0);
  `embed_grad`/`bias_grad` are `torch.empty` — safe only because the embed
  kernel's unconditional exclusive-owner stores cover every element
  (initcheck-validated, M11 memo §5.4; re-validated for the split, M13 memo
  §5.6); keep allocation and coverage in lockstep if either changes.
- Preserve CUDA-only behavior unless explicitly implementing a CPU fallback.
  `__init__.py` intentionally exposes no `SpartonHead` when CUDA is
  unavailable.
- Preserve tensor contracts unless the task explicitly changes them:
  `hidden` is `[B, S, D]`, decoder/embed weights are `[V, D]`, optional bias
  is `[V]`, attention mask is `[B, S]`, and output sparse reps are `[B, V]`.
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
- Do not casually change autotune config lists, tile-size heuristics,
  `torch.library.custom_op` signatures, fake registrations, or autograd setup.
  These affect compilation, graph capture, memory behavior, and gradients.
  In the split backward, the uniform kernel's maskless-load fast branch is
  gated on `hidden_dim % BLOCK_D == 0` at compile time — a new config whose
  BLOCK_D breaks divisibility silently takes the masked path (correct but
  slower); a non-multiple-of-SUB CHUNK is caught by the host assert.
- There are two forward-style reduction helpers: one returns max values plus
  indices for autograd, and one returns only values. Keep their intended
  memory tradeoff clear when editing.
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
  in a command someone can rerun (appendix of rerun commands). Entry-gated
  milestones state their decision rule and its denominators in the doc
  itself, and "closing without kernel work is a recorded outcome, not a
  failure" (proven twice: M12 closed at its gate; M13 passed its gate and
  shipped).
- **Memos record what actually happened**, including deviations from the plan
  and what was deliberately not validated. An honest "known gaps" section is
  mandatory. For performance memos, the spine is the analytic model and an
  expected-vs-measured table per change — a reader must be able to see *why*
  each change worked, not only that it did; for multi-candidate work, record
  the full variant walk with each step's profiler-confirmed mechanism and
  each kill's number (M13 memo §5.4 is the exemplar). State the regime map
  (do_bench vs ncu vs nsys) in the memo header.
- **Every number in a committed document was produced by a command run in
  that session, and quotes its run of record.** If prose was drafted before
  the measurement, correct the draft to the measured value before
  committing. Quote both numbers when two regimes disagree, with the regime
  named. Derived figures name their derivation; estimates are labeled
  estimates; an isolated unpreserved measurement may not stand as a result —
  label it as unpreserved or re-run it (M11 memo §8 item 7; the M13 review
  retired one such figure, memo §8 item 8).
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
  (incl. the §6.1 IR-stage visibility recipe and the §6.2
  layout/lowering-attribution lessons added at M13), failure modes. Read it
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

- The dev environment is the Docker image `nvcr.io/nvidia/pytorch:26.05-py3`
  (NGC PyTorch container; ships the torch 2.12 nightly + Triton 3.6.0 +
  CUDA 13.2 toolchain the venv layers over via `--system-site-packages`).
  The workspace-defect notes below describe this container as mounted on
  this host.
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
- For backward work specifically, the M13 memo carries the current
  mechanism evidence and the validated gather/traffic model (the M11 memo
  carries the segmented design it split); `scripts/bench_backward.py`
  is the harness and `legacy_fused_sparton_bwd` (the M11 segmented design)
  the A/B reference. The Core Kernel Invariants section's three split
  invariants are the correctness surface of any edit there.
- For training behavior, start from `training/model.py` and `training/train.py`;
  avoid importing training modules unless the optional Hugging Face
  dependencies are needed for the task.
- **Performance-track status (both tracks closed, 2026-06-13; no planned
  milestone follows — v4 §3 "Beyond M13"):** the forward is
  tensor-pipe-bound at 92–94% with no scheduling slack (M12 closed without
  kernel work; terminal residual: per-cycle pipe efficiency + L2 pressure
  at the autotuned tile shape — `docs/sparton_milestone12_forward_memo.md`;
  launcher v2 stays deferred, v4 §6). The backward runs the M13 split
  design (real records 1.46–1.60× over M11; the synthetic f=0.10 short-run
  regime carries a recorded 6–16% regression with no real-data
  representative; residual: uniform-pass LTS ≈ 61–67% (config-dependent)
  vs the embed kernel's 82–104%, plus the short-run regimes capped by the
  mixed fraction — `docs/sparton_milestone13_backward_memo.md`). Reopening
  either track starts from the relevant memo's residual-bottleneck note
  and a fresh profile of the artifact as it ships, per the
  Performance-Optimization Loop.

## Training and Hugging Face References

- `SpladeModel` supports `head="torch"`, `head="compiled"`, and
  `head="sparton"` (plus `sparton_backend=` for explicit backend pinning).
  Keep these modes behaviorally aligned when changing model code.
- `train.py` defaults to `FacebookAI/xlm-roberta-base` and
  `nthakur/swim-ir-cross-lingual` with languages `de,es,fr`.
- Full training downloads large Hub assets and can be expensive. Do not run it
  casually as validation; prefer small synthetic or smoke probes
  (`scripts/probe_training_smoke.py`) unless the user explicitly asks for
  a training run. The xlm-roberta-base weights and the swim-ir `de` split are
  cached locally since the M10 tier-2 runs; a steady-state 150-step tier-2
  run costs ~25 s on this host (M13 memo §6/§7) — the cold-start first run
  is much slower.
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
  `/workspace/venvs/sparton/bin/python -m py_compile src/sparton/*.py training/*.py tests/*.py scripts/*.py`
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
  `scripts/soak_optimized_correctness.py` (use `--quick` while iterating,
  the full sweep as the gate).
- For changes that touch autograd, AMP, or the training path, rerun
  `scripts/probe_training_smoke.py`.
- For backward-kernel changes, rerun `scripts/bench_backward.py` with
  `--impls current,legacy` over {uniform, zipf} and the captured-real
  bundles in `tests/data/bundles/` (regenerate via
  `scripts/capture_index_distributions.py` if absent — it reuses existing
  files; regenerated bundles are *different records*, so cross-session
  comparisons to old transcripts break and baselines are re-measured
  rather than compared); uniform-only evidence is
  never sufficient for a backward change (M11 rule of record).
- For new or changed kernels whose ownership semantics differ from their
  predecessor (atomics→plain stores, `torch.empty` outputs, complementary
  multi-kernel writers), run `compute-sanitizer`
  racecheck/memcheck/initcheck on a small shape through the harness
  (commands in the M11 memo §5.4; the M13 §5.6 run used
  `B4 S33 D64 V2048`, which also exercises the non-divisible-D mask path).
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
  stores) and `hidden_grad` atomics are reduced to ~chunk-partial sums
  (the M13 split preserves the atomic structure), but their accumulation
  order still varies (measured per-call relative spread: gradient-norm
  ≤ 1.2e-7, element-sensitive loss-proxy ≤ 4.2e-6 — M11 memo §7, M13 memo
  §5.5). At training scale the chaotic early regime amplifies this to a
  measured same-config 150-step loss spread of **16–38%** depending on
  backend and statistic (3 seed-matched repeats per backend, M13 memo §6
  — supersedes the M10-era "~20%" estimate; per-call determinism
  improvements do not shrink it). Establish same-config noise bands
  before reading meaning into cross-config training differences; do not
  promise bitwise-reproducible training.
- The M13 split backward trades the synthetic `f = 0.10` short-run regime
  (−6…−16%, ~45 µs/call; no real capture exhibits it) for 46–60% gains on
  captured-real records. The trade is recorded **pending maintainer
  ratification** (M13 memo §8 item 12); if rejected, the revert is one
  change — re-wire `fused_sparton_bwd_op` to `segmented_sparton_bwd` and
  re-run the M13 §5.6 gate block.
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
