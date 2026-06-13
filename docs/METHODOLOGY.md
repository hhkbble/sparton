# Sparton methodology: working method + kernel-optimization technique

The reusable working method that produced the M2→M13 arc, merged with the
kernel-optimization technique reference it relied on. **Repo facts, invariants,
environment, validation commands, and task routing** live in
[`../AGENTS.md`](../AGENTS.md) (loaded into every session); the **built system**
is in [ARCHITECTURE.md](ARCHITECTURE.md); the **development history and evidence
of record** in [DEVELOPMENT.md](DEVELOPMENT.md). This document is the method:
*how* to work and *how* to optimize Triton/Gluon kernels. Rules cite the
milestone that taught them so you can audit the evidence in DEVELOPMENT.md rather
than trust the rule.

How to use it: the **Process layer** (§A) is the default discipline for any
non-trivial task; the **Performance-Optimization Loop** (§A.3) and the
**Technique layer** (§B) are its specialization for Triton/Gluon performance work
— read §B before any kernel performance work. The **worked examples** (§C) are
the case studies that prove each method.

---

# §A — Process layer

## §A.1 The Operating Loop

The cycle that produced M9–M13; follow it for any non-trivial task.

1. **Orient** (read README, CHANGELOG, ARCHITECTURE.md, the active milestone
   section), then **probe before designing**: when the task starts from a
   suspicion or a review finding, write a small disposable probe that
   demonstrates the behavior before writing the plan. Never put a claim in a plan
   or doc that you have not executed (the M9 review provenance is the model:
   every finding has a rerun recipe — [DEVELOPMENT.md](DEVELOPMENT.md) M9).
2. **Plan as ordered, independently-landable tasks**, each naming its files, its
   behavior change, the tests it adds, and a runnable gate with an expected
   result. State scope constraints up front as "will NOT touch" lists (M9: no
   kernel-body changes, no autotune-config changes, no schema changes, no new
   dependencies) — they make reviews tractable and prevent drive-by churn.
3. **Bug fixes are red→green**: write the regression test first, watch it fail
   against unmodified code, capture the failing output verbatim for the memo,
   then fix. Never leave a commit boundary red — the red evidence lives in the
   memo, not in history ([DEVELOPMENT.md](DEVELOPMENT.md) M9 F1).
4. **Run the gate after every task**, not only at the end. A gate is a command
   plus a number (test count, ms window, ratio, tolerance) — never an adjective.
5. **Document at completion**: milestone memo with evidence, CHANGELOG entries,
   status note in the design doc, and corrections to any prose drafted before the
   measurements existed.
6. **Commit once per milestone/task** with a concise subject and a detailed
   what/why/validated-how body. Commit pre-existing unrelated worktree changes
   separately first so each commit is attributable.

### Session conduct (how a fast session is actually run)

The M13 close was a single session covering entry evidence, a kernel promotion,
and the review; these are the habits that made it fast without cutting evidence:

- **Pipeline GPU time against authoring time.** GPU jobs run in the background
  (serially — see §A.2) while docs, tests, and the memo are written; the only
  forbidden overlap is editing `src/` or `scripts/` while a background or
  *queued* process will import them — a chained second run imports the edited
  tree and silently corrupts the A/B (M11). Docs and tests are always safe to
  edit. Sequence kernel-adjacent edits into campaign windows: land them before a
  measurement campaign starts or after its last run of record (the cache-key
  rule, §A.6).
- **Write the memo incrementally, measured content only.** Create the skeleton
  early; fill each section the moment its numbers land, citing the transcript in
  the same edit; leave unmeasured sections as explicit stubs. Prose written ahead
  of results is the error class the Documentation System forbids — at M13 the one
  sentence drafted ahead of T1's outcome had to be retracted within minutes.
- **Measure the unit cost of an expensive planned step once before scheduling
  around it.** The M13 plan budgeted ~20 minutes per tier-2 training run from a
  cold-start estimate; one measured run showed 24 s steady-state, which changed
  the campaign's shape. One probe beats a schedule built on an assumed cost.
- **Deposit as you go.** Every decision-carrying run is teed to a transcript at
  launch time, not reconstructed later; disposable probes, analysis scripts,
  profiles, and transcripts live under the gitignored `tests/data/runs/<label>/`
  (convention in `tests/data/README.md`) and the memo cites them by path. **Never
  deposit to `/tmp`**: background-task stdout buffers land there transiently, the
  mount is `noexec` and gets wiped — nothing in `/tmp` may be cited or relied on
  (the M13 review agents' scratch went there and none of it survived as evidence;
  the recorded findings did because they were in the memo). `scripts/` is only for
  validated, reusable tooling — a probe that proves durable is promoted there with
  repo-relative paths and run end-to-end at promotion (the M13 traffic model and
  IR-dump tool are the precedents). Documents must never rely on artifacts outside
  the repository: anything load-bearing either lives in-repo or has an in-repo
  regeneration script.
- Copy the repo's exemplar for the artifact you are creating (§A.6) instead of
  designing from scratch; the shapes are proven.
- Give long sweeps a `--quick` subset for development; run the full sweep only as
  the gate — and know what the subset cannot prove (§A.4: pinned axes).
- A pytest run launched immediately after a heavy background GPU job can fail
  transiently (subprocess-based tests); rerun and classify before debugging
  ([DEVELOPMENT.md](DEVELOPMENT.md) M11 §8).
- Autotune and Inductor caches are persistent (`cache_results=True`,
  `TORCHINDUCTOR_CACHE_DIR`): first runs pay compile/tune cost, reruns are cheap.
  Judge timings accordingly and don't fear re-running gates.
- Historical results live in memos — cite them instead of re-running history, but
  re-measure anything that gates the current decision.

## §A.2 Evidence, Measurement, and Gates

- Use the hardened environment prefix (see [`../AGENTS.md`](../AGENTS.md)
  Environment) for every CUDA/Triton command; run benchmarks serially.
- **Judge benchmarks on the second consecutive run** (warm caches). Recorded
  baselines of record (dev shape `B=32,S=128,D=768,V=30522` fp16) are in
  [DEVELOPMENT.md](DEVELOPMENT.md) M12/M13 tables; rerun the row yourself before
  using it as a gate.
- **Measure A-vs-A before judging A-vs-B**: repeat the same configuration and use
  that spread as the noise band. Autotune config-selection jitter between
  processes is part of the band (±5% on borderline cells — M11 §5.2); a
  cross-impl difference inside the same-impl band is noise, not signal. A single
  wild cell inside an otherwise-consistent run is classified before any cell of
  record is taken from it — a third consecutive run plus an interference check
  separates transient host noise from signal (M13 §3: two contaminated cells,
  classified, no record taken from that run).
- **Keep measurement regimes separate**: `triton.testing.do_bench` (L2-flushed)
  is the latency of record; `ncu` serializes and flushes, so its durations are
  inflated — use it for structure, counters, and ratios; `nsys` for kernel
  inventory and single-regime per-call shares; sanitizer-run timings are
  meaningless. Never compare numbers across regimes. When an op-level do_bench
  number must be reconciled with a kernel-level sum, *measure* the bridge (M13
  §5.2: the do_bench-vs-GPU-sum gap is itself a quoted number), never assume it
  away.
- **Deposit a transcript for every number that carries a decision** (tee
  ncu/bench output to a file under `tests/data/runs/<label>/`; generated bundles
  live in `tests/data/bundles/`). The largest finding class in the M11 review was
  decision-carrying numbers with no preserved log; an unpreserved spot-run may
  not be quoted as a result — label it or re-run it (M11 §8 item 7).
- **A profile is evidence about the exact artifact profiled.** If the code, its
  config family, or its selected config changes after the profile — even by an
  edit that "shouldn't matter" — mechanism gates must be re-discharged on the
  shipped form. The M13 review caught the E5 gate discharged against an
  intermediate variant's profile; the production re-profile landed at a different
  config and a different top unit (M13 §8 item 7; the same failure class as M11
  §8 item 7).
- **"Pre-registered" means committed before the measurement.** The claim is
  auditable only by the commit boundary; a rule fixed in-session before its runs
  is honest but is written as exactly that, not as pre-registration (M13 §8
  item 11). Pre-register decision rules and exit numbers in a commit (or in the
  already-committed design doc) before candidate work starts; apply them as
  written; a missed clause is recorded as a deviation, never reinterpreted
  silently.
- **Classify a failure before fixing it.** Four verdicts are possible: a real
  defect (non-contiguous `embed_grad`, M9 F1); expected behavior that the gate
  mis-asserts (GradScaler-skipped early fp16 steps are normal AMP scale
  calibration — the M10 smoke's gate was rewritten, not the code); an
  out-of-contract input (near-tie index mismatches are allowed by the index
  contract); or behavior that is in-contract but the contract itself needs a
  maintainer ruling — escalate, then record the ruling in code comments and docs
  (the M11 mask-factor finding became a contract note, not a code fix; the M13
  synthetic-sparse trade is recorded pending ratification, M13 §8 item 12).
  Fixing before classifying produces wrong fixes.
- Perf claims need a mechanism, not just a delta: when a result surprises (cuBLAS
  unusually slow on a stress shape), say why or flag it as unexplained in the memo
  rather than letting the ratio stand alone.
- A gate that can be expressed as a script should be one (see
  `scripts/soak_optimized_correctness.py`): availability check, summary line,
  non-zero exit listing every failure. Scripts that write files default their
  outputs under `tests/data/` (gitignored; CLI-overridable) — never `/tmp` and
  never absolute paths outside the repo (`capture_index_distributions.py` →
  `tests/data/bundles/`, `dump_backward_ir.py` → `tests/data/ir_dump/` are the
  precedents).

## §A.3 The Performance-Optimization Loop

The extension of the Operating Loop for performance work, proven across M11
(backward restructure), M12 (forward no-go), and M13 (backward split).
Kernel-level techniques live in §B (read it before any Triton/Gluon performance
work; §6.1–§6.2 carry the repo-added IR and layout-attribution recipes); this
section is the repo-proven process around them.

1. **Name the binding resource before designing** (§3.3): profile the current
   implementation (ncu, direct op calls on the main thread) and identify the unit
   that binds. Then ask the altitude question — what class of change can move that
   binder? The arc has a worked example of each answer (§C): M11's binder was an
   algorithm-level *operation count* (L2 reduction sectors — only changing which
   operations exist helped; TMA and Gluon could not); M12's presumed SM-side
   scheduling gap was *disproved* by the first profile of the production kernel
   (tensor pipe already saturated — the milestone closed without kernel work,
   which is a valid outcome); M13's binder was a *lowering coupling* (the gather's
   vector width was anchored to the scan tile's layout — neither tuning nor pure
   occupancy work could move it; a structural kernel split could). Beware
   inherited priors: M12's "SM-side" prior came from a sibling benchmark kernel
   and did not transfer to production; profile the artifact you intend to change.
2. **Write the analytic traffic model first, as a runnable script.** Per-buffer
   formulas in the problem dims and block params, validated against counters
   before any candidate is built (M11 §3 matched measured sectors to four
   significant figures; M13 §4 to ≤1% on the decision counters across four
   shapes). Compute distribution statistics (run counts, mixed fractions, active
   fractions) from the *actual inputs* — same seeds, same bundle records — never
   estimated. Deposit the script with its output (the M13 model is
   `scripts/m13_traffic_model.py`, output of record
   `tests/data/m13_traffic_model_out.txt`) so the model re-runs against future
   counters. The model then prices each candidate's ceiling before you invest in
   it, arbitrates surprises, and becomes the memo's expected-vs-measured spine.
   Pre-register the validation bar; a residual above it is recorded with a named
   mechanism (M13: the +30% query-shape L1 residual was intra-warp sector
   coalescing under index collisions), never absorbed.
3. **Benchmark on realistic data distributions.** Synthetic-uniform inputs missed
   both decisive properties of real batches (dense scores; 20–46% of vocab entries
   sharing one argmax position — M11 §4). Capture real distributions once
   (`capture_index_distributions.py` bundles, with distribution stats recorded at
   capture time), replay them in the harness (`bench_backward.py`), and keep
   synthetic sources for regimes real data does not cover. The data may also
   overturn design assumptions — measure the distribution before trusting the
   plan's picture of it. Know which regimes are synthetic-only and say so when a
   result depends on one (the `f ≪ 1` regime has no real capture; the λ-probe
   brackets a dense→collapsed transition with nothing usable inside — M13 §7).
4. **Prototype behind a registry; production stays untouched.** Candidates live in
   `scripts/` with the production op's exact signature — the *full* signature
   contract including output optionality (the M13 prototypes' `bias_grad`
   placeholder-vs-None mismatch survived every quick gate, M13 §8 item 2) —
   selected by `--impls`; the harness numerically verifies every cell against
   production *before* timing it, so a broken prototype cannot produce a timing
   row. Wire the winner into the op only at promotion; retain the loser of record
   as an explicit, test-pinned legacy reference. Commit the prototypes at the
   decision point so the matrix of record stays reproducible from history (the
   M11 practice; M13 skipped it and the review recorded the gap).
5. **Pre-register numeric decision rules and exit numbers** (committed before
   measurement), including an early-stop for timeboxed alternatives (the B2b rule,
   M11 §5.1; the M13 variant budget), so skipping work is defensible — and record
   the decision-criteria walk in the memo even for the candidates you killed: each
   kill is justified by a number, not a vibe (M13 §5.1: three of four candidates
   killed by arithmetic before any code existed).
6. **Time at the op level for the verdict; profile at the kernel level for
   structure.** The op-level closure allocates exactly what production allocates —
   buffer fills, sorts, and host passes count. The host side is in-bounds for
   "kernel" optimization: M11's unlock included `torch.sort` plus a fused prep
   kernel; fusing seven elementwise launches into one was worth more than any
   kernel tweak at small shapes (M11 §5.2). Use one nsys pass for single-regime
   per-call shares (sort and fill kernels never match a Triton `-k` regex —
   without it the op-vs-kernel gap gets mis-attributed; M13 §2.1).
7. **Audit the autotune key — then price what you find.** A performance-relevant
   argument missing from the key silently reuses a wrong config (`seq_len`, M11
   §5.2: ~7% on documents). The same audit at M13 found the embed kernel's
   identical omission costs 0.69% — inside the noise band, classified immaterial,
   no fix (M13 §2.2). The audit is mandatory; the fix is not — the price decides.
   Note Triton ≥3.6 keys include argument dtypes automatically. Selections are
   read from the autotuner's cache/`best_config` host-side (cache-hit selections
   print nothing under `TRITON_PRINT_AUTOTUNING=1` — M12 lesson).
8. **Read the IR when the question is about lowering — and know the limits of
   static reading** (§6.1–§6.2): `CompiledKernel.asm` / `nvdisasm` answer
   instruction-form questions (vector widths, scan lowering, spills) at zero GPU
   cost — `scripts/dump_backward_ir.py` is the standing tool for the backward
   family; confirm the lowering before benchmarking a config family. When the
   static census and runtime counters disagree, stop reading listings and
   attribute empirically: compile the suspect region in isolation (differential
   compilation) and compute bytes-per-warp-instruction from sector counts ÷
   executed-load counts — at M13 these settled in minutes what three rounds of
   instruction-counting could not (M13 §5.4).
9. **Treat measured equivalences as mechanism evidence.** Two configs from
   different occupancy classes timing identically killed occupancy-alone as a
   lever at M13 (§2 item 1); a vectorized-but-register-heavy config tying a
   scalar-but-occupant config localized the real constraint to the layout
   coupling. When the tuner keeps flip-flopping between two configs, that tie is
   data about the binder, not noise to ignore.
10. **Single-source any parameter two cooperating kernels must agree on.** The M13
    split's uniform/mixed predicates complement only at one shared chunk
    granularity; letting each kernel autotune its own silently dropped
    contributions (caught by the harness's verify-before-time gate — M13 §5.4
    item 2). The fix shape: one kernel owns the choice, the other reads it
    host-side (`best_config`), and an assert makes the compatibility
    self-enforcing.
11. **Sanitize kernels whose ownership semantics changed** (atomics→stores,
    `torch.empty` outputs, multiple writers with complementary predicates):
    `compute-sanitizer` racecheck/memcheck/initcheck on a small shape — initcheck
    mechanically validates empty-allocation claims, one harness run sweeps every
    autotune config under the sanitizer, and a non-divisible-D small shape
    exercises the masked-tail paths (M11 §5.4; M13 §5.6).
12. **Per-iteration discipline**: accept a tuning change only with a
    profiler-confirmed mechanism (§4.3), re-verify correctness after every kernel
    edit (the harness does this per cell), and stop optimizing when the remaining
    gap has a named, recorded bottleneck — the memo's residual-bottleneck note is
    the entry evidence for the next milestone, so write it with the same
    provenance care as a result (M11 §6; M13 §9).

## §A.4 Testing Doctrine

- **Tests must prove what they appear to prove.** The suite was green for an
  entire milestone while only ever exercising the optimized backend's fallback
  policy, because every test shape was "tiny" (M9 F3). The class recurred at M13:
  the newly-promoted fast path activates only when destination runs reach the
  chunk size, and no test shape came close — the suite stayed green while only
  the scan path executed (M13 §8 item 9). When a code path's activation depends on
  input properties (shape tiers, policy pruning, dtype, run structure), construct
  the activating input, **assert the activation property itself**, then assert
  outputs — `test_optimized_forward_nontiny_shapes` asserts the candidate set;
  `test_backward_uniform_chunk_path_matches_closed_form` asserts the sorted-key
  run structure before checking closed-form gradients. Budget a standing question
  at every promotion: *which inputs make the new path the one that executes, and
  does any test construct them?*
- **Conditional expectations must pin what they condition on.** A closed-form
  expectation computed from the kernel's own saved outputs is the right shape for
  backward tests at random non-tiny shapes (autograd through the reference
  re-litigates contract-legal near-tie choices), but it masks forward bugs unless
  the same test pins those saved outputs — reference scores within tolerance plus
  the index contract (`test_fused_backward_nontiny_shapes`; M11 §8 item 6).
- **Match assertion strength to input class**: deterministic constructed cases
  (intentional ties, masked winners, dyadic-rational patterns) assert exact
  equality — they pin tie policy. Random-input cases assert the contract
  (`assert_index_contract`), because backends with different accumulation
  precision legitimately disagree at near-ties. Converting one into the other in
  either direction is a bug.
- Every validation rule has a `pytest.raises(..., match=...)` test with a stable
  message substring. Error-message templates (see `_validation.py`:
  `sparton {backend} forward: {arg}{rule}; got {actual}`) are part of the API —
  tests depend on them; change them deliberately.
- Single-source gate logic: when a gate script and a test overlap, the test
  imports the script's function (`test_training_parity_smoke_autocast` reuses
  `probe_training_smoke.run_mode`; the backward harness's synthetic-input
  contract is pinned by importing `make_synthetic_case`) instead of duplicating
  it.
- Mark expensive coverage `@pytest.mark.slow` — including tests that autotune
  kernel families at non-tiny shapes; the default `pytest -q` runs everything,
  `-m "not slow"` is the documented quick loop. Don't let the quick loop lose
  meaning by marking cheap tests slow.
- **A quick subset that pins a contract-relevant axis cannot prove that axis.**
  `bench_backward.py --quick` pins `bias=on`; the M13 prototypes' `bias_grad`-
  optionality bug survived four quick gates and surfaced only at the full matrix
  (M13 §8 item 2). When iterating with a reduced matrix, enumerate which contract
  axes it holds fixed and run the full matrix before any decision that depends on
  one of them.
- Subprocess tests use absolute paths (`_REPO_ROOT`, `_SRC_PATH` in
  `tests/test_sparton_kernel.py`), never CWD-relative ones. The suite must pass
  under `python -m pytest`, the venv `pytest` console script, and
  `pytest /workspace/sparton/tests` from a foreign working directory — run all
  three after touching test infrastructure.
- Parametrized cases live in named tables with explicit ids (`FORWARD_CASES`,
  `BACKWARD_CASES`, `VALIDATION_ERROR_CASES`, `NONTINY_FORWARD_CASES`);
  `strict_parametrization_ids` is enabled.
- Gate availability with fixtures/helpers (`_forward_for_backend`,
  `_optimized_gluon_availability`), not device-name checks — capability, not
  hardware identity.
- Tensors built for gradient tests must be autograd leaves: an arithmetic result
  like `-torch.ones(..., requires_grad=True)` is a non-leaf whose `.grad` stays
  `None`; use `torch.full`/`torch.ones` directly (M11 red→green note).
- A small-shape repro that passes does not clear a property that depends on
  scale-coupled choices (autotune selections, granularities): the M13 granularity
  bug passed its small repro because both tuners happened to agree there (M13 §8
  item 3). Repro at the failing scale, or pin the coupled choice in the repro.

## §A.5 Milestone Review

Before a milestone closes, run an adversarial review of the diff **and the
evidence chain**, then fix or record every finding the review cannot refute —
silence is not a verdict. The proven shape (M11, M13): independent reviewers per
dimension — kernel/op correctness, every quoted number checked against its
preserved transcript, doctrine compliance — with findings verified or refuted
independently before acting. Both passes caught classes the green gates could not
see: M11 — a contract-level gap, an unpreserved flattering figure, a test-gate
hole; M13 — a mechanism gate discharged against an intermediate variant's profile,
a pytest-activation hole for the newly-promoted path, and double-digit
number-level drifts in a memo written the same day. Two standing conclusions: the
author cannot see their own drift, so the transcript audit is not optional; and
the number-vs-transcript dimension pays for itself every time it runs. (Default to
inline `Agent` reviewers per dimension, not workflow orchestration, unless asked.)

Findings about the contract itself go to the maintainer for a ruling (§A.2,
fourth verdict) rather than being "fixed". When a promotion proceeds despite a
missed pre-registered clause, the protocol is: check the cited authority's exact
wording (the M13 paraphrase was stricter than the regression-without-gain
criterion it cited — historically design v1 §9, quoted in
[DEVELOPMENT.md](DEVELOPMENT.md) M13 §5.5), record the miss as a deviation in the
memo, brief the review to attack
specifically that decision, name the revert path, and request maintainer
ratification if the miss touches a regime or contract of record (M13 §8 items 4
and 12). Corrections to the milestone's own documents are themselves recorded in
the memo's deviations section — the review is part of the evidence, not a cleanup
to hide.

## §A.6 House Style

Copy the repo's best instance of a pattern instead of inventing a new shape:

| Artifact | Exemplar |
|---|---|
| Gate/sweep script | `scripts/soak_optimized_correctness.py` (docstring contract, availability gate in `main`, `--quick`, summary line, non-zero exit with failure list) |
| Training/integration probe | `scripts/probe_training_smoke.py` (reusable `run_mode`, per-mode gates) |
| Perf decision harness | `scripts/bench_backward.py` (impl registry, per-cell verification before timing, determinism protocol, provenance header, documented synthetic-input contract and its `--quick` limits) |
| Capture-to-disk distribution tooling | `scripts/capture_index_distributions.py` (stats recorded at capture; bundles under `tests/data/bundles/` with rerun recipe; overrides recorded in bundle metadata) |
| Runnable analytic model | `scripts/m13_traffic_model.py` (+ its output of record `tests/data/m13_traffic_model_out.txt`): exact stats from actual inputs, expected-vs-measured table per buffer per shape, time floors with named anchors |
| Contract checker + test tables | `tests/test_sparton_kernel.py` (`assert_index_contract`, case tables) |
| Activation-pinning test | `tests/test_sparton_kernel.py::test_backward_uniform_chunk_path_matches_closed_form` (constructed activating input, the activation property asserted, closed-form outputs) |
| Input validation + error template | `src/sparton/_validation.py` |
| Architecture reference | [ARCHITECTURE.md](ARCHITECTURE.md) (built-system shape: platform facts, contracts, layering rules, kernel designs of record, measured state, deferred work) |
| Development memo | [DEVELOPMENT.md](DEVELOPMENT.md) — M13 is the current exemplar (model spine, variant walk with per-step mechanisms, decision ledger, review-as-evidence deviations); M11 the analytic-model shape; M9 the finding→fix table + red→green; M10 the gate-by-gate decision record |

Rules:

- **Symmetry rule**: the Nth implementation of an existing pattern mirrors the
  structure of the others byte-for-byte where semantics allow. The three forward
  wrappers are intentionally line-for-line parallel; the hybrid F1 bug existed
  precisely because hybrid lacked the wrapper the others had.
- **One-seam changes**: new cross-backend behavior is one shared helper called at
  exactly one layer (`autocast_canonicalize` at the top of each wrapper;
  `_bwd_shared_stages` under both backward designs), never N divergent copies.
- **No silent fallbacks.** The single sanctioned exception is default-backend
  resolution (one-time `RuntimeWarning`, M10). Explicit selections raise with the
  reason. Do not add a second exception — data-dependent algorithm dispatch inside
  an op would also need a host sync, which is why M11/M13 shipped fixed kernel
  families (the M13 split launches both kernels unconditionally with complementary
  device-side predicates — no dispatch).
- Diagnostics go through `logging.getLogger("sparton")` at DEBUG. Never `print`
  from library code; a regression test enforces silent import.
- Comments state constraints the code cannot show (TMA alignment origin, the
  16-bit `dtype_name` assumption, run-boundary composition invariants, the
  complement-granularity requirement) — never mechanics, never change-narration.
  Cross-reference comments about kernel twins go above the decorator stack, never
  inside `@triton.jit`/`@gluon.jit` bodies (kernel-body bytes affect compiled-
  source hashes). Above-decorator placement is NOT cache-safe either: Triton's JIT
  cache key includes the function's starting line number (verified on 3.6 at the
  M12 close — a +2-line comment edit re-keyed the kernel's compile and autotune
  caches), so never edit such comments while a measurement campaign is in flight;
  land them before the runs or after the last run of record.
- Asserts that guard a structural invariant carry the invariant in a comment and
  fail loudly with named bounds (`_bwd_shared_stages`' int32 bounds; the
  `GRANULE % SUB` complement assert) — an invariant enforced only by convention on
  today's config list is one config edit away from a silent-drop bug (M13 §8
  item 9).
- Retired implementations are either deleted or promoted to an explicit,
  test-pinned reference with a role comment naming its removal condition
  (`legacy_fused_sparton_bwd`: the M11 segmented design since M13) — never left as
  silent dead code, and never two legacy copies (the baton passes: the displaced
  design becomes the reference, the older one is deleted in the same change).
- Capability dispatch for fatal-failure APIs (Gluon MMA families abort the process
  at LLVM selection) uses static whitelists probed in subprocesses
  (`probe_mma_matrix.py`), never try/except fallback.
- Python: match the file's existing style; type annotations on new public
  functions; keyword-only for new optional constructor args; English comments.

## §A.7 Documentation authoring discipline

The repo's documents and their roles (README, AGENTS.md, ARCHITECTURE.md,
DEVELOPMENT.md, METHODOLOGY.md, CHANGELOG.md) are listed in
[`../AGENTS.md`](../AGENTS.md); keeping each in its role is part of every task's
definition of done. The authoring discipline:

- **Architecture/reference docs must let a cold agent execute without
  re-deriving**: per-task file lists, behavior specs, tests to add, runnable gates
  with expected results, and an explicit rejected/deferred section with rationale
  (so good ideas aren't re-litigated and bad ones aren't re-tried). Ground every
  claim in a command someone can rerun (an appendix of rerun commands).
  Entry-gated milestones state their decision rule and its denominators in the
  doc itself, and "closing without kernel work is a recorded outcome, not a
  failure" (proven twice: M12 closed at its gate; M13 passed its gate and
  shipped).
- **Memos record what actually happened**, including deviations from the plan and
  what was deliberately not validated. An honest "known gaps" section is
  mandatory. For performance memos, the spine is the analytic model and an
  expected-vs-measured table per change — a reader must be able to see *why* each
  change worked, not only that it did; for multi-candidate work, record the full
  variant walk with each step's profiler-confirmed mechanism and each kill's
  number (M13 §5.4 is the exemplar). State the regime map (do_bench vs ncu vs
  nsys) in the memo header.
- **Every number in a committed document was produced by a command run in that
  session, and quotes its run of record.** If prose was drafted before the
  measurement, correct the draft to the measured value before committing. Quote
  both numbers when two regimes disagree, with the regime named. Derived figures
  name their derivation; estimates are labeled estimates; an isolated unpreserved
  measurement may not stand as a result — label it as unpreserved or re-run it
  (M11 §8 item 7; the M13 review retired one such figure, §8 item 8).
- Documents form a supersession chain, never silent replacement: a superseded doc
  gets a status line naming its successor and what it remains authoritative for.
- Commits: concise subject + detailed body covering what changed, why, and how it
  was validated (with the actual gate results).

# §B — Technique layer: kernel optimization on Triton and Gluon

Scope: Triton language/compiler kernels and Triton's experimental Gluon kernel
language (not NVIDIA Triton Inference Server). Triton and Gluon share a tile-based
GPU programming model and Python/JIT workflow but sit at different abstraction
levels: Triton hides layout, memory-allocation, data-movement, and asynchrony
behind the compiler; Gluon exposes those details, useful when Triton is close but
the bottleneck requires explicit control of register/shared-memory layouts, async
pipelines, tensor-core pipelines, or hardware-specific scheduling. The
optimization work falls into a small loop: (1) build a correct baseline;
(2) benchmark with stable shapes/dtypes/strides/warmups and fixed versions;
(3) classify the bottleneck; (4) tune the high-impact knobs first (tile sizes,
`num_warps`, `num_stages`, `num_ctas`, register caps, program ordering,
accumulation dtype, vectorization/alignment, masks, fusion boundaries); (5) use
profilers to validate the *hypothesis*, not just the timing delta; (6) only then
add lower-level techniques.

## §2 Triton vs. Gluon: when to use which

| Use Triton when... | Use Gluon when... |
|---|---|
| You need a custom operator quickly and compiler-generated layouts are good enough. | The critical bottleneck is layout, shared memory, async copy, tensor memory, warp specialization, or architecture-specific scheduling. |
| The kernel is mostly elementwise, reduction-like, small GEMM-like, softmax-like, or fusion-heavy. | You need fine control over register/thread/warp/CTA distribution or bank-conflict behavior. |
| Portability and code size matter more than the last few percent. | You are chasing near-peak on a fixed target GPU generation. |
| `triton.autotune` over tile/meta-parameters is enough to expose the frontier. | Triton's abstraction hides the mechanism you need to change. |

Gluon shares Triton's compiler stack and tile-based SPMD model but exposes tile
layouts, memory allocation, data movement, and asynchrony, at the cost of
explicit responsibility for more hardware details.

## §3 Baseline optimization methodology

### §3.1 Define the performance contract

Capture before tuning: shapes, dtypes, strides, alignment, batch sizes, and
realistic input distributions; target hardware (Ampere/Hopper/Blackwell, AMD
CDNA/gfx, clocking, shared-memory size, expected Tensor Cores/MFMA/WMMA);
correctness tolerances by dtype and operation order (keep a simple reference +
randomized edge-case tests); the throughput metric (GB/s for memory-bound,
TFLOP/s or tensor-core utilization for GEMM-like, latency for small kernels,
end-to-end model impact for fused kernels).

### §3.2 Use a reproducible benchmark harness

Warm up JIT compilation separately from timing; synchronize around timing; use
enough repetitions; pin shapes and config keys when autotuning; log Triton
version, driver/runtime versions, GPU model, clock/power mode, environment
variables, and the selected autotune config.

### §3.3 Classify the bottleneck first

| Symptom | Likely bottleneck | First actions |
|---|---|---|
| Low arithmetic intensity, high DRAM traffic | Memory-bound | Fuse ops, reduce stores/loads, improve coalescing, cache-friendly program order, stage reusable data. |
| Low tensor-core/MFMA active cycles | Compute pipeline underfed | Increase tile reuse, tune `BLOCK_M/N/K`, async copy/TMA, improve layouts, pipeline prologue/steady-state/epilogue. |
| Low occupancy from too many registers | Register pressure | Reduce tile size, split accumulators, `maxnreg` carefully, reduce live ranges, reconsider unrolling/pipelining. |
| Many shared-memory conflicts | Shared-memory layout | Swizzled/shared layouts, adjust vector width, alignment, layout mapping. |
| High synchronization or barrier stalls | Pipeline/scheduling | Reduce barriers, finer staging, warp specialization or persistent scheduling. |
| Wide variance across blocks | Load imbalance | Persistent kernels, grouped scheduling, CLC on Blackwell, split work more evenly. |

## §4 Common Triton optimization techniques

### §4.1 Tile shape and program mapping

The most important knobs are the meta-parameters defining tile shape (`BLOCK_M`,
`BLOCK_N`, `BLOCK_K`, vector width, reduction block size, group size for program
ordering). Practical rules: start from known-good tile families for the operation
class, then tune around them; for GEMM-like kernels tune `BLOCK_M/N/K` jointly
with `num_warps`/`num_stages`; increase tile size only while occupancy, register
pressure, shared-memory footprint, and masking overhead stay acceptable; prefer
program-ordering that reuses one operand in L2 across neighboring CTAs; for
reductions tune the reduction tile separately from the output tile.

### §4.2 Autotuning

`triton.autotune` evaluates multiple `triton.Config` entries and re-evaluates when
selected key arguments change; `triton.Config` controls meta-arguments plus
`num_warps`, `num_stages`, `num_ctas`, `maxnreg`. Use `key=[...]` capturing
shape/dtype/stride changes that affect performance; keep the search space small
enough for CI or precompute per architecture; for kernels that write outputs
during tuning use reset/restore hooks or scratch outputs; log selected configs
with `TRITON_PRINT_AUTOTUNING=1`; use early pruning or performance models when the
Cartesian product grows too large.

### §4.3 Occupancy, register pressure, and software pipelining

`num_warps` changes warps per Triton program; `num_stages` controls compiler
software pipelining (especially matmul-style loops on SM80+); `maxnreg` caps
registers but can induce spills (validate with a profiler). Loop: increase tile
reuse until register/shared-memory pressure hurts occupancy; sweep `num_warps`
per tile family (larger is not automatically better); sweep `num_stages` (more
stages hide latency but increase shared memory and live ranges); check generated
resource usage and profiler stalls before accepting a timing improvement.

### §4.4 Memory access and fusion

Memory-bound: make loads/stores contiguous/coalesced; minimize rereads and
materialized temporaries by fusing elementwise epilogues; use masks only where
needed; align pointer offsets and vectorization; avoid unnecessary dtype
conversions in the hot path. Compute-bound: keep accumulators in registers and
write once; use tensor-core-friendly tile dimensions/dtypes; fuse epilogues only
when the added register pressure does not reduce tensor-core utilization.

### §4.5 Persistent kernels and cache-aware grouped scheduling

Persistent kernels keep a bounded set of CTAs resident and iterate over multiple
work tiles. Use persistence when the number of tiles is large and scheduling
overhead or load imbalance matters, the kernel benefits from cache-aware grouped
work assignment, and the persistent loop stays simple enough to avoid
register/synchronization blowups. Avoid it when the simple tiled kernel already
saturates the device, or the persistent loop reduces occupancy or complicates
correctness more than it helps.

## §5 Common Gluon-specific techniques

- **§5.1 Explicit tensor layouts.** Tensors require layouts mapping elements to
  CTAs/warps/lanes/registers; `BlockedLayout` (parameters `size_per_thread`,
  `threads_per_warp`, `warps_per_cta`, `order`) is the baseline. Choose register
  ownership to match the access pattern, avoid unnecessary physical registers,
  match tensor-core operand layouts, convert/compose layouts only when the data
  movement pays back.
- **§5.2 Shared-memory layout and bank conflicts.** Stage global memory into
  shared memory when reuse justifies the extra traffic/barriers; use swizzled
  shared layouts for bank-conflict reduction; validate conflicts with profilers
  (do not assume from code shape); keep the footprint compatible with occupancy.
- **§5.3 Async copy, TMA, and pipeline staging.** Use `cp.async`-style
  global-to-shared staging (Ampere/Hopper), TMA multidimensional copies
  (Hopper/Blackwell), explicit prologue/steady-state/epilogue stages, and
  barriers/mbarriers only at required producer-consumer boundaries. Good signals:
  higher tensor-core utilization, lower memory-dependency stalls, stable
  occupancy. Bad signals: stages raise register/shared-memory pressure enough to
  lower throughput; barrier or producer/consumer imbalance dominates.
- **§5.4 Warp specialization.** Assign different warps to different roles (memory
  producers, compute consumers) to overlap independent work and reduce the
  per-warp critical path; documented Hopper-or-newer for NVIDIA. Use it when async
  movement and compute genuinely overlap, producer/consumer have different
  instruction streams, and the extra synchronization/shared-memory traffic is
  smaller than the hidden latency.
- **§5.5 Tensor-core/MFMA pipelines and low precision.** Use architecture-native
  matrix instructions (NVIDIA WGMMA/TCGen05, AMD MFMA/WMMA); keep accumulator
  live ranges under control (split/slice when register pressure dominates);
  specialize low-precision paths around scale/dequantization pipelines rather than
  bolting scaling onto an FP16 kernel late; remap workgroups for chiplet/XCD
  locality on AMD when profiler evidence supports it.
- **§5.6 Dynamic work distribution on Blackwell.** Cluster Launch Control (CLC,
  SM100+) lets a block that finishes early cancel a pending cluster and take its
  work; the key optimization is issuing CLC during the TMA prologue and checking
  the result after tile completion to hide CLC latency behind compute. Use it when
  there is real inter-block load imbalance on Blackwell-class hardware and the
  request/check overlaps with useful work.

## §6 Tooling checklist

| Tool | Best use | Notes |
|---|---|---|
| `triton.testing.do_bench` | Fast kernel microbenchmarking | Good for local iteration; still log shape/config/version. |
| `triton.autotune` + `triton.Config` | Search tile/compiler knobs | Include tile sizes, `num_warps`, `num_stages`, `num_ctas`, `maxnreg`; use correct keys. |
| `TRITON_PRINT_AUTOTUNING=1` | See selected configs | Useful for regression debugging and CI logs (silent on cache hits). |
| Triton debugging ops | Compile-time and runtime checks | `static_print`, `static_assert`, `device_print`, `device_assert` (the last needs `TRITON_DEBUG=1`). |
| `TRITON_INTERPRET=1` | CPU-side functional debugging | Useful before profiling; no `bfloat16`, limited indirect access. |
| FpSan | Structural equivalence for FP kernels | Optimized-vs-reference under sanitized semantics, not IEEE accuracy. |
| Proton | Triton/Gluon profiling, intra-kernel scopes | Used by the persistent matmul tutorial; DSL examples in the Triton repo. |
| NVIDIA Nsight Compute / `ncu` | Kernel-level performance counters | Occupancy, SM/memory throughput, warp stalls, instruction mix, tensor-core activity. |
| NVIDIA Compute Sanitizer | Correctness debugging | `memcheck`, `racecheck`, `initcheck`, `synccheck`. |
| ROCm Compute Profiler / `rocprof` | AMD profiling | Speed-of-light, memory chart, roofline, baseline comparison; raw counters/traces. |
| Triton-Viz | Visualize program behavior | Memory-access visualization/education; not a replacement for hardware counters. |
| PTXAS/compiler inspection | NVIDIA codegen/resource debugging | Register count, spills, generated PTX/SASS, ptxas-option experiments. |

### §6.1 IR-stage visibility in practice (recipe added at M11)

Triton (and Gluon, sharing the pipeline) keeps every compilation stage of a
launched kernel in memory: walk `jit_fn.device_caches` (the function under an
`@triton.autotune` wrapper is `.fn`) to the `CompiledKernel`s and read the `.asm`
dict — keys `ttir`, `ttgir`, `llir`, `ptx`, `cubin`, `source`; `nvdisasm -c` on
the cubin yields SASS. `TRITON_KERNEL_DUMP=1` (+`TRITON_DUMP_DIR`) dumps the same
to disk; `MLIR_ENABLE_DUMP=1` adds per-pass IR. What each stage answers:

- **TTGIR** — the optimization-relevant stage: chosen `#blocked`/shared layouts
  (`sizePerThread`, `threadsPerWarp`, `order`), `tt.scan`/reduce lowering,
  pipelining structure, swizzles. For Gluon kernels the layouts are user-chosen,
  so this is where to verify they survived.
- **PTX/SASS** — instruction mix and final forms: load vector widths
  (`LDG.E.128` vs scalar), atomic forms (`REDG.E.ADD.F32x4` = 4-wide vectorized
  reduction; `REDUX.*` = warp-level reduce), spills (`ld.local/st.local`),
  predication, barrier count.

Use it as a zero-GPU-cost "confirm the lowering" step between picking a config
family and benchmarking it, and to settle ncu surprises by reading the
instruction form instead of inferring it from counter arithmetic. M11 examples:
the backward's 4-wide vectorized `REDG` (inferred indirectly from instruction
counts in T1; one SASS grep proves it), and the `tl.cumsum` shuffle-tree whose
cost was discovered by per-kernel timing — visible immediately as `tt.scan` +
SHFL chains in TTGIR/SASS.

### §6.2 Layout and lowering attribution in practice (lessons added at M13)

Triton chooses tensor layouts from the *consuming* operations, and that choice
propagates back into load vector widths. Lessons measured on the sparton backward
([DEVELOPMENT.md](DEVELOPMENT.md) M13 §2/§5.4; regenerate the per-config IR/SASS
dumps any time with `scripts/dump_backward_ir.py` → `tests/data/ir_dump/`):

- **Reductions are layout-agnostic; scans are not.** A tile feeding only `tl.sum`
  vectorizes freely (`LDG.E.128` at ordinary register budgets); a tile feeding
  `tl.cumsum` along the row axis anchors a row-per-thread or narrow layout — the
  measured outcome on a (CHUNK, BLOCK_D) gather tile was scalar `ld.global.b32` on
  every config except the one-thread-per-row shape, which paid `BLOCK_D` floats of
  live scan state per thread (255 registers → 1 CTA/SM). Vectorize-or-occupancy
  was a structural trade, not a tuning trade.
- **Branch-local SSA separation is not layout separation.** Giving the hot branch
  its own `tl.load` (separate SSA value from the scan branch's load) compiled to
  vectorized *sites* in PTX/SASS, but the executed instruction mix stayed
  scalar-class (runtime bytes-per-instruction ≈ unchanged). The working fix
  classes are structural: a branch-free hot kernel (split kernels with
  complementary predicates), or folding the suppression into load *masks* — a
  per-row mask broadcast over the vector axis preserves the wide form, while a
  branch around the load re-anchors it.
- **`while` walks do not software-pipeline; `tl.range` for-loops do.** The
  persistent `while` + sentinel-exit pattern serializes each iteration's
  keys→check→tile round trips; converting the hot pass to a bounded for-loop
  (device-side trip count, e.g. an active-entry counter written by an upstream
  kernel — no host sync) is what unlocked pipelined/vectorized execution. Keep
  sentinel economics with masks, not loop breaks.
- **Attribute empirically when static reading stalls.** Two tools settle what
  instruction listings cannot: (1) *differential compilation* — compile the
  suspect region in isolation (a throwaway kernel with only the hot branch) and
  diff its census against the full kernel's; (2) *bytes-per-warp-instruction* —
  `sectors × 32 ÷ smsp__inst_executed_op_global_ld` from one ncu pass tells you
  the executed width regardless of what the SASS sites suggest (≈3–4 B/lane =
  scalar, ≈16 B/lane = v4). Three rounds of instruction-counting failed to
  explain what these settled in minutes.
- **Measured config equivalences are evidence.** Two autotune configs from
  different occupancy classes (16.6% vs 24.8% warps) timing identically eliminates
  occupancy-alone as the binder; a vectorized-but-1-CTA config tying a
  scalar-but-3-CTA config localizes the constraint to the layout coupling above.
  Read autotuner ties as information about the bottleneck, not as jitter to
  suppress.

## §7 Common failure modes

- **Tuning only wall time.** A faster microbenchmark can still rely on lower
  clocks, cache luck, or shape-specific artifacts.
- **Over-fusion.** Fusion reduces memory traffic but can increase registers enough
  to reduce occupancy or tensor-core utilization.
- **Too-large tiles.** Larger tiles improve reuse until they cause spills,
  shared-memory pressure, masks, or lower occupancy.
- **Autotune key mismatch.** If key arguments omit a performance-relevant
  stride/shape/dtype, Triton may reuse a bad config.
- **Independently tuned copies of a shared parameter.** When two cooperating
  kernels must agree on a partitioning parameter (a chunk/granule size whose
  predicates must complement), letting each autotune its own silently drops or
  double-counts work at the disagreement boundary — and a small-shape repro can
  pass because both tuners happen to agree there. Single-source the choice (one
  kernel tunes, the other reads `best_config` host-side) and assert the
  compatibility ([DEVELOPMENT.md](DEVELOPMENT.md) M13 §5.4 item 2).
- **Ignoring tails.** Non-power-of-two dimensions and ragged batches often
  dominate real workloads.
- **Assuming Gluon layout changes are free.** Layout conversions and explicit
  shared-memory staging have real cost.
- **Misusing async pipelines.** More stages do not help if barriers, live ranges,
  or producer/consumer imbalance dominate.
- **Profiler perturbation.** Nsight Compute and ROCm Compute Profiler can replay
  kernels or collect counters in multiple passes; benchmark separately from
  detailed profiling.

## §8 Practical optimization playbooks

**Memory-bound elementwise or reduction kernel:** (1) simple Triton baseline;
(2) benchmark GB/s vs expected bandwidth; (3) tune block size/vector width and
`num_warps`; (4) fuse adjacent ops; (5) improve coalescing/alignment; (6) check
masks and tails; (7) read profiler memory throughput, cache hit rate, warp stalls;
(8) move to Gluon only if explicit layout/shared-memory control is needed.

**GEMM-like kernel:** (1) start from Triton matmul tiling; (2) autotune
`BLOCK_M/N/K`, `GROUP_M`, `num_warps`, `num_stages`; (3) check tensor-core/MFMA
utilization and occupancy; (4) add cache-aware program ordering; (5) consider
persistent scheduling; (6) move to Gluon for explicit operand layouts,
shared-memory swizzles, async/TMA, WGMMA/TCGen05/MFMA control, or warp
specialization; (7) for low precision, design the scale/dequant pipeline as part
of the main tile pipeline.

**Attention or fused sequence kernel:** (1) identify whether the dominant path is
memory traffic, softmax/reduction latency, or matmul utilization; (2) tile by
sequence length and head dimension to keep working sets in registers/shared
memory; (3) use numerically stable online reductions; (4) fuse epilogues while
controlling register pressure; (5) profile tails, causal masks, variable sequence
lengths, KV-cache layout; (6) consider Gluon for explicit layout, warp-level
reductions, async copy/TMA, warp specialization.

## §9 Recommended source trail

1. Triton: `triton.autotune` — <https://triton-lang.org/main/python-api/generated/triton.autotune.html>
2. Triton: `triton.Config` — <https://triton-lang.org/main/python-api/generated/triton.Config.html>
3. Triton tutorial: Matrix Multiplication — <https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html>
4. Triton tutorial: Persistent Matmul — <https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html>
5. Triton: Debugging Triton — <https://triton-lang.org/main/programming-guide/chapter-3/debugging.html>
6. Triton: Floating-Point Sanitizer — <https://triton-lang.org/main/programming-guide/chapter-3/fpsan.html>
7. Gluon tutorial: Introduction — <https://triton-lang.org/main/getting-started/tutorials/gluon/intro.html>
8. Gluon tutorial: Tensor Layouts — <https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html>
9. Gluon tutorial: Async Copy — <https://triton-lang.org/main/getting-started/tutorials/gluon/async-copy.html>
10. Gluon tutorial: Warp Specialization — <https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html>
11. Gluon tutorial: Cluster Launch Control — <https://triton-lang.org/main/getting-started/tutorials/gluon/cluster-launch-control.html>
12. AMD ROCm blog: From Naive to Near-Peak GEMM with Gluon — <https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html>
13. AMD ROCm workload optimization/profiling — <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html>
14. NVIDIA Nsight Compute Profiling Guide — <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
15. NVIDIA Compute Sanitizer — <https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html>
16. Triton Proton intra-kernel example — <https://github.com/triton-lang/triton/blob/main/third_party/proton/tutorials/intra_kernel/example_dsl.py>
17. Triton-Viz — <https://github.com/Deep-Learning-Profiling-Tools/triton-viz>
18. TritonForge paper — <https://arxiv.org/abs/2512.09196>

---

# §C — Worked examples (the method proving itself)

Three milestones illustrate the Performance-Optimization Loop's first step —
*name the binding resource, then ask what class of change can move it* — landing
on three different answers. Full evidence in [DEVELOPMENT.md](DEVELOPMENT.md).

- **M11 — the binder was an algorithm-level operation count.** The shared backward
  was bound by L2 reduction-sector traffic: one atomic lane per active `(b,v,d)`
  element, `f·B·V·D/8` sectors, ~97% of reduction traffic and invariant under any
  v-major restructuring (the analytic model matched measured sectors to four
  significant figures). TMA and Gluon could not touch it — only changing *which
  operations exist* helped: host-side destination sort + a chunked segmented
  reduction emitting ~two partial-sum atomics per destination run instead of one
  per contribution (264× L2 red-sector cut on real query records). The host side
  (sort + a fused prep kernel that collapsed seven elementwise launches) was the
  larger win at small shapes. (M11 §1–§6.)
- **M12 — the prior was disproved by the first profile of the artifact.** The
  forward's presumed binder was SM-side scheduling slack (drains, barrier stalls,
  wave tails) inherited from a sibling GEMM bring-up kernel at 63.9% pipe. The
  first ncu of the *production* kernel showed it tensor-pipe-bound at 92–94% with
  the L2 fabric at 89–91% and SM active/elapsed = 99.6% — no bubbles to fill. The
  binder is operation count (pipe-work + L2 traffic at the selected tile), not
  scheduling; a persistent/warp-specialized rewrite cannot move it. The milestone
  **closed without kernel work** — a valid, recorded outcome. Lesson: profile the
  artifact you intend to change; inherited priors from sibling kernels do not
  transfer. (M12 §1–§5.)
- **M13 — the binder was a lowering coupling.** The segmented hidden-grad kernel
  sat at 3.2× its traffic floor with a SASS-confirmed mechanism: the gather load's
  vector width is layout-coupled to the `tl.cumsum` tile, so every autotune config
  either scalarizes the gather (L1TEX issue-bound) or pays 255 registers for one
  CTA/SM (occupancy-bound) — two configs from different occupancy classes timing
  identically proved occupancy-alone was not the lever. Neither tuning nor pure
  occupancy work could move it; a *structural kernel split* (a branch-free,
  pipelined, vectorized streaming reduction for single-destination chunks plus a
  segmented scan for run-boundary chunks, complementary at one shared granularity)
  could — 1.46–1.60× on captured-real records. The embed kernel, a measured
  gather-dominated sibling running at 82–104% LTS, was the existence proof that
  the floor was reachable. (M13 §1–§5.)


