# Sparton Remaining-Work Design v4 (post-M11)

Date: 2026-06-12.
Status: **active guide for subsequent backend development.**

Supersession chain: [v1](sparton_gluon_remaining_work_design.md) remains
authoritative for platform facts and the original measured evidence (sm_120
MMA matrix, Gluon API survey, environment defects, profiler notes, the GEMM
bring-up counters and break-even model); [v2](sparton_remaining_work_design_v2.md)
for the post-M8 review findings (F1–F20), the design-evolution ledger, the
architecture rules of record (§4, including the policy-bank mechanics that
the deferred launcher v2 would replace — §6), and the executed M9/M10
specifications;
[v3](sparton_remaining_work_design_v3.md) for the executed **M11 plan of
record** and the post-M10 state snapshot and floor-ratio derivations (its
§1) — the specification whose deviation ledger lives in the
[M11 memo](sparton_milestone11_backward_memo.md), which is itself the
evidence of record for the backward's mechanism, analytic traffic model,
and residual bottlenecks. This document replaces v3's forward-looking
content: it re-grounds the remaining work in the post-M11 measured state
and restructures it around the method that M11 proved out (the
Performance-Optimization Loop in `AGENTS.md`, with
[the kernel-optimization reference](triton_gluon_kernel_optimization.md)).
Method rules live in those documents and are not restated here.

Every number below was measured in the 2026-06-11/12 sessions on the
validation platform (RTX 5090, Triton 3.6.0, torch 2.12 nightly) and is
cited to its run of record (Appendix A). Derived figures are labeled with
their derivation. Rule of record: **historical numbers orient; anything
that gates a current decision is re-measured by the entry task that
consumes it.**

---

## 1. Where we are after M11 (verified)

### 1.1 State summary

- `optimized` (Gluon TMA + `mma_v2` fused forward, policy-autotuned) is the
  default backend where available; explicit selections never fall back
  (M10, unchanged).
- All three backends share the **M11 segmented backward** inside the
  unchanged `sparton::fused_sparton_bwd` op: `bwd_prep_kernel` →
  exclusive-owner `embed_grad_kernel` (zero atomics, `torch.empty`
  outputs) → `torch.sort` + `bwd_gather_payload_kernel` →
  `segmented_hidden_grad_kernel` (~two partial-sum atomics per destination
  run; `seq_len` in its autotune key). The pre-M11 kernel is retained as
  `legacy_fused_sparton_bwd` — the test-pinned A/B reference of record,
  with a role comment naming its removal condition (§2).
- Suite: **132 tests** (quick loop 109). Standing gate scripts:
  `soak_optimized_correctness.py` (384-case forward soak),
  `probe_training_smoke.py` (300-step AMP parity), and the M11 backward
  toolchain — `capture_index_distributions.py` (real-distribution bundles),
  `bench_backward.py` (registry harness, per-cell verification before
  timing, determinism protocol), `ncu_backward_target.py` (direct-op
  profiling target).
- `compute-sanitizer` works on this host since the post-M11 restart; the
  M11 racecheck/memcheck/initcheck gates ran clean on the promoted tree.
- Determinism: `embed_grad`/`bias_grad` are structurally deterministic
  (plain stores); `hidden_grad` order-nondeterminism remains but with ~60×
  fewer atomics (per-call proxy spread ≤ 4.2e-6; the M13 split preserves
  the atomic structure and the band — M13 memo §5.5). The training-scale
  band was re-measured at M13-T0: same-config 150-step loss spread 16–38%
  depending on backend and statistic (M13 memo §6); AGENTS.md's sharp
  edge now carries the measured band.

### 1.2 Measured snapshot (runs of record)

Dev shape `B=32, S=128, D=768, V=30522`, fp16:

| metric | M10 gate run 2 | M11 gate run 2 |
|---|---:|---:|
| optimized forward + bias | 0.900 ms | 0.880 ms |
| optimized fwd+bwd + bias | 2.325 ms | 1.949 ms |
| implied optimized backward | 1.425 ms | **1.069 ms** |

The dev-shape floor ratio of record is **1.02×**: the post-M8 review run
measured the optimized forward at 0.898 ms against a 0.880 ms full-V
cuBLAS GEMM in the same table (v2 §1.2). Neither the M10 nor the M11 gate
run re-quoted a floor; their forwards (0.900 / 0.880 ms) are consistent
with the review run within the same-config band. The dev forward is
essentially done; M12-T0 re-measures forward and floor in one preserved
run before any ratio gates a decision *(done — dev 1.029×, M12 memo §3)*.

Canonical `splade-code-06B` grid (bf16, `D=1024, V=151936`, all-ones
masks): forward (kernel unchanged since M8) runs at **≈1.05–1.12× the
per-row GEMM floor**. Provenance requires care: the preserved same-run
evidence is the M8 memo grid (`opt+b` vs `gemm ms` columns → 1.094–1.123×
across the nine rows); v3 §1.2 recorded 1.054–1.113× (best 8×512, worst
16×768) from the M10 gate run, whose per-row gemm column was **not
preserved** — per the evidence rules it orients but may not gate
*(resolved at M12: re-measured in one preserved run, 1.099–1.117× — memo
§3)*. The M11
implied backward is 1.546–3.720 ms per row, −32…−41% vs M10. Derived
shares: the backward is now **~17–52% of optimized fwd+bwd across the
grid** (M11 implied-bwd ÷ (M10 forward + M11 implied-bwd), mixed-run) and
**~55% on the dev shape** (1.069 / 1.949, same M11 run); v3's figures were
24–64% and 61%. Captured-real records (V=250002): backward 3.3–4.4 ms
post-M11 (legacy: 5.90–7.93 ms).

Two residual costs, both with named mechanisms:

1. **Forward gap to the GEMM floor: ≈5–12% above the floor (≈5–11% of
   forward time) on grid rows.** In absolute terms roughly 0.3–0.5 ms at
   8×512 and 1.6–1.8 ms at 16×768, depending on which recorded run anchors
   the floor (derived from the §1.2 ratios; re-derived in one run at
   M12-T0). The only counter evidence for the forward family is the
   **GEMM bring-up kernel** (v1 §3.5: tensor pipe 63.9% vs cuBLAS
   86.6% at identical occupancy/instruction family ⇒ the gap is intra-CTA
   pipelining quality). **The production fused kernel has never been
   ncu-profiled** — closing that is the first step of any forward work
   (M12-T0). *[Superseded at M12: the profile exists (memo §2) and
   overturned this picture — tensor pipe 92–94%, no scheduling slack.]*
2. **Backward residual (M11 memo §6):** the segmented hidden-grad kernel is
   bound by embed-gather latency (no unit above ~35% on the deposited real
   doc record);
   the embed/bias kernel is L2-throughput-bound on its g/idx stream
   re-reads (LTS 82.6% on the dev shape). Atomics are no longer a
   bottleneck anywhere (L2 reduction sectors down 27.7× dev / 51.7× corner
   / 264× real query record). The 16 `steps150`-document harness cells at
   1.35–1.43× vs legacy (vs the 1.5× clause) are the recorded deviation
   this residual explains. *[Resolved at M13: the residual was recovered —
   the split backward takes those cells to 1.568–1.596× vs the segmented
   design (≈ 2.1–2.3× vs the M2 legacy the clause was written against, composed
   from the preserved per-design ranges);
   the embed-kernel attribution was corrected (binder = hidden-row gather
   re-reads, already at the traffic floor) — M13 memo §2/§5.]*

Host-side launch overhead of the optimized forward: ~0.119 ms/call
wall-minus-GPU at `8×128×768×1280`, ~0.051 ms of it rebuilding the
22-descriptor bank every call (v2 §1.3). Unchanged since — no launcher work
has happened. Irrelevant at ≥1 ms GPU times — which is every documented
workload; it would bind only for a small-shape latency-critical caller of
the head itself, which nothing in the repo exercises. The scheduled fix
(launcher v2) is **deferred** by maintainer decision; rationale and
revival triggers in §6, overhead kept visible by M12-T4.

### 1.3 What M11 changed in this plan's assumptions

1. **No track dominates universally anymore.** v3 could order "backward
   first" from a single share figure; post-M11 the forward gap and the
   backward residual are the same order of magnitude, and which is larger
   depends on the shape (dev: backward ~55%; 16×768: forward gap
   ~1.6–1.8 ms vs backward 3.72 ms with an unknown recoverable fraction).
   Both kernel tracks below are therefore **entry-gated on fresh profiles
   and analytic ceilings**, and the two entry tasks fill the two columns
   of one comparative table (§3 preamble) so the ordering is decided by
   measurement, not by this document.
2. **The altitude question now has a worked example on each side**
   (AGENTS.md Performance-Optimization Loop, step 1). The backward's binder
   was an operation count (L2 reduction sectors) — grid restructuring, TMA,
   and a Gluon port could not touch it, and B2b was rejected on that
   structural evidence. The forward's known gap is SM-side pipelining
   quality (cuBLAS reference at the same occupancy and instruction family)
   — exactly the binder class where lowering-level tools (persistent
   scheduling, warp specialization) pay off. M12's kernel work remains
   plausible *because* its binder class differs from the one that
   bounded B2a/B2b. *[Superseded at M12: the production kernel's own
   profile put its binder in the operation-count class after all (pipe
   saturated at 92–94%; the v1 inference was drawn from the GEMM bring-up
   kernel at 63.9% and did not transfer) — memo §5.]*
3. **Numeric exit gates are set from a validated analytic model, not
   before one exists.** M11's 1.5× clause was written before the traffic
   model and the doc-record cells near-missed it at 1.35–1.43× for a
   reason the model later named (embed-gather latency, not atomics — the
   premise the clause was sized for). M12/M13 exit numbers are therefore
   fixed at the end of their entry tasks, from the model, and
   pre-registered in the milestone memo before candidate work starts.
4. **Autotune-selection jitter is part of the noise band** (±5% between
   processes on borderline cells, M11 memo §5.2/§5.3). This invalidates
   v3's M12-T1 gate "identical policy selection to today's autotuner"
   as stated — near-tie selections legitimately flip between runs. The
   parity gate for any revived launcher v2 is restated accordingly in its
   deferred entry (§6): candidate-set/ranking parity (deterministic) plus
   winner-time parity within a measured A-vs-A band.
5. **Real-distribution evidence is standing infrastructure, and the data
   already falsified one assumption** (tier-2 representations are dense,
   `f = 1.0`; what real batches add is index collisions, not sparsity).
   The sparse regime (`f ≪ 1`, late-stage FLOPS-regularized training)
   remains covered only by synthetic `--active-fraction 0.10` sources — a
   recorded gap that M13-T0 may close cheaply, or re-record.
   *(Re-recorded at M13-T0: a 3-attempt shortened-warmup probe brackets
   the λ transition — dense at 1e-2, collapsed to f = 0 at 1e-1 — without
   landing inside it; ~25 s/attempt, so a λ-bisection is cheap if the
   regime ever gates a decision. M13 memo §7.)*

---

## 2. Architecture rules in force

`AGENTS.md` carries the durable rules (layering, symmetry, one-seam
changes, no-silent-fallback with its single exception, index contract,
binary mask contract, op/schema stability, `torch.empty`-vs-coverage
lockstep). Load-bearing for the milestones below:

- **Backward swap is schema-safe** (proven by M11): a backward change that
  keeps the saved-tensor set swaps inside `sparton::fused_sparton_bwd`
  without touching schemas or autograd wiring; a changed saved-tensor set
  requires a new op name. The same rule governs any M13 change.
- **No data-dependent algorithm dispatch inside an op** — it would need a
  host sync and a second fallback seam. M11 shipped one kernel family for
  this reason; M13 inherits the constraint.
- **Backward prototypes live only in the `bench_backward.py` registry**
  (production-op signature, selected by `--impls`, numerically verified
  per cell before any timing). Production is wired winner-only at
  promotion.
- **Policy-bank mechanics** (v2 §4.2) remain the document of record for
  how the optimized forward selects configurations today — and stay in
  force while launcher v2 is deferred (§6). Whatever eventually replaces
  the mechanism (a revived launcher v2, or a T3 rewrite that folds in the
  single-pair launch) must update v2 §4.2's status note when it does.
- **D2 (kernel-body duplication** between the optimized forward and
  `bench_gluon_gemm.py`) stays deliberately unresolved until the M12-T3
  decision point: dedupe into a shared `@gluon.jit` helper only if the
  rewritten production mainloop and the benchmark still share structure;
  if M12 closes without a rewrite, discharge D2 by re-affirming the
  cross-reference comments instead. **Discharged 2026-06-13** (M12 closed
  without a rewrite; comments re-affirmed — M12 memo §5); re-open only if
  a future milestone reopens the kernel body.
- **`legacy_fused_sparton_bwd` removal condition** (role comment at the
  kernel): it stays until a later milestone supersedes the M11 comparison
  evidence. If M13 promotes a new backward, the baton passes explicitly:
  the segmented design becomes the new test-pinned A/B reference and the
  M2-era legacy kernel is deleted in the same change — never two silent
  legacy copies.

---

## 3. Milestone plan

Numbering continues from v3. Execution order inside and across milestones:

1. **Unconditional tasks:** M12-T0 (forward entry evidence), M12-T4
   (measurement additions), and M13-T0 (backward entry evidence +
   training-scale debt). Launcher v2 (M12-T1) is **deferred** by
   maintainer decision — rationale, revival triggers, and the retained
   gate design in §6. M13-T0 reuses M12-T0's re-measured grid table when
   it exists (one provenance for the shared baselines) and re-measures
   otherwise — a convenience, not a dependency.
2. **Kernel-rewrite work (M12-T2/T3, M13-T1/T2) is conditional**, each
   behind its own track's entry rule with its own denominator (M12-T0:
   ≥10% of forward time; M13-T0: ≥10% of backward time — stated in the
   tasks). A track whose entry rule fails closes with a short memo
   instead of kernel work; closing without kernel work is a recorded
   outcome, not a failure (v3 allowed it for M12; v4 extends it to M13).
3. **Cross-track ordering when both entry rules pass:** execute in
   descending order of modeled recoverable ms per call on the shapes the
   models name, read from the comparative table that the two entry tasks
   fill — M12-T0 contributes the re-measured shared baselines and the
   forward column, M13-T0 the backward column. If only one rule passes,
   ordering is moot; if neither, the performance work ends with two
   closing memos (see Beyond M13).

### M12 — Forward track: entry-gated scheduling rewrite

Update, 2026-06-13: **M12 is complete — closed without kernel work.**
T0's re-measured evidence fails the pre-registered rule's second clause:
the production forward kernel is tensor-pipe-bound at **92.3–94.4%**
utilization with the L2 fabric simultaneously at 89–91% and DRAM at the
compulsory byte floor — there are no scheduling bubbles for a
persistent/warp-specialized rewrite to fill (rule (i) passed on five grid
rows at 10.16–10.51%, margins inside the run-to-run noise band; rule (ii)
failed by ~18 points). T2/T3 were not entered; T4 landed
(`--mask-density`, `bench_host_overhead.py`, plus the T0 tooling
`ncu_forward_target.py`); D2 was discharged by re-affirmation; the
launcher fold-in trigger (§6, trigger b) never fired. The one-provenance
state table, validated traffic model, decision walk, and the forward
track's terminal residual-bottleneck note (per-cycle pipe efficiency +
L2 pressure at the autotuned 64×64×32 tile shape — a tile-shape question,
not a scheduling one) live in the
[M12 memo](sparton_milestone12_forward_memo.md).

Goal: rewrite the mainloop only if fresh counter evidence says the
remaining forward gap is real, SM-side, and worth ≥10% of forward on a
shape that matters. (Launcher v2, formerly this milestone's unconditional
task, is deferred — §6.)

Will NOT touch: backward kernels or the backward op; `_validation.py`;
op schemas (`sparton::optimized_fwd` keeps its schema throughout); the
selection/launch mechanism — decorator autotune and the descriptor bank
stay as-is while launcher v2 is deferred, except that a T3 rewrite may
fold in the single-pair launch per its §6 revival trigger; training code;
dependencies. T3 is the only task that changes the kernel.

#### M12-T0 Entry evidence: first production-forward profile + analytic model + decision

The Performance-Optimization Loop steps 1–2 applied to the forward; v3's
prose entry gate made executable. Half a day to a day.

- New `scripts/ncu_forward_target.py` mirroring
  `ncu_backward_target.py` (parameterized main-thread direct calls to the
  optimized forward, NVTX range `fwd_direct/`, CLI shape/dtype/bias);
  `ncu_targets.py` stays the fixed-dev-shape hybrid-era launcher it is.
- ncu the **production** `sparton_optimized_forward_kernel` (first time):
  dev shape (fp16) plus the best- and worst-ratio grid corners (8×512,
  16×768, bf16). Record tensor-pipe %, SOL compute/memory, DRAM bytes,
  stall profile, occupancy, regs — same table discipline as M11 memo §2.
  Log selected policies with `TRITON_PRINT_AUTOTUNING=1`. nsys: kernel
  inventory per forward call. Deposit transcripts (done at execution under
  the session-local `/root/profiles/m12/`; today's convention is the
  gitignored `tests/data/runs/<label>/`).
- Re-measure the canonical grid and dev row **with the `gemm ms` column in
  the same run** (`bench_sparton_baseline.py --optimized-policy on`, run 2
  of 2) so per-row floor ratios and fwd/bwd shares carry one provenance —
  replacing §1.2's mixed-run derivation.
- Analytic model before any candidate work: per-row forward floor =
  max(tensor-pipe-bound GEMM time, A+B(+C) byte floor) + epilogue traffic
  (start from v1 §3.3's break-even model); validate against the counters
  (M11 memo §3 standard: the model must explain the measured bytes/ratios
  before it is used to predict candidates).
- **Decision rule (pre-registered):** proceed to T2/T3 only if (i) the
  re-measured gap to floor is **≥10% of forward time on at least one
  canonical grid row**, AND (ii) the counters name an SM-side binder the
  persistent/warp-specialized family addresses: tensor-pipe utilization
  below **74%** on that row (the v1 §3.5 tensor-pipe equivalent of the M7
  85%-of-cuBLAS gate) with the dominant stall in the
  scheduling/barrier/pipe-wait family (stall metric named in the memo) —
  not a memory-side or operation-count binder. If either fails: write the
  closing memo, discharge D2 by re-affirming the cross-reference comments,
  and close M12 kernel work. Fix the T3 exit numbers from the validated
  model at this point (§1.3 item 3).
- Output: the re-measured shared state table (grid + dev: forward, f+b,
  gemm columns, one provenance) and the **forward column** of §3's
  comparative table (modeled recoverable ms per shape); M13-T0 adds the
  backward column.
- Gate: counter tables + model-vs-measured residuals recorded in the M12
  memo with deposited transcripts; an explicit go/no-go entry in the memo.

#### M12-T1 Launcher v2 — deferred (maintainer decision, 2026-06-12)

The task number is retained so existing references resolve; the work is
not scheduled. Deferred until a latency user exists or until the T3
rewrite forces the kernel signature open anyway — full rationale, revival
triggers, and the gate design of record for any revival are in §6
("Launcher v2"). F9 remains a recorded finding; M12-T4's latency rows
keep the measured overhead visible.

#### M12-T2 `gl.warp_specialize` subprocess probe (entry gate for the WS variant)

Still unprobed on sm_120 (v1 §3.6 item 2). New
`scripts/probe_warp_specialize.py`, subprocess-isolated in the
`probe_mma_matrix.py` house shape: compile and run a minimal
producer/consumer warp-specialized kernel; record register budgets and
barrier interplay. Gate: the probe exits 0 having printed a verdict line
plus a register/barrier table, **or** records the fatal/unusable verdict
— either outcome completes the task; the verdict selects T3's variant. A
fatal or unusable result demotes T3 to the CTA-barrier persistent variant
only — it does not close T3.

#### M12-T3 Persistent (+ warp-specialized) mainloop (conditional on T0)

- Persistent flat tile loop with L2-aware rasterization; producer/consumer
  empty/full barriers replacing the per-iteration CTA `gl.barrier()`
  (WS variant only if T2 passes); epilogue unchanged. The O1 kernel's
  per-s-tile pipeline drain and embed-tile re-read disappear naturally in
  the persistent formulation — measure the actual gain against the T0
  model, don't assume it.
- This rewrite is the **D2 decision point** (§2). It is also the recorded
  revival trigger for the deferred launcher v2 (§6): the rewrite opens
  the kernel signature anyway, so decide here — and record the decision —
  whether the new kernel takes a single descriptor pair (folding in the
  launcher simplification and deleting the 22-slot bank) or keeps the
  bank mechanism.
- Per-iteration discipline (loop steps 8–10): confirm lowering by reading
  TTGIR/SASS before benchmarking a config family (method ref §6.1, zero
  GPU cost); accept tuning changes only with a profiler-confirmed
  mechanism; audit the decorator-autotune key if the rewrite adds
  performance-relevant arguments.
- Gates: full suite, shape soak, training smoke, benchmark grid; exit
  number fixed at T0 from the model (target family: ≤ **1.05× the
  re-measured per-row GEMM floor on every grid row**); v1 §9 rejection
  criteria apply (>5% regression anywhere → rejected without a >10% win
  elsewhere); nsys shows a single kernel launch per forward; ncu DRAM
  reads within 1.3× the analytic A+B floor; tensor-pipe utilization
  recorded against the 86.6% cuBLAS reference. Exit names the residual
  bottleneck (loop step 10).

#### M12-T4 Measurement-set additions (unconditional)

- Add `--mask-density` to `scripts/bench_sparton_baseline.py` (default
  1.0 — today's all-ones behavior unchanged) and record a forward sweep
  over densities {25, 75, 100}% on the dev shape plus one grid row —
  v1 §10.3's sweep, landed on the forward side (`bench_backward.py`
  already carries it for the backward).
- Record small-shape/latency rows (the v2 Appendix A item 5 shape family,
  e.g. `8×128×768×1280`) including the wall-minus-GPU host share per
  backend — the standing documentation of the deferred F9 overhead (§6),
  and the baseline any future latency user would revive launcher v2
  against.
- Gate: the recorded tables land in `scripts/README.md` (or a
  benchmarks doc it links) with run-of-record provenance. Record, don't
  threshold — these are documentation of record, not pass/fail numbers.

### M13 — Backward residual track (entry-gated)

Update, 2026-06-13: **M13 is complete — entry rule passed, kernel work
executed, split backward promoted.** T0's validated traffic model
(per-buffer residuals ≤ 1% on the decision counters; the embed kernel
measured *at* its floor, correcting the M11 §6 attribution — its binder
is hidden-row gather re-reads, not g/idx streams) priced the segmented
kernel at 3.2× its L2 floor on real records with a SASS-confirmed
mechanism: the gather's vector width is layout-coupled to the cumsum
tile (scalarize, or pay 255 regs for 1 CTA/SM). Conservative recoverable:
38% of the doc-record backward — the ≥10% rule passed ~4×. T1/T2
delivered the **split segmented backward** (branch-free pipelined
vectorized uniform-chunk pass — production profile 1.43 ms at 56
regs/73.5% occupancy on the doc record vs the 1.34 ms modeled
conservative floor — plus the segmented scan on run-boundary chunks
only, complementary at a shared 64-entry granule): captured-real backward
1.46–1.60× vs M11 (steps150 docs 1.568–1.596×), every canonical grid row
within band or improved (dev implied bwd 1.061 → 0.982 ms), synthetic
f=0.10 short-run cells −6…−16% (recorded deviation, v1 §9's
regression-without-gain criterion; no real capture exhibits the regime).
Sanitizer clean; AMP smoke unchanged; suite 134. The A/B baton passed
(segmented design = the test-pinned reference, M2-era kernel deleted).
Training-scale debt closed (tier-2 parity holds; same-config band
measured 16–38%, replacing the "~20%" sharp edge). Sparse-regime capture
probed and not materialized (λ-transition brackets to f=0); synthetic
f=0.10 stays the regime of record. Evidence:
[M13 memo](sparton_milestone13_backward_memo.md).

Goal: decide — with a model, on real distributions — whether the
segmented backward's named residual (embed-gather latency; g/idx stream
re-reads) is worth a kernel change, and if so, take it behind the M11
machinery. The M11 memo §6 is the entry evidence; its residual-bottleneck
note was written to be this milestone's input.

Will NOT touch: forward kernels and wrappers; op schemas while the
saved-tensor set is unchanged (§2); `_validation.py`; dependencies. The
production op stays untouched until promotion — candidates live in the
`bench_backward.py` registry.

#### M13-T0 Entry evidence: analytic gather ceiling + training-scale debt

Unconditional (runs even if the kernel decision is "close"):

- **Analytic ceiling for the residual** (loop step 2): per-buffer formulas
  for the segmented kernel's gather traffic and the embed kernel's stream
  re-reads in `(B, S, V, D, f, run-count)`; validate against the deposited
  M11 transcripts plus fresh ncu on one real query and one real doc record
  (`ncu_backward_target.py --bundle ...`). The model must state the floor
  the gather path cannot go below and what fraction of today's 3.3–4.4 ms
  real-record times sits above it.
- **Candidate hypotheses to size against the model, not to build yet**
  (each is unverified until the model prices it): wider/vectorized gather
  loads (confirm current load form in SASS first — §6.1 recipe, zero GPU
  cost); fusing `bwd_gather_payload_kernel` into the segmented kernel (one
  fewer stream pass); occupancy/register work on the segmented kernel
  (168 regs/thread at M11); d-tile blocking to reuse gathered rows.
- **Sparse-regime data (optional, timeboxed):** probe whether a
  regularizer-active capture is cheap — `train.py` exposes `--lambda_l1`
  and `--reg_warmup_steps`, and `capture_index_distributions.py` exposes
  `--train-steps`, so a shortened-warmup recipe may yield an `f ≪ 1`
  bundle without the 10000-step default warmup. If it does not materialize
  cheaply, the synthetic `--active-fraction 0.10` sources remain the
  sparse regime of record and the gap stays recorded (M11 memo §9). Note:
  the segmented kernel's sentinel quick-exit already makes sparse inputs
  cheap, so this is a representativeness item, not a risk item.
- **Training-scale debt (closes two recorded gaps from M11/M10):**
  1. Tier-2 150-step `train.py` parity rerun on the promoted backward
     (hybrid vs optimized, seed-matched — the M11 known-gap item; the AMP
     smoke remains the gate of record, this is the deferred
     belt-and-suspenders run).
  2. Re-measure the same-config training-loss spread (≥3 same-config
     150-step runs): the "~20%" band in AGENTS.md's sharp edge predates
     M11 and is flagged there as not re-measured. Replace it with the
     post-M11 measured band in AGENTS.md and the M13 memo.
- **Decision rule (pre-registered):** proceed to T1/T2 only if the
  validated model says **≥10% of the measured backward time is
  recoverable on at least one canonical grid row or captured-real
  record** (backward times exist for both: the implied-backward grid
  column and the harness's real-record cells), AND the §3 cross-track
  ordering places M13 next (or M12-T3 is already complete or closed).
  Otherwise: closing memo recording the ceiling and the residual's price,
  and the track ends with the named bottleneck on file. Fix T2's exit
  numbers from the model now (§1.3 item 3).

#### M13-T1 Candidates in the registry (conditional)

- Implement the surviving candidates from T0 in `bench_backward.py`'s
  registry with the production op's exact signature (`--impls`), full
  semantics (fp32 accumulation, bias/no-bias, `scores == 0 → zero
  gradient`, binary-mask contract), per-cell verification before timing —
  the M11 T3 pattern, including the pre-registered early-stop rule for
  timeboxed alternatives.
- Decision matrix: {uniform, zipf, captured-real ×2 bundles (+ sparse
  bundle if T0 produced one)} × densities × fp16/bf16 × bias modes, two
  consecutive runs, run 2 of record; decision criteria in the M11 order
  (real-distribution performance, synthetic non-regression, complexity,
  determinism delta).

#### M13-T2 Promotion (conditional)

- Wire the winner inside `fused_sparton_bwd_op` (schema-safe swap if the
  saved-tensor set is unchanged; else new op name per §2). Pass the A/B
  baton explicitly (§2): the segmented design becomes the test-pinned
  legacy reference; delete the M2-era kernel in the same change.
- Gates (machinery identical to M11 T4): full gradient matrix (vs the
  closed-form expectation from saved `(scores, idx)` with pinned forward
  outputs, and A/B vs the segmented kernel); `probe_training_smoke.py`
  unchanged; full suite + shape soak; `bench_backward.py --impls
  current,legacy` over all sources — uniform-only evidence is never
  sufficient; performance exit numbers fixed at T0; determinism protocol
  before/after; `compute-sanitizer` racecheck/memcheck/initcheck if
  ownership semantics change (now runnable on this host — no deferral
  path needed); the adversarial milestone review before close
  (`AGENTS.md` Milestone Review).

### Beyond M13

Future-hardware and out-of-scope items are unchanged from v1 §9 (WGMMA /
TCGen05 / TMEM `load_max` on real sm_90a/sm_100 hardware, FP8 formats,
clusters, TMA gather/scatter for backward, CUTLASS, split-K). Revisit only
on a hardware or Triton-capability change, re-running the v1 probe matrix
first. There is no planned milestone beyond M13: if both kernel tracks
close at their entry gates, the project's performance work ends with two
closing memos and named residuals — a valid terminal state.

---

## 4. Validation matrix v4

Standing gates (every milestone exit, hardened env, serial):

```bash
ENV='TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas CPATH=/usr/local/cuda-13.2/include TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor'
PY=/workspace/venvs/sparton/bin/python
env $ENV PYTHONPATH=src $PY -m py_compile src/sparton/*.py training/*.py tests/*.py scripts/*.py
env $ENV $PY -m pytest -q                          # full, incl. slow (132 at M11 exit)
env $ENV $PY -m pytest -q -m "not slow"            # quick loop (109 at M11 exit)
env $ENV PYTHONPATH=src $PY -u scripts/soak_optimized_correctness.py        # forward-surface changes
env $ENV PYTHONPATH=src $PY -u scripts/probe_training_smoke.py              # autograd/AMP changes
env $ENV PYTHONPATH=src $PY -u scripts/bench_sparton_baseline.py --optimized-policy on   # grid of record
env $ENV PYTHONPATH=src $PY -u scripts/bench_sparton_baseline.py \
    --batch-sizes 32 --seq-lens 128 --dim 768 --vocab 30522 --dtype fp16 --optimized-policy on  # dev row
env $ENV PYTHONPATH=src $PY -u scripts/bench_backward.py --impls current,legacy \
    --sources uniform,zipf,real --bundle tests/data/bundles/swimir_de_steps0.pt \
    --bundle tests/data/bundles/swimir_de_steps150.pt --active-fraction 0.10     # backward changes
env $ENV PYTHONPATH=src $PY -c "import sparton"    # stdout-silent
git diff --check
```

(Bundles live gitignored in `tests/data/bundles/`;
`capture_index_distributions.py` regenerates them if absent and reuses
them if present — regenerated bundles contain different records, so
re-measure baselines rather than comparing across the swap.)

M12 additions: production-forward counter tables vs the analytic model
(T0); nsys launch-count and ncu DRAM-floor checks plus the v1 §9
rejection criteria (T3); mask-density sweep and latency rows including
the host-overhead share (T4). The launcher-v2 parity and host-overhead
gates live with its deferred entry (§6) and apply only on revival.

M13 additions: model-vs-counter validation on real records (T0);
per-cell-verified decision matrix over all sources, run 2 of record (T1);
gradient matrix + A/B vs segmented, determinism protocol before/after,
sanitizer on ownership-semantics changes, exit numbers as fixed at T0
(T2); tier-2 150-step parity rerun and the training-scale determinism
band re-measured, with AGENTS.md's sharp edge updated (T0).

Index and score tolerances are unchanged (v2 §6.2/§6.3; the tie-aware
`assert_index_contract` remains the contract of record). The binary-mask
backward contract (M11 ruling, memo §9) is pinned by the existing tests;
any M13 candidate inherits it.

---

## 5. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| F9 host overhead (~0.119 ms/call) stays in place for any future latency user | low — no such user identified | deferred with recorded revival triggers and gate design (§6); M12-T4 keeps the overhead share measured and visible |
| Production-forward profile contradicts the GEMM-kernel priors (binder not SM-side) | medium | that is what T0 exists to find; the decision rule then closes T3 without kernel work, with the memo recording the real binder |
| `gl.warp_specialize` immature/fatal on sm_120 | medium | M12-T2 subprocess probe before any T3 work; CTA-barrier persistent variant is the fallback |
| Persistent rewrite churns the validated forward kernel for <5% | medium | T0 entry gate + model-derived exit numbers + v1 §9 rejection criteria + shape soak; closing-without-rewrite is a recorded outcome |
| M13 gather ceiling sits close to current performance | medium | model-first: the ceiling is computed before any candidate is built; the closing memo records the price of the residual |
| Captured bundles lost (host cleanup) or unrepresentative beyond swim-ir/XLM-R | low | regeneration recipe in `scripts/README.md`; tokenizer/corpus generality is a recorded limitation (M11 memo §9); harness accepts any capture-script bundle |
| Sparse-regime capture needs training-recipe changes that don't converge cheaply | low | timeboxed probe; synthetic `f = 0.10` stays the regime of record; sentinel quick-exit already covers sparse cost |
| Backward changes shift training numerics | low–medium | A/B vs segmented + AMP smoke + determinism protocol with the post-M11 baseline; tier-2 parity rerun lands at M13-T0 regardless |
| Triton upgrade moves Gluon APIs mid-milestone | medium | shim + `VALIDATED_TRITON` warning in place; re-run the M7 ratio gate (`bench_gluon_gemm.py --require-ratio 85`), epilogue probe, and soak on any bump |
| Two-process measurement interference / post-GPU-job pytest flakes | low | serial runs; classify-then-rerun rule (AGENTS.md); never edit `src/`/`scripts/` while a queued process will import them |

---

## 6. Rejected / deferred

Deferred by maintainer decision, 2026-06-12 (supersedes v3's "launcher v2
proceeds independently"):

- **Launcher v2 (M12-T1: owned two-phase selector; delete the 22-slot
  descriptor bank and `POLICY_ID` if-chain; fixes F9)** — **deferred
  until a latency user exists or until the T3 rewrite forces the kernel
  signature open anyway.** Three-part rationale:
  1. **No identified latency scenario at this seam.** In every documented
     workload (Trainer-based training, batched encoding) the head runs
     behind a backbone forward at shapes where its GPU time is ≥1 ms and
     host launch work overlaps it. The 0.119 ms/call binds only for a
     small-shape, latency-critical caller of the head itself — nothing in
     the repo exercises or documents such a caller, and in an online-
     serving scenario the backbone forward would dominate the budget
     anyway.
  2. **The replacement is unproven against the incumbent.** An owned
     selector still pays the shared wrapper/op/autograd machinery
     (~0.013 ms/call wall-minus-GPU on the naive path — derived from v2
     Appendix A item 5: 0.035 wall vs 0.022 GPU) plus per-call
     winner-descriptor construction (~0.005 ms, scaling the measured
     0.051 ms / 22-descriptor rebuild to one pair) and a cache lookup.
     The paper estimate lands at ~0.018 ms against the ≤0.02 ms/call
     target — under it with no margin, and no prototype was ever built.
     The assumed ~6× host-overhead win could plausibly be ~2–3×, or miss
     the gate.
  3. With (1) and (2) unresolved, churning the validated production
     selection path is speculative-benefit churn — it fails the
     risk/benefit bar that every other task in this plan is held to.
  F9 stays a recorded finding; M12-T4 records the wall-minus-GPU host
  share per shape so the cost stays visible. **Revival triggers:** (a) a
  real latency/small-shape user appears; (b) the M12-T3 rewrite proceeds
  — fold the single-pair launch into the new kernel there rather than
  reviving this as a standalone task; (c) a Triton autotune-API change
  forces the mechanism open. **Gate design of record on revival:**
  prototype first and measure the actual host floor before committing to
  the ≤0.02 ms/call target (re-scope if the prototype misses it);
  deterministic candidate-set/ranking parity vs the autotune
  `early_config_prune` path on canonical keys (tiny-problem fallback
  collapse included); winner-time parity within a measured A-vs-A band —
  not exact selection match, which the ±5% jitter makes unachievable
  (§1.3 item 4); full suite + shape soak; grid/dev rows within ±5% of a
  same-session pre-change baseline; keep the multi-slot kernel until
  parity is proven; update v2 §4.2's status note at the switch. v3 §3
  M12-T1 carries the original mechanism spec.

Decided during M11 — do not re-litigate (evidence in the M11 memo):

- **B2b (Gluon/TMA backward port)** — rejected structurally: the binder
  was L2 reduction-sector volume, which TMA cannot touch; the gathered
  rows cannot use TMA; neither gradient contraction is MMA-shaped
  (memo §5.1).
- **Tensor-core one-hot `hidden_grad` accumulation** for short-S shapes —
  rejected on compute cost (`B·V·S·D` MACs regresses documents; memo §6).
- **Adaptive per-distribution kernel dispatch inside the backward op** —
  rejected: host sync + a second fallback seam (AGENTS invariant).
- **Bitwise-deterministic backward as a gate** — still rejected;
  determinism is a recorded metric and tiebreaker. The M11 swap improved
  it as a side effect (embed/bias now structural); revisit only on a user
  requirement.
- **Non-binary (weighted) mask gradients** — out of contract by maintainer
  ruling (M11 memo §9); supporting them is an extension (backward change +
  `head="torch"` autograd tests), not a bug fix. A value-scan validation
  was rejected (metadata-only validation layer).

Carried forward from v3 §6 (rationale unchanged):

- **Hybrid-side performance fixes** (v1 §9 items 7–9) — deferred
  indefinitely; hybrid is the compatibility path post-promotion.
- **Descriptor-bank caching** — still rejected, including now that
  launcher v2 is deferred: if the overhead ever matters, the right fix is
  the bank *removal* above, not caching the bank.
- **Kernel-body dedup (D2)** — **discharged by re-affirmation at the M12
  close, 2026-06-13** (no rewrite; cross-reference comments updated — M12
  memo §5). Re-open only if a future milestone reopens the kernel body.
- **Restricting hybrid to fp16/bf16** — rejected; fp32 hybrid is permitted
  legacy (`test_validation_allows_fp32_hybrid`).
- **CI setup** — out of scope (no CUDA runner); §4 is the gate mechanism.
- **WGMMA/TCGen05/FP8/clusters/TMA-gather/CUTLASS/split-K** — unchanged v1
  §9 rejections (hardware/scope; fatal aborts for the first two here).
- **Migrating custom ops to inferred schemas** — rejected (no
  `Optional[Tensor]` return support in schema inference).

Still deferred, new home in this document:

- **torch 2.7.1 floor suite run** (v1 §3.4 / v2 risks): one full suite run
  on the declared minimum torch before any release that advertises the
  floor. Not attached to M12/M13; it is a release gate, and no release is
  planned by this document.
- **Capturing a long-trained sparse checkpoint at full warmup** — only the
  shortened-warmup probe is in scope (M13-T0); a 10000-step training run
  is not justified by any current gate.

---

## Appendix A. Numbers provenance

Transcript paths under `/root/` are session-local artifacts of the runs
that produced them (the repo's self-containment rule post-dates them);
every row's regeneration recipe is the named script/command, and the
bundles of record now live in `tests/data/bundles/`.

| Number | Source (run of record) |
|---|---|
| Dev shape M10: fwd 0.900 / f+b 2.325 ms | M10 gate run 2, [M10 memo](sparton_milestone10_promotion_memo.md) |
| Dev floor 0.880 ms with same-run forward 0.898 ms (1.02×) | post-M8 review run, [v2](sparton_remaining_work_design_v2.md) §1.2 (re-quoted by the M9 memo; the M10/M11 gate runs quote no floor) |
| Dev shape M11: fwd 0.880 / f+b 1.949 / implied bwd 1.069 ms (and hybrid 1.152 / 2.262) | M11 gate run 2, [M11 memo](sparton_milestone11_backward_memo.md) §6 |
| Grid floor ratios: 1.099–1.117× (M12 run of record); 1.094–1.123× (M8 same-run); 1.054–1.113× (v3) | **current ratios of record: M12 memo §3** (`grid_bf16_run2.txt`); M8 memo grid `opt+b`/`gemm ms` columns (preserved same-run); v3 §1.2's range came from the M10 grid run whose per-row gemm column was **not preserved** — orients, may not gate |
| Grid implied backward 1.546–3.720 ms, −32…−41% vs M10 | M11 memo §6 table |
| Backward share ~17–52% grid / ~55% dev | **re-derived on one provenance at M12** (memo §3: 18.5–52.3% grid / 54.9% dev); originally derived mixed-run (M11 implied-bwd ÷ (M10 forward + M11 implied-bwd)) |
| Forward absolute gap per grid row | **M12 run of record** (memo §3): 0.126–1.628 ms/call (gap 8.97–10.51% of forward); supersedes the v4-draft anchors derived from the contested ratio range |
| Real-record backward 3.3–4.4 ms (legacy 5.90–7.93); speedups: queries 2.15–2.39×, steps150 docs 1.35–1.43×, synthetic 1.09–1.29× | M11 memo §5.3 / §6 gate ledger |
| L2 red sectors 96.57M→3.49M (dev), 330.63M→6.40M (corner), 408.03M→1.55M (real query) | M11 memo §6; transcripts `/root/profiles/m11/` |
| Residual: segmented kernel no unit >35% (real doc); embed kernel LTS 82.6% (dev) | M11 memo §6 (`bwd_real_r1doc_segmented.txt`, `bwd_after_dev_fp16.txt`) |
| Determinism: hidden-grad norm spread 1.14e-7; proxy 4.12e-6; embed exactly 0 | M11 memo §7 |
| Autotune selection jitter ±5% between processes | M11 memo §5.2/§5.3 |
| Host overhead 0.119 ms/call, bank 0.051 ms | review probe, [v2](sparton_remaining_work_design_v2.md) §1.3 + Appendix A item 5 |
| GEMM bring-up kernel 63.9% tensor pipe vs cuBLAS 86.6% (same occupancy/instruction family) | [v1](sparton_gluon_remaining_work_design.md) §3.5 — **GEMM benchmark kernel, not the production forward** |
| Distribution stats: f = 1.0000, top1 collision 0.184–0.198, query collision factor ≈10417 | M11 memo §3/§4 (capture bundles) |
| Suite 132 / quick 109; soak 384/384; smoke parity 0.21–0.24% (bf16) | M11 memo §6 gate ledger |
| M13 gather model: seg L2-read/red residuals ≤ 1% on four shapes; seg floor 1.04/1.34 ms vs 3.31 measured (doc) | [M13 memo](sparton_milestone13_backward_memo.md) §4 (model promoted to `scripts/m13_traffic_model.py`, output of record `tests/data/m13_traffic_model_out.txt`; counter transcripts were session artifacts) |
| M13 baselines: real-record bwd 3.29–4.32 ms (run 3 of record; run-2 contamination classified); A-vs-A ≤ 2.45% | M13 memo §3 (`bwd_baseline_run{1,2,3}.log`) |
| M13 split backward: real cells 1.46–1.60× vs segmented; grid implied bwd 1.430–3.522 ms (dev 0.982); synthetic f=0.10 −6…−16% | M13 memo §5.5/§5.6 (`bench_decision_fixed_run2.log`, `gate_ab_run2.log`, `gate_grid_run2.log`, `gate_dev_run2.log`) |
| Uniform-pass floor attainment: 1.37 ms vs 1.34 modeled (doc, ncu regime) | M13 memo §5.4 (`bwd_segv3_doc_r1.txt`) |
| Training-scale same-config band 16–38%; tier-2 parity holds | M13 memo §6 (`tier2_*_s42_run{1..3}.log`) |
| Embed-key audit: doc-tuned vs query-tuned 0.69% (immaterial) | M13 memo §2.2 (`embed_key_audit_*.log`) |

## Appendix B. Rerun commands

The standing-gate block in §4, plus:

- Forward profiling (M12-T0): ncu/nsys command shapes from v1 §11 against
  the new `scripts/ncu_forward_target.py` (NVTX `fwd_direct/`); backward
  profiling via `scripts/ncu_backward_target.py` (invocation in
  `scripts/README.md`); `TRITON_PRINT_AUTOTUNING=1` for selection
  logging.
- Sanitizer (M13-T2, if ownership semantics change): the three
  `compute-sanitizer` invocations recorded in the M11 memo §5.4.
- Host-overhead probe (M12-T4's host-share rows; revived launcher v2):
  the 300-call wall-vs-`do_bench` method of v2 Appendix A item 5 at
  `8×128×768×1280` fp16.
- M7 ratio gate on any Triton bump: `bench_gluon_gemm.py --dtype {fp16,bf16}
  --include-block-n-256 --require-ratio 85` plus `probe_gluon_epilogue.py`.
- Bundle regeneration: `capture_index_distributions.py --train-steps {0,150}`
  (defaults to `tests/data/bundles/swimir_de_steps{0,150}.pt`; reuses an
  existing file, `--force` to regenerate).
