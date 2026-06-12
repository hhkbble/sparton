# Sparton Remaining-Work Design v3 (post-M10)

Date: 2026-06-12.
Status: **superseded as the forward plan by
[sparton_remaining_work_design_v4.md](sparton_remaining_work_design_v4.md)
(2026-06-12, post-M11).** This document remains authoritative for the
executed **M11 plan of record** (§3 M11 — the specification whose deviation
ledger lives in the [M11 memo](sparton_milestone11_backward_memo.md)) and
for the post-M10 state snapshot and floor-ratio derivations in §1. The
incomplete forward work (M12) moved to v4 in refined form — re-grounded in
the post-M11 measurements, with a production-forward profiling entry task,
a jitter-aware launcher parity gate, and the new entry-gated backward
residual track (M13); its original wording here is retained for the record
but is no longer the specification.

Supersession chain: [v1](sparton_gluon_remaining_work_design.md) remains
authoritative for platform facts and the original measured evidence (sm_120
MMA matrix, Gluon API survey, environment defects, profiler notes, ncu
counters); [v2](sparton_remaining_work_design_v2.md) remains authoritative
for the post-M8 review findings (F1–F20), the design-evolution ledger, the
architecture rules of record (§4), and the executed M9/M10 specifications
and gates. This document carries the **incomplete** work forward — the
backward track and the forward scheduling/launcher track — refined with what
M9 and M10 measured and built. Method rules live in `AGENTS.md`; this
document does not restate them.

Everything numeric below was measured in the M9/M10 sessions of 2026-06-12
on the validation platform (RTX 5090, Triton 3.6.0, torch 2.12 nightly) or
is cited to the document that measured it (Appendix A).

---

## 1. Where we are after M10 (verified)

### 1.1 State summary

- `optimized` (Gluon TMA + `mma_v2` fused forward, policy-autotuned) is the
  **default backend** where available; default resolution falls back to
  hybrid with a one-time warning elsewhere; explicit selections never fall
  back (M10).
- All three backends share one Triton backward
  (`fused_sparton_bwd_kernel_with_bias`) — unchanged since M2's no-bias fix.
- AMP works on every backend (`autocast_canonicalize`, M10); the index
  contract, input validation, and wrapper symmetry are in place (M9).
- Suite: 114 tests (quick loop 98); standing gate scripts:
  `scripts/soak_optimized_correctness.py` (384-case shape soak) and
  `scripts/probe_training_smoke.py` (AMP training parity).
- Tier-2 training infrastructure exists and is locally cached:
  transformers 5.11.0 + accelerate 1.14.0 in the venv, xlm-roberta-base
  weights and the swim-ir `de` split downloaded, `training/train.py`
  smoke-validated (150-step runs, both backends, bf16 Trainer AMP).

### 1.2 Measured snapshot (M10 gate runs, warm second run)

Dev shape `B=32, S=128, D=768, V=30522`, fp16:

| metric | hybrid | optimized | optimized vs GEMM floor (0.880 ms) |
|---|---|---|---|
| forward + bias | 1.181 ms | 0.900 ms | 1.02× |
| fwd+bwd + bias | 2.642 ms | 2.325 ms | — |
| implied backward | ~1.46 ms | ~1.43 ms | — |

Canonical `splade-code-06B` grid (bf16, `D=1024, V=151936`): optimized
forward runs at **1.05–1.11× the per-row GEMM floor** (best 8×512 at 1.054×,
worst 16×768 at 1.113×); full table in the M10 memo.

**The backward is the largest remaining lever, and its share is
shape-dependent**: implied backward time (opt f+b minus opt fwd) ranges
1.43–5.4 ms across measured shapes — **61% of optimized fwd+bwd on the dev
shape, 24–64% across the grid** (largest share at small-S rows where the
forward is cheap relative to the `B·V` gradient work; smallest at 16×768).
This refines v2's single "~62%" figure; the priority conclusion stands.

Host-side launch overhead of the optimized forward (review probe, v2 §1.3 /
Appendix A): ~0.119 ms/call wall-minus-GPU at `8×128×768×1280`, of which
~0.051 ms is rebuilding the 22-descriptor bank. Irrelevant at ≥1 ms GPU
times; dominant for small/latency workloads.

Backward kernel counters (v1 §3.5, ncu, dev shape — the kernel is unchanged,
so they remain the structural evidence): 6.0% SOL compute / 9.6% DRAM,
167 MB read / 66 MB written — **latency/atomic-bound**, enormous headroom
without new hardware features.

### 1.3 What M9/M10 changed in this plan's assumptions

1. Promotion happened **before** the optimization tracks (v2's reorder,
   executed): both remaining milestones are now pure performance work on a
   promoted, gated default — they can be scheduled freely and must not
   regress the standing gates.
2. The tier-2 setup gives M11 something v2 could only wish for: **real index
   distributions from real tokenized batches** through a real backbone, with
   no new downloads.
3. M10 quantified backward atomic non-determinism (~20% same-config loss
   spread in chaotic training regimes — M10 memo): M11's harness must
   measure determinism, and aggregation work may improve it as a side
   effect worth recording.

---

## 2. Architecture rules in force

`AGENTS.md` carries the durable rules (layering, symmetry, one-seam changes,
no-silent-fallback with its single exception, index contract, op/schema
stability). Three rules are load-bearing for the milestones below:

- **Backward swap is schema-safe**: autograd `setup_context`/`backward` are
  backend-private, so a new backward op can be swapped inside the existing
  registrations without touching forward op schemas. A new op name
  (`sparton::optimized_bwd` or versioned) is required **if and only if** the
  saved-tensor set changes.
- **Policy-bank mechanics** (v2 §4.2) are the document of record for how the
  optimized forward selects configurations today; M12's launcher v2 replaces
  that mechanism and must update v2 §4.2's status note when it does.
- **D2 (kernel-body duplication** between the optimized forward and the GEMM
  benchmark) stays deliberately unresolved until M12's mainloop rewrite —
  that rewrite is the decision point: dedupe into a shared `@gluon.jit`
  helper only if the rewritten production mainloop and the benchmark still
  share structure; otherwise re-affirm the cross-reference comments.

---

## 3. Milestone plan

Numbering continues from v2. M11 before M12: the backward headroom
(latency/atomic-bound kernel at 6% utilization) dwarfs the forward's
remaining 5–11% gap to the GEMM floor, and M12's entry gate explicitly asks
whether the remaining forward gap is still worth chasing after M11 lands.

### M11 — Backward track (primary)

Update, 2026-06-12: **M11 is complete** — the shared backward is now the
three-kernel segmented backward (prep + exclusive-owner embed/bias kernel +
sorted segmented-scan hidden-grad kernel), swapped inside the unchanged
`sparton::fused_sparton_bwd` op; the pre-M11 kernel is retained as
`legacy_fused_sparton_bwd`, the A/B reference of record. Evidence, gate
ledger, and the analytic traffic model live in the
[M11 memo](sparton_milestone11_backward_memo.md). Two recorded deviations
from this section's spec: B2b was never built (the T3 counters showed the
binder is L2 reduction-sector volume, which TMA cannot touch and which has
no MMA shape — the B3/T5 mechanism was pulled into the decision probe
instead), and `steps150` document records land at 1.35–1.43× vs the 1.5×
exit clause (queries 2.15–2.39×, synthetic 1.13–1.30×, no cell regresses
anywhere — promotion proceeded with the deviation recorded; memo §8). The
T5 verdict and residual bottleneck are recorded at the end of the memo's
§6 gate ledger; pre-existing gaps surfaced by the milestone's adversarial
review (notably non-binary-mask gradients) are in memo §9.

Goal: replace or improve the shared Triton backward so that fwd+bwd drops
materially on realistic index distributions, without regressing the
correctness matrix, AMP behavior, or uniform-distribution performance.

#### M11-T1 Re-profile the baseline (entry evidence, half a day)

The v1 §3.5 counters predate M9/M10 (same kernel, older stack). Re-collect
on the current default path before designing: ncu on
`fused_sparton_bwd_kernel_with_bias` (direct op call on the main thread —
autograd's worker thread does not inherit NVTX ranges, v1 §2.5), dev shape +
one grid corner (`16×512`, bf16), recording SOL compute/DRAM, DRAM bytes,
atomic stall shares, and occupancy. Gate: numbers recorded in the M11 memo;
if utilization is no longer ≲10%, revisit this milestone's premise before
proceeding.

#### M11-T2 Distribution-aware backward harness

Uniform-random indices understate atomic conflicts on hot tokens; **no
backward change is accepted on uniform evidence alone** (v1 §8 rule,
restated). Build the harness before any kernel work:

- New `scripts/capture_index_distributions.py`: loads the cached
  xlm-roberta-base via `training/model.py` (`head="sparton"`), optionally
  fine-tunes for a configurable number of steps (default 150, the validated
  tier-2 recipe), then runs forward passes over real tokenized swim-ir `de`
  batches and saves per-batch `(hidden_shape, max_scores, max_idx, mask)`
  plus tokenizer metadata to a `.pt` bundle under a `--out` path. Follows
  the gate-script house shape (availability gate, `--quick`, summary line,
  non-zero exit). Rationale for capture-to-disk: backward benchmarking must
  not pay a backbone forward per measurement, and the bundle makes runs
  reproducible across sessions.
- New `scripts/bench_backward.py` (or an extension of
  `bench_sparton_baseline.py` if it stays small): times the backward op
  directly (build grad buffers, call `fused_sparton_bwd_op`-equivalent per
  backend-under-test) over three distribution sources × mask densities
  {25%, 75%, 100%}:
  1. uniform-random indices (today's tests' regime),
  2. Zipfian over vocab (`--zipf-s` default 1.1, documented as a proxy),
  3. captured real bundles from T1's script (untrained and 150-step-trained
     variants — training sharpens the distribution toward hot tokens).
- **Determinism protocol** (recorded metric, not a gate): per configuration,
  5 same-input repeats; report max relative spread of `embed_grad` /
  `hidden_grad` norms and of a fixed loss proxy. Establishes the
  non-determinism baseline that aggregation may improve (M10 memo residual).
- Gate: harness runs end-to-end on all three sources; baseline numbers for
  the current kernel recorded (these are the comparison targets for T3).

#### M11-T3 Decision probe: B2a (Triton aggregation) vs B2b (Gluon port)

Two prototypes, one decision, timeboxed (suggested: 2–3 focused days each;
stop early if one is clearly dominant on the harness):

- **B2a — Triton, same kernel family**: per-CTA aggregation of `d_bias` and
  `d_embed` in registers/shared memory before atomics (cutting atomic
  traffic by up to the `BLOCK_B` factor), wider/coalesced `BLOCK_D` loads,
  re-tuned block shapes for the current GPU. No new technology; the 6%-util
  kernel leaves room for a large win without leaving Triton.
- **B2b — Gluon port**: TMA loads for the gathered hidden rows and embed
  tiles, same aggregation strategy, `mma_v2` only where a real contraction
  exists (the backward is gather/scatter-shaped, so most gains come from
  memory orchestration, not MMA).
- Both prototypes must implement the full semantics: `g = grad_out *
  exp(-scores)` where `scores > 0`, `scores == 0 → zero gradient`, fp32
  accumulation, bias and no-bias.
- Decision criteria, in order: (1) measured backward time on the
  **captured-real** distributions at 75% mask density; (2) uniform-regime
  non-regression; (3) implementation/maintenance complexity (a 10% extra win
  does not justify a second Gluon kernel family if B2a gets close); (4)
  determinism delta as a tiebreaker.

#### M11-T4 Productionize the winner

- Wire the winning kernel behind the existing autograd registrations (all
  three backends share it, as today). Apply the op-naming rule (§2) if
  saved tensors change.
- Correctness gates: full gradient matrix vs the PyTorch reference and vs
  the current backward (bias/no-bias × fp16/bf16, tiny + non-tiny shapes,
  masked rows, `scores == 0 → zero gradient`); `probe_training_smoke.py`
  passes unchanged (AMP + GradScaler); full suite + shape soak green
  (forward untouched, but the soak is cheap insurance).
- Performance exit gates: backward ≥ **1.5× faster** than the current kernel
  on the captured-real harness configurations; no configuration (any
  source × density) regresses by more than 5%; fwd+bwd re-recorded on the
  dev shape and full grid (the `opt f+b ms` column) in the M11 memo.
- Determinism: report the protocol numbers before/after; document any
  improvement in the memo and `AGENTS.md`'s sharp-edge note if the ~20%
  band shrinks materially.

#### M11-T5 (conditional) B3 deeper aggregation

Only if T4's profiling still shows atomic conflict dominant: duplicate-index
`d_hidden` aggregation, segmented/sorted accumulation, or a two-pass
formulation. Entry requires harness evidence naming the remaining
bottleneck; same gates as T4.

### M12 — Forward scheduling and launch overhead (entry-gated)

Entry gate: after M11, re-examine whether ≥10% of forward time is
recoverable on shapes someone cares about (grid rows sit at 1.05–1.11× of
the GEMM floor; the dev shape at 1.02× is nearly done). If the answer is no,
record that in a short memo and close the milestone without kernel work —
launcher v2 (T1) may still proceed independently for small-shape/latency
users.

#### M12-T1 Launcher v2 (independent of the entry gate; fixes F9)

Replace the decorator-autotune + 22-descriptor-bank mechanism with an owned
two-phase selector:

- Selection: `derive_optimized_forward_policies` already ranks candidates;
  benchmark them once per `(B, S, D, V, dtype)` key with the existing
  `do_bench` infrastructure, memoize in-process (optional on-disk cache
  mirroring `cache_results=True` semantics).
- Launch: build **one** descriptor pair per call for the selected policy and
  invoke a single-pair kernel; delete the 22-arg bank, the `POLICY_ID`
  if-chain, and `make_descriptor_bank`'s padding logic once parity is
  proven. Keep the multi-slot kernel until then.
- Gates: identical policy selection to today's autotuner on the canonical
  grid + dev keys (assert by comparing selected `(policy, key)` pairs);
  host overhead ≤ **0.02 ms/call** measured by the v2 Appendix A probe
  method; full suite + shape soak green; benchmark rows within ±5%.
- Documentation: update v2 §4.2's status note (the policy-bank mechanics it
  documents are replaced) and the AGENTS Project Map line for
  `_gluon_policy_runtime.py` if helpers are deleted.

#### M12-T2 `gl.warp_specialize` probe (entry gate for T3)

Still unprobed on sm_120 (v1 §3.6 item 2). Subprocess-isolated probe in the
house shape (`probe_mma_matrix.py` pattern): compile + run a minimal
producer/consumer warp-specialized kernel; record register budgets and
barrier interplay. A fatal or unusable result demotes T3 to the
CTA-barrier persistent variant only.

#### M12-T3 Persistent (+ warp-specialized) mainloop

- Persistent flat tile loop with L2-aware rasterization; producer/consumer
  empty/full barriers replacing the per-iteration CTA `gl.barrier()`;
  epilogue unchanged. The O1 kernel's per-s-tile pipeline drain and
  embed-tile re-read disappear naturally in the persistent formulation —
  measure the actual gain, don't assume it.
- Targets: forward ≤ **1.05× the per-row GEMM floor on every grid row**
  (today: 1.05–1.11×); tensor-pipe utilization toward the 86.6% cuBLAS
  reference (v1 §3.5).
- This rewrite is the **D2 decision point** (§2).
- Gates: full suite, shape soak, smoke, benchmark grid; v1 §9 rejection
  criteria apply (>5% regression anywhere → rejected without a >10% win
  elsewhere); nsys shows a single kernel launch per forward; ncu DRAM reads
  within 1.3× the analytic A+B floor (v2's M9-era gate, still the right
  structural check).

#### M12-T4 Measurement set additions

Mask-density sweep (v1 §10.3) and small-shape/latency recordings (with and
without launcher v2) folded into the benchmark documentation of record.

### Beyond M12

Future-hardware and out-of-scope items are unchanged from v1 §9 (WGMMA /
TCGen05 / TMEM `load_max` on real sm_90a/sm_100 hardware, FP8 formats,
clusters, TMA gather/scatter for backward, CUTLASS, split-K). Revisit only
on a hardware or Triton-capability change, re-running the v1 probe matrix
first.

---

## 4. Validation matrix v3

Standing gates (every milestone exit, hardened env, serial):

```bash
ENV='TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas CPATH=/usr/local/cuda-13.2/include TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor'
PY=/workspace/venvs/sparton/bin/python
env $ENV PYTHONPATH=src $PY -m py_compile src/sparton/*.py training/*.py tests/*.py scripts/*.py
env $ENV $PY -m pytest -q                          # full, incl. slow (114 at M10 exit)
env $ENV $PY -m pytest -q -m "not slow"
env $ENV PYTHONPATH=src $PY -u scripts/soak_optimized_correctness.py        # forward-surface changes
env $ENV PYTHONPATH=src $PY -u scripts/probe_training_smoke.py              # autograd/AMP changes
env $ENV PYTHONPATH=src $PY -u scripts/bench_sparton_baseline.py --optimized-policy on   # grid of record
env $ENV PYTHONPATH=src $PY -u scripts/bench_sparton_baseline.py \
    --batch-sizes 32 --seq-lens 128 --dim 768 --vocab 30522 --dtype fp16 --optimized-policy on  # dev row
env $ENV PYTHONPATH=src $PY -c "import sparton"    # stdout-silent
git diff --check
```

M11 additions: the T2 harness across {uniform, zipf, captured-real} ×
mask densities {25, 75, 100}% × {fp16, bf16} × bias/no-bias; gradient matrix
vs reference **and** vs the current backward on tiny + non-tiny shapes;
`scores == 0 → zero gradient`; the determinism protocol (recorded). M12
additions: selection-parity assertion (launcher v2), nsys launch-count and
ncu DRAM-floor checks (persistent kernel), mask-density sweep.

Index and score tolerances are unchanged (v2 §6.2/§6.3 — the tie-aware
`assert_index_contract` remains the contract of record; backward changes do
not touch it but the soak re-run proves that).

---

## 5. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Captured-real distributions unavailable or unrepresentative | medium | capture script is part of M11-T2 with cached model/data (no downloads); Zipfian proxy documented; uniform never sufficient alone |
| B2a and B2b both miss the 1.5× gate | medium | gate is on realistic distributions where atomic relief is largest; if missed, record measured ceiling in memo and re-scope (B3 or accept) rather than promote a marginal kernel |
| `gl.warp_specialize` immature/fatal on sm_120 | medium | M12-T2 subprocess probe before any T3 work; CTA-barrier persistent variant is the fallback |
| Launcher v2 selects differently than the autotuner | low | explicit selection-parity gate on canonical keys before the bank is deleted |
| Triton upgrade moves Gluon APIs mid-milestone | medium | shim + `VALIDATED_TRITON` warning already in place; re-run the M7 ratio gate + epilogue probe + soak on any bump |
| Backward changes shift training numerics | low–medium | gradient matrix vs current backward, AMP smoke, and the determinism protocol quantify any shift; tier-2 150-step parity rerun if grads change beyond tolerance |
| Adaptive default on non-Gluon platforms drifts untested | low | unchanged M10 design; fallback-warning test pins it; no further fallback may be added (AGENTS invariant) |

---

## 6. Rejected / deferred (carried forward, with current rationale)

- **Hybrid-side performance fixes** (v1 §9 items 7–9: bias fusion,
  tile-count tuning, slice-copy elimination) — deferred indefinitely:
  post-promotion, hybrid is the compatibility path; optimization spend there
  double-pays. Reconsider only if promotion is reverted.
- **Descriptor-bank caching** — superseded by M12-T1 launcher v2 (removes
  the bank instead of caching it).
- **Kernel-body dedup (D2)** — deferred to the M12-T3 decision point (§2).
- **Restricting hybrid to fp16/bf16** — still rejected; fp32 hybrid is
  permitted legacy, validated by `test_validation_allows_fp32_hybrid`.
- **CI setup** — still out of scope (no CUDA runner); the Validation command
  set is the gate mechanism.
- **WGMMA/TCGen05/FP8/clusters/TMA-gather/CUTLASS/split-K** — unchanged v1
  rejections (hardware/scope; fatal aborts on this GPU for the first two).
- **Migrating custom ops to inferred schemas** — still rejected (no
  `Optional[Tensor]` return support in schema inference; explicit strings
  stay).
- **Bitwise-deterministic backward as a gate** — rejected for M11:
  atomic-free formulations cost more than the determinism is worth for
  training workloads; determinism is a recorded metric and a tiebreaker
  only. Revisit if a user requirement appears.

---

## Appendix A. Numbers provenance

| Number | Source |
|---|---|
| Dev-shape forward/fwd+bwd/floor (0.900 / 2.325 / 0.880 ms; hybrid 1.181 / 2.642) | M10 gate run 2, [M10 memo](sparton_milestone10_promotion_memo.md) |
| Grid rows and 1.05–1.11× floor ratios; backward share 24–64% (61% dev) | derived from the M10 grid table (same memo) |
| Host launch overhead 0.119 ms / bank 0.051 ms | review probe, [v2](sparton_remaining_work_design_v2.md) §1.3 + Appendix A item 5 |
| Backward kernel counters (6.0% compute, 167 MB read) | [v1](sparton_gluon_remaining_work_design.md) §3.5 (kernel unchanged since; re-verified at M11-T1) |
| Soak 384/384, smoke parity 0.03–0.24%, suite 114 | M10 memo validation ledger |
| Same-config training loss spread ~20% | M10 memo, tier-2 table |

## Appendix B. Rerun commands

The standing-gate block in §4 plus, for the new M11/M12 tooling once it
lands, the invocations documented in `scripts/README.md` (capture script,
backward harness, warp-specialize probe). Profiling command shapes are
unchanged from v1 §11 (ncu/nsys, hardened env, serial; backward profiled via
direct op calls on the main thread).
