# Milestone 13 memo — backward residual track: entry evidence and decision

Date: 2026-06-13.
Plan of record: [design v4](sparton_remaining_work_design_v4.md) §3 M13.
Entry evidence consumed: [M11 memo](sparton_milestone11_backward_memo.md) §6
(residual-bottleneck note), [M12 memo](sparton_milestone12_forward_memo.md)
§3 (one-provenance grid/dev table — the shared baselines and the forward
column of the cross-track comparison).

All `do_bench` figures are means (Triton 3.6 default). Latency-of-record
cells come from `triton.testing.do_bench` at the op level; ncu durations are
serialized and appear only as structure/counter evidence; nsys is the
single-regime source for sort/fill attribution. Transcripts:
`/root/profiles/m13/` (ncu/nsys/IR), `/root/m13_runs/` (bench/training
logs).

## 1. Decision

**T0 verdict: GO — the pre-registered ≥10% rule passes by ~4× margin.**
The validated traffic model (§4; per-buffer residuals ≤ 1% on the
decision-carrying counters) prices the segmented hidden-grad kernel at
3.2× its L2-traffic floor on the captured-real doc record (3.31 ms
measured vs 1.04/1.34 ms strict/conservative floor), with a named,
SASS-confirmed mechanism: the gather load's vector width is
layout-coupled to the `tl.cumsum` tile, so every autotune config either
scalarizes the gather (`ld.global.b32`, L1TEX issue-bound at 66–71%) or
pays 255 registers for one CTA/SM (latency-bound at 16.6% occupancy).
Candidate-relative recoverable time on the doc record: **1.63 ms
conservative = 38.1% of the measured 4.27 ms backward** (§5.2). One
candidate survived sizing and T1 evolved it into the promoted design
(§5.4): the **split segmented backward** — a branch-free, pipelined,
vectorized streaming-reduction kernel for single-destination chunks
(variant profile: 1.37 ms vs the 1.34 ms modeled conservative floor on
the doc record; the promoted kernel's own profile: 1.43 ms at 56
regs/73.5% occupancy with the mixed pass at 370 µs — §5.5 E5 row) plus the segmented scan covering only run-boundary
chunks, complementary at one shared 64-entry granularity. **T2 promoted
it** (schema-safe swap inside `sparton::fused_sparton_bwd`): captured-real
records improve 1.46–1.60× over the M11 segmented design (steps150 docs
1.568–1.596× — the M11 deviation cells now clear even the old 1.5×
clause), every canonical grid row holds or improves (dev implied backward
1.061 → 0.982 ms), and the synthetic `f = 0.10` short-run cells regress
6–16% (~45 µs/call) — a recorded deviation with a named mechanism,
sanctioned by v1 §9's regression-without-gain criterion (§5.5, §8). The
baton passed: the M11 segmented kernel is the test-pinned A/B reference
behind `legacy_fused_sparton_bwd`; the M2-era atomic kernel is deleted.
The embed kernel was *not* touched: fresh counters show it already at
its traffic floor, and they correct the M11 §6 attribution (its binder
is gather re-reads, not g/idx streams). Residual bottleneck at exit, for
any future milestone (Loop step 10): the uniform pass runs at LTS ≈ 61–67%
(config-dependent) against the embed kernel's demonstrated 82–104% — the remaining gap is
latency exposure in the persistent for-loop's serialized
keys→uniformity→tile chain plus the short-run regimes (dense-dev,
synthetic-sparse) where the mixed fraction caps the split's benefit; the
backward's floor-level term is now the embed kernel itself plus
`torch.sort` (≈ 3.2% of the doc op).

## 2. T0 entry evidence — fresh counters on the promoted segmented backward

Direct-op profile via `benchmarks/ncu_backward_target.py` (NVTX
`bwd_direct/`, main thread, `--launch-skip 4 --launch-count 4` over the
four-kernel regex `bwd_prep_kernel|embed_grad_kernel|
bwd_gather_payload_kernel|segmented_hidden_grad_kernel` — the
`benchmarks/README.md` invocation's `regex:sparton_bwd` matches only the
retired legacy kernel name and profiles nothing on this family; corrected
at close). Metric set = the M11 §2 union plus M13 additions
(`lts__t_sectors_op_{read,write,atom}`,
`l1tex__t_sectors_pipe_lsu_mem_global_op_ld`,
`smsp__inst_executed_op_global_ld`, local-memory bytes for spill checks,
`--section Occupancy`). Transcripts:
`/root/profiles/m13/bwd_{dev_fp16,corner_16x512_bf16,real_q_steps0r0_fp16,real_doc_steps150r1_fp16}.txt`.
ncu durations are serialized (structure only); the per-call shares below
them come from the nsys pass (§2.1).

**Segmented hidden-grad kernel** (the dominant kernel on every shape):

| metric | dev fp16 | corner bf16 | query r0 fp16 | doc r1 fp16 |
|---|---|---|---|---|
| duration (ncu) | 765 µs | 2.59 ms | 2.80 ms | 3.31 ms |
| selected config (grid → blocks) | (4096,6)×128 → BD=128, 4w, 168 regs | (4096,16)×128 → BD=64, 4w, 168 regs | (4096,12)×256 → BD=64, 8w, 255 regs | (4096,6)×128 → BD=128, 4w, 168 regs |
| achieved / theoretical occupancy | 24.7 / 25% | 24.8 / 25% | 16.6 / 16.7% | 24.8 / 25% |
| SM / L1TEX / LTS / DRAM throughput % | 37.7 / 71.3 / 30.0 / 4.5 | 48.0 / 52.0 / 29.8 / 7.6 | 25.5 / 30.4 / 33.1 / 13.5 | 29.6 / 66.6 / 28.2 / 11.4 |
| L1 global-ld sectors / LTS read sectors | 66.6 M / 48.9 M | 226.4 M / 170.2 M | 211.7 M / 209.9 M | 268.2 M / 201.3 M |
| L2 hit rate / red sectors | 95.9% / 3.49 M | 93.3% / 6.40 M | 88.8% / 1.55 M | 89.0% / 12.46 M |
| DRAM read | 68.2 MB | 372.4 MB | 750.8 MB | 746.6 MB |
| global-ld warp instructions | 20.4 M | 70.9 M | 18.0 M | 79.2 M |
| stall: long_scoreboard (cyc/issue) | 5.4 | 4.7 | 4.9 | 10.0 |
| local-memory bytes (spills) | 0 | 0 | 0 | 0 |

Two structural facts the M11 note could not see:

1. **The autotuner's selected config moved on the doc key since M11** (the
   M12 cache re-key forced a re-tune): M11's doc transcript recorded the
   255-reg 8-warp `BLOCK_D=64` config at 16.6% occupancy (3.27 ms); today
   the doc key selects the 168-reg 4-warp `BLOCK_D=128` config at 24.8%
   occupancy — and lands at 3.31 ms, the same time within jitter. Two
   configs from different occupancy classes (16.6% vs 24.8%) deliver the
   same duration: occupancy alone is demonstrably not the lever between
   these operating points (constraining candidate (c), §5).
2. **No unit saturates on any shape, but the highest unit is not the one
   M11 named.** On the currently-selected dev/doc configs the top unit is
   L1TEX at 66–71% (the LSU/sector-request pipe), with LTS at 28–30% and
   DRAM ≤ 11.4%; the M11 "no unit above ~35%" reading was taken on the
   255-reg config (whose L1TEX is 30%, reproduced today on the query
   shape). Bytes-per-load arithmetic (L1 sectors × 32 B ÷ ld
   instructions) says the two config families lower the gather to
   different widths — doc/dev family ≈ 108 B/warp-inst (≈ 3.4 B/thread:
   narrow loads), query family ≈ 375 B/warp-inst (≈ 11.7 B/thread: wide
   loads) — settled by reading the SASS (§2.2), not inferred further from
   counters.

**Embed-grad kernel** (second-largest; exclusive-owner stores):

| metric | dev fp16 | corner bf16 | query r0 fp16 | doc r1 fp16 |
|---|---|---|---|---|
| duration (ncu) | 214 µs | 759 µs | 626 µs | 895 µs |
| selected config (grid → blocks) | (12,477)×256 → BV=64, BD=64, 8w, 128 regs | (16,4748)×128 → BV=32, BD=64, 4w, 127 regs | (12,7813)×128 → BV=32, BD=64, 4w, 128 regs | (12,3907)×256 → BV=64, BD=64, 8w, 128 regs |
| LTS throughput / hit rate | 82.2% / 91.6% | 104.5% / 88.5% | 57.4% / 68.9% | 84.7% / 85.3% |
| L1 global-ld sectors / LTS read sectors | 49.3 M / 37.1 M | 164.7 M / 158.5 M | 157.3 M / 56.5 M | 200.8 M / 147.3 M |
| DRAM read / write | 12.6 / 53.2 MB | 32.2 / 583.9 MB | 32.6 / 726.7 MB | 36.3 / 727.9 MB |
| achieved occupancy | 32.6% | 32.7% | 32.3% | 32.6% |

The fresh sector split **corrects the M11 §6 attribution**: the embed
kernel's L2 read traffic is dominated by the *hidden-row gather*, not by
the g/idx streams. On the doc record the gather accounts for ≈ 147 M of
the 171 M total LTS sectors; the g/idx stream re-reads
(`N·8 B·⌈D/BLOCK_D⌉` = 384 MB = 12 M sectors at the selected
`BLOCK_D=64`) are ≈ 7% of L2 reads. The kernel *is* L2-throughput-bound
(84.7% on doc, 104.5% on the corner — both near the fabric's achievable
rate), but on gather re-reads of `hidden` rows (requested `f·N·D·elt`,
compulsory only `B·S·D·elt` — re-read factor ≈ V_active/S), which only
caching, not d-tile stream hoisting, can absorb. Candidate (d)'s embed
half is re-priced accordingly (§5).

Prep and payload kernels: 12–50 µs everywhere (prep DRAM-bound at 56–84%,
payload latency-bound at 40–87% occupancy with `long_scoreboard`/
`lg_throttle` stalls — both ≤ 1.2% of the op, §2.1; neither is a lever).

### 2.1 Per-call shares in one regime (nsys, doc record)

`nsys` pass over `ncu_backward_target.py --launches 8` on the doc record
(`/root/profiles/m13/bwd_inventory_doc.{nsys-rep,sqlite,txt}`); per-call
averages over the 8 measured launches (the NVTX window catches only the
first async call, so attribution uses the last-8 instances per kernel):

| component | µs/call | share |
|---|---:|---:|
| `segmented_hidden_grad_kernel` | 2969 | 73.7% |
| `embed_grad_kernel` | 849 | 21.1% |
| `torch.sort` (CUB onesweep×4 + histogram + exclsum + iota) | 131 | 3.2% |
| `bwd_gather_payload_kernel` | 40 | 1.0% |
| `bwd_prep_kernel` | 34 | 0.8% |
| `hidden_grad` zero-fill | 3 | 0.1% |
| GPU total | 4026 | 100% |

Correction to the M11 sort note: "`torch.sort` 10–25 µs" was measured at
the dev shape (≈ 1 M keys); at the real-record 4 M keys the sort family
costs 131 µs/call — still only 3.2% of the op, not a lever. The op-level
`do_bench` doc cell (≈ 4.27–4.32 ms, §3) sits ≈ 6% above the nsys GPU sum
(launch gaps for the 9-kernel-per-call chain plus L2-flush effects);
regimes are kept separate as always.

### 2.2 Autotune-key audit (Loop step 7): embed kernel's missing `seq_len`, priced

`embed_grad_kernel`'s key is `['batch_size', 'vocab_size', 'hidden_dim']`
(`_backend_hybrid.py:490`) — no `seq_len` — and real query/doc records
share `(B, V, D)`, so whichever side autotunes first serves both. The
config-capture probe confirmed it structurally (the doc call cache-hits
the query-tuned entry, `bwd_selected_configs.txt`): the harness's
record ordering gives doc cells the query-tuned config (BV32/4w/s3),
while a doc-only process tunes BV64/8w/s2. Priced at the op level
(`/root/m13_runs/embed_key_audit_{docfirst,queryfirst}.log`): doc-first
4.055 ms vs query-first 4.083 ms — **0.69%, inside the 2.45% A-vs-A
band**. Classified: real structural sharing, immaterial cost — the embed
kernel's tile economics don't depend on S (S enters only the gather
stride), unlike the segmented kernel where the same key omission cost
~7% (M11 §5.2). No fix; recorded here as the audit of record.

## 3. Re-measured op-level baselines (decision denominators)

Three consecutive `bench_backward.py --impls current --sources
uniform,zipf,real` runs (both M11 bundles, `--active-fraction 0.10`,
`--determinism`; 88 cells each, 0 failures), teed to
`/root/m13_runs/bwd_baseline_run{1,2,3}.log`.

**Run-2 contamination, classified:** run 2 carried two cells far outside
every other run's band — `steps0:r4:doc` bf16/bias-on at 13.817 ms (runs
1/3: 3.883/3.883 ms) and `steps0:r0:query` fp16/bias-on at 3.861 ms (runs
1/3: 3.330/3.329 ms). `nvidia-smi` showed no other compute process;
runs 1 and 3 agree on every one of the 88 cells to ≤ 2.45%. Verdict:
transient host interference during run 2 (the M11 memo §8 item 5 failure
class), not a property of the code or data. **Run 3 is the run of
record**; the A-vs-A noise band is max 2.45% / median 0.150% (run 1 vs
run 3, 88 cells), under the standing ±5% autotune-jitter umbrella.

Run-3 ranges (per bundle/side, over {fp16, bf16} × {bias on, off}):

| source | cells | op backward ms |
|---|---:|---|
| steps0 queries (B16, S24–40) | 16 | 3.285–3.363 |
| steps0 docs (B16, S192–256) | 16 | 3.876–4.011 |
| steps150 queries (B16, S24–40) | 16 | 3.292–3.362 |
| steps150 docs (B16, S192–256) | 16 | 4.235–4.318 |
| synthetic uniform+zipf (dev shape, f=0.10) | 24 | 0.298–0.327 |

Consistent with the M11 gate ledger's 3.3–4.4 ms real-record family.
Determinism protocol (5 same-input repeats per cell, maxima over run 3):
`‖hidden_grad‖` spread ≤ 1.14e-07, `‖embed_grad‖` spread exactly 0, loss
proxy ≤ 2.60e-06 — the M11 §7 band reproduces; this is the "before" side
for any T2 comparison.

Grid/dev denominators are **reused** from the M12 memo §3 one-provenance
table (implied backward 1.543–3.822 ms across the canonical grid, 1.061 ms
dev; `/root/profiles/m12/grid_bf16_run2.txt`, `dev_fp16_run2.txt`),
sanctioned by v4 §3 item 1 — same-day provenance, and the backward op is
unchanged since.

## 4. Analytic model — gather traffic and time floors

Deposited as a runnable script with its output
(`/root/m13_runs/probes/m13_traffic_model.py`,
`/root/m13_runs/m13_traffic_model_out.txt`). All distribution statistics
(destination-run count, mixed-chunk fraction, active fraction) are computed
from the *actual inputs* — the same seeds and bundle records the §2 ncu
runs used — never estimated. Notation: `N = B·V`, `elt` = 2 B (fp16/bf16),
`T_d = ⌈D/BLOCK_D⌉`, sector = 32 B; `chunks` = live chunks `⌈f·N/CHUNK⌉`,
`runs` = distinct destination keys among active entries, `m` = chunks whose
first and last key differ.

### 4.1 Per-buffer formulas and validation

Segmented hidden-grad kernel:

| buffer | formula (sectors) |
|---|---|
| L2 read | gather `f·N·D·elt/32` + streams `3·(N·4/32)·T_d` + prev/next `2·(m·CHUNK·4/32)·T_d` |
| L2 red | `(chunks + 2·runs)·D/8` — one chunk-partial atomic per live chunk per d-column plus ≤ 2 run-boundary partials per run |
| DRAM read (compulsory) | embed table `V·D·elt` + streams once `3·N·4`; measured/compulsory = κ, the L2-capacity re-fetch factor (reported, not predicted) |

Embed-grad kernel:

| buffer | formula |
|---|---|
| L1 load sectors | hidden gather `f·N·D·elt/32` + g/idx streams `(N·8/32)·T_d` |
| L2 write sectors | stores `V·D·4/32 + V·4/32` (exclusive-owner, exact) |
| DRAM write | `V·D·4` modulo the L2-resident writeback tail (lazy eviction) |

Expected-vs-measured (fresh §2 counters; full table in the deposited
output):

| shape | seg L2 read | seg red | emb L1 load | emb L2 write |
|---|---:|---:|---:|---:|
| dev fp16 | +0.7% | +0.8% | +1.1% | −0.0% |
| corner bf16 | +0.9% | +0.5% | +0.4% | −0.0% |
| query r0 | +0.1% | +0.0% | **+29.7%** | −0.0% |
| doc r1 | −0.1% | +0.2% | +1.6% | −0.0% |

Every formula meets the pre-registered ≤ 5% bar except the embed kernel's
L1 sectors on the query record, whose +30% residual has a named mechanism:
with S = 24 and idx-top1 share 0.33, gathered hidden rows collide so
heavily that lanes within one warp-instruction hit the *same* sector and
coalesce at request time — the byte model over-counts requests exactly
where collisions are strongest (measured L1→L2 gather absorption α = 0.77
on the query vs 0.27–0.30 on dev/doc, 0.04 on the corner). The red-sector
formula `(chunks + 2·runs)·D/8` lands ≤ 0.8% on all four shapes — the M11
formula's "2 per run" is completed by the chunk-partial term, which
dominates when runs are long (`L ≫ CHUNK`): partial-sum atomics are per
*chunk*, not per run. κ (DRAM re-fetch): 1.45 dev (47 MB table; mostly
L2-resident, residue is sort-buffer traffic), 1.20 corner (311 MB table),
1.94–1.96 real records (384 MB table ≫ 96 MB L2). DRAM-write residuals on
the embed kernel (+5.5–6.6% real/corner, +76% dev) are the L2-resident
writeback tail — at dev scale `embed_grad` (94 MB) largely never leaves L2
inside the measurement window.

### 4.2 Time floors

Per kernel `t_floor = max(L2 term, DRAM term)` with measured-anchored
rates: L2 fabric **6.6 TB/s** (M12 memo §4's measured rate at 89–91%
busy; the corner embed kernel actually clears this anchor by 12% — the
floor is conservative in the safe direction), DRAM **1.5 TB/s** (the prep
kernel demonstrates 84% of nameplate in this same kernel family). A
conservative-attainment variant prices L2 at 5.1 TB/s (≈ 70% of implied
peak) for the decision arithmetic. ncu durations are used only as the
same-regime comparison column.

| kernel @ shape | t_floor (strict / conservative) | measured (ncu) | headroom |
|---|---|---|---|
| segmented @ doc r1 | 1.04 / 1.34 ms | 3.31 ms | **2.0–2.3 ms** |
| segmented @ query r0 | 1.03 / 1.33 ms | 2.80 ms | 1.5–1.8 ms |
| segmented @ corner | 0.86 / 1.11 ms | 2.59 ms | 1.5–1.7 ms |
| segmented @ dev | 0.25 / 0.33 ms | 0.77 ms | 0.4 ms |
| embed @ doc r1 | 0.83 / 1.08 ms | 0.89 ms | **at floor** |
| embed @ query r0 | max(0.39, 0.51 DRAM) | 0.63 ms | ~0.1 ms (DRAM-bound: V·D·4 store) |
| embed @ corner | 0.86 / 1.12 ms | 0.76 ms | beats the anchor |
| embed @ dev | 0.19 / 0.25 ms | 0.21 ms | at floor |

Two facts carry the decision:

1. **The embed kernel is the existence proof for the segmented kernel's
   headroom.** It is a *gather-dominated* workload (≈ 86% of its L2 reads
   are gathered hidden rows, §2) running at 82–104% LTS with `LDG.E.128`
   loads at 33% occupancy — i.e., on this GPU, this op's gather pattern
   demonstrably saturates the L2 fabric when the loads are wide and
   enough CTAs are resident. The claim "the segmented kernel can approach
   its L2 floor" therefore rests on a measured sibling, not on theory.
2. **Why the segmented kernel sits at 3.2× its floor — the
   vectorization⇔occupancy coupling** (§2 + SASS/TTGIR evidence,
   `/root/profiles/m13/ir_dump/`): the gather load's vector width is
   layout-coupled to the `tl.cumsum` tile. Configs with more than one
   thread per CHUNK-row scalarize the gather to `ld.global.b32`
   (43 scalar sites; ≈ 78 useful gather bytes per warp-instruction,
   derived as modeled gather bytes ÷ measured instructions — the
   request-side sector arithmetic of §2 gives 108 B/inst for the same
   kernel; 4× the instructions for the same bytes either way, L1TEX pipe
   at 66–71%), and the *only*
   config shape that vectorizes (one thread per row: CHUNK = threads,
   e.g. 256/BD64/8w) must hold `BLOCK_D` floats per thread — 255
   registers → 1 CTA/SM → 16.6% occupancy → latency-bound at L1TEX 30%.
   The autotuner is choosing between two failure modes of the same
   structure; both land ≈ 3.3 ms on the doc shape (M11's 255-reg
   selection at 3.27 ms; today's 168-reg selection at 3.31 ms). The
   measured equivalence of the 16.6%- and 24.8%-occupancy operating
   points also kills occupancy-alone as a lever (§5, candidate (c)).

## 5. Candidate sizing and the decision walk

### 5.1 The four v4 §3 hypotheses, priced by the model

Each candidate was sized against the validated model **before any was
built**; kills are justified by a number, and the walk is recorded even
for kills (M11 §5.1 discipline).

**(a) Wider/vectorized gather loads, within the current kernel
structure — KILLED by evidence already on file.** The SASS census
settles the lowering: in the entire 8-config family exactly one config
vectorizes the gather (CHUNK=256/BD64/8w: one thread per CHUNK-row,
`ld.global.v4.b32`), and it costs `BLOCK_D` floats of live cumsum state
per thread — 255 registers, 1 CTA/SM. Every other config (≥ 2 threads
per row, including the in-list 128/64/8w) scalarizes to `b32`. The
autotuner has effectively *already run this experiment per shape*: M11's
doc selection was the vectorized config (3.27 ms), today's is the scalar
config (3.31 ms) — equal within jitter. Vectorization inside this
structure just trades the L1TEX-issue wall for the occupancy wall.

**(b) Fuse `bwd_gather_payload_kernel` into the segmented kernel —
KILLED by arithmetic.** The payload kernel is 40 µs = 1.0% of the doc
op (§2.1). Fusing removes it and one `N·8 B` round trip but makes the
segmented kernel resolve `perm` per d-column: `N·8·T_d` extra int64
loads plus randomly-ordered 4 B `g` gathers (sector-inflated 8×) —
≈ +27% segmented-kernel L2 bytes on the doc shape against a ≤ 1.2%
saving. Net negative at every `T_d > 1`; also blocked as a sort-key
payload pack by the int32 key width (`B·S` bound asserted in
`_bwd_shared_stages`).

**(c) Occupancy/register work alone — KILLED by a measured
equivalence.** The two operating points the autotuner alternates
between — 255 regs/16.6% occupancy and 168 regs/24.8% occupancy —
time identically on the doc shape (3.27 vs 3.31 ms across the M11/M13
runs of record). A +50% occupancy change with no vector-width change
moved nothing; `maxnreg` variants of the same structure have no priced
mechanism to do better.

**(d) d-tile blocking — KILLED on both kernels by the corrected
attribution.** In the segmented kernel the gather does not multiply
with `T_d` (disjoint slices); only the keys/g/v streams do, and they
are 4.5–9% of its L2 bytes — halving them is worth ≤ 3% of the kernel.
In the embed kernel, §2's sector split corrected M11's premise: the
g/idx streams this candidate hoists are ≈ 7% of L2 reads (the gather
dominates), and the kernel already runs at its floor (0.89 vs 0.83 ms
on doc) — there is nothing for stream-hoisting to recover. Composable
freebie if a winning structure happens to permit it; not a candidate.

**(a)+(c) jointly — the one KEEP: `seg_v2`, a restructure that breaks
the vectorize⇔occupancy coupling.** The mechanism, named: assign
CHUNK-rows to *warps* (not single threads), lanes split the row's
`BLOCK_D` columns in `v4` vectors, cumsum becomes hierarchical
(in-lane sequential over the warp's rows, cross-warp prefix via a small
shared-memory exchange) — wide loads at tens-of-registers state, ≥ 3
CTAs/SM. Ceiling: the kernel's traffic floor (§4.2) — conservative
1.34 ms on the doc record vs 3.31 measured. The embed kernel is the
existence proof that this op's gather pattern saturates the fabric
when issued wide at ≥ 33% occupancy (§4.2 item 1). Atomic count is
CHUNK-preserving (the `(chunks + 2·runs)·D/8` formula keys on CHUNK,
not on the intra-chunk thread layout), so the M11 red-sector win is
untouched by construction.

### 5.2 Decision rule applied (pre-registered in v4 §3)

Rule: proceed to T1/T2 only if the validated model says ≥ 10% of
measured backward time is recoverable on ≥ 1 canonical grid row or
captured-real record, AND M12-T3 is complete or closed (it closed
2026-06-13). Recoverable is computed candidate-relative: kernels the
candidate does not touch are charged at their *measured* per-call cost,
only the segmented kernel drops to its modeled floor.

Doc record steps150:r1 (the rigorous chain — every component from the
§2.1 nsys pass; the op denominator is the §3 run-of-record cell for the
*profiled* record, which the harness labels `steps150:r4:doc:B16xS192`
fp16/bias-on = 4.269 ms — a review-pass correction: the figure first
quoted here, 4.288 ms, belongs to a different doc cell; the swap moves
the share by +0.1 pp in the non-flattering direction):

| variant | T_model(op) | recoverable | share of backward |
|---|---:|---:|---:|
| conservative (seg → 1.341 ms, L2 @ 5.1 TB/s) | 2.641 ms | 1.628 ms | **38.1%** |
| strict floor (seg → 1.036 ms, L2 @ 6.6 TB/s) | 2.336 ms | 1.933 ms | 45.3% |

(T_model(op) = nsys GPU sum 4.026 with the segmented kernel replaced by
its floor, plus the measured 0.243 ms do_bench-vs-GPU-sum gap;
denominator 4.269 ms, run 3.)

Corroborating estimates (segmented-kernel share scaled from the same-
regime ncu sums; labeled estimates, not runs of record): query r0
≈ 1.3 ms recoverable ≈ 40%; dev grid row ≈ 0.48 ms ≈ 45% of the M12
implied-backward 1.061 ms; grid 16×512 (the counter-validated corner)
≈ 1.7 ms ≈ 46% of 3.704 ms.

**Decision: GO.** The rule passes by ~4× margin on the rigorous shape
and on every estimated one. T1 builds `seg_v2` only; (a), (b), (c), (d)
are closed above with their numbers.

### 5.3 T2 exit numbers (fixed now, from the conservative model) and T1 early-stop

Pre-registered before any candidate code exists:

- **E1** steps150 doc cells (the M11 1.35–1.43× deviation cells):
  op-level ≥ **1.45×** vs the §3 run-3 baseline (conservative op floor
  2.66 ms + 9% engineering margin on a 4.29 ms baseline).
- **E2** query cells ≥ **1.25×**; **E3** steps0 doc cells ≥ **1.40×**.
- **E4** no synthetic cell regresses > 5% (v1 §9 criterion); dev-row
  implied backward improves ≥ 1.20×; no canonical grid row regresses
  > 5%.
- **E5 mechanism gate** (Loop step 10): the winner's ncu on the doc
  record shows the replacement kernel at LTS ≥ **65%** (vs 28.2%
  today) with ≥ 64-bit gather loads in SASS — a time win without the
  predicted mechanism does not promote.
- **E6** determinism: `embed_grad` spread exactly 0; loss-proxy spread
  ≤ 1.0e-5 (M11 §7 band family; §3 baseline 2.6e-6).
- **E7** machinery gates unchanged from v4 §3 M13-T2 (gradient matrix,
  A/B vs segmented, AMP smoke, full suite, all-sources harness run ×2,
  sanitizer if ownership semantics change).

**T1 early-stop (pre-registered):** `seg_v2` gets at most three
structural variants (warp-row ownership with smem cross-warp prefix;
in-lane multi-row sequential scan variant; plus a config sweep of the
winner). If the best variant does not reach ≥ 1.25× on the doc-record
`--quick` cells, kernel work stops, the attainment gap vs the model is
recorded as the milestone's finding, and the track closes without
promotion — the floor stands as the documented ceiling either way.

### 5.4 T1 execution: the variant walk

All variants live in `benchmarks/bwd_prototypes.py` (production-op
signature, per-cell verified vs `current` before every timing row);
quick-cell logs `/root/m13_runs/bench_segv{2,3}*_quick*.log`, kernel
profiles `/root/profiles/m13/bwd_segv{2,2b,3,31}_doc_r1.txt`. Every accepted step has a
profiler-confirmed mechanism; every rejected one a recorded number.

1. **v1 `seg_v2` — branch-local loads: 1.12–1.13× doc, insufficient.**
   The hot uniform branch's tile load was given its own SSA value so the
   scan could not anchor it. A branch-A-only compilation artifact proves
   Triton emits the fully-vectorized form for exactly this code
   (`ir_dump/segv2__*`), and red-sector counts prove branch A executes —
   but the executed full-kernel instruction mix stayed scalar-class
   (80.36 M ld warp-instructions, ≈ 107 B/inst, L1TEX 52.6%): inside the
   live branch context the vectorized lowering does not survive to
   execution. Time moved 3.31 → 2.84 ms (ncu) and no further. Lesson
   recorded: branch-local SSA separation is not layout separation.
2. **v3.0 — split kernels (uniform-stream + mixed-scan): killed by its
   own first quick run, instructively.** Verification failed on every
   non-tiny cell: the two kernels' uniform/mixed predicates complement
   only at a *shared* chunk granularity, and each autotuner had selected
   its own CHUNK — chunks the two granularities classified differently
   were silently dropped (the harness's verify-before-time gate caught
   it; a small-shape repro passed because both tuners happened to agree
   there). Fix: the mixed kernel is not autotuned; it runs at the uniform
   kernel's `best_config` granularity.
3. **v3.0-profile — the streaming kernel reaches the model's floor.**
   On the doc record `seg_v3_uniform_kernel` ran **1.37 ms at 64
   regs/thread, 66% occupancy, LTS 67.6%, ≈ 376 B per load-instruction**
   (wide loads executing) — within 3% of the §4.2 conservative floor
   (1.34 ms) and 2.4× faster than the production kernel. The branch-free
   `for`-loop form is what unlocked it (pipelinable, reduction-anchored
   layout). The op, however, gained only 1.05×: the mixed pass cost
   1.86 ms — it had inherited the uniform winner's 255-register shape
   and replicated its granule walk per d-column.
4. **v3.1 — 1D mixed walk + device-side live bound: query 1.47×, doc
   1.14×, sparse synthetic −33%.** The mixed pass became a 1D granule
   walk with an inner d-loop (keys traffic ÷ T_d) and sub-tiled scans —
   legal because chunk-local partials compose across tile boundaries
   (the production run-boundary invariant). A prototype prep kernel
   gained a one-atomic active-entry counter so the uniform pass walks
   `⌈n_active/CHUNK⌉` chunks instead of the whole sentinel suffix (no
   host sync; the v3.0 sparse cells had regressed 2.5× on that walk).
   Doc stayed low because the mixed pass still cost 1.30 ms at 255 regs
   doing ~3% of the work (`bwd_segv31_doc_r1.txt`).
5. **Granule pinned to 64 — the unlock: doc 1.48×.** The mixed fraction
   scales with the shared granule (`m ≈ runs·CHUNK/N`), so letting the
   uniform tuner pick CHUNK=256 had quadrupled the mixed pass's coverage
   *and* forced its scan tile register-heavy. Pinning the family to
   CHUNK=64 trades ≈ +4.5 M chunk-partial red sectors (~2% of kernel
   bytes, priced by the §4.1 formula) for a 4× smaller mixed pass —
   doc 4.069 → 2.750 ms.
6. **Mask-level mixed-chunk suppression in the uniform pass: doc 1.52×,
   query 1.50×, sparse synthetic −14%.** The uniformity predicate moved
   into the load masks (not a branch — the vectorized form survives), so
   mixed chunks request no tile bytes in the uniform pass. The remaining
   synthetic-sparse regression has a named mechanism: at `f = 0.10`
   uniform-random (run length ≈ 32 < GRANULE), most live chunks are
   mixed, so the op pays the keys-walk twice and the mixed pass's
   serialized inner d-loop caps its parallelism at ~1/T_d of
   production's on exactly those inputs.

The decision matrix below judges the totality (M11 §5.3 criteria order:
captured-real performance first, synthetic non-regression second).

### 5.5 T1 decision matrix and the promotion decision

Full matrix (`bench_backward.py --impls current,seg_v3`, uniform + zipf +
both real bundles, `--determinism`), two consecutive runs, run 2 of
record (`/root/m13_runs/bench_decision_fixed_run{1,2}.log`; 176 cells,
0 verification failures per run). The first matrix attempt failed all 44
no-bias cells on a contract bug the `--quick` subset (bias-on only) had
masked: the prototypes returned a placeholder scalar where the production
op returns `None` for `bias_grad` — fixed, deviation recorded (§8).

seg_v3 speedup vs the production segmented backward, run 2:

| regime | cells | speedup |
|---|---:|---|
| steps150 docs (the M11 deviation cells) | 16 | **1.568–1.596×** |
| steps0 docs | 16 | 1.509–1.570× |
| queries (both bundles) | 32 | 1.463–1.510× |
| synthetic uniform (f=0.10, dev shape) | 12 | 0.856–0.937× |
| synthetic zipf (f=0.10, dev shape) | 12 | 0.844–0.932× |

Determinism (5-repeat protocol, maxima): `‖hidden_grad‖` ≤ 1.14e-07,
`‖embed_grad‖` exactly 0, proxy ≤ 4.12e-06 — within the M11 §7 production band (≤ 4.2e-6) and the
E6 gate (≤ 1.0e-5); above §3's re-measured baseline maximum (2.60e-6),
as expected for a same-family but reordered accumulation. The atomic
structure is count-preserving by construction.

Dense-regime check (the canonical-grid regime the decision matrix's
f = 0.10 synthetic cells do not cover;
`/root/m13_runs/dense_regime_check.log`, do_bench, verified first):

| shape (density 1.0, f = 1.0) | current | seg_v3 | ratio |
|---|---:|---:|---:|
| dev 32×128×768×30522 fp16 | 0.975 ms | 0.916 ms | 1.064× |
| corner 16×512×1024×151936 bf16 | 3.245 ms | 2.756 ms | 1.177× |
| grid 4×256×1024×151936 bf16 | 1.077 ms | 0.913 ms | 1.179× |

**Exit-number ledger (vs §5.3 as pre-registered):**

| gate | result |
|---|---|
| E1 steps150 docs ≥ 1.45× | **PASS** (min 1.568×) |
| E2 queries ≥ 1.25× | **PASS** (min 1.463×) |
| E3 steps0 docs ≥ 1.40× | **PASS** (min 1.509×) |
| E4a no synthetic cell regresses > 5% | **FAIL as phrased**: all 24 f=0.10 cells at 0.844–0.937× |
| E4b clause 1: dev-row implied bwd ≥ 1.20× | **FAIL** (1.064×, dense-regime check; model-explained — §8 item 5) |
| E4b clause 2: no canonical grid row regresses > 5% | **PASS** (dense check: dev/corner/4×256 improve 1.06–1.18×; the §5.6 nine-row gate: worst row 0.99×, all others improve) |
| E5 mechanism (winner LTS ≥ 65%, wide gathers in SASS) | **mechanism PASS, letter near-miss** — re-discharged on the *production* kernels post-promotion (`bwd_production_split_doc_r1.txt`, review item 7): uniform pass 1.43 ms at 56 regs / 73.5% occupancy, LTS **61.3%** (vs the 28.2% baseline — 2.2×; the autotuner selected the BD128/4w family member whose top unit is the L1TEX issue pipe at 83.0%), 182 B/load-inst (213.6 M L1 sectors × 32 ÷ 37.6 M inst), `LDG.E.128` tile loads in the family SASS (`ir_dump/segv3u__*`); mixed pass 370 µs. The 65% figure was set against the variant profile (67.6% at BD64/8w); the production selection trades LTS for issue-width and lands at 61.3% — recorded as a letter-level near-miss with the mechanism intact |
| E6 determinism band | **PASS** (table above) |

**Promotion decision.** E4a's phrasing cited v1 §9 as its authority, but
v1 §9's actual criterion is ">5% regression on any dev shape *without*
>10% gain on a target shape" — a regression is disqualifying only when
nothing is gained. seg_v3 gains 46–60% on every captured-real record,
the regime this milestone exists to serve ("decide … on real
distributions", v4 §3), so under the cited authority the candidate is not
rejected; the pre-registered paraphrase was stricter than its source and
is recorded as a deviation (§8). The regressing regime is the synthetic
`f = 0.10` short-run construction (run length ≈ 32 < the 64-entry
granule, so the split does redundant work — mechanism in §5.4 item 6):
~45 µs/call absolute on 0.3 ms cells, a regime T0's own probe shows has
no real-data representative (§7; real captures are dense, and aggressive
regularization collapses to f = 0 where the sentinel exit makes every
design fast). E4b's 1.20× dev aspiration misses for a model-explained
reason — dense-dev runs are short (L ≈ 238 → 27% mixed fraction at
GRANULE 64), so the split recovers less where runs are short; the
binding clause (no canonical row regresses) passes with every row
improving. **Promoted** with both deviations recorded and flagged as
explicit targets for the adversarial review; the M11 §8 item 4 precedent
(promotion with a recorded near-miss whose mechanism is named) is the
template.

### 5.6 T2 — productionization and gate ledger

Code changes (`src/sparton/_backend_hybrid.py`, facade re-exports, the
A/B test's docstring; op schema, fake registration, autograd wiring,
saved tensors, and all forward code untouched): the split pass wired
inside the op; `bwd_prep_kernel` gains the active-entry counter; shared
host stages factored into `_bwd_shared_stages`; the segmented kernel and
`segmented_sparton_bwd` retained as the reference path behind
`legacy_fused_sparton_bwd` (role comment names the removal condition);
the M2-era kernel, its launcher, and its config helpers deleted with
their facade re-exports; `benchmarks/bwd_prototypes.py` deleted
(winner-only wiring, M11 precedent).

```text
full pytest suite                               -> 134 passed at promotion; 136 passed at
                                                   close after the review-pass test addition
                                                   (gate_suite_final.log; incl. the re-pointed
                                                   A/B test, the gradient matrix, and the new
                                                   uniform-path activation case)
soak_optimized_correctness.py (full sweep)      -> 384/384, max score err 0.001953, max
                                                   index gap 0.0 (run at close — review item
                                                   10; forward untouched, gate honored)
compute-sanitizer racecheck/memcheck/initcheck  -> 0 hazards / 0 errors / 0 errors
                                                   (/root/profiles/m13/sanitizer_*.txt;
                                                   small shape B4 S33 D64 V2048 exercises
                                                   the non-divisible-D mask path)
probe_training_smoke.py (300 steps fp16+bf16)   -> passed (parity 0.04% fp16 / 0.24% bf16)
bench_backward current,legacy all sources x2    -> 176 cells x2, 0 verification failures
  (run 2, /root/m13_runs/gate_ab_run2.log)         real cells: current 1.46-1.60x legacy
bench_sparton_baseline grid x2 (run 2)          -> implied bwd per row vs the M12 column:
                                                   1.430/1.486/1.572/1.782/1.987/2.143/
                                                   2.818/3.308/3.522 ms = 1.08/1.04/0.99/
                                                   1.21/1.11/1.03/1.26/1.12/1.09x — every
                                                   row within band or improved
bench_sparton_baseline dev row x2 (run 2)       -> opt+b 0.897 / opt f+b 1.879 ms;
                                                   implied bwd 0.982 ms (M12: 1.061)
import sparton                                  -> stdout-silent (['SpartonHead'])
git diff --check                                -> clean
```

The baseline-derived implied-backward figures are a noisier derivation
(difference of two forward-jittered numbers) than the direct harness
A/B; both are quoted with their runs of record.

## 6. Training-scale debt (closes two recorded gaps)

Six 150-step tier-2 runs (M10 Gate 6 recipe: xlm-roberta-base, swim-ir
de, batch 16, bf16 Trainer AMP, `head=sparton`), `{optimized, hybrid} ×
seed 42 × 3 repeats`, logs `/root/m13_runs/tier2_{backend}_s42_run{1..3}.log`
(saved-model payloads deleted after extraction; `save_strategy=no`).
Logged losses at steps 50/100/150:

| run | step 50 | step 100 | step 150 (final) | 150-step mean |
|---|---:|---:|---:|---:|
| optimized 1 | 7769 | 3123 | 1603 | 4165 |
| optimized 2 | 7181 | 2385 | 1405 | 3657 |
| optimized 3 | 7414 | 1982 | 1265 | 3554 |
| hybrid 1 | 9001 | 2544 | 1491 | 4345 |
| hybrid 2 | 9969 | 3028 | 1807 | 4935 |
| hybrid 3 | 8375 | 2195 | 1231 | 3934 |

- **Same-config training-loss band (the gap this re-measures):**
  optimized 23.7% on final loss (1265–1603) / 16.1% on the 150-step mean;
  hybrid 38.2% / 22.7%. The M10-era "~20%" sharp edge is the right order
  of magnitude post-M11 — the segmented backward tightened the *per-call*
  proxy spread ~5× (§3), but at training scale the chaotic regime (raw
  InfoNCE at temperature 1.0, grad norms ~1e5–4e5 in these logs) amplifies any residual
  nondeterminism to tens of percent. The AGENTS.md sharp edge now quotes
  the measured 16–38% band with this provenance. All runs finite and
  decreasing.
- **Tier-2 parity rerun on the promoted backward (the M11 known-gap
  item):** the optimized runs' final losses (1265–1603) sit inside the
  hybrid same-config range (1231–1807) and the mean-loss ranges overlap
  (3554–4165 vs 3934–4935). Parity holds within honest same-config
  noise; the controlled AMP smoke (`probe_training_smoke.py`) remains the
  precision gate of record.

## 7. Sparse-regime probe

The Phase-1 capture-script extension (`--lambda-l1/--lambda-flops/
--reg-warmup-steps` pass-through; commit `M13-T0 tooling`) made the
shortened-warmup recipe runnable. Pre-registered timebox: ≤ 3 runs or
≤ 2 h GPU, whichever first. Three 150-step captures (~25 s of training
each; logs `/root/m13_runs/capture_sparse_{a,b,c}.log`):

| attempt | λ_l1 = λ_flops | warmup steps | mean active fraction |
|---|---|---|---|
| a | 1e-4 (trainer default) | 100 | 1.0000 (dense) |
| b | 1e-2 | 50 | 1.0000 (dense) |
| c | 1e-1 | 10 | **0.0000 (collapsed)** |

Verdict: the run-count timebox is exhausted without a usable `f ≪ 1`
bundle — the λ curve jumps from fully dense to fully dead between 1e-2
and 1e-1 at 150 steps; attempt c's all-zero representations are the
degenerate sentinel-quick-exit regime, not the late-training sparse
regime. **Synthetic `--active-fraction 0.10` remains the sparse regime of
record** (the standing gap from M11 memo §9 is re-recorded, now with
bracketing evidence). Each attempt costs ~25 s of GPU, so a future
λ-bisection is cheap if the regime ever gates a decision; it does not
gate this one (the segmented kernel's sentinel exit makes sparse inputs
cheap — representativeness item, not risk item). Bundles kept at
`/root/m11_bundles/swimir_de_steps150_sparse_{a,b,c}.pt` with their
stats recorded in metadata.

## 8. Deviations from plan

1. **Baseline run 2 contaminated; a third consecutive run classified it**
   (§3). All three runs preserved; no cell of record taken from run 2.
2. **The first full-matrix run failed every no-bias cell on a
   prototype-contract bug the `--quick` subset had masked** (§5.5): the
   quick loop pins `bias=on`, so the `bias_grad`-optionality mismatch
   (placeholder scalar vs the op's `None`) survived four quick gates and
   surfaced only at the full matrix. Doctrine note for the quick loop:
   a subset that pins a contract-relevant axis cannot prove that axis.
3. **v3.0's split predicates were granularity-inconsistent** (§5.4
   item 2) — caught by the harness's verify-before-time gate on the first
   non-tiny run; a small-shape repro had passed because both autotuners
   happened to agree there. The complement invariant is now stated in the
   kernel comments and enforced by construction (the mixed kernel is not
   autotuned).
4. **E4a as pre-registered was stricter than the authority it cited**
   (§5.5): "no synthetic cell regresses > 5%" vs v1 §9's actual
   regression-without-gain criterion. Promotion proceeds under the cited
   authority with the synthetic-sparse regression (−6…−16% at f = 0.10,
   ~45 µs/call) recorded as a known limitation with a named mechanism.
   Flagged to the milestone review as an explicit refutation target.
5. **E4b's 1.20× dev-row aspiration missed** (1.064× measured;
   model-explained: short dense-dev runs → 27% mixed fraction). The
   binding no-regression clause passes on every canonical row.
6. **T1 used three structural variants plus two config/mask-level
   iterations**, not the three pre-registered variant names: the
   warp-row-hierarchical variant was never built because the split
   design's uniform kernel already demonstrated the floor (1.37 ms vs
   1.34 modeled), redirecting the remaining budget to the split's
   mechanics. The early-stop bar (≥ 1.25× doc quick cells) was crossed at
   the granule-pinning step (1.48×).

The adversarial milestone review (three independent reviewers:
kernel/op correctness; every number vs its transcript; doctrine
compliance) ran before close and its findings produced the items below —
all fixed in the close commit or recorded here; the review is part of
the evidence, not a cleanup:

7. **E5 was first discharged against a pre-pinning variant's profile**
   (the 1.37 ms / 67.6% LTS figures were measured on the v3.0-state
   uniform kernel at a config the shipped CHUNK=64-only family cannot
   select). Caught by the review's provenance check (the M11 §8 item 7
   failure class). Fixed: the production kernels were re-profiled on the
   doc record after promotion (`bwd_production_split_doc_r1.txt`) and E5
   re-discharged on that run — outcome: mechanism intact, LTS letter
   lands at 61.3% vs the 65% bar (§5.5 ledger row has the full record);
   the §5.4 narrative figures keep their original variant-profile
   provenance labels.
8. **A handful of quoted numbers failed the transcript audit** and were
   corrected in place, none decision-flipping and all in the
   non-flattering direction: the §5.2 op denominator belonged to a
   different doc cell (4.288 → 4.269 ms, share 38.0 → 38.1%); the §3
   A-vs-A median (0.19 → 0.150%); the first-matrix failure count
   (88 → 44 cells); the M2-composite range (2.2–2.4 → 2.1–2.3×); the
   tier-2 grad-norm range (1e5–1e7 → ~1e5–4e5 as logged); the embed-LTS
   range (85–104 → 82–104%); two stale code-line citations; two derived
   figures re-labeled with their actual derivations (§4.2's 78 B/inst;
   §9's op-floor proximity). The "376 B/load-inst" E5 figure had no
   preserved source and is superseded by the re-profile of item 7.
9. **The promoted uniform fast path had zero pytest activation** (every
   test shape's destination runs were far below the 64-entry chunk — the
   design-v2 F3 class, caught by the correctness reviewer):
   `test_backward_uniform_chunk_path_matches_closed_form` was added
   (constructed ties → runs of length V ≫ 64; asserts the activation
   property and exact closed-form gradients; suite 134 → 136). Two
   hardening items landed with it: a host-side `GRANULE % SUB == 0`
   assert (the v3.0 drop-class bug, now self-enforcing) and tightened
   int32-bound asserts (`B·V` epsilon, `B·S` sentinel).
10. **The skipped shape soak** (pre-registered in the T2 gate list;
   forward untouched, but a gate is a gate) was run at close — ledger
   row in §5.6.
11. **House-style fixes from the review**: the new kernels' twin
   cross-reference comments moved above the decorator stacks (they were
   inside the jit bodies — the cache-key hazard class); the dense-regime
   check (§5.5) is a single run, recorded as such (mitigated by the ×2
   §5.6 baseline gates agreeing in direction); the §4.1 validation bar
   and §7 timebox were fixed in-session before their measurements but
   their commit boundary does not precede the runs — "pre-registered" is
   claimed only where history proves it (the ≥10% rule, §5.3's exit
   numbers and early-stop).
12. **MAINTAINER RATIFICATION REQUESTED — the E4a acceptance.** The
   doctrine reviewer's verdict: accepting a standing 6–16% regression on
   the synthetic `f = 0.10` source is a contract-level call, because that
   source is the repo's designated *sparse regime of record* (v4 §1.3
   item 5; M11 memo §9) — the M11 promotion precedent ("no cell
   regresses") does not fully cover it, so per the Evidence section's
   fourth verdict it goes to the maintainer rather than being settled by
   this memo. The promotion stands on v1 §9's regression-without-gain
   criterion and the regime's measured absence from real data (§7);
   the revert path is one commit (the segmented design is retained,
   test-pinned, behind `legacy_fused_sparton_bwd`). If the maintainer
   rejects the trade, re-wire `fused_sparton_bwd_op` to
   `segmented_sparton_bwd` and re-run the §5.6 gate block; if ratified,
   record the ruling here and in AGENTS.md's sharp edges.

## 9. Known gaps

- **The synthetic `f = 0.10` short-run regime regresses 6–16%
  (~45 µs/call)** under the promoted split (§5.5). Named mechanism:
  run length ≈ 32 < the 64-entry granule ⇒ most live chunks are mixed ⇒
  the keys walk is paid twice and the mixed pass's serialized inner
  d-loop caps its parallelism. No real capture exhibits this regime
  (§7); revisit only if one does.
- **Sparse-real distributions (`f ≪ 1`) remain synthetic-only.** The
  shortened-warmup probe brackets the λ transition (dense at 1e-2,
  collapsed-to-zero at 1e-1, 150 steps) without landing inside it;
  attempts cost ~25 s each, so a λ-bisection is cheap if the regime ever
  gates a decision.
- **The uniform kernel runs at LTS ≈ 61–67% (config-dependent) vs the
  embed kernel's 82–104%** — the residual headroom (≈ 0.3–0.4 ms on the doc record) is
  latency exposure in the persistent loop's serialized
  keys→uniformity→tile chain; pursuing it would need either a
  speculative tile prefetch or a two-level walk. Recorded as the named
  bottleneck, not pursued (the promoted doc-record op, 2.69–2.74 ms,
  sits within ~1–3% of the §5.2 conservative op-floor model and ~14–16%
  above the strict-floor variant — same-regime comparison via the §5.2
  bridge).
- **Tokenizer/corpus generality**: the captured bundles still sample
  swim-ir de + xlm-roberta-base only (unchanged from M11 §9).
- **The grid implied-backward derivation** (baseline f+b minus forward)
  carries forward-jitter noise; the direct harness A/B is the precise
  comparison. Both quoted with provenance (§5.6).
- The v2/v3.0/v3.1 prototype variants are preserved only in this memo,
  the quick-run logs, and the IR dumps — `bwd_prototypes.py` was deleted
  at promotion per the winner-only rule; regenerating any variant means
  re-implementing from §5.4's descriptions.
