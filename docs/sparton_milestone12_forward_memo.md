# Milestone 12 memo — forward track (entry evidence; closed without kernel work)

Date: 2026-06-13. Plan of record: design v4 §3 M12
([sparton_remaining_work_design_v4.md](sparton_remaining_work_design_v4.md)).
Method references: `AGENTS.md` and
[triton_gluon_kernel_optimization.md](triton_gluon_kernel_optimization.md)
(§3.3 bottleneck classification). Platform: RTX 5090 (sm_120), torch 2.12
nightly, Triton 3.6.0, ncu 2026.1.1. Benchmark cells are
`triton.testing.do_bench` op-level **means** (Triton 3.6's default
`return_mode`; the M11 memo's header says "medians" for the same helper —
a mislabel inherited from there, flagged in §7 item 5) judged on run 2 of
two consecutive runs; ncu numbers are kernel-level (serialized,
multi-pass) and are never compared against them. Transcripts for every decision-carrying
number: `/root/profiles/m12/`.

## 1. Decision

**M12 closes without kernel work.** The pre-registered entry rule (v4
M12-T0) fails on its second clause: the production forward kernel is
**tensor-pipe-bound at 92.3–94.4% pipe utilization** on every profiled
shape — including the rule-(i)-passing grid rows — with the L2 fabric
simultaneously at 89–91% and DRAM reads at the compulsory byte floor. The
persistent / warp-specialized rewrite (T2/T3) targets scheduling bubbles
(pipeline drains, barrier stalls, wave tails); the counters show those
bubbles do not exist (SM active/elapsed = 99.6% at 16×512; dominant warp
stall is *waiting for the execution pipe*, i.e. compute saturation). T2 and
T3 are not entered; D2 is discharged by re-affirming the cross-reference
comments (`_backend_optimized_gluon.py`, `bench_gluon_gemm.py`); the
launcher-v2 revival trigger (b) — "T3 opens the kernel signature" — never
fires, so the M12-T1 deferral stands untouched.

One-paragraph mechanism: the remaining 9.0–10.5% wall-time gap between the
optimized forward and the same-run full-V cuBLAS GEMM is **per-cycle
tensor-pipe efficiency at the autotuned 64×64×32 tile shape, plus the L2
operand-traffic pressure that shape implies** — the kernel holds the tensor
pipe *more* active than cuBLAS's 86.6% reference (v1 §3.5) yet finishes
later, i.e. it issues more pipe-work for the same math; and its L2
requested traffic scales as `A·V/BLOCK_N + B_m·B·S/BLOCK_M` (validated to
≤0.6% below), running the L2 fabric at ~6.6 TB/s. Recovering the gap would
mean a tile-shape/epilogue redesign (larger tiles lose on wall time at
these shapes — the autotuner's revealed preference, consistent with the
M7/E5 history), not a scheduling rewrite. That is the **named residual
bottleneck and terminal state of the forward track** (Performance Loop
step 10): dual saturation at the selected policy; remaining upside ≤10.5%
of forward with no in-scope mechanism.

## 2. T0 entry evidence — first profile of the production forward kernel

Profiled via the new `benchmarks/ncu_forward_target.py` (NVTX `fwd_direct/`,
main-thread raw-op calls, autotune warmed outside the range;
`--launch-skip 1 --launch-count 1`). The kernel had never been ncu-profiled
before this session (only the GEMM bring-up kernel had counters, v1 §3.5).
Raw transcripts `fwd_{dev_fp16,4x768_bf16,16x512_bf16}.txt`.

| metric | dev 32×128×768×30522 fp16 | 4×768×1024×151936 bf16 | 16×512×1024×151936 bf16 |
|---|---:|---:|---:|
| duration (ncu regime) | 944.9 µs | 4.51 ms | 12.02 ms |
| SOL compute (SM) | 94.39% | 92.33% | 92.26% |
| **tensor pipe (highest pipe)** | **94.4%** | **92.3%** | **92.3%** |
| SOL memory (L2-bound) | 88.87% | 91.18% | 90.93% |
| DRAM throughput | 2.83% | 3.64% | 1.50% |
| DRAM read / write | 53.28 MB / 0 | 319.51 / 8.61 MB | 335.78 / 23.04 MB |
| L2 read sectors (= bytes) | 188.65 M (6.04 GB) | 938.74 M (30.04 GB) | 2496.27 M (79.88 GB) |
| L2 hit rate | 99.01% | 98.92% | 99.55% |
| dominant stall (cycles/warp) | exec-pipe wait, 7.0 | exec-pipe wait, 5.5 | exec-pipe wait, 5.5 |
| warp cycles / issued inst | 13.53 | 12.45 | 12.43 |
| achieved occupancy | 24.67% | 16.64% | 16.64% |
| regs/thread (spill bytes) | 119 (0) | 113 (0) | 113 (0) |

Autotune selections (recovered from the Autotuner cache by deposited
probes — `fwd_selected_configs.txt` for the profile shapes, see §7 item 2;
`fwd_selected_configs_grid.txt` for **all nine grid keys plus dev**, added
at the review pass): every key selects the same
`64×64×32 / 2×2 warps` tile family — `POLICY_ID 10` (3 stages) or
`POLICY_ID 9` (4 stages), with the stage count flipping between processes
at near-ties (the known ±5% selection jitter; the tile family never
changes). `BLOCK_M = BLOCK_N = 64` feeds the §4 traffic formulas.

nsys inventory (dev, `--launches 8`, `fwd_inventory_dev.txt`): exactly one
`sparton_optimized_forward_kernel` launch per op call (11 launches for
3 warmup + 8 profiled; 99.1% of GPU time), zero fills and zero per-call
memcpys — the `torch.empty` outputs are fully covered by the kernel's
stores. The 3 tiny memsets/D2H copies in the trace belong to the script's
one-time summary-line reads, not to the op.

Premise check (Loop step 1): the binder is **not** SM-side idleness.
SM active cycles / elapsed = 31.74 M / 31.87 M = **99.6%** at 16×512 —
there is no wave-tail or drain slack for a persistent schedule to smooth;
total idle from *all* causes is below the 9–10.5% gap.

## 3. Re-measured shared state (one provenance; replaces v4 §1.2's mixed-run derivation)

`bench_sparton_baseline.py --optimized-policy on`, two consecutive runs,
run 2 of record (`grid_bf16_run{1,2}.txt`, `dev_fp16_run{1,2}.txt`);
per-row floor = the same-run `gemm ms` column (the byte-floor term is
0.17–0.19 ms at every grid row — at least 7× below the GEMM term, §4).

Run 2 of record (bf16 grid; dev fp16 row appended):

| B×S | opt fwd ms | gemm ms | ratio | gap % of fwd | recoverable ms | implied bwd ms | bwd share of f+b |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4×256 | 1.405 | 1.279 | 1.099 | 8.97 | 0.126 | 1.543 | 52.3% |
| 4×512 | 2.818 | 2.538 | 1.110 | 9.94 | 0.280 | 1.551 | 35.5% |
| 4×768 | 4.245 | 3.799 | 1.117 | **10.51** | 0.446 | 1.555 | 26.8% |
| 8×256 | 2.825 | 2.538 | 1.113 | **10.16** | 0.287 | 2.159 | 43.3% |
| 8×512 | 5.619 | 5.106 | 1.100 | 9.13 | 0.513 | 2.200 | 28.1% |
| 8×768 | 8.434 | 7.569 | 1.114 | **10.26** | 0.865 | 2.200 | 20.7% |
| 16×256 | 5.631 | 5.059 | 1.113 | **10.16** | 0.572 | 3.549 | 38.7% |
| 16×512 | 11.235 | 10.092 | 1.113 | **10.17** | 1.143 | 3.704 | 24.8% |
| 16×768 | 16.878 | 15.250 | 1.107 | 9.65 | 1.628 | 3.822 | 18.5% |
| dev 32×128 (fp16) | 0.873 | 0.848 | 1.029 | 2.86 | 0.025 | 1.061 | 54.9% |

The implied-backward column (`opt f+b` − `opt fwd`, same run) reproduces
the M11 gate values (grid 1.546–3.720, dev 1.069) within noise; the
backward share is **~18–52% across the grid and ~55% on the dev shape** —
the v4 §1.2 derived range, now on one provenance. This table is the
**forward column of the §3 comparative table**: recoverable forward time is
0.13–1.63 ms/call on grid rows (bounded above by the gap; the true
recoverable fraction is smaller still, since the floor assumes cuBLAS-level
pipe efficiency the epilogue-fused kernel cannot reach — §5). M13-T0 adds
the backward column.

A-vs-A noise band: run-1 gaps are 8.09–9.71% (zero rows ≥10%); run-2 gaps
8.97–10.51% (five rows ≥10%). Per-row run-to-run spread on gap% is
0.04–2.1 points — **the 10% bar sits inside the same-config noise band on
every row**. Rule (i) is judged mechanically on the run of record (pass),
and §5 records the sub-noise margin; the verdict does not depend on it
because rule (ii) fails by ~18 points, far outside any band.

## 4. Analytic traffic model (validated before the decision)

Per-row formulas in `(B, S, D, V, elt)` and the selected policy's blocks
(`BLOCK_M = BLOCK_N = 64`): compulsory reads `A = B·S·D·elt` (hidden),
`B_m = V·D·elt` (embed), `bias = V·elt`; outputs `B·V·(elt + 8)`.
Requested (pre-L2) reads, from the kernel's CTA structure (grid
`(B, ⌈V/BLOCK_N⌉)`, per-CTA s-tile × k-tile loops):

- hidden: `A · ⌈V/BLOCK_N⌉` (every v-tile CTA re-reads its batch row), and
- embed: `B_m · B · ⌈S/BLOCK_M⌉` (every CTA re-reads its embed slice once
  per s-tile).

| shape | predicted L2 reads | measured (sectors×32 B) | residual | predicted compulsory DRAM | measured DRAM read | residual |
|---|---:|---:|---:|---:|---:|---:|
| dev fp16 (477 v-tiles, 2 s-tiles) | 6.001 GB | 6.037 GB | +0.6% | 53.23 MB | 53.28 MB | +0.1% |
| 4×768 bf16 (2374, 12) | 29.87 GB | 30.04 GB | +0.6% | 317.76 MB | 319.51 MB | +0.6% |
| 16×512 bf16 (2374, 8) | 79.66 GB | 79.88 GB | +0.3% | 328.25 MB | 335.78 MB | +2.3% |

The model matches the requested-traffic counters to ≤0.6% (the residual is
the unmodeled mask/bias epilogue loads) and the DRAM counters sit at
100–102% of the compulsory floor: **the L2 absorbs the re-reads almost
completely** (hit ≈ 99%), at the price of running the L2 fabric at
89–91% busy (79.88 GB / 12.02 ms ≈ 6.6 TB/s at 16×512). DRAM writes
against the output term: 16×512 measures 23.04 vs 24.31 MB predicted
(−5%); dev measures 0 — the 9.77 MB output set is L2-resident at kernel
end (the v1 §11 caveat); 4×768 measures 8.61 vs 6.08 MB predicted — a
2.5 MB excess that stays **unexplained** (recorded; no claim in this memo
depends on the write side). Floor term: `max(gemm_ms, bytes/BW)` — at
nameplate-class ~1.79 TB/s DRAM bandwidth the byte term is 0.17–0.19 ms
across the grid's 314–337 MB compulsory traffic, i.e. **7× (4×256, gemm
1.279 ms) to 81× (16×768, gemm 15.250 ms) below the same-run GEMM term**,
so the per-row floor of record is the `gemm ms` column.

Expected-vs-measured conclusion: every counter the persistent-rewrite
thesis depends on is explained without scheduling slack — traffic is at
the modeled floor, occupancy is barrier/SM-limited exactly as designed,
spills are zero, and the only stall that matters is the pipe itself. The
lone unreconciled residual (the 4×768 write excess above) sits on the
write side, which no clause of the decision rule reads.

## 5. Decision-criteria walk (pre-registered rule, v4 M12-T0)

**Rule (i)** — gap ≥10% of forward on ≥1 canonical grid row, run 2 of
record: **PASS**, on five rows (4×768: 10.51%; 8×768: 10.26%; 16×512:
10.17%; 8×256 and 16×256: 10.16%). Recorded with the §3 caveat: the margin
above 10.0% (0.16–0.51 points) is smaller than the measured run-to-run
spread (up to 2.1 points); run 1 had zero passing rows.

**Rule (ii)** — on a passing row, tensor-pipe utilization **< 74%** with
the dominant stall in the scheduling/barrier/pipe-wait family and not a
memory-side or operation-count binder: **FAIL, decisively.** Both profiled
passing rows (4×768, 16×512) measure tensor-pipe **92.3%** — 18 points
*above* the threshold (dev: 94.4%); the dominant stall ("waiting for the
execution pipe", 5.5–7.0 cycles/warp) signals pipe *saturation*, the exact
opposite of fillable bubbles; and the memory side is simultaneously at
89–91% L2-fabric busy with DRAM at the compulsory floor. The three
unprofiled passing rows (8×256, 8×768, 16×256) select the same
`64×64×32 / 2×2` tile family (captured for all nine grid keys —
`fwd_selected_configs_grid.txt`), sit in the same §4 traffic regime, and
have gap ratios within 0.4% of the profiled rows; the verdict extrapolates
to them via the model (§8 item 1 records the absence of per-row counters
as a gap).

**Verdict: NO-GO.** Per the pre-registered rule: closing memo (this
document), D2 discharged by re-affirmation (comment updates at
`src/sparton/_backend_optimized_gluon.py` above the decorator stack and
`benchmarks/bench_gluon_gemm.py`), M12 kernel work closed; T2
(`gl.warp_specialize` probe) is not run — it exists solely as the entry
gate for T3's WS variant. Correction from the review pass: the comment
edits exclude the comment *bytes* from the compiled-source hash, but
Triton 3.6's JIT cache key also includes the function's **starting line
number**, so the +2-line edit re-keyed the kernel's compile and autotune
caches anyway (verified: `cache_key` changed). No evidence is tainted —
every run of record predates the edit, and the full suite ran on the
edited tree — but the original "outside compiled-source hashes" safety
claim was wrong as a mechanism; the durable rule (never edit above-kernel
comments mid-campaign) is now in `AGENTS.md`. The altitude reading
(AGENTS.md Performance-Optimization Loop, step 1): the forward's binder
turned out to be in the same class as the backward's was at M11 —
operation count (pipe-work and L2 traffic that exist by construction at
the selected tile shape) — not lowering-level scheduling; the v1 §3.5
inference that the gap was "intra-CTA pipelining quality" was drawn from
the GEMM bring-up kernel at 63.9% pipe and does not transfer to the
production kernel at 92–94%.

## 6. T4 — measurement-set additions (record, don't threshold)

**Mask-density sweep** (v1 §10.3, landed on the forward side via the new
`--mask-density` flag; run 2 of record, transcripts
`density_{dev,8x512}_{0.25,0.75,1.0}_run{1,2}.txt`):

| density | dev opt fwd ms | 8×512 opt fwd ms |
|---:|---:|---:|
| 0.25 | 0.877 | 5.546 |
| 0.75 | 0.878 | 5.545 |
| 1.00 | 0.879 | 5.550 |

Forward time is flat across density (peak-to-peak 0.23% on the dev row,
0.09% at 8×512) — the mask is an epilogue multiply, not control flow —
and density is correctly absent from the forward autotune key (Loop
step 7 audit). The fwd+bwd column falls at low
density (dev opt f+b 1.824 at 0.25 vs 1.930 at 1.0) because the backward's
work scales with active rows; recorded, not gated.

**Host-overhead rows** (new `bench_host_overhead.py`; run 2 of record,
`host_overhead_run{1,2}.txt`; wall/GPU/host ms):

| backend | 8×128×768×1280 fp16 | dev 32×128×768×30522 fp16 |
|---|---|---|
| hybrid | 0.099 / 0.032 / 0.067 | 1.154 / 1.152 / 0.002 |
| naive | 0.032 / 0.021 / 0.010 | 1.271 / 1.251 / 0.020 |
| optimized | 0.159 / 0.038 / **0.121** | 0.910 / 0.878 / 0.032 |

The v2 Appendix A item 5 record (optimized 0.183 wall / 0.064 GPU →
0.119 ms host) reproduces: the optimized host share is **0.121 ms/call**
at the small shape and 0.002–0.032 ms across the two runs at the dev
shape (overlapped, noise-dominated) — F9 stands documented exactly as the
launcher-v2 deferral assumed (v4 §6). The GPU
column at the small shape (0.038 vs the v2-era 0.064) differs across the
stack eras; not investigated (record-only scope, §8 item 5).

## 7. Deviations from plan

1. **ncu corner shapes changed** from the planned 8×512/16×768 (v3-era
   best/worst ratios, unpreserved-M10 provenance) to **4×768 and 16×512**:
   the re-measured run-2 ratios are uniform (1.099–1.117) and rule (i)
   passed on different rows than the old ratios suggested. Rule (ii)'s own
   wording requires the counters to be read *"on that row"*, so the
   profiled corners moved to passing rows (the execution plan carried this
   contingency; v4's text does not spell it out). The originally named
   corners were not profiled — they fail rule (i) in the run of record.
2. **Forward autotune selections were not captured by
   `TRITON_PRINT_AUTOTUNING`** — every profile-shape key was already in
   the on-disk autotune cache from earlier sessions (`cache_results=True`),
   so nothing re-tuned and nothing printed. Recovered by a disposable
   probe reading the `Autotuner.cache` after a cache-hit call
   (`fwd_selected_configs.txt`). Method note for future campaigns: a
   cache-hit selection is silent; read the cache, don't clear it.
3. **Rule (ii)'s stall-family wording** listed `math_pipe_throttle`-class
   stalls in the "scheduling family"; the measured dominant stall
   (execution-pipe wait) is nominally in that family but signals
   *saturation*, the opposite of the under-utilized case the clause was
   written for. The unambiguous `<74%` threshold clause carried the
   decision; recorded so the next rule-writer separates "pipe idle,
   waiting at barriers" from "pipe full".
4. T2 (`probe_warp_specialize.py`) was **not built**: it is the entry gate
   for T3's WS variant only, and T3 was closed at T0. The sm_120
   `gl.warp_specialize` viability question remains open and returns to
   v1 §3.6's unresolved list (§8 item 4).
5. **Review-pass corrections to this memo's own first draft** (inline
   adversarial review, two independent reviewers — numbers-vs-transcripts
   and doctrine/consistency): the unreproducible "15–50× below the GEMM
   term" multiplier was replaced by the derived 7–81× range with its
   bandwidth assumption stated; the 4×768 DRAM-write excess (8.61 vs
   6.08 MB) is now reconciled in §4 instead of silently omitted; the
   density/host-share figures are quoted exactly instead of rounded
   ("≤0.2%" → 0.23%/0.09%; "≈0" → 0.002–0.032 ms); the noise-band lower
   bound was corrected (0.05 → 0.04 points); `do_bench` cells were
   relabeled means (the "medians" header was inherited from the M11 memo,
   which carries the same mislabel — flagged, left to a future correction
   there); the "outside compiled-source hashes" claim was corrected to
   the real mechanism (§5) and the durable comment-editing rule added to
   `AGENTS.md`; and the all-keys selection capture
   (`fwd_selected_configs_grid.txt`) replaced the §5 policy-family
   extrapolation sub-claim with measured fact.

## 8. Known gaps (deliberately not validated)

1. Only 3 of the 10 measured shapes were ncu-profiled (dev + two of the
   five rule-(i)-passing rows); the verdict on 8×256/8×768/16×256 is
   model-extrapolated (same selected tile family — captured for all keys,
   `fwd_selected_configs_grid.txt` — same regime, ratios within 0.4%),
   not per-row counter-measured.
2. The 86.6% cuBLAS tensor-pipe reference is the v1 §3.5 dev-shape
   measurement (2026-06-11 stack); it was not re-collected. The
   pipe-efficiency comparison in §1 mixes that reference with this
   session's pipe figures — directionally robust (the kernel exceeds it
   while being slower), but not a same-session A/B.
3. The byte-floor term uses a nameplate-class DRAM bandwidth estimate
   (~1.79 TB/s); it is ≥7× below the GEMM term on every grid row, so its
   precision cannot affect the floor.
4. `gl.warp_specialize` on sm_120 remains unprobed (deviation 4); the
   probe design is retained in the plan history should a future milestone
   need it.
5. The host-overhead GPU-column shift at the small shape (0.038 vs
   v2-era 0.064 ms) is recorded, unexplained, and harmless to every
   decision in this memo (the host column, which carries F9, reproduces).
6. nsys inventory at the dev shape only.

## 9. Gate ledger (hardened env, serial; transcripts `/root/profiles/m12/`)

```text
py_compile (src, training, tests, benchmarks)   -> clean
pytest -q -m "not slow" (post-tooling)          -> 111 passed (109 + 2 new)
pytest -q (full, at close)                      -> 134 passed
import sparton                                  -> stdout-silent (['SpartonHead'])
git diff --check                                -> clean
grid bf16 x2 (run 2 of record)                  -> grid_bf16_run{1,2}.txt
dev fp16 x2 (run 2 of record)                   -> dev_fp16_run{1,2}.txt
ncu production forward x3 shapes               -> fwd_{dev_fp16,4x768_bf16,16x512_bf16}.txt
autotune selections (cache probes)              -> fwd_selected_configs.txt (profile shapes),
                                                   fwd_selected_configs_grid.txt (all 9 grid keys + dev)
nsys inventory (dev)                            -> fwd_inventory_dev{.txt,.nsys-rep}
density sweep 3x2x2                             -> density_*.txt (forward flat: 0.23%/0.09% peak-to-peak)
host overhead x2                                -> host_overhead_run{1,2}.txt (F9: 0.121 ms)
shape soak / training smoke                     -> not run: no forward-correctness-surface
                                                   or autograd change in this milestone
                                                   (comment-only src edit)
```
