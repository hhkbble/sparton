# Sparton Remaining-Work Design v2 (post-M8)

Date: 2026-06-12.
Status: **active guide for subsequent backend-refactor development.**

This document supersedes
[sparton_gluon_remaining_work_design.md](sparton_gluon_remaining_work_design.md)
(referred to below as "v1") as the forward plan. v1 remains authoritative for
platform facts that have not changed: the sm_120 MMA availability matrix and
fatal-abort behavior (v1 §2.2), the local Gluon API survey (v1 §2.3), the
environment-defect root cause and hardened env (v1 §2.4), profiler usage notes
(v1 §2.5), and the ncu counter evidence for the hybrid path and the original
GEMM bring-up kernel (v1 §3.5).

Everything below is grounded in a line-by-line review of the three commits
that produced the current state — `4408271` (v1 design + validated
benchmarks), `6d56ed6` (M3–M5: backend split, router, naive backend, merged
benchmark), `e5e9643` (M6–M8: Gluon shim, policy runtime, optimized forward) —
plus fresh runtime probes executed on 2026-06-12 in this workspace (Appendix
A). The review's goal, reflected in the milestone order below: **after M9 (and
nothing else), the project should be in a clean production-ready state**, with
all later milestones being performance work that can be scheduled freely.

---

## 1. Confirmed current state (review-verified)

### 1.1 Backends, modules, and layering

```text
src/sparton/
  __init__.py                  # exports SpartonHead only when CUDA is available
  sparton_kernel.py            # facade/router: resolve_backend, SpartonHead, re-exports,
                               #   lazy __getattr__ for optimized symbols
  _backend_hybrid.py           # default backend (faithful M3 move of the original file,
                               #   incl. its import-time side effects and dead code)
  _backend_naive_triton.py     # M5 tl.dot fused forward, bounded autotune (10 configs,
                               #   key=(S,D,V)), hybrid backward via its own autograd
  _backend_optimized_gluon.py  # M8 Gluon fused forward (TMA + mma_v2 + fused epilogue),
                               #   policy-autotuned, hybrid backward via its own autograd
  _gluon_runtime.py            # lazy Gluon import shim, sm_80+ -> mma_v2 whitelist,
                               #   VALIDATED_TRITON="3.6.0" warning, autotune fallback
  _gluon_policy_runtime.py     # lazy host helpers: policy->triton.Config, POLICY_ID
                               #   mapping, config pruning, dtype mapping, descriptor banks
  _runtime_policy.py           # pure-Python (import-safe anywhere): DeviceProfile,
                               #   ProblemSpec, GluonGemmPolicy, candidate universes,
                               #   validity pruning, ranking, tiny-problem fallback
```

Layering as implemented: `SpartonHead.forward` → bound forward callable →
`torch.library` custom op → kernel launch. The op names and schemas are the
stable layer:

```text
sparton::fused_sparton_fwd (Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)   # hybrid
sparton::fused_sparton_bwd (...)                                                  -> (Tensor, Tensor, Tensor?)
sparton::naive_fwd         (Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)
sparton::optimized_fwd     (Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)
```

All three forwards register autograd that calls `fused_sparton_bwd_op` (the
single backward implementation). `naive`/`optimized` are reached through
Python wrappers (`naive_forward`, `optimized_forward`) that force input
contiguity before the op; **hybrid is reached through the raw op with no
wrapper** — see finding F1.

### 1.2 Verified performance and memory snapshot

Re-measured during this review (hardened env, serial, warm autotune caches),
`triton.testing.do_bench`:

Dev shape `B=32, S=128, D=768, V=30522`, fp16, all-ones mask:

| metric | hybrid | naive (autotuned) | optimized | full-V cuBLAS GEMM |
|---|---|---|---|---|
| forward + bias | 1.176 ms | 1.301 ms | **0.898 ms** | 0.880 ms |
| forward no-bias | 1.075 ms | 1.299 ms | 0.896 ms | — |
| peak extra memory | 140.50 MiB | 9.86 MiB | 9.86 MiB | — |
| fwd+bwd + bias | 2.637 ms | — | — | — |

Canonical `naver/splade-code-06B` grid (`D=1024, V=151936`, bf16, B∈{4,8,16},
S∈{256,512,768}) — from the M8 memo, spot-confirmed here: optimized beats
hybrid on **all nine rows** (e.g. B=16/S=512: 11.49 vs 15.33 ms) and runs at
roughly 1.10× the per-row GEMM floor.

Two consequences that reshape the plan:

1. The optimized forward is already **at ~102% of the GEMM floor** on the dev
   shape and ~110% on the big bf16 grid. v1's M9 exit gate ("≥ hybrid forward
   on at least one realistic shape") is already exceeded by the O1 kernel;
   persistent/warp-specialized scheduling is now a ~10% tail-chase, not a
   feasibility question.
2. With the forward at 0.9 ms, the unchanged backward (~1.46 ms at 6%
   compute / 9.6% DRAM utilization, v1 §3.5) is now **~62% of optimized
   fwd+bwd time**. The backward track is the largest remaining lever and is
   ordered before forward scheduling work below.

### 1.3 Validation ledger (this review, 2026-06-12)

```text
py_compile over src/sparton, training, tests, benchmarks  -> passed
hardened-env python -m pytest -q                          -> 47 passed, 15 warnings in 11.5 s
                                                             (warnings are upstream torch deprecations)
non-tiny optimized correctness probe (7 shapes, fp16+bf16,
  V-tail, batch-crossing S-tail, single-K-tile)           -> scores pass everywhere;
                                                             strict index equality FAILS at near-ties (F2)
index-mismatch classification probe                       -> every mismatch is a legitimate near-tie
                                                             (chosen logit within 1 fp16 ULP of ref max;
                                                             exact bf16 ties in the bf16 case); hybrid
                                                             matches the fp16 reference exactly
non-contiguous-input backward probe                       -> hybrid embed_grad WRONG (F1);
                                                             naive/optimized correct
torch.compile(fullgraph=True) capture of SpartonHead      -> passes for hybrid, naive, optimized
host-overhead probe (B=8,S=128,D=768,V=1280)              -> optimized ~0.119 ms host overhead/call,
                                                             ~0.051 ms of it descriptor-bank rebuild
error-surface probe                                       -> mixed dtypes: bare AssertionError (optimized) /
                                                             mid-kernel CompilationError (naive);
                                                             D=10: context-free "strides must be 16-byte aligned"
API check (Triton 3.6.0)                                  -> gluon.autotune absent (shim fallback correct);
                                                             gl.NVMMASharedLayout public; gl.max has no
                                                             return_indices parameter
```

---

## 2. Design evolution ledger (v1 plan → shipped implementation)

Future agents should treat these as **decisions, not drift**. They are
consistent with v1's stated principles (measured gates, no silent fallbacks,
hybrid default) even where they replaced v1's concrete mechanism.

| # | v1 said | What shipped | Assessment |
|---|---|---|---|
| E1 | M4 (router) then M5 (naive) as separate milestones | Landed together in `6d56ed6`; `backend` kwarg arrived with M5 | Fine; gates of both were met |
| E2 | Naive backend is a fixed-config debug baseline | M8 added bounded Triton autotune (10 configs, key=`(S,D,V)`), original config kept as a candidate | Good: makes the naive column a fair baseline; keeps M5 semantics |
| E3 | Canonical perf evidence on dev shape `B=32,S=128,D=768,V=30522` fp16 | Canonical benchmark became `naver/splade-code-06B` dims (`D=1024, V=151936`, bf16) on a fixed B×S grid; dev shape kept as a compat wrapper | Good: realistic target model; dev shape remains the ncu/floor anchor |
| E4 | M7 = subprocess-per-config GEMM sweep + policy generator | In-process Triton autotune over a bounded 12-policy universe, runtime device/problem pruning (`early_config_prune`), `--require-ratio` gate | Good: the production selection mechanism is itself the gate. Subprocess isolation was for *bring-up* deadlock safety; candidates are now pre-validated |
| E5 | M7 85% gate expected from intra-CTA pipelining work | Gate passed (86.45% fp16 / 86.34% bf16) by the **small-tile `64x64x64/3/2x2` policy** (2 CTAs/SM), not by pipelining | Important: occupancy was demoted in v1 §9 on cuBLAS-comparison grounds, but it is what actually closed the gap. Update priors: tile-shape sweep ≥ pipelining quality on this GPU |
| E6 | M8 = fixed-config O1 kernel | Policy-autotuned O1: fixed 11-policy production universe as 22 descriptor kernel args, constexpr `POLICY_ID` selects the pair, runtime-derived candidates prune the config list, tiny problems collapse to the `64³/3/2x2` fallback | Sound given Triton's autotune API; carries real costs (per-call bank rebuild, 22-arg signature, duplicated if-chain) — see F9/D2 |
| E7 | Epilogue expected via `gl.max(..., return_indices=True)` | Explicit `gl.reduce` over `(value, row_index)` with a strict-`>`-plus-lowest-index combine; local `gl.max` has no `return_indices` at all | Correct and the right shape for the contract; recheck on Triton upgrades |
| E8 | `gluon.autotune` assumed | Absent in 3.6.0; shim exports `triton.autotune` fallback | Correct |
| E9 | Whitelist maps each CC major explicitly | Simplified to `major >= 8 → mma_v2`, `< 8 → error` | Same behavior; fine |
| E10 | Optimized gating "on RTX 5090" | Tests/probes gate on capability + importable Gluon (`is_gluon_backend_available`), benchmarks stay machine-specific | Good portability hygiene |
| E11 | (not in v1) | `_gluon_policy_runtime.py` extracted so the backend and the GEMM benchmark share host-side plumbing; kernel bodies intentionally kept separate | Right call; see D2 for the kernel-body duplication rationale |

The numerics also evolved silently (nobody decided it, the architecture did):
hybrid computes logits in the input dtype (cuBLAS fp16/bf16 + fp16 bias add),
while naive/optimized accumulate in fp32 and apply bias/mask/max in fp32.
naive and optimized agree with each other and are *more* precise than hybrid;
the consequence is F2 below. This is adopted as intended behavior in §4.3.

---

## 3. Review findings

Severity: **[blocker]** must fix for a production-ready state; **[should]**
fix in M9, low risk; **[note]** recorded, scheduled later or accepted.

### 3.1 Correctness

- **F1 [blocker] Hybrid backward silently computes wrong `embed_grad` for
  non-contiguous `hidden`.** `SpartonHead` (hybrid) calls
  `fused_sparton_fwd_op` directly; autograd saves the raw tensors; the
  backward kernel computes flat offsets assuming dense `[B,S,D]` strides
  (`_backend_hybrid.py:566` onward). Probe: `hidden = base[:, ::2, :]` →
  `embed_grad` wrong, while `hidden_grad` *happens* to be right because
  `torch.zeros_like` on a non-dense view returns a contiguous buffer that the
  kernel's dense offsets index correctly. naive/optimized are immune because
  their wrappers call `.contiguous()` before the op, so autograd saves
  contiguous tensors. Forward output is correct in all cases (matmul handles
  strides), which makes the gradient corruption silent. Fix in M9-T1.

### 3.2 Contract and coverage gaps

- **F2 [blocker] The index contract is mis-specified by the tests.** The
  documented contract (v1 §10.1) is "exact equality where the winning logit is
  positive and non-tied; ambiguous otherwise", but the tests assert plain
  `torch.equal` and only pass because the tiny seeded shapes avoid near-ties.
  On realistic shapes (probe: `B=8,S=128,D=768,V=1283` fp16; `B=3,S=345,
  D=768,V=2048` bf16) optimized/naive legitimately pick different near-tie
  winners than the input-dtype reference: every mismatch's chosen logit is
  within one fp16 ULP of the reference max (fp16) or exactly equal in
  reference precision (bf16). A tie-aware index assertion (spec in §6.2) and
  realistic-shape tests are required; the contract itself is fine.
- **F3 [blocker] No test exercises the optimized backend's non-fallback
  policies.** Every optimized pytest case is "tiny" under
  `_runtime_policy._is_tiny_problem` (`_runtime_policy.py:249`: `M<1024 or
  N<1024 or K<64`), so the suite only ever compiles/runs the `64³/3/2x2`
  fallback. The production policy bank is exercised solely by benchmarks and
  the epilogue probe. The review's non-tiny probes pass (scores everywhere,
  ties per F2), so this is a coverage gap, not a latent bug — but pytest green
  currently proves much less than it appears to. Fix in M9-T3.

### 3.3 Error surfaces (production UX)

- **F4 [blocker] Invalid inputs fail with internals, not contracts.**
  Probe-verified: `hidden` fp16 + `embed` fp32 → naive raises a mid-kernel
  `CompilationError`, optimized raises a **bare `AssertionError` with no
  message**; `D=10` (row stride not 16-byte aligned) → optimized raises
  `AssertionError: strides must be 16-byte aligned` with no mention of which
  input or what the requirement derives from. Missing validations: dtype
  equality across `hidden`/`embed`/`bias`, supported-dtype set per backend
  (optimized is fp16/bf16-only by descriptor `element_bitwidth=16`;
  `D * itemsize % 16 == 0` for TMA), shape/device checks with input names.
  Fix in M9-T2.

### 3.4 Hygiene, staleness, metadata

- **F5 [should] `AGENTS.md` is stale after M8**: line ~131 still instructs
  that `backend="optimized"` "should continue to fail clearly until the Gluon
  milestones are implemented and validated"; the Project Map and Core Kernel
  Invariants sections do not mention the four new modules or the optimized
  path. M8 did not touch `AGENTS.md` at all.
- **F6 [should] Dead code in `_backend_hybrid.py`** (faithfully moved in M3,
  never cleaned): unused imports `gc`, `time`, `torch.amp`, `gradcheck`,
  `torch._dynamo` (lines 1–10); ~40 lines of commented-out `FusedSparton`
  autograd class (line 601+); a commented-out stale autotune block; the
  vestigial 4-tuple return `return hidden_grad, embed_grad, bias_grad, None`
  (`_backend_hybrid.py:598`) whose fourth element no caller uses.
- **F7 [should] Import-time side effects in `_backend_hybrid.py`**:
  `torch.set_float32_matmul_precision('high')` (line 12) mutates **global**
  process state on import of a library module; `print(f"Sparton using device:
  ...")` (line 15) writes to stdout on import (a known sharp edge);
  `SpartonHead.load` prints `"no bias"` (`sparton_kernel.py:98`). For the
  supported fp16/bf16 paths the precision knob is irrelevant (it governs fp32
  matmuls only), so removing it is safe for every documented path; treat any
  fp32-hidden user as out of contract (validated in M9-T2).
- **F8 [should] Wrapper asymmetry** (root cause of F1):
  `_forward_op_for_backend` returns the raw op for hybrid but wrappers for
  naive/optimized. One rule should hold: *every* public entry goes through a
  per-backend wrapper that validates and canonicalizes inputs; ops assume
  canonical inputs.
- **F11 [should] Training integration nits**: `training/model.py:121` error
  message names `'pytorch'`/`'triton'` though accepted values are
  `torch|compiled|sparton`; `SpartonSpladeHead` cannot select a Sparton
  backend (always default).
- **F12 [should] Packaging metadata is wrong for any release**: `pyproject`
  declares `license = {text = "MIT"}` while `LICENSE` is Apache-2.0 (v1
  flagged this as do-not-silently-fix; M9 fixes it *deliberately*); author
  placeholder "Your Name <you@example.com>".
- **F13 [should] Test-infra fragility**:
  `test_bench_gluon_gemm_autotune_configs_follow_policy_generator` does `from
  benchmarks.bench_gluon_gemm import ...`, which resolves only because
  `python -m pytest` puts the CWD on `sys.path` (namespace package). Running
  the `pytest` console script from anywhere else breaks collection.
- **F14 [note] Shim niceties**: `_gluon_runtime._load_gluon` imports
  `NVMMASharedLayout` from the private `language._layouts` module
  (`_gluon_runtime.py:107`) although `gl.NVMMASharedLayout` is public
  (verified); `is_gluon_backend_available` loads the Gluon namespace but
  discards it instead of caching into `_STATE` (`_gluon_runtime.py:90`).
- **F16 [note]** `_problem_from_autotune_args` hardcodes `dtype_name="fp16"`
  (`_backend_optimized_gluon.py:50`). Harmless today (fp16 and bf16 are both
  2 bytes, and Triton's autotune cache key includes tensor-arg dtypes via the
  output pointer), but the intent deserves a comment/assert rather than a lie.
- **F20 [note]** `resolve_backend` error for a bad `SPARTON_BACKEND` value
  does not say the value came from the environment variable.

### 3.5 Measured non-blockers (scheduled, not M9)

- **F9 [note → M12] Per-call host overhead of the optimized launch path.**
  Every `optimized_forward` call rebuilds all 22 `TensorDescriptor`s
  (~0.051 ms) plus autotuner dispatch — ~0.119 ms host time/call measured.
  Irrelevant on ≥1 ms shapes; dominant for small/latency workloads. Addressed
  by the launcher-v2 design in M12 (one descriptor pair per call after
  selection), not by patching the bank.
- **D2 [accepted] Kernel-body duplication** between
  `_backend_optimized_gluon.py` and `benchmarks/bench_gluon_gemm.py` (mainloop
  + the 11/12-way descriptor if-chain). Deliberately **not** deduplicated in
  M9: the two kernels already differ in epilogue and barrier-reset structure,
  M12 will rewrite the production mainloop (persistent/WS) and they diverge
  further; a shared `@gluon.jit` mainloop helper would churn validated kernel
  code right before a planned rewrite. Cross-reference comments are added
  instead (M9-T4). Revisit only if a third copy ever appears.

---

## 4. Architecture rules going forward

### 4.1 Layering and the symmetry rule

After M9-T1/T2, the invariant is:

```text
SpartonHead.forward
  └─ <backend>_forward(hidden, embed, bias, mask)      # public per-backend wrapper
       ├─ _validate_inputs(...)                        # shared, raises contract errors
       ├─ .contiguous() canonicalization (all inputs)
       └─ sparton::<backend>_fwd custom op             # stable schema; assumes canonical inputs
            └─ kernel launch (+ autograd registration that saves the op's inputs)
```

- Wrappers are the only public callables (`hybrid_forward`, `naive_forward`,
  `optimized_forward`); `SpartonHead` binds wrappers, never raw ops.
- Op schemas never change; new behavior = new op name (v1 §4.3 rule stands).
- Backends never import each other except `_backend_hybrid`'s backward op
  (the single backward implementation until M11).
- Only `_gluon_runtime` may import `triton.experimental.gluon.*`;
  `_runtime_policy` must stay importable with no triton/torch at module level
  (it is imported by tests on CPU-only machines).

### 4.2 Policy-bank mechanics (document of record)

How the optimized forward actually selects a configuration — previously only
discoverable by reading three files:

1. `_runtime_policy.optimized_forward_policy_universe()` returns a **fixed,
   ordered 11-policy universe** (fallback `64x64x64/3/2x2/sw128` first, then
   the production candidates, `BLOCK_N ≤ 128`). The kernel's signature bakes
   in 11 descriptor *pairs* (22 args); `POLICY_ID` (constexpr) selects a pair
   via an if-chain, so each compiled variant dead-codes the other 21 args.
2. At decoration time, the universe becomes 11 `triton.Config`s
   (`POLICY_ID=i` + the policy's block/warp/stage constexprs).
3. At launch, the autotuner (key `B,S,D,V`; tensor dtypes are appended to the
   cache key by Triton) calls `early_config_prune`, which derives the **active
   candidate set** from the real `DeviceProfile` and `ProblemSpec`
   (`derive_optimized_forward_policies`): resource-invalid policies drop,
   tiny problems (`M<1024 or N<1024 or K<64`) collapse to the fallback alone,
   the rest are ranked (wave fill, occupancy, tails). Empty set → hard error
   (no silent fallback).
4. The host builds descriptors for **all 11** policies each call
   (`make_descriptor_bank`), because any pruned-in config may win.

Constraints that must hold for any future policy added to the universe:
`BLOCK_K * itemsize ≥ swizzle_byte_width`; stage memory
`NUM_STAGES*(BLOCK_M+BLOCK_N)*BLOCK_K*2B + barriers ≤ 101376 B`;
`BLOCK_N ≤ 128` in the production universe (the 128×256 evidence policy lives
only in the benchmark universe, slot 12). The universe length is asserted at
import (`_backend_optimized_gluon.py:26`); changing it means changing the
kernel signature and the if-chain together.

### 4.3 Numerics and index contract (adopted)

- **Scores**: hybrid produces input-dtype logits (cuBLAS + input-dtype bias
  add); naive/optimized accumulate in fp32 and apply bias/mask/max/log1p in
  fp32 before casting the stored score to the input dtype. The fp32 path is
  *more* accurate and is the intended behavior for all future backends.
  Cross-backend score agreement is within the existing tolerances
  (fp16 2e-3, bf16 5e-2).
- **Indices**: zero-baseline, strict-`>` semantics are unchanged: the index is
  meaningful only where the returned score is > 0; within a backend, ties
  resolve to the lowest sequence index. **Across implementations with
  different accumulation precision, the winner at a near-tie (gap within
  input-dtype rounding) is explicitly unspecified.** The testable contract is
  §6.2: wherever the score is positive, the returned index must point at a
  masked logit within tolerance of the true maximum. Backward correctness is
  insensitive to which near-tie winner was chosen (gradient flows through a
  position whose logit differs by ≤1 ULP).

---

## 5. Milestone plan

Numbering continues from M8. Mapping to v1: new **M9** is new (this review);
**M10** ≈ v1 M11 (promotion) — deliberately moved before performance work;
**M11** ≈ v1 M10 (backward); **M12** ≈ v1 M9 (persistent/WS forward) plus the
launcher v2. Rationale for the reorder: §1.2 — the forward already clears
every promotion-relevant perf bar, and the backward is now the dominant cost,
so promotion should not wait on either optimization track.

### M9 — Production readiness (the cleanup milestone)

Goal: with M9 complete and **no later milestone**, the repository is a clean,
correct, honestly-documented production state: hybrid default, optimized as a
validated opt-in, no silent-wrong-answer paths, contract-named errors, tests
that prove what they appear to prove, accurate docs/metadata.

Scope discipline: **no kernel-body changes, no autotune-config changes, no
schema changes, no new dependencies.** Every task below is independently
land-able; the listed order minimizes rebasing (T1 → T2 → T3 are sequential;
T4–T6 are parallel after T1).

#### M9-T1 Wrapper symmetry and the F1 fix

- `_backend_hybrid.py`: add `hybrid_forward(hidden, embed, bias, mask)`
  mirroring `naive_forward` byte-for-byte in structure: `.contiguous()` on
  `hidden`/`embed`/`mask` (+ `bias` when present), then
  `return fused_sparton_fwd_op(...)`. Do not touch the op or kernels.
- `sparton_kernel.py`: `_forward_op_for_backend("hybrid")` returns
  `hybrid_forward`; add `hybrid_forward` to the facade imports and `__all__`.
  Keep `fused_sparton_fwd_op` exported (compat).
- `resolve_backend`: when the invalid value came from `SPARTON_BACKEND`
  (i.e. `backend is None`), say so in the `ValueError` message (F20).
- Tests (extend `tests/test_sparton_kernel.py`): parametrized over all three
  backends — build non-contiguous `hidden` (`base[:, ::2, :]`), non-contiguous
  `embed`/`bias` (row/element slices of larger tensors), run forward+backward,
  assert score and **all three gradient** parities vs the reference computed
  on contiguous clones (this is the F1 regression test; it must fail on
  pre-T1 hybrid).
- Gate: new tests green; full suite green; one benchmark row
  (`--batch-sizes 32 --seq-lens 128 --dim 768 --vocab 30522 --dtype fp16`)
  within ±5% of §1.2 (contiguity canonicalization is a no-op for the
  benchmark's already-contiguous inputs).

#### M9-T2 Input validation with contract-named errors

- New `src/sparton/_validation.py` (no triton import), one entry point:

  ```python
  def validate_forward_inputs(hidden, embed, bias, mask, *, backend: str) -> None
  ```

  Checks, each raising `ValueError`/`TypeError` that names the offending
  argument, the actual value, and the requirement:
  - `hidden.ndim == 3`, `embed.ndim == 2`, `mask.ndim == 2`,
    `mask.shape == hidden.shape[:2]`, `embed.shape[1] == hidden.shape[2]`,
    `bias is None or bias.shape == (embed.shape[0],)`;
  - all tensors on the same CUDA device;
  - `hidden.dtype == embed.dtype` (and `bias.dtype` when present);
  - dtype ∈ {fp16, bf16} for `naive`/`optimized`; hybrid additionally permits
    fp32 (existing, documented-as-legacy behavior);
  - `optimized` only: `(hidden.shape[2] * hidden.element_size()) % 16 == 0`,
    with the error text explaining the TMA 16-byte stride-alignment origin and
    naming `D`; same check for `embed`.
  - mask dtype: accept bool/integer/floating; document (README + docstring)
    that non-binary masks weight logits (defined behavior, not the HF
    attention-mask contract) — v1 §5 stance, now written at the entry point.
- Call it at the top of each of the three wrappers.
- Tests: one `pytest.raises(..., match=...)` per rule per relevant backend,
  including the two probe reproducers (mixed dtype; `D=10` on optimized).
- Gate: the M9-T2 error-surface probe (Appendix A) now prints contract errors
  for every case; suite green. Wrapper overhead stays negligible (pure-Python
  checks, no device syncs; spot-check with the host-overhead probe).

#### M9-T3 Honest validation: tie-aware index assertions + non-tiny coverage

- Add to `tests/test_sparton_kernel.py` (or a `tests/_asserts.py` helper):

  ```python
  def assert_index_contract(scores, idx, hidden, embed, bias, mask, *, atol, rtol):
      """Wherever score > 0, the chosen index must hold a masked logit within
      tolerance of the per-(b, v) reference max; deterministic-tie tests keep
      exact equality separately."""
      masked = reference_masked_logits(hidden, embed, bias, mask)   # input dtype
      ref_max = masked.max(dim=1).values
      chosen = masked.gather(1, idx.unsqueeze(1)).squeeze(1)
      active = scores.float() > 0
      gap = (ref_max.float() - chosen.float())[active]
      assert (gap <= atol + rtol * ref_max.float()[active].abs()).all()
  ```

- Keep the existing deterministic semantic-case tests (intentional exact ties,
  masked-winner cases) on `torch.equal` — those pin the strict-`>` policy.
  Replace `torch.equal(idx, expected_idx)` in the *random-input* forward tests
  with `assert_index_contract` (hybrid may keep exact equality: probe shows it
  matches the input-dtype reference bit-for-bit).
- Add `@pytest.mark.slow` non-tiny optimized+naive tests with exactly the
  review-probe shapes (they are chosen to cover every structural edge):
  `B=8,S=128,D=768,V=1283` fp16 bias+no-bias (V-tail, full policy bank);
  `B=3,S=345,D=768,V=2048` fp16+bf16 (S % BLOCK_M ≠ 0 → A-tile rows cross the
  batch boundary — the v1 §7.3 correctness rule); `B=9,S=120,D=64,V=1024`
  (single K-tile); `B=2,S=513,D=1024,V=1536` no-bias (long S). Scores via
  `assert_close`, indices via `assert_index_contract`.
- Document the §4.3 index contract in README (one short paragraph under
  Backend Selection).
- Gate: full suite green including `-m slow` on the CUDA workspace; quick runs
  documented as `python -m pytest -q -m "not slow"`.

#### M9-T4 Library hygiene (behavior-preserving except where stated)

- `_backend_hybrid.py`: delete unused imports (`gc`, `time`, `torch.amp`,
  `gradcheck`, `torch._dynamo`); delete the commented-out `FusedSparton` class
  and stale commented autotune block; drop the vestigial fourth return value
  of `fused_sparton_bwd_with_bias` **and** its unpacking in
  `fused_sparton_bwd_op` (private helper; facade re-export signature is the
  function object itself, callers in-repo only).
- **Deliberate behavior changes** (each gets its own CHANGELOG line):
  - remove `torch.set_float32_matmul_precision('high')` — a library import
    must not mutate global matmul precision. Supported dtypes are unaffected
    (the knob governs fp32 matmuls); any fp32-hidden hybrid user now gets
    torch's default precision, which is the user's setting to make.
  - replace `print(f"Sparton using device: ...")` with `logger =
    logging.getLogger("sparton")` + `logger.debug(...)`; keep the `DEVICE`
    symbol and export.
  - replace `SpartonHead.load`'s `print("no bias")` with
    `logger.debug("SpartonHead.load: checkpoint has no bias for this head")`.
  - update the two tests/docs that account for the import-time print
    (`AGENTS.md` sharp-edges list; any command-output comparisons).
- `_gluon_runtime.py`: use public `gl.NVMMASharedLayout`; make
  `is_gluon_backend_available` cache the loaded namespace into `_STATE` (F14).
- `_backend_optimized_gluon.py:44`: comment + assert documenting why
  `dtype_name="fp16"` is safe (both supported dtypes are 16-bit; autotune
  cache keys include tensor dtypes) (F16).
- Add the D2 cross-reference comments at both kernel bodies ("structure
  intentionally mirrors <other file>; see design v2 §3.5/D2 before
  deduplicating").
- Gate: suite green; `import sparton` on the CUDA box prints nothing at
  default logging config; benchmark row within ±5%.

#### M9-T5 Documentation and metadata truth

- `AGENTS.md`: Project Map gains the four new modules and their one-line
  roles; Core Kernel Invariants describes the three-backend layering (§4.1)
  and replaces the stale "optimized should fail clearly" line with "optimized
  is an experimental opt-in; hybrid remains the default until the M10 gates
  pass"; benchmarks map row points at this document; sharp-edges list drops
  the import-print/`"no bias"` entries (fixed in T4) and the license-mismatch
  entry (fixed below).
- `pyproject.toml`: `license = {text = "Apache-2.0"}` to match `LICENSE` and
  upstream (the v1 "don't silently fix" rule is satisfied: this is the
  explicit, changelogged fix); leave author/URL fields unless the repo owner
  states otherwise (upstream attribution is plausible-intentional — flag in
  the PR description, do not guess).
- `training/model.py`: fix the invalid-head message to name
  `'torch'/'compiled'/'sparton'`; add optional `sparton_backend: str | None =
  None` threaded to `SpartonHead(..., backend=...)` (default `None` keeps
  today's behavior exactly).
- `README.md`: index-contract paragraph (T3); correct the optimized
  requirements line (sm_80+, Triton with Gluon, validated 3.6.0).
- `CHANGELOG.md`: entries for every T1–T6 user-visible change under the
  landing date.
- Gate: `git diff` review of docs; AGENTS validation commands re-run verbatim.

#### M9-T6 Test-infra hardening

- `tests/conftest.py`: insert the repo root at the front of `sys.path`
  (computed from `__file__`) so `import benchmarks.bench_gluon_gemm` works
  under any pytest invocation (F13); keep `pythonpath = ["src"]` in
  `pyproject` as-is.
- Verify `pytest -q` (console script) and `python -m pytest -q` both collect
  and pass from the repo root.
- Gate: both invocations green.

#### M9 exit checklist (run in order, hardened env, serial)

```bash
ENV='TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas CPATH=/usr/local/cuda-13.2/include TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor'
env $ENV PYTHONPATH=src python -m py_compile src/sparton/*.py training/*.py tests/*.py benchmarks/*.py
env $ENV python -m pytest -q                      # full, incl. slow
env $ENV python -m pytest -q -m "not slow"        # the documented quick loop
env $ENV PYTHONPATH=src python -u benchmarks/bench_sparton_baseline.py \
    --batch-sizes 32 --seq-lens 128 --dim 768 --vocab 30522 --dtype fp16 --optimized-policy on
#   -> hybrid/optimized columns within ±5% of §1.2
env $ENV PYTHONPATH=src python -c "import sparton"   # no stdout output
git diff --check
```

Plus: M9 milestone memo under `docs/` (per AGENTS documentation discipline)
recording the F-list → fix mapping and the exit-checklist transcript.

### M10 — Promotion decision for the optimized forward

Entry condition: M9 complete. The candidate configuration is **optimized
forward + existing Triton backward** (the M8 shape) — promoting before the
backward/scheduling tracks is deliberate: every gate below is already
plausibly met, and hybrid remains one constructor argument away.

Gates (all required; adapted from v1 §12.3):

1. Correctness matrix green (§6) including `-m slow`, fp16+bf16, bias/no-bias.
2. Forward faster than hybrid on the canonical grid (all 9 rows) and the dev
   shape, by the recorded margins (re-measured at gate time).
3. Fwd+bwd net ≥ hybrid on the same shapes (same backward ⇒ follows from 2;
   measure anyway and record).
4. Memory gate: peak extra ≤ 2× outputs on dev shape and grid corners.
5. **Shape soak**: scripted sweep over S ∈ {1, 7, 64, 127, 128, 129, 255,
   511}, B ∈ {1, 2, 5}, D ∈ {768, 1024}, V ∈ {30522, 151936}, bias ∈ {y, n},
   dtype ∈ {fp16, bf16}, random masks incl. all-zero rows — scores vs
   reference + index contract. (Cheap: reuses the §6.2 checker; catches
   policy-bank edge interactions the fixed test shapes cannot.)
6. **Training-integration smoke, tier 1 (mandatory, no downloads)**:
   synthetic head-only training — random hidden states, contrastive-style
   loss + FLOPS-style sparsity regularizer, 300 optimizer steps, fp16 AMP and
   bf16 autocast variants; assert finite grads/loss throughout and
   hybrid-vs-optimized loss curves match within noise (same seed).
   Tier 2 (recommended, requires installing `transformers`/`accelerate` into
   the venv per AGENTS rules): a few-hundred-step `training/train.py` run on a
   small dataset slice with `head="sparton"`, `sparton_backend="optimized"`
   vs `"hybrid"`.
7. Docs: README flips the recommendation; CHANGELOG records the default
   change and the rollback (`backend="hybrid"` / `SPARTON_BACKEND=hybrid`).

Mechanics of the switch: `resolve_backend`'s default (and only it) changes
from `"hybrid"` to `"optimized"` **gated on
`_gluon_runtime.is_gluon_backend_available()` at `SpartonHead` construction —
if unavailable, default resolution falls back to hybrid with a one-time
`warnings.warn`** (explicit env/kwarg `optimized` still hard-errors, no silent
fallback for explicit requests; only the *default* is environment-adaptive).
This keeps CPU-less/old-Triton installs importable and is the single
deviation from "no fallbacks", confined to default resolution and loudly
warned.

### M11 — Backward track (now the largest lever)

Evidence: backward ≈ 1.46 ms on the dev shape at 6.0% compute / 9.6% DRAM
utilization, 167 MB read / 66 MB written (v1 §3.5) — latency/atomic-bound,
~62% of optimized fwd+bwd.

1. **Distribution harness first** (entry gate): extend
   `bench_sparton_baseline.py` (or a sibling) with a backward benchmark over
   index distributions: uniform (today's tests), Zipfian over vocab with hot
   tokens (documented proxy), and — when available — indices captured from a
   real SPLADE checkpoint forward (naver/splade-v3 is license-gated; a
   few-hundred-step trained model from M10-tier-2 is the fallback source).
   Uniform-random understates atomic conflicts; no backward change is
   accepted on uniform evidence alone (v1 §8 rule, restated).
2. **Two prototypes, one decision** (timeboxed): (a) *Triton B2a* — keep the
   kernel family, add per-CTA aggregation of `d_bias`/`d_embed` in
   registers/smem before atomics and re-tune block shapes (the 6%-util kernel
   leaves enormous room without new technology); (b) *Gluon B2b* — direct
   port with TMA loads and the same aggregation. Pick by measured fwd+bwd on
   the harness; correctness matrix identical to forward gates plus
   `scores == 0 → zero gradient` and AMP smoke.
3. B3 (deeper aggregation, duplicate-`d_hidden` handling) only if the harness
   shows atomic conflict still dominant after B2x.
4. Exit: backward ≥ 1.5× faster than current on the realistic-distribution
   harness without regressing uniform; gradient matrix green; fwd+bwd
   re-recorded in the M11 memo. New op name (`sparton::optimized_bwd` or a
   versioned name) if and only if saved-tensor needs change — autograd
   `setup_context` is backend-private, so swapping the backward op inside
   existing autograd registrations is schema-safe.

### M12 — Forward scheduling and launch overhead (tail-chase, optional)

Entry gate: profiling shows ≥10% recoverable forward time on shapes someone
cares about, after M10/M11. Candidates, in measured-priority order:

1. **Launcher v2 (also fixes F9)**: replace decorator autotune with an owned
   two-phase selector — derive candidates (`derive_optimized_forward_policies`
   already ranks), benchmark once per `(B,S,D,V,dtype)` key with the existing
   do_bench infra, memoize, then build **one** descriptor pair per call and
   call a single-pair kernel (22-arg bank and if-chain deleted). This is the
   elegant end-state the bank approximated under Triton's autotune API;
   keep `cache_results`-style on-disk memoization optional. Exit: host
   overhead ≤ 0.02 ms/call; identical kernel selection on the canonical grid.
2. **Persistent + warp-specialized mainloop** (v1 O2/O3): `gl.warp_specialize`
   probe on sm_120 first (still unprobed — v1 §3.6 item 2); persistent flat
   tile loop with L2-aware rasterization; producer/consumer barriers replace
   the per-iteration CTA barrier; epilogue unchanged. Targets: forward ≤
   1.05× the per-row GEMM floor on the canonical grid (currently ~1.10×);
   tensor-pipe utilization toward the 86.6% cuBLAS reference. The per-s-tile
   pipeline drain and embed-tile re-read of the O1 kernel disappear naturally
   in the persistent formulation; measure, don't assume.
3. Mask-density sweep (v1 §10.3) folded into the M12 measurement set.

Each candidate keeps v1 §9's rejection criteria (measured A/B on dev + grid,
>5% regression anywhere → rejected without a >10% win elsewhere).

---

## 6. Validation matrix v2

### 6.1 Shapes and cases (every backend change re-runs this)

- Unit/semantic: existing tiny seeded cases + deterministic tie/mask cases
  (exact-equality assertions, pin strict-`>` and zero-baseline).
- Structural (slow-marked): the four §M9-T3 shapes covering V-tail,
  batch-crossing S-tail, single-K-tile, long-S; fp16+bf16 × bias/no-bias.
- Canonical perf: dev shape (fp16) + splade-code-06B grid (bf16) via
  `bench_sparton_baseline.py`; record, don't gate (gates live in milestones).
- Soak (promotion and backward milestones): §M10 gate 5 sweep.
- Memory: allocator peak-vs-outputs per backend on dev shape (≤2× outputs for
  fused backends; hybrid recorded for reference).
- Training: §M10 gate 6 tiers.

### 6.2 Index assertion of record

The tie-aware checker from M9-T3 **is the index contract**: for every
position with score > 0, the masked input-dtype logit at the returned index
is within `atol + rtol·|max|` of the per-(b,v) maximum (fp16 2e-3/2e-3, bf16
5e-2/5e-2); positions with score == 0 are unconstrained (zero-baseline).
Exact index equality remains asserted only in deterministic constructed
cases and (empirically, as long as it holds) for hybrid vs the input-dtype
reference.

### 6.3 Tolerances

Unchanged from v1 §10.1: scores fp16 `atol=rtol=2e-3`, bf16 `5e-2`; gradients
fp16 `2e-3` (vs fp32-reference autograd on contiguous clones).

---

## 7. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| M9 hygiene edits regress hybrid perf via accidental behavior change | low | T-gates re-run the dev-shape benchmark row (±5%); no kernel/config edits allowed in M9 |
| Promotion (M10) exposes a policy-bank edge the soak missed | medium | rollback is one kwarg/env; default-resolution fallback warns loudly; hybrid path untouched |
| Adaptive default (M10) surprises a user expecting hard failure | low | confined to default resolution; explicit `optimized` still hard-errors; CHANGELOG + README |
| Realistic index distributions unavailable for M11 | medium | Zipfian proxy documented as proxy; tier-2 training run produces real indices; no promotion of backward on uniform-only evidence |
| `gl.warp_specialize` immature on sm_120 (M12) | medium | probe before committing; launcher v2 and persistent-without-WS are independent wins |
| Triton upgrade moves Gluon APIs | medium | shim + `VALIDATED_TRITON` warning already in place; M9-T4 keeps shim the only Gluon importer; re-run M7 ratio gate + epilogue probe on any bump |
| `Tensor?` returns on real torch 2.7.1 floor | low | unchanged from v1 §3.4: one suite run on 2.7.1 before advertising no-bias training on the floor; split-op fallback documented |

---

## 8. Rejected / deferred (with rationale)

- **Deduplicating the Gluon kernel bodies now** — rejected for M9 (D2):
  validated-kernel churn immediately before the M12 rewrite; helpers already
  share all host-side plumbing.
- **Fixing F9 by caching descriptor banks** — superseded by launcher v2
  (M12), which removes the bank instead of caching it.
- **Restricting hybrid to fp16/bf16** — rejected; fp32 hybrid works today and
  may have silent users. Validated as "permitted, legacy, not
  benchmark-covered" (M9-T2).
- **Hybrid-side performance fixes** (v1 §9 items 7–9: bias fusion, tile-count
  tuning, slice-copy elimination — `_backend_hybrid.py:248`) — deferred
  indefinitely: after M10, hybrid is the compatibility path, not the perf
  path; spending optimization effort there double-pays. Reconsider only if
  promotion is rejected.
- **CI setup** — out of scope: no CUDA runner available to this repo; the
  documented hardened-env command set remains the gate mechanism.
- **WGMMA/TCGen05/FP8/clusters/TMA-gather/CUTLASS/split-K** — unchanged from
  v1 §9 (hardware/scope rejections all still hold on this platform).
- **Migrating custom ops to inferred schemas** — still rejected (no
  `Optional[Tensor]` return support in schema inference; explicit strings
  stay).

---

## Appendix A. Review probe provenance (2026-06-12)

Probes were run from the repo root with the hardened env (v1 §2.4), serially.
They are small and intentionally disposable (superseded by the M9-T2/T3 tests
that encode the same checks); rerun recipes:

1. **Non-tiny optimized correctness** — 7 shapes
   (`8×128×768×1280|1283 fp16 ±bias`, `3×345×768×2048 fp16+bf16`,
   `9×120×64×1024 fp16`, `2×513×1024×1536 fp16 no-bias`), inputs
   `randn×0.05`, mask density 75%: scores `assert_close` pass on all; strict
   `torch.equal` on indices fails with 2–67 mismatches per shape.
2. **Mismatch classification** — for every mismatch, compare the reference
   masked logit at the kernel's index against the reference max:
   max gap 0.000122 (= 1 fp16 ULP at that magnitude) on fp16, 0.0 on bf16;
   hybrid: 0 mismatches. naive and optimized mismatch identically (both
   fp32-accumulate).
3. **Non-contiguous backward** — `hidden = base[:, ::2, :]` (strides
   (160,32,1)), fp16, all backends: hybrid `embed_grad` max-abs-diff vs
   reference ≈ 1e0-scale wrong (assert fails), naive/optimized pass; all
   `hidden_grad`s pass (see F1 for why).
4. **torch.compile capture** — `torch.compile(head, fullgraph=True)` forward
   for each backend: all pass (fake registrations correct).
5. **Host overhead** — 300-call wall vs `do_bench` GPU time at
   `8×128×768×1280` fp16: hybrid 0.097/0.026 ms, naive 0.035/0.022 ms,
   optimized 0.183/0.064 ms; descriptor-bank build alone 0.051 ms (22
   descriptors).
6. **Error surfaces** — fp32 `embed` against fp16 `hidden`: naive
   `CompilationError` (mid-kernel), optimized bare `AssertionError`; `D=10`
   optimized: `AssertionError: strides must be 16-byte aligned`.
7. **API check** — `gluon.autotune` absent; `gl.NVMMASharedLayout` public;
   `gl.max` has no `return_indices` parameter (Triton 3.6.0).

## Appendix B. Rerun commands

```bash
ENV='TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas CPATH=/usr/local/cuda-13.2/include TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor'
PY=/workspace/venvs/sparton/bin/python

# Baseline validation (M9 exit checklist is the superset):
env $ENV PYTHONPATH=src $PY -m py_compile src/sparton/*.py training/model.py training/train.py tests/*.py benchmarks/*.py
env $ENV $PY -m pytest -q
env $ENV PYTHONPATH=src $PY -c "import torch, sparton.sparton_kernel as sk; print(torch.ops.sparton.optimized_fwd.default._schema)"

# Canonical benchmarks:
env $ENV PYTHONPATH=src $PY -u benchmarks/bench_sparton_baseline.py --optimized-policy on
env $ENV PYTHONPATH=src $PY -u benchmarks/bench_sparton_baseline.py \
    --batch-sizes 32 --seq-lens 128 --dim 768 --vocab 30522 --dtype fp16 --optimized-policy on

# M7 ratio gates (re-run on any Triton bump):
env $ENV PYTHONPATH=src $PY -u benchmarks/bench_gluon_gemm.py --dtype fp16 --include-block-n-256 --require-ratio 85
env $ENV PYTHONPATH=src $PY -u benchmarks/bench_gluon_gemm.py --dtype bf16 --include-block-n-256 --require-ratio 85
env $ENV PYTHONPATH=src $PY -u benchmarks/probe_gluon_epilogue.py

# Profiling: unchanged from v1 §11 / Appendix B.
```
