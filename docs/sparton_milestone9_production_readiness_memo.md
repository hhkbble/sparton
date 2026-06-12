# Sparton Milestone 9 Production Readiness Memo

Date: 2026-06-12.

## Summary

Milestone M9 from
[sparton_remaining_work_design_v2.md](sparton_remaining_work_design_v2.md)
is complete. Every review finding scheduled for M9 is fixed, with tests that
encode the fix; the milestone deliberately contains **no kernel-body changes,
no autotune-config changes, no custom-op schema changes, and no new
dependencies**. Hybrid remains the default backend; `optimized` remains an
experimental opt-in pending the M10 promotion decision.

The test suite grew from 47 to 105 tests (90 in the `-m "not slow"` quick
loop). The dev-shape benchmark is unchanged within noise before and after the
milestone.

## Finding → fix mapping

| Finding (design v2 §3) | Fix | Where |
|---|---|---|
| F1 wrong gradients for non-contiguous inputs on the hybrid autograd path | `hybrid_forward` wrapper canonicalizes (validates + `.contiguous()`) before the op, so autograd saves dense tensors, matching naive/optimized; red→green regression test across all three backends | `_backend_hybrid.py` (`hybrid_forward`), `sparton_kernel.py` routing, `test_forward_backward_handles_noncontiguous_inputs` |
| F2 index near-tie contract mis-specified by tests | Tie-aware `assert_index_contract` helper (design §6.2) replaces exact equality in the random-input naive/optimized tests; deterministic constructed cases and the (bit-exact) hybrid random case keep `torch.equal`; contract documented in README and AGENTS | `tests/test_sparton_kernel.py`, `README.md`, `AGENTS.md` |
| F3 no coverage of non-fallback optimized policies | Six `@pytest.mark.slow` non-tiny shapes (V-tail, batch-crossing S-tail, single-K-tile, long-S; fp16+bf16) for naive and optimized, with an explicit assertion that the runtime-derived candidate set is larger than the fallback alone | `test_naive_forward_nontiny_shapes`, `test_optimized_forward_nontiny_shapes` |
| F4 internals-leaking error surfaces | `src/sparton/_validation.py` with `validate_forward_inputs` called first in all three wrappers; contract-named `ValueError`/`TypeError`; per-rule tests including both probe reproducers (mixed dtype, unaligned D) | `_validation.py`, three backend wrappers, validation test section |
| F5 stale AGENTS.md | Project map lists all seven backend modules; layering invariant and index-contract rule added; stale "optimized should fail" line replaced; sharp edges refreshed | `AGENTS.md` |
| F6 dead code in `_backend_hybrid.py` | Unused imports, commented `FusedSparton` block + `fused_mlm_splade`, stale commented autotune block, commented debug print removed; `fused_sparton_bwd_with_bias` returns a 3-tuple | `_backend_hybrid.py` |
| F7 import-time side effects | `set_float32_matmul_precision('high')` removed; device banner and `load()` "no bias" print moved to the `"sparton"` logger at DEBUG; silent import enforced by `test_import_emits_no_stdout` | `_backend_hybrid.py`, `sparton_kernel.py` |
| F8 wrapper asymmetry | All three backends now route `SpartonHead` through wrappers with identical structure (validate → contiguous → op) | `sparton_kernel.py` |
| F11 training nits | Invalid-head message names `'torch', 'compiled', or 'sparton'`; `SpladeModel(..., sparton_backend=...)` and `--sparton_backend` threaded to `SpartonHead(..., backend=...)` (default `None` = unchanged behavior) | `training/model.py`, `training/train.py` |
| F12 license metadata | `pyproject.toml` license now Apache-2.0, matching `LICENSE` (deliberate, changelogged); authors/URL ownership fields intentionally untouched | `pyproject.toml` |
| F13 test-infra fragility | `tests/conftest.py` inserts the repo root on `sys.path`; subprocess tests use absolute `_REPO_ROOT`/`_SRC_PATH`; verified under `python -m pytest`, the `pytest` console script from the repo root, and `pytest /workspace/sparton/tests` from a foreign CWD | `tests/conftest.py`, `tests/test_sparton_kernel.py` |
| F14 shim niceties | Public `gl.NVMMASharedLayout` replaces the private `language._layouts` import; `is_gluon_backend_available` caches the loaded namespace into `_STATE` | `_gluon_runtime.py` |
| F16 hardcoded `dtype_name="fp16"` | Intent comment plus a host-side `element_size() == 2` assert in `_launch_optimized_fwd` | `_backend_optimized_gluon.py` |
| F20 backend-error provenance | `resolve_backend` errors name the `SPARTON_BACKEND` environment variable or the `backend` argument as the source; both paths tested | `sparton_kernel.py` |
| Deferred: F9 (per-call descriptor-bank rebuild) | Scheduled as the M12 launcher-v2 work, per design | — |
| Deferred: D2 (kernel-body duplication) | Cross-reference comments added above both decorator stacks (outside the JIT bodies); dedup deliberately rejected before the M12 rewrite | `_backend_optimized_gluon.py`, `scripts/bench_gluon_gemm.py` |

## F1 red→green evidence

The regression test was run against the unmodified hybrid path first (with the
hybrid case temporarily falling back to the raw op). With `hidden`, `embed`,
and `bias` all non-contiguous slices, the hybrid case failed on the first
gradient comparison while naive/optimized passed:

```text
FAILED tests/test_sparton_kernel.py::test_forward_backward_handles_noncontiguous_inputs[hybrid]
  assert_close(hidden.grad.float(), ref_hidden.grad.float(), atol=2e-3, rtol=2e-3)
  AssertionError: Tensor-likes are not close!
  Mismatched elements: 94 / 160 (58.8%)
  Greatest absolute difference: 3.6494140625
1 failed, 2 passed
```

(The review probe with only `hidden` non-contiguous surfaced the corruption in
`embed.grad` instead — same root cause: the op saved raw tensors and the
backward kernel computes flat offsets that assume dense strides; which
gradient corrupts first depends on which inputs are non-dense.) After adding
`hybrid_forward`, all three parametrizations pass on scores and on all three
gradients.

## Deliberate behavior changes

1. `import sparton` no longer prints the device banner; `SpartonHead.load` no
   longer prints `"no bias"`. Both are DEBUG records on
   `logging.getLogger("sparton")`.
2. The import-time `torch.set_float32_matmul_precision('high')` global is
   removed. fp16/bf16 paths are unaffected (verified by the benchmark gate);
   fp32 hybrid users now inherit the application's own setting.
3. `pyproject.toml` license metadata: MIT → Apache-2.0 (matches `LICENSE`).
4. `fused_sparton_bwd_with_bias` returns 3 values instead of 4 (vestigial
   trailing `None` removed; only in-repo caller updated; symbol re-exported).
5. Invalid inputs to the forward wrappers now raise contract-named
   `ValueError`/`TypeError` instead of reaching kernels/descriptors.
6. `scripts/bench_sparton_baseline.py` hybrid columns measure
   `hybrid_forward` (the user path) rather than the raw op.

## Plan deviations (recorded)

- The tails tests (`test_*_forward_handles_tails`) were listed for conversion
  to the tie-aware checker in the implementation plan but **kept on exact
  index equality**: their inputs are deterministic dyadic-rational patterns
  with no near-ties, so exact equality is stable and strictly stronger —
  consistent with the design rule that deterministic constructed cases keep
  `torch.equal`.
- The five subprocess tests additionally moved to absolute
  `_REPO_ROOT`/`_SRC_PATH` paths (discovered by the foreign-CWD invocation
  gate; same F13 fragility class).

## Benchmark A/B and gates

Dev shape `B=32, S=128, D=768, V=30522`, fp16, warm caches, second consecutive
run of each configuration:

| configuration | hyb+b ms | naive+b ms | opt+b ms |
|---|---|---|---|
| §1.2 recorded baselines (raw-op hybrid) | 1.176 | 1.301 | 0.898 |
| step-0 preflight re-run (raw-op hybrid) | 1.178 | 1.258 | 0.898 |
| after T1 wrapper switch | 1.178 | 1.259 | 0.898 |
| after T4 hygiene | 1.178 | 1.302 | 0.899 |
| exit checklist (final) | 1.177 | 1.260 | 0.898 |

All within ±5% of the recorded baselines (hybrid and optimized within 0.2%;
naive run-to-run spread ~3% predates M9).

## Exit checklist transcript (2026-06-12, hardened env, serial)

```text
--- 1. py_compile src/sparton/*.py training/*.py tests/*.py scripts/*.py ---
py_compile: passed
--- 2. pytest full (incl. slow) ---
105 passed, 15 warnings in 14.36s
--- 3. pytest quick loop (-m "not slow") ---
90 passed, 15 deselected, 15 warnings in 13.51s
--- 4. benchmark dev-shape row (x2, judge run 2) ---
| 32 | 128 | 1.176 | 1.076 | 0.880 | 33.7 | 2.631 | 140.50 | 1.259 | 1.259 | 9.86 | 0.898 | 0.896 | 9.86 | 9.31 | 238.5 | 3483317 |
| 32 | 128 | 1.177 | 1.077 | 0.880 | 33.8 | 2.635 | 140.50 | 1.260 | 1.259 | 9.86 | 0.898 | 0.897 | 9.86 | 9.31 | 238.5 | 3479908 |
--- 5. import silence ---
import sparton: stdout empty (passed)
--- 6. git diff --check ---
clean
```

Additional invocation-mode gates: `python -m pytest -q` (105 passed), venv
`pytest -q -m "not slow"` from the repo root (90 passed), and
`pytest -q -m "not slow" /workspace/sparton/tests` from `/tmp` (90 passed)
all collect and pass. A `torch.compile(fullgraph=True)` capture test per
backend is part of the suite (slow-marked); a pre-test smoke confirmed
capture with validation in the traced path.

The warning count moved from 15 to 16 transiently during T4 because the new
fp32-hybrid test triggers its own dynamo config copy (the same upstream
`FutureWarning` attributed to a second test); the final suite reports 15
warnings, all upstream torch deprecations.

## Known gaps deliberately not closed

- Empty tensors (`B`, `S`, or `V` == 0) are documented as unspecified in
  `_validation.py`, not validated.
- Raw-op callers (`scripts/ncu_targets.py`, anyone importing
  `fused_sparton_fwd_op` directly) bypass validation by design; the ops
  assume canonical inputs (AGENTS invariant).
- `training/` changes are `py_compile`-checked only: `transformers` is not
  installed in the venv. The `sparton_backend` threading is exercised for
  real at the M10 tier-2 training smoke.
- F9 (per-call descriptor-bank rebuild, ~0.05 ms host) and D2 (kernel-body
  duplication) are scheduled for M12, not M9.
