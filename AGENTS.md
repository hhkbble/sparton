# Repository Instructions

This file applies to the entire `/workspace/sparton` repository. Treat it as
the source-grounded operating guide for future agents and contributors. It
encodes both the repository facts and the working method that produced the
M8→M10 arc; the method sections (Operating Loop, Evidence and Measurement,
Testing Doctrine, House Style, Documentation System) apply to every task,
not only kernel work. Rules cite the memo or design section that taught them
so you can audit the evidence rather than trust the rule.

## Project Map

- `src/sparton/` is the installable Python package. Its public surface is
  currently `SpartonHead`, exported from `src/sparton/__init__.py` only when
  CUDA is available.
- `src/sparton/sparton_kernel.py` is the public facade and backend router
  (`resolve_backend`, `SpartonHead`, re-exports, lazy `optimized` symbols).
  Backend implementations live beside it:
  - `_backend_hybrid.py` — compatibility backend (compiled tiled matmul +
    Triton reduction + the single Triton backward used by all backends);
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
- `tests/` contains the pytest 9 kernel/reference test suite. Pytest is
  configured in `pyproject.toml`.
- `benchmarks/` contains validated probe/benchmark/gate scripts (MMA
  availability, Gluon GEMM ratio gate, merged backend baselines, shape soak,
  training smoke, profiler launchers); usage in `benchmarks/README.md`. The
  forward plan lives in `docs/sparton_remaining_work_design_v3.md`;
  `docs/sparton_remaining_work_design_v2.md` remains authoritative for the
  post-M8 review findings and M9/M10 provenance, and
  `docs/sparton_gluon_remaining_work_design.md` for platform facts and
  measured evidence.
- There is currently no lint config, typecheck config, CI config, or lockfile.
  The documented command set in Validation is the gate mechanism.

## Orientation Before Changes

- Read `README.md` for user-facing behavior, setup, examples, and documented
  project status.
- Read `CHANGELOG.md` for recent repository changes before planning or editing.
- For milestone work, read the active design doc's milestone section
  (`docs/sparton_remaining_work_design_v3.md`) before planning: it contains
  file-level task specs, gates, and recorded decisions. Do not re-derive what
  it already settles; do not silently contradict it — evolve it and record
  the evolution (see Documentation System).
- Read other design notes under `docs/` only when relevant to the requested
  work. Do not use `AGENTS.md` as a project changelog.
- Check `git status --short` and the last few commits before editing.

## The Operating Loop

The cycle that produced M9 and M10; follow it for any non-trivial task.

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
  never run two Triton/Inductor-compiling processes concurrently.
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
  M10 memo table; rerun the row yourself before using it as a gate.
- **Measure A-vs-A before judging A-vs-B**: repeat the same configuration and
  use that spread as the noise band. The atomic-add backward makes even
  same-seed training runs differ (~20% loss spread in chaotic regimes — M10
  memo, tier 2); a cross-backend difference inside the same-backend band is
  noise, not signal.
- **Keep measurement regimes separate**: `triton.testing.do_bench`
  (L2-flushed) is the latency of record; `ncu` serializes and flushes, so its
  durations are inflated — use it for structure, counters, and ratios; `nsys`
  for kernel inventory. Never compare numbers across regimes (v1 §2.5 records
  a published mistake from mixing them).
- **Classify a failure before fixing it.** Three verdicts are possible: a
  real defect (non-contiguous `embed_grad`, M9 F1), expected behavior that
  the gate mis-asserts (GradScaler-skipped early fp16 steps are normal AMP
  scale calibration — the M10 smoke's gate was rewritten, not the code), or
  an out-of-contract input (near-tie index mismatches are allowed by the
  index contract). Fixing before classifying produces wrong fixes.
- Perf claims need a mechanism, not just a delta: when a result surprises
  (cuBLAS unusually slow on a stress shape), say why or flag it as
  unexplained in the memo rather than letting the ratio stand alone.
- A gate that can be expressed as a script should be one (see
  `benchmarks/soak_optimized_correctness.py`): availability check, summary
  line, non-zero exit listing every failure.

## Testing Doctrine

- **Tests must prove what they appear to prove.** The suite was green for an
  entire milestone while only ever exercising the optimized backend's
  fallback policy, because every test shape was "tiny" (design v2 F3). When a
  code path's activation depends on input properties (shape tiers, policy
  pruning, dtype), assert the activation itself in the test —
  `test_optimized_forward_nontiny_shapes` asserts the derived candidate set
  is larger than the fallback before checking outputs.
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
  `probe_training_smoke.run_mode`) instead of duplicating it.
- Mark expensive coverage `@pytest.mark.slow`; the default `pytest -q` runs
  everything, `-m "not slow"` is the documented quick loop. Don't let the
  quick loop lose meaning by marking cheap tests slow.
- Subprocess tests use absolute paths (`_REPO_ROOT`, `_SRC_PATH` in
  `tests/test_sparton_kernel.py`), never CWD-relative ones. The suite must
  pass under `python -m pytest`, the venv `pytest` console script, and
  `pytest /workspace/sparton/tests` from a foreign working directory — run
  all three after touching test infrastructure.
- Parametrized cases live in named tables with explicit ids
  (`FORWARD_CASES`, `VALIDATION_ERROR_CASES`, `NONTINY_FORWARD_CASES`);
  `strict_parametrization_ids` is enabled.
- Gate availability with fixtures/helpers (`_forward_for_backend`,
  `_optimized_gluon_availability`), not device-name checks — capability, not
  hardware identity.

## House Style

Copy the repo's best instance of a pattern instead of inventing a new shape:

| Artifact | Exemplar |
|---|---|
| Gate/sweep script | `benchmarks/soak_optimized_correctness.py` (docstring contract, availability gate in `main`, `--quick`, summary line, non-zero exit with failure list) |
| Training/integration probe | `benchmarks/probe_training_smoke.py` (reusable `run_mode`, per-mode gates) |
| Contract checker + test tables | `tests/test_sparton_kernel.py` (`assert_index_contract`, case tables) |
| Input validation + error template | `src/sparton/_validation.py` |
| Forward design doc | `docs/sparton_remaining_work_design_v3.md` (active plan); `docs/sparton_remaining_work_design_v2.md` for the review-born shape (findings + evolution ledger) |
| Milestone memo | `docs/sparton_milestone9_production_readiness_memo.md` (finding→fix table, red→green evidence), `docs/sparton_milestone10_promotion_memo.md` (gate-by-gate decision record) |

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
  the reason. Do not add a second exception.
- Diagnostics go through `logging.getLogger("sparton")` at DEBUG. Never
  `print` from library code; a regression test enforces silent import.
- Comments state constraints the code cannot show (TMA alignment origin, the
  16-bit `dtype_name` assumption, D2 cross-references) — never mechanics,
  never change-narration. Cross-reference comments about kernel twins go
  above the decorator stack, never inside `@triton.jit`/`@gluon.jit` bodies
  (kernel-body bytes affect compiled-source hashes).
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
  `hidden_grad`, `embed_grad`, and `bias_grad` in `float32`.
- Preserve CUDA-only behavior unless explicitly implementing a CPU fallback.
  `__init__.py` intentionally exposes no `SpartonHead` when CUDA is unavailable.
- Preserve tensor contracts unless the task explicitly changes them:
  `hidden` is `[B, S, D]`, decoder/embed weights are `[V, D]`, optional bias is
  `[V]`, attention mask is `[B, S]`, and output sparse reps are `[B, V]`.
- The mask semantics in both PyTorch and Triton paths are part of correctness:
  logits are masked over sequence positions before ReLU, `log1p`, and max over
  the sequence dimension. Non-binary masks weight logits — defined behavior,
  not the HF attention-mask contract (documented in `_validation.py`).
- Numerics contract: forward output follows hidden/logit dtype; `naive`/
  `optimized` accumulate logits in fp32 (more precise than hybrid's
  input-dtype logits — intended); backward gradient buffers are `float32`.
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
| Active design doc (`docs/sparton_remaining_work_design_v3.md`) | The forward plan: milestones, gates, architecture rules, recorded decisions | Milestone completion gets a dated status note pointing at the memo; superseding it means a new doc plus a supersession note in the old one |
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
  mandatory.
- **Every number in a committed document was produced by a command run in
  that session.** If prose was drafted before the measurement, correct the
  draft to the measured value before committing. Quote both numbers when two
  regimes disagree, with the regime named.
- Documents form a supersession chain, never silent replacement: the old doc
  gets a status line naming its successor and what it remains authoritative
  for.
- Commits: concise subject + detailed body covering what changed, why, and
  how it was validated (with the actual gate results).

## Reference Material

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
- Avoid running multiple Triton/Inductor-compiling processes concurrently when
  validating or benchmarking; serialize runs for attributable results.

## Task Routing

- For public API or packaging work, start from `pyproject.toml`,
  `src/sparton/__init__.py`, and `SpartonHead`.
- For kernel correctness/performance work, start from
  `src/sparton/sparton_kernel.py` and the relevant `_backend_*.py`; build a
  small PyTorch reference before changing Triton/Gluon kernels or
  custom-op/autograd wiring; probe risky APIs in subprocesses first.
- For training behavior, start from `training/model.py` and `training/train.py`;
  avoid importing training modules unless the optional Hugging Face
  dependencies are needed for the task.
- For the next planned milestones (M11 backward track, M12 forward
  scheduling/launcher), read the corresponding design v3 §3 sections; v3
  §1.2 and the M10 memo's residual-risks section carry their entry evidence.

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
- The backward kernel accumulates with atomic adds: training is
  non-deterministic run-to-run even with fixed seeds (same-config 150-step
  runs differed ~20% in final loss in the chaotic early regime — M10 memo).
  Establish same-config noise bands before reading meaning into cross-config
  training differences; do not promise bitwise-reproducible training.
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
