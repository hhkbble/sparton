# Repository Instructions

This file applies to the entire `/workspace/sparton` repository. Treat it as
the source-grounded operating guide for future agents and contributors. It
encodes the durable repository facts, invariants, environment, and validation
gates. The working method that produced the M8→M13 arc — the Operating Loop,
Evidence and Measurement, the Performance-Optimization Loop, Testing Doctrine,
Milestone Review, and House Style — now lives in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). Rules cite the milestone (see
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)) that taught them so you can audit
the evidence rather than trust the rule.

How to use this file: read Project Map + Orientation always; then the
section for your task class (Task Routing names the entry points); read
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the working method and copy its
House Style exemplar for any artifact you create; run the Validation commands
that match what you touched. When in doubt about process, the Operating Loop in
METHODOLOGY.md is the default and the others are its specializations.

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
  - `_backend_optimized.py` — the **`optimized`** backend (op `sparton::optimized_fwd`),
    the default where available (M10; pure Triton since the post-M13 promotion that
    replaced the original Gluon kernel and removed Gluon from the package). A persistent
    grid-stride forward over `(batch, vocab-tile)` tiles in stock `@triton.jit`: host-side
    TMA descriptors (`triton.tools.tensor_descriptor`, no device-side creation → no
    `triton.set_allocator`), `tl.dot`, `D`/`V` as `constexpr` (per-model fold of the loop
    bounds + the persistent div/mod; B/S stay runtime). Self-contained tile-policy space
    (this module — `OptimizedTile` / `_CANDIDATE_TILES` / `_tile_valid`) + a self-implemented
    **measured** autotuner keyed on `(D, V, dtype, arch)` — NOT B/S
    (`SPARTON_OPTIMIZED_AUTOTUNE`, default on; off → the small `_analytic_tile` + the manual
    `SPARTON_OPTIMIZED_WARP_SPECIALIZE` flag). One kernel carries both argmax epilogues behind
    a `WARP_SPECIALIZE` constexpr: the 2-result `tl.reduce` combine (homogeneous default) and a
    `tl.max`+masked-`tl.min` form (the only WS-compatible shape on Triton 3.7.1 — auto-WS
    rejects a 2-result reduce); the tuner enables WS only where it measures faster, timed with
    plain `triton.testing.do_bench`. Design of record:
    [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §5.1;
  - `_backend_runtime.py` — `is_optimized_backend_available()`: the default-resolution gate
    (CUDA + capability sm_90+ + importable `triton.tools.tensor_descriptor`); imports no
    kernel code;
  - `_validation.py` — shared autocast canonicalization and input-contract
    validation used by the per-backend forward wrappers.
- `training/` is a Hugging Face training/benchmark example, not a separate
  package. `training/model.py` wraps Hugging Face MLM backbones, and
  `training/train.py` wires dataset loading, tokenization, contrastive loss,
  sparsity regularization, and `Trainer`.
- `tests/` contains the kernel/reference pytest suite (135 tests after the
  post-M13 Gluon removal; the quick loop `-m "not slow"` is 111). Pytest (>=9) is configured
  in `pyproject.toml`. `tests/data/` is the repository-local home for
  *generated* data and run artifacts (never Hugging Face downloads): small
  fixtures committed, large artifacts gitignored and regenerable by a
  `scripts/` script that reuses existing files — conventions and the
  current contents of record in `tests/data/README.md`.
- `scripts/` contains validated probe/benchmark/gate scripts: merged backend
  baselines (with `--mask-density`), shape soak, training smoke, host-overhead recorder,
  the forward and backward direct-op profiling targets, the backward
  toolchain (index-distribution capture with regularizer pass-through and
  reuse-if-exists outputs, distribution-aware backward harness with
  per-cell verification), the M13 analytic traffic model
  (`m13_traffic_model.py`), and the backward IR/config dump tool
  (`dump_backward_ir.py`); usage in `scripts/README.md`. Only validated,
  reusable tooling lives here — disposable probes go to the gitignored
  `tests/data/runs/<label>/` until they prove durable (Operating Loop).
- Document map: `docs/` holds three consolidated references —
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (the built system: platform
  facts, contracts, layering rules, kernel designs of record, measured state,
  deferred work); [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) (the development
  memo of record, M2→M13: evidence, decisions, deviations, known gaps —
  including the M13 backward's validated gather/traffic model, the M12 forward
  no-go, and the M11 segmented design the split supersedes); and
  [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) (the working method +
  kernel-optimization technique). Both kernel tracks are closed; no milestone
  is planned after M13.
- There is currently no lint config, typecheck config, CI config, or lockfile.
  The documented command set in Validation is the gate mechanism.

## Orientation Before Changes

- Read `README.md` for user-facing behavior, setup, examples, and documented
  project status.
- Read `CHANGELOG.md` for recent repository changes before planning or editing.
- For milestone/kernel work, read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
  (the built system, its rules, and the deferred/out-of-scope sections) and the
  relevant [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) milestone section before
  planning: together they carry the file-level decisions, gates, and recorded
  outcomes. Do not re-derive what they settle; do not silently contradict them —
  evolve the doc and record the evolution (see Documentation System). Both
  kernel tracks are closed at M13, so new performance work starts from a
  milestone's residual-bottleneck note and a fresh profile
  ([`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §A.3).
- Read other design notes under `docs/` only when relevant to the requested
  work. Do not use `AGENTS.md` as a project changelog.
- Check `git status --short` and the last few commits before editing.

## Working method, performance loop, testing, and review

The working-method sections that used to live here — **The Operating Loop**
(+ session conduct), **Evidence, Measurement, and Gates**, **The
Performance-Optimization Loop**, **Testing Doctrine**, **Milestone Review**,
and **House Style** — now live in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md), the single home for the working
method and kernel-optimization technique (METHODOLOGY.md absorbed them from
this file and merged in the former `triton_gluon_kernel_optimization.md`).

Read METHODOLOGY.md before any non-trivial task, and its **§B technique layer**
before any Triton performance work. The one-line orientation it expands:
probe before designing; plan as ordered, independently-landable tasks with
explicit "will NOT touch" lists; bug fixes are red→green; run a gate after every
task (a command plus a number, never an adjective); for performance, **name the
binding resource and write the analytic traffic model first**, benchmark on real
distributions, prototype behind a registry, and pre-register decision rules;
run an adversarial milestone review of the diff **and** the evidence chain
before closing; and copy the repo's exemplar (METHODOLOGY.md House Style) for any
artifact you create. The Core Kernel Invariants below and the Validation
commands are the safety surface those method sections operate within.

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
     granularities silently drop contributions — DEVELOPMENT.md M13 §5.4 item 2).
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
  (initcheck-validated, DEVELOPMENT.md M11 §5.4; re-validated for the split,
  DEVELOPMENT.md M13 §5.6); keep allocation and coverage in lockstep if either changes.
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
  M11 review; DEVELOPMENT.md M11 §9). Non-binary values weight logits in the forward as
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
- `optimized` is the default backend where available (M10 promotion; pure Triton
  since the post-M13 promotion — CUDA sm_90+ plus importable
  `triton.tools.tensor_descriptor`); `hybrid` is the
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

Six documents with distinct roles; keeping them in role is part of every task's
definition of done. The **authoring discipline** for them (design-doc
executability, memo honesty, every-number-traces-to-a-command, the supersession
chain, commit-body rules) lives in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §A.7.

| Document | Role | Update trigger |
|---|---|---|
| `README.md` | User-facing setup, examples, behavior, high-level status | User-visible behavior changes |
| `AGENTS.md` (this file) | Durable repository facts, invariants, environment, validation; pointers to the docs below | A rule changes or a new durable lesson is learned; never task history |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The built system: platform facts, contracts, layering rules, kernel designs of record, measured state, deferred work | The built architecture or a measured-state figure changes |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | Development memo of record (M2→M13): finding→fix mappings, gate transcripts, red→green captures, analytic models, deviations, known gaps | A milestone or substantial debugging session completes |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | The working method + kernel-optimization technique (Operating Loop, Perf Loop, Testing Doctrine, Milestone Review, House Style, tooling) | A durable method lesson is learned; never task history |
| `CHANGELOG.md` | Dated user-visible changes (short summaries): behavior, API/schema, packaging, validation infrastructure, fixes, milestones | Every task that changes any of those |

## Reference Material

- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) §B is the kernel-optimization
  technique reference (originally the user-provided
  `triton_gluon_kernel_optimization.md`, adopted at M11 and merged into
  METHODOLOGY.md): bottleneck-classification-first profiling (§3.3), autotune
  hygiene, occupancy/register tuning loops, Gluon techniques, the tooling
  checklist (incl. the §6.1 IR-stage visibility recipe and the §6.2
  layout/lowering-attribution lessons added at M13), and failure modes. Read it
  before any Triton performance work.
- Upstream project metadata in this repo points at
  `https://github.com/thongnt99/sparton`; the local remote is different.
- The README citation references:
  `https://arxiv.org/abs/2603.25011`
- Hugging Face references used by the examples:
  `https://hf.co/FacebookAI/xlm-roberta-base`,
  `https://hf.co/naver/splade-v3`, and
  `https://hf.co/datasets/nthakur/swim-ir-cross-lingual`.
- For version-sensitive Hugging Face or Triton behavior, verify current
  installed versions and upstream docs before relying on stale local notes.

## Environment

- The dev environment is the Docker image `nvcr.io/nvidia/pytorch:26.05-py3`
  (NGC PyTorch container; ships the torch 2.12 nightly + **Triton 3.7.1** + CUDA 13.2
  toolchain the venv layers over via `--system-site-packages`). The optimized
  backend's pure-Triton kernel is validated on this Triton 3.7.1 / sm_120 stack.
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
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §2.4.
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
  small PyTorch reference before changing Triton kernels or
  custom-op/autograd wiring; probe risky APIs in subprocesses first; follow
  the Performance-Optimization Loop for perf work.
- For backward work specifically, [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)
  M13 carries the current mechanism evidence and the validated gather/traffic
  model (M11 carries the segmented design it split); `scripts/bench_backward.py`
  is the harness and `legacy_fused_sparton_bwd` (the M11 segmented design)
  the A/B reference. The Core Kernel Invariants section's three split
  invariants are the correctness surface of any edit there.
- For training behavior, start from `training/model.py` and `training/train.py`;
  avoid importing training modules unless the optional Hugging Face
  dependencies are needed for the task.
- **Performance-track status (both tracks closed, 2026-06-13; no planned
  milestone follows — ARCHITECTURE.md §6.8):** the forward is
  tensor-pipe-bound at 92–94% with no scheduling slack (M12 closed without
  kernel work; terminal residual: per-cycle pipe efficiency + L2 pressure
  at the autotuned tile shape — DEVELOPMENT.md M12; launcher v2 stays
  deferred, ARCHITECTURE.md §6.7). The backward runs the M13 split design
  (real records 1.46–1.60× over M11; the synthetic f=0.10 short-run regime
  carries a recorded 6–16% regression with no real-data representative;
  residual: uniform-pass LTS ≈ 61–67% (config-dependent) vs the embed
  kernel's 82–104%, plus the short-run regimes capped by the mixed fraction
  — DEVELOPMENT.md M13). Reopening either track starts from the relevant
  milestone's residual-bottleneck note and a fresh profile of the artifact
  as it ships, per the Performance-Optimization Loop (METHODOLOGY.md §A.3).
  Post-M13, the **`optimized` forward was reimplemented in pure Triton** and all Gluon
  was removed from the package (the Gluon `optimized` and the interim
  `experiment`/`experiment_gluon` backends are gone; backends are now
  `{hybrid, naive, optimized}`). The pure-Triton kernel is a persistent forward with
  host-side TMA + `tl.dot`, a measured self-tuner keyed on `(D, V, dtype, arch)` (not
  B/S), and warp specialization as a tuned dimension; it reaches ≈parity with the old
  Gluon optimized at large/throughput shapes (the single Gluon tile was +7…16% slower)
  and wins the serving (≈0.68×) and small-V (≈0.65×) regimes. This was a forward-backend
  replacement + re-baseline, **not** a production-track reopen — both forward/backward
  production tracks stay closed. See ARCHITECTURE.md §5.1 and DEVELOPMENT.md
  "Post-M13 — optimized promoted to a pure-Triton TMA forward".

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
  run costs ~25 s on this host (DEVELOPMENT.md M13 §6/§7) — the cold-start
  first run is much slower.
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
  (commands in DEVELOPMENT.md M11 §5.4; the M13 §5.6 run used
  `B4 S33 D64 V2048`, which also exercises the non-divisible-D mask path).
- For training changes, use a tiny local or sliced dataset smoke test before
  any full Hub-backed training run.

## Known Sharp Edges

- Importing `sparton` is silent on stdout; diagnostics go through the
  `"sparton"` `logging` logger at DEBUG level. A regression test enforces the
  silent import.
- The backward is not bitwise-deterministic run-to-run. Since M11,
  `embed_grad`/`bias_grad` are exactly deterministic (exclusive-owner plain
  stores) and `hidden_grad` atomics are reduced to ~chunk-partial sums
  (the M13 split preserves the atomic structure), but their accumulation
  order still varies (measured per-call relative spread: gradient-norm
  ≤ 1.2e-7, element-sensitive loss-proxy ≤ 4.2e-6 — DEVELOPMENT.md M11 §7,
  M13 §5.5). At training scale the chaotic early regime amplifies this to a
  measured same-config 150-step loss spread of **16–38%** depending on
  backend and statistic (3 seed-matched repeats per backend, DEVELOPMENT.md
  M13 §6 — supersedes the M10-era "~20%" estimate; per-call determinism
  improvements do not shrink it). Establish same-config noise bands
  before reading meaning into cross-config training differences; do not
  promise bitwise-reproducible training.
- The M13 split backward trades the synthetic `f = 0.10` short-run regime
  (−6…−16%, ~45 µs/call; no real capture exhibits it) for 46–60% gains on
  captured-real records. The trade is recorded **pending maintainer
  ratification** (DEVELOPMENT.md M13 §8 item 12); if rejected, the revert is
  one change — re-wire `fused_sparton_bwd_op` to `segmented_sparton_bwd` and
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
