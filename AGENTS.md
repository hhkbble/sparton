# Repository Instructions

This file applies to the entire `/workspace/sparton` repository. Treat it as
the source-grounded operating guide for future agents and contributors.

## Project Map

- `src/sparton/` is the installable Python package. Its public surface is
  currently `SpartonHead`, exported from `src/sparton/__init__.py` only when
  CUDA is available.
- `src/sparton/sparton_kernel.py` contains the Triton and PyTorch custom-op
  implementation for the fused SPLADE-style projection head.
- `training/` is a Hugging Face training/benchmark example, not a separate
  package. `training/model.py` wraps Hugging Face MLM backbones, and
  `training/train.py` wires dataset loading, tokenization, contrastive loss,
  sparsity regularization, and `Trainer`.
- `tests/` contains the pytest 9 kernel/reference test suite. Pytest is
  configured in `pyproject.toml`.
- `benchmarks/` contains validated probe/benchmark scripts for the backend
  refactor (MMA availability, Gluon GEMM microbenchmark, hybrid baselines,
  profiler launchers); usage in `benchmarks/README.md`, interpretation in
  `docs/sparton_gluon_remaining_work_design.md`.
- There is currently no lint config, typecheck config, CI config, or lockfile.

## Orientation Before Changes

- Read `README.md` for user-facing behavior, setup, examples, and documented
  project status.
- Read `CHANGELOG.md` for recent repository changes before planning or editing.
- Read task-specific design notes under `docs/` only when they are relevant to
  the requested work. Do not use `AGENTS.md` as a project changelog.

## Documentation Discipline

- At task completion, consider whether `CHANGELOG.md` needs an entry. Update
  the current dated section, or create one for the commit date, for user-visible
  behavior changes, public API/schema changes, packaging or dependency changes,
  validation/test infrastructure changes, notable fixes, and completed
  milestones.
- For completed milestones, design-review outcomes, substantial debugging
  sessions, or decisions that future agents need to audit, add a dated or
  clearly named memo under `docs/`. Keep implementation history and task notes
  in those memos, not in `AGENTS.md` or `README.md`.
- Keep `README.md` focused on user-facing setup, examples, behavior, and
  high-level project status. Keep `AGENTS.md` focused on durable development
  guidance and repository invariants.
- When making a git commit, use both a concise subject line and a detailed body
  that explains what changed, why it changed, and how it was validated.

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
  needed for the task. At audit time, `torch` and `triton` were present,
  `datasets` was present, and `transformers`/`accelerate` were missing.
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
  `src/sparton/sparton_kernel.py`; build a small PyTorch reference before
  changing Triton kernels or custom-op/autograd wiring.
- For training behavior, start from `training/model.py` and `training/train.py`;
  avoid importing training modules unless optional Hugging Face dependencies are
  installed.

## Core Kernel Invariants

- Main runtime path: `SpartonHead.forward` calls `fused_sparton_fwd_op`, which
  calls `fused_sparton_fwd_with_indices`, tiled matmul helpers, and
  `reduce_seq_max_log1p_relu_with_indices`.
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
  the sequence dimension.
- Be careful around dtype and accumulation behavior. The forward output follows
  hidden/logit dtype; backward gradient buffers are `float32`.
- Do not casually change autotune config lists, tile-size heuristics,
  `torch.library.custom_op` signatures, fake registrations, or autograd setup.
  These affect compilation, graph capture, memory behavior, and gradients.
- There are two forward-style reduction helpers: one returns max values plus
  indices for autograd, and one returns only values. Keep their intended memory
  tradeoff clear when editing.
- If adding a `backend` argument later, preserve current behavior as the
  `hybrid` baseline and do not introduce silent hardware-feature fallbacks.

## Training and Hugging Face References

- `SpladeModel` supports `head="torch"`, `head="compiled"`, and
  `head="sparton"`. Keep these modes behaviorally aligned when changing model
  code.
- `train.py` defaults to `FacebookAI/xlm-roberta-base` and
  `nthakur/swim-ir-cross-lingual` with languages `de,es,fr`.
- Full training downloads large Hub assets and can be expensive. Do not run it
  casually as validation; prefer small synthetic or smoke probes unless the user
  explicitly asks for a training run.
- `FacebookAI/xlm-roberta-base` is a Transformers MLM model with
  `xlm-roberta` architecture.
- The README quick-start references `naver/splade-v3`; at audit time that model
  was gated and tagged `license:cc-by-nc-sa-4.0`.
- `nthakur/swim-ir-cross-lingual` is a multilingual text retrieval/question
  answering dataset tagged `license:cc-by-sa-4.0`.
- If touching Hugging Face integration, verify current Hub metadata and local
  package versions instead of relying only on README examples.

## Validation

- For docs-only changes, review the diff and status:
  `git diff -- README.md AGENTS.md CHANGELOG.md docs/`
  `git status --short`
- For brand-new untracked docs, use `git diff --no-index`, for example:
  `git diff --no-index -- /dev/null CHANGELOG.md`
- Syntax check for the current Python files:
  `/workspace/venvs/sparton/bin/python -m py_compile src/sparton/__init__.py src/sparton/sparton_kernel.py training/model.py training/train.py tests/conftest.py tests/test_sparton_kernel.py`
- Import check from source:
  `PYTHONPATH=src /workspace/venvs/sparton/bin/python -c "import sparton; print(sparton.__all__)"`
- Pytest suite:
  `TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas /workspace/venvs/sparton/bin/python -m pytest -v`
- CUDA/Triton probes in this workspace should include:
  `TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas`.
- For kernel changes, compare against a direct PyTorch reference on small CUDA
  tensors and check both forward values and backward gradients. Include masked
  sequence positions, bias and no-bias cases if relevant, multiple `S`/`V`
  shapes, and at least one nontrivial tile boundary.
- For training changes, use a tiny local or sliced dataset smoke test before
  any full Hub-backed training run.

## Known Sharp Edges

- `pyproject.toml` declares `license = {text = "MIT"}`, while `LICENSE` and the
  upstream repository indicate Apache-2.0. Do not silently "fix" this in
  unrelated work.
- The local Git remote is `https://github.com/hhkbble/sparton.git`; the README
  and package metadata reference `https://github.com/thongnt99/sparton` and the
  citation references `https://github.com/thongnt99/lsr-kernel`.
- `training/model.py` error text for invalid heads mentions older names
  (`pytorch`, `triton`) even though the accepted values are `torch`,
  `compiled`, and `sparton`.
- `SpartonHead.load` prints `"no bias"` when a bias is absent; preserve or
  change this only as part of an explicit logging/API cleanup.
- Importing `sparton_kernel.py` prints the active device. Account for this in
  tests or command output comparisons.
- `SpartonHead` has no CPU fallback. CPU-only environments should use PyTorch
  reference paths, not the Sparton kernel.

## Editing Expectations

- Inspect the source path you are changing before editing. Prefer actual runtime
  behavior over README claims.
- Keep changes surgical. Do not reformat large files, change public signatures,
  add dependencies, or alter kernel/training behavior outside the requested
  scope.
- Use English for comments and explain non-obvious intent, not mechanics.
- Preserve existing style unless a focused cleanup is part of the task.
- Never overwrite user changes. Check `git status --short` before editing when
  worktree state matters.
