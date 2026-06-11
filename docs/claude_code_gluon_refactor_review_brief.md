# Claude Code Brief: Gluon Refactor Review and Remaining Design

This is a task brief for Claude Code. It is not the final remaining-work
design document. Use it to review the repository, complete the remaining
feasibility checks, identify missed optimization points, and then write a new
design document under `docs/`.

## Current State

Start from the actual repository state, not from assumptions in prior design
notes.

- Latest major commit: `48b55fa Fix no-bias backward and add milestone docs`.
- Prior commit: `6e19af3 add warning when CUDA is not available`.
- Completed milestones:
  - PyTorch semantic reference and pytest coverage for current Sparton
    score/index behavior.
  - `bias=None` backward fix for the current hybrid path.
  - Optional-bias custom-op schemas:
    - `sparton::fused_sparton_fwd(... Tensor? bias ...) -> (Tensor, Tensor)`
    - `sparton::fused_sparton_bwd(... Tensor? bias ...) -> (Tensor, Tensor, Tensor?)`
- Not yet started:
  - backend extraction into `_backend_hybrid.py`;
  - backend routing and a public/backend-aware wrapper;
  - naive Triton fused forward;
  - Gluon runtime compatibility shim;
  - standalone Gluon GEMM/MMA microbenchmark;
  - optimized Gluon forward;
  - optimized Gluon backward.

During planning, the current baseline was re-run with:

```bash
TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas /workspace/venvs/sparton/bin/python -m pytest -q
```

Observed result:

```text
11 passed, 1 warning
```

There was also an earlier transient TorchInductor/Triton subprocess compile
failure on the BF16 biased forward case that passed on focused rerun and full
rerun. Treat first-run compiler/cache failures as operational evidence to
classify, not as deterministic kernel correctness failures unless reproducible.

## Required Reading

Read and reconcile these files before writing the new design:

- `docs/sparton_gluon_current_platform_design.md`
- `docs/sparton_gluon_design_review.md`
- `docs/sparton_milestone2_bias_none_backward_memo.md`
- `CHANGELOG.md`
- `README.md`
- `pyproject.toml`
- `src/sparton/__init__.py`
- `src/sparton/sparton_kernel.py`
- `tests/conftest.py`
- `tests/test_sparton_kernel.py`
- `training/model.py`

Treat `docs/sparton_gluon_design_review.md` as the correction layer over the
original Gluon design. Do not blindly continue from the first design document.
Where the review and the original design disagree, prefer the review unless
your new source/runtime checks prove otherwise.

## Fixed Contracts To Preserve

Preserve the current public and mathematical behavior unless the new design
explicitly calls out a later, gated API change.

- `SpartonHead` currently has no backend argument.
- `SpartonHead` is exposed from `sparton` only when CUDA is available.
- Current runtime path is the hybrid path:
  - compiled PyTorch/TorchInductor tiled matmul;
  - Triton sequence reduction for mask, max/argmax, ReLU, and `log1p`;
  - current Triton backward.
- The hybrid backend must remain the default until another backend is proven by
  correctness, memory, and performance evidence.
- Do not promote `optimized` to default until a standalone Gluon MMA/GEMM
  microbenchmark compiles, runs, and has acceptable throughput on the target
  GPU.
- Public backend names should stay architecture-neutral, but private Gluon
  compatibility logic may dispatch across installed API namespaces when the
  APIs diverge.
- Preserve score/index/mask semantics exactly:
  - logits are multiplied by mask values before reduction;
  - running max starts from baseline zero;
  - index updates use strict `>`;
  - all-negative, all-zero, or fully masked outputs have score zero and index
    zero;
  - `bias=None` returns no bias gradient;
  - backward uses saved scores and indices.

## Feasibility Checks To Complete

Reconfirm local runtime facts before drawing conclusions. The planning snapshot
was:

```text
Python: 3.12.3
Torch: 2.12.0a0+0291f960b6.nv26.04.48445190
Triton: 3.6.0
CUDA available: True
Torch CUDA: 13.2
GPU: NVIDIA GeForce RTX 5090
CUDA capability: (12, 0)
SM count: 170
warp size: 32
max threads per block: 1024
max threads per SM: 1536
shared memory per block: 49152
shared memory per block opt-in: 101376
shared memory per SM: 102400
```

Use this prefix for CUDA/Triton probes in this workspace:

```bash
TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas PYTHONPATH=src /workspace/venvs/sparton/bin/python
```

Re-run and record baseline validation:

```bash
PYTHONPATH=src /workspace/venvs/sparton/bin/python -m py_compile src/sparton/__init__.py src/sparton/sparton_kernel.py training/model.py training/train.py tests/conftest.py tests/test_sparton_kernel.py
TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas /workspace/venvs/sparton/bin/python -m pytest -q
TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas PYTHONPATH=src /workspace/venvs/sparton/bin/python -c "import torch; import sparton.sparton_kernel; print(torch.ops.sparton.fused_sparton_fwd.default._schema); print(torch.ops.sparton.fused_sparton_bwd.default._schema)"
```

Also run focused BF16 and no-bias probes if full pytest exposes a compiler
failure. Separate deterministic failures from transient compiler/cache issues.

Complete the remaining Gluon feasibility work:

- Inspect installed Gluon source/API symbols under the local Triton package, not
  only online documentation.
- Confirm current API names for descriptors, TMA, barriers/fences, and MMA.
- Find or build a known-good standalone Gluon GEMM/MMA microbenchmark on the
  RTX 5090 before designing fused optimized production kernels.
- Validate descriptor layout for `hidden.reshape(B*S, D)` and conceptual
  `embed.T` without physically transposing `embed`.
- Verify synchronization protocol and whether TMA/async copies and MMA can be
  combined in the local stack.
- Use `nsys` and `ncu` when a runnable kernel exists. During planning both were
  available at `/usr/local/bin/nsys` and `/usr/local/bin/ncu`.
- Verify whether the declared minimum `torch>=2.7.1` safely supports
  `Tensor?` custom-op schemas. If not proven, document the compatibility risk
  and possible split-op fallback.

## Optimization Topics To Consider

Look for additional optimization ideas, but keep them bounded by this repo, the
current hardware, the installed runtime, and the existing public contracts.

Hybrid preservation and cleanup:

- mechanically extract the current implementation into `_backend_hybrid.py`;
- add backend routing only after moved hybrid behavior is proven unchanged;
- keep `hybrid` as default;
- make layout and contiguity requirements explicit;
- preserve current autotune configs unless a measured change justifies editing
  them;
- keep custom-op, fake registration, and autograd behavior stable.

Naive Triton forward:

- implement as a correctness/debug baseline before Gluon fusion;
- do not materialize `[B, S, V]`, `[B, S, V_tile]`, or `[B*S, V_tile]`;
- match zero-baseline, strict-`>` index, mask multiplication, bias, and tail
  semantics exactly;
- use it to isolate Gluon-specific bugs, not as the expected production winner.

Optimized Gluon forward:

- build a runtime policy generator from shape and numeric hardware resources;
- prove GEMM-only throughput before fusing reduction;
- evaluate TMA/async staging, warp specialization, persistent scheduling,
  cooperative scheduling, and pingpong-like scheduling;
- keep the no-materialization guarantee for production forward;
- track register pressure, shared-memory usage, occupancy, HBM traffic, L2
  behavior, and kernel launch count.

Backward:

- keep current Triton backward as the baseline;
- implement Gluon direct atomic backward as correctness-first only after
  optimized forward is stable;
- consider local aggregation for `d_bias` and `d_embed`;
- consider duplicate-index aggregation for `d_hidden` only after profiling
  realistic index distributions;
- do not make Gluon backward default until correctness and performance are
  demonstrated against the current backward.

Do not add optimizations that cannot be evaluated in the current environment or
were intentionally removed from the first production scope:

- FP8, FP6, or FP4 paths;
- cluster multicast;
- native TMA gather/scatter for backward;
- CUTLASS or C++ integration;
- unsupported architecture-specific production paths;
- multi-CTA cluster kernels as required production machinery.

## New Design Document To Produce

After the review and feasibility checks, write a new design document under
`docs/` with a distinct name, for example:

```text
docs/sparton_gluon_remaining_work_design.md
```

That new document should replace the old design as the guide for subsequent
development. It should include:

- confirmed current state and exact validation results;
- corrected milestone order;
- remaining feasibility evidence and unresolved checks;
- backend architecture and dispatch plan;
- public API stance, including whether and when to add a backend argument;
- optimization candidates and rejection criteria;
- correctness, memory, and performance validation matrix;
- profiling plan using available tools;
- assumptions, risks, and explicit gates for promoting backends.

Keep the design decision-complete enough for implementation, but do not make it
overly rigid. Claude Code should do its own source/runtime research and update
the plan when evidence contradicts earlier notes. Do not drift into unrelated
training, packaging, licensing, or repository cleanup unless it directly affects
the backend refactor.

## Working Tree Boundaries

- Do not touch the untracked `CLAUDE.md`; it currently contains only
  `@AGENTS.md`.
- Do not rewrite prior design docs except to correct a factual error discovered
  during the new review.
- A `CHANGELOG.md` update is not required for this instruction-only brief. If
  subsequent work changes public behavior, tests, packaging, dependencies, or
  durable project guidance, update `CHANGELOG.md` then.

## Expected Final Report

When the new remaining-work design is complete, report:

1. What was concluded or changed.
2. The files and code paths reviewed.
3. Validation commands run and exact results.
4. Remaining risks, uncertainties, and follow-up checks.
