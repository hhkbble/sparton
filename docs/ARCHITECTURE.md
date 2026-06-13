# Sparton Architecture

This document describes the **built system** as it ships today: the three
backends, the platform facts they run on, the mathematical and API contract
they implement, the layering rules that hold across them, the kernel designs
of record, the measured performance state with its named residuals, and the
validation gates.

It supersedes and consolidates the family of design/review/platform notes that
guided the M1–M13 development arc. For the development history — milestone
specifications, promotion gates, red→green evidence, deviation ledgers — see
[DEVELOPMENT.md](DEVELOPMENT.md). For the working method (the operating loop,
the performance-optimization loop, kernel-optimization technique) see
[METHODOLOGY.md](METHODOLOGY.md). Durable repository facts, invariants, and
operating rules live in [../AGENTS.md](../AGENTS.md); user-facing setup and
examples in [../README.md](../README.md); dated changes in
[../CHANGELOG.md](../CHANGELOG.md).

Numbers in this document are quoted verbatim from the measurement runs of
record (see [DEVELOPMENT.md](DEVELOPMENT.md) for provenance). Where two
measurement regimes disagree the regime is named; the latest measured state is
preferred for the current architecture, and platform facts come from the
original platform survey.

---

## 1. Overview

`SpartonHead` (in `src/sparton/`) replaces the MLM projection head of a
SPLADE-style model with a CUDA-only fused projection head that avoids
materializing the full `[B, S, V]` logits tensor. It exposes three
implementation backends that share one numerics contract, one forward op
schema family, and one shared backward:

| Backend | Role | Forward implementation |
|---|---|---|
| `optimized` | **Default** where available (CUDA sm_80+ with importable `triton.experimental.gluon`); the production fast path | Single Gluon kernel: TMA staging + `mma_v2` mainloop + fused max/argmax/ReLU/log1p epilogue; bounded policy autotune |
| `hybrid` | Compatibility path; must stay behaviorally stable | Compiled (TorchInductor) tiled matmul per vocab tile + Triton sequence-reduction kernel; materializes per-tile `[B, S, V_tile]` logits |
| `naive` | `tl.dot` fused-forward debug baseline | Single Triton kernel with `tl.dot`, bounded autotune; isolates fusion semantics from Gluon-specific failure modes |

All three forwards delegate backward to the same op (`fused_sparton_bwd_op`),
which runs the **M13 split segmented backward** (§5). `optimized` was promoted
to the default after the M10 gates (forward faster than hybrid on every
measured shape, output-only memory, full correctness matrix green); promotion
gates and evidence are in [DEVELOPMENT.md](DEVELOPMENT.md) M10.

### Module map (`src/sparton/`)

```text
src/sparton/
  __init__.py                  # exports SpartonHead only when CUDA is available
  sparton_kernel.py            # public facade/router: resolve_backend, SpartonHead,
                               #   re-exports, lazy __getattr__ for optimized symbols
  _backend_hybrid.py           # hybrid forward (compiled tiled matmul + Triton reduction)
                               #   AND the shared backward used by all backends:
                               #   the M13 split segmented backward
                               #   (split_segmented_sparton_bwd, wired into
                               #   sparton::fused_sparton_bwd) plus the retained M11
                               #   segmented design behind legacy_fused_sparton_bwd
  _backend_naive_triton.py     # tl.dot fused-forward debug baseline, bounded autotune
  _backend_optimized_gluon.py  # Gluon TMA + mma_v2 fused forward, policy autotune (default)
  _gluon_runtime.py            # the ONLY module allowed to import triton.experimental.gluon;
                               #   lazy shim + capability whitelist + VALIDATED_TRITON warning
  _gluon_policy_runtime.py     # lazy host-side policy/config/descriptor helpers, shared by
                               #   the optimized backend and the GEMM benchmark
  _runtime_policy.py           # pure-Python policy generation (no torch/triton at module
                               #   level; importable by tests on CPU-only machines)
  _validation.py               # shared autocast canonicalization + input-contract validation
```

Adjacent, **not** part of the installable package: `training/` is a Hugging
Face training/benchmark example (`training/model.py` wraps MLM backbones,
`training/train.py` wires dataset/loss/sparsity/`Trainer`); `tests/` holds the
kernel/reference pytest suite; `scripts/` holds validated probe/benchmark/gate
tooling.

---

## 2. Platform & environment facts

This section is the single authoritative copy of the platform facts the system
was built and validated against. Treat them as version-sensitive: re-verify
installed `torch`/`triton` versions and upstream docs before relying on stale
notes, and re-run the Gluon capability probe on any GPU or Triton change.

### 2.1 Runtime snapshot (validation platform)

```text
Python 3.12.3
Torch  2.12.0a0+0291f960b6.nv26.04.48445190
Triton 3.6.0                         (target: GPUTarget(backend='cuda', arch=120, warp_size=32))
CUDA   13.2 (torch.version.cuda)
GPU    NVIDIA GeForce RTX 5090, CC (12, 0), 170 SMs
warp size 32, max threads/block 1024, max threads/SM 1536
shared memory: 49152 B/block default, 101376 B/block opt-in, 102400 B/SM
total memory 34190458880 B (~31.8 GiB), L2 cache 100663296 B (96 MiB)
```

The dev environment is the NGC container image
`nvcr.io/nvidia/pytorch:26.05-py3` (torch 2.12 nightly + Triton 3.6.0 + CUDA
13.2 toolchain). The L2 size (96 MiB) is large enough to hold the entire
`hidden` activation for typical dev shapes (`4096 x 768` fp16 = 6 MiB), which
matters for GEMM scheduling policy.

### 2.2 MMA availability matrix on sm_120 (proven by compile+run probes)

| Gluon MMA family | Local API | Result on RTX 5090 |
|---|---|---|
| `ampere.mma_v2` (mma.sync) | `triton.experimental.gluon.language.nvidia.ampere.mma_v2` | **Works.** Tiny GEMM tile max-abs-err 6e-6 vs fp32 matmul |
| `hopper.warpgroup_mma` (WGMMA) | `...nvidia.hopper.warpgroup_mma` | **Fatal abort**: `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.wgmma.commit_group.sync.aligned` |
| `blackwell.tcgen05_mma` + tensor memory | `...nvidia.blackwell.tcgen05_mma` | **Fatal abort**: `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.ld` (an earlier probe saw `...wait.st`) |

Decisive consequences for the architecture:

1. **`mma_v2` is the only tensor-core path for the optimized backend on this
   GPU.** It is the same `mma.sync` instruction family that `tl.dot` lowers to
   on sm_120, so Gluon's edge here is *manual control of staging, layouts,
   synchronization, and scheduling* — not access to a bigger MMA instruction.
2. The Gluon front end accepts all three families; failures occur at LLVM
   instruction selection as **fatal process aborts, not catchable Python
   exceptions**. Capability dispatch must therefore happen *before*
   compilation from a static whitelist — "try one, fall back on error" is not
   implementable. (This is why the optimized backend ships with a whitelist,
   not a try/except.)
3. Blackwell tensor-memory `load_max` (an N-dimension max fused into the TMEM
   load — a perfect fit for this kernel's epilogue) exists in the local API but
   is datacenter-Blackwell-only; future-hardware material, excluded.

### 2.3 Local Gluon API surface + capability whitelist (Triton 3.6.0)

`_gluon_runtime.py` is the only module allowed to import
`triton.experimental.gluon.*`; kernels import project-local names from it. It
is a **lazy** import (nothing Gluon-related loads unless a Gluon backend is
requested, keeping `triton>=3.3.1` valid for hybrid users) and records
`VALIDATED_TRITON = "3.6.0"`, warning once when run against a different Triton
because `experimental` namespaces move.

The verified local API surface, all under `triton.experimental.gluon`:

- Host-side descriptor: `gluon.nvidia.hopper.TensorDescriptor` (re-exported by
  `gluon.nvidia.blackwell`). Constraints enforced in `__post_init__`: base
  16-byte aligned; all non-last strides 16-byte aligned; last dim contiguous
  (`strides[-1] == 1`); layout must be `NVMMASharedLayout`; element bitwidth in
  {8, 16, 32}; `block_shape[-1] >= swizzle_byte_width / elem_bytes` (for fp16 +
  128-byte swizzle this means **BLOCK_K >= 64**; BLOCK_K=32 requires a
  64-byte-swizzle descriptor).
- Device language `gluon.language` (`gl`): `allocate_shared_memory`,
  `shared_memory_descriptor.{load,store,slice,index,permute,reshape}`,
  `warp_specialize`, `arange/load/store/reduce`, `gl.max/sum/min`, atomics,
  layouts (`BlockedLayout`, `SliceLayout`, `DotOperandLayout`,
  `NVMMADistributedLayout`, `NVMMASharedLayout.get_default_for`,
  `SwizzledSharedLayout`, `PaddedSharedLayout`), `gl.barrier()`.
  `gl.NVMMASharedLayout` is public; `gl.max` has **no** `return_indices`
  parameter (the epilogue argmax uses an explicit `gl.reduce` over
  `(value, row_index)`).
- TMA: `gl.nvidia.hopper.tma.async_copy_global_to_shared(desc, coord, barrier,
  smem)`, `async_copy_shared_to_global`, `store_wait`. (The tutorial-era
  `tma.async_load` does not exist locally.)
- mbarrier: `gl.nvidia.ampere.mbarrier.{allocate_mbarrier, init, wait, arrive,
  invalidate, MBarrierLayout}` plus `gl.nvidia.hopper.mbarrier.expect`.
  Barriers are rank-1 `[1]` shared memdescs.
- `hopper.fence_async_shared(cluster=False)`.
- `gluon.autotune` is **absent** in 3.6.0; the shim exports a `triton.autotune`
  fallback.

**Capability whitelist** (`select_mma_family`): `torch.cuda.get_device_capability()`
major `>= 8 → "mma_v2"`; major `< 8 → RuntimeError` ("requires CUDA capability
sm_80 or newer"). The whitelist is private, architecture-aware dispatch; public
backend names stay architecture-neutral. WGMMA/TCGen05 variants would be future
work behind the same shim, gated on separate validation on real sm_90a/sm_100
hardware.

### 2.4 Workspace defects and the hardened-env prefix

Two deterministic workspace defects make cache-cold TorchInductor compiles fail
unless the environment is hardened:

1. **The NVIDIA-built Triton wheel ships no bundled CUDA headers.**
   `triton/backends/nvidia/include/` does not exist, but
   `triton/backends/nvidia/driver.py` hardcodes it as the only include dir for
   building its `cuda_utils` C extension. Any *cold* build fails with
   `fatal error: cuda.h: No such file or directory`. Fixed by
   `CPATH=/usr/local/cuda-13.2/include`.
2. **`/tmp` is mounted `noexec`.** The default TorchInductor cache root is
   `/tmp/torchinductor_<user>`; a successfully built `cuda_utils.so` redirected
   there fails to `dlopen` (`failed to map segment from shared object`). Fixed
   by `TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor` (relocates the
   derived `triton/<device>` cache onto an executable filesystem).

`cuda_utils` is built once per process at the first Triton driver
initialization, in whatever `TRITON_CACHE_DIR` is active then — which is why
runs that touch a plain Triton kernel first (warm `~/.triton/cache`) pass while
async-compile workers on a cache miss fail. The hardened prefix used for every
CUDA/Triton command (and the additional `TRITON_PTXAS_PATH` this workspace
needs for ptxas):

```bash
TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas \
CPATH=/usr/local/cuda-13.2/include \
TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
PYTHONPATH=src /workspace/venvs/sparton/bin/python
```

Operational rule: do not run two Triton/Inductor-compiling processes
concurrently when comparing results; serialize for attributable timings.

### 2.5 Profilers and measurement regimes

- `nsys` 2026.2.1 (`/usr/local/bin/nsys`): **works** — kernel inventory,
  launch counts, stream overlap, single-regime per-call shares.
- `ncu` 2026.1.1 (`/usr/local/bin/ncu`): **works** (host counter restriction
  was lifted). ncu serializes launches and flushes caches between kernels, so
  per-kernel durations are inflated — use it for structure, counters, and
  ratios, never as a latency of record. On this GPU the DRAM byte counters are
  `dram__bytes_op_read.sum` / `dram__bytes_op_write.sum` (`dram__bytes_read.sum`
  reports `n/a`). NVTX ranges are thread-local: profile backward by calling the
  bwd op directly on the main thread (autograd's worker thread does not inherit
  the range).
- `triton.testing.do_bench` (L2-flushed) is the **latency of record**. Never
  compare numbers across regimes.
- `/usr/local/cuda-13.2/bin/compute-sanitizer` works on this host since the
  post-M11 restart (racecheck/memcheck/initcheck). Under stock WSL2/WDDM it
  fails with "Device not supported" — that means the host needs its debugger
  interface enabled, not that the code is wrong.

### 2.6 Packaging floors

- Declared `torch>=2.7.1`, `triton>=3.3.1`, Python `>=3.10`.
- **`triton.experimental.gluon` does not exist in Triton 3.3.1** — it first
  appears in 3.4.0; the validated version is 3.6.0. The declared `triton` floor
  stays valid for the **hybrid backend only**; Gluon code is imported lazily and
  fails with a clear error naming the required Triton when the namespace is
  missing.
- `torch>=2.7.1` is sufficient for the `Tensor?` custom-op schemas the ops use:
  the ops pass **explicit** schema strings (not inferred), and 2.7.1's schema
  parser + custom-op runtime tolerate optional tensor parameters and `None`
  returns. The ops must **not** be migrated to inferred schemas — schema
  inference has no `Optional[Tensor]` return support. A release advertising
  no-bias training on the floor version should run the suite once against real
  torch 2.7.1 first; the documented fallback (never needed) is split
  bias/no-bias ops.

---

## 3. Mathematical & API contract

### 3.1 Operation

Inputs and outputs (tensor contracts; all CUDA):

```text
hidden:  [B, S, D]
embed:   [V, D]            (decoder/embedding weight, row-major)
bias:    [V] or None
mask:    [B, S]
->
scores:  [B, V]            (sparse representation; dtype follows hidden/logit dtype)
indices: [B, V]            (int64)
```

For each batch row `b` and vocabulary id `v`:

```text
raw[b, s, v]        = dot(hidden[b, s, :], embed[v, :]) + bias[v]
masked_raw[b, s, v] = raw[b, s, v] * mask[b, s]
running_max[b, v]   = max_s masked_raw[b, s, v], with baseline 0
indices[b, v]       = first s that strictly improves running_max
scores[b, v]        = log(1 + max(0, running_max[b, v]))
```

The baseline `0` is intentional: the downstream activation is ReLU then
`log1p`, so if all valid logits are negative the score is zero and the index is
not semantically meaningful. Tie-breaking uses a **strict `>`** update,
matching the kernel: a position updates running state only if its value
strictly exceeds the current running value, so within a backend ties resolve to
the lowest sequence index.

Backward, from the saved score and index:

```text
g[b, v] = grad_out[b, v] * exp(-scores[b, v])  if scores[b, v] > 0 else 0

d_bias[v]              += sum_b g[b, v]
d_embed[v, d]          += sum_b g[b, v] * hidden[b, indices[b, v], d]
d_hidden[b, idx, d]    += g[b, v] * embed[v, d]     (idx = indices[b, v])
```

### 3.2 Index contract of record (tie-aware)

The returned index for a vocabulary entry is **meaningful only where its score
is greater than zero** (zero-baseline policy). Within one backend, ties resolve
to the lowest sequence index.

**Across backends with different accumulation precision, the winner at a
near-tie is explicitly unspecified.** `naive`/`optimized` accumulate logits in
fp32; `hybrid` produces input-dtype logits. So at near-ties (logit gaps within
input-dtype rounding) different backends may legitimately return different
winners — every observed mismatch's chosen logit was within one fp16 ULP of the
reference max (fp16) or exactly equal in reference precision (bf16); hybrid
matches the input-dtype reference bit-for-bit.

The testable contract (the assertion of record, `assert_index_contract`):
**wherever the score is positive, the masked input-dtype logit at the returned
index is within `atol + rtol·|max|` of the per-`(b,v)` maximum** (fp16
2e-3/2e-3, bf16 5e-2/5e-2); positions with score == 0 are unconstrained.
Backward correctness is insensitive to which near-tie winner was chosen
(gradient flows through a position whose logit differs by ≤1 ULP). Exact index
equality is asserted only in deterministic constructed cases (intentional ties,
masked winners) — those pin the strict-`>` policy.

### 3.3 Mask contract (binary {0,1}), and the backward-exactness ruling

`mask` is the standard tokenizer `attention_mask`: a binary `{0, 1}` (or
boolean) `[B, S]` tensor. Logits are masked over sequence positions (`logits *
mask`) before ReLU, `log1p`, and max over the sequence dimension — this masking
is part of correctness in both the PyTorch reference and the kernels.

**Maintainer ruling (M11 review): under the binary mask contract the shared
backward is exact, and no `mask[b, idx]` factor is needed** — a masked winner
forces score 0, and the `scores > 0` guard zeroes its gradient. Non-binary mask
values weight logits in the forward as an implementation property, but they are
**outside the contract**: the backward does not differentiate the mask factor.
Supporting weighted masks would be an extension (a backward change plus tests
against the `head="torch"` autograd path), not a bug fix. Mask values are
deliberately not validated — `_validation.py` checks are metadata-only; a value
scan would need a device sync.

### 3.4 Numerics contract

- Forward output follows the hidden/logit dtype. `naive`/`optimized` accumulate
  logits in **fp32** and apply bias/mask/max/log1p in fp32 before casting the
  stored score — *more* precise than hybrid's input-dtype logits, and the
  intended behavior for all future backends. Cross-backend score agreement is
  within tolerance (fp16 2e-3, bf16 5e-2).
- Backward gradient buffers (`hidden_grad`, `embed_grad`, `bias_grad`) are
  accumulated in **float32**. `hidden_grad` is zero-filled (atomic
  accumulation; untouched rows stay 0); `embed_grad`/`bias_grad` are
  `torch.empty` — safe only because the embed kernel's unconditional
  exclusive-owner stores cover every element (initcheck-validated). Allocation
  and coverage must stay in lockstep if either changes.
- The backward is **not bitwise-deterministic** run-to-run: `embed_grad`/
  `bias_grad` are exactly deterministic (exclusive-owner plain stores) but
  `hidden_grad` atomic accumulation order still varies (§6 records the bands).

### 3.5 Supported dtypes

| Backend | Supported `hidden`/`embed`/`bias` dtype |
|---|---|
| `optimized` | fp16, bf16 (descriptor `element_bitwidth=16`; also requires `D * element_size % 16 == 0` for TMA) |
| `naive` | fp16, bf16 |
| `hybrid` | fp16, bf16, fp32 (fp32 is permitted legacy, not benchmark-covered) |

### 3.6 Custom op schemas and autograd saved-tensor set

Op names and schemas are the stable layer. The hybrid op names
(`sparton::fused_sparton_fwd`/`fused_sparton_bwd`) stay bound to their
implementation forever (compat with anything that captured them); each backend
registers its own forward op so schemas may diverge without touching hybrid.
All three forwards register autograd that calls the single backward op
`fused_sparton_bwd_op`.

```text
sparton::fused_sparton_fwd (Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)   # hybrid
sparton::fused_sparton_bwd (Tensor grad_out, Tensor max_scores, Tensor max_idx, Tensor hidden,
                            Tensor embed, Tensor? bias, Tensor mask)               -> (Tensor, Tensor, Tensor?)
sparton::naive_fwd         (Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)   # naive
sparton::optimized_fwd     (Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)   # optimized
```

**Autograd saves**: max scores, max indices, hidden states, decoder weights,
bias, and mask. A backward swap that keeps this saved-tensor set is schema-safe
inside the op (proven by M11 and M13). A change to the saved-tensor set
requires a new op name.

---

## 4. Architecture layering & rules

### 4.1 The layering of record

`SpartonHead.forward` resolves a backend **once at construction** (the instance
binds the per-backend wrapper; the env var `SPARTON_BACKEND` is read once at
import, never per call) and `forward` just calls the bound wrapper:

```text
SpartonHead.forward
  └─ <backend>_forward(hidden, embed, bias, mask)      # public per-backend wrapper
       ├─ _validation.autocast_canonicalize(...)       # mirrors torch.autocast: casts fp32
       │                                                #   master params to the autocast dtype
       ├─ _validation.validate_forward_inputs(...)      # shared contract checks; raises named errors
       ├─ .contiguous() canonicalization (all inputs)
       └─ sparton::<backend>_fwd custom op              # stable schema; assumes validated, contiguous inputs
            └─ kernel launch (+ autograd that saves the op's inputs;
               backward → fused_sparton_bwd_op)
```

Wrappers (`hybrid_forward`, `naive_forward`, `optimized_forward`) are the only
public callables; `SpartonHead` binds wrappers, never raw ops. The ops assume
validated, contiguous inputs — **raw-op callers (e.g. profiling targets) bypass
validation by design**; do not "fix" that by validating inside the ops.

`autocast_canonicalize` exists because the custom ops are not autocast-registered:
under an active CUDA autocast region it casts `hidden`/`embed`/`bias` to the
autocast dtype (fp16/bf16) the way `torch.matmul` would, and leaves everything
unchanged outside autocast (`mask` is never cast). This is what makes standard
AMP training with fp32 master parameters work on every backend.

### 4.2 Cross-backend rules

- **Symmetry rule.** The Nth implementation of a pattern mirrors the others
  byte-for-byte where semantics allow. The three forward wrappers are
  intentionally line-for-line parallel; the historical hybrid no-contiguity bug
  existed precisely because hybrid lacked the wrapper the others had.
- **One-seam changes.** New cross-backend behavior is one shared helper called
  at exactly one layer (`autocast_canonicalize` at the top of each wrapper; the
  shared backward under all three forwards), never N divergent copies.
- **Backend isolation.** Backends never import each other except that the
  naive/optimized backends import the shared backward op from
  `_backend_hybrid`. Only `_gluon_runtime` may import
  `triton.experimental.gluon.*`; `_runtime_policy` must stay importable with no
  torch/triton at module level.
- **No silent fallbacks.** The single sanctioned exception is *default
  resolution*: with no `backend` argument and no `SPARTON_BACKEND`, an
  unavailable optimized backend falls back to hybrid with a one-time
  `RuntimeWarning` (M10). An explicitly selected backend that is unavailable
  **raises with the reason**. No second exception may be added — data-dependent
  algorithm dispatch inside an op would also need a host sync, which is why the
  backward ships fixed kernel families (§4.3).
- **No data-dependent algorithm dispatch inside an op.** It would need a host
  sync and a second fallback seam. The backward launches its kernels
  unconditionally with complementary device-side predicates — no dispatch.
- **Backward swap is schema-safe.** A backward change that keeps the
  saved-tensor set swaps inside `sparton::fused_sparton_bwd` without touching
  forward op schemas or autograd wiring (proven by M11→M13). A changed
  saved-tensor set requires a new op name.
- **Diagnostics** go through `logging.getLogger("sparton")` at DEBUG; library
  code never `print`s (a regression test enforces silent import).

### 4.3 Policy-bank mechanics (optimized forward)

How the optimized forward selects a configuration today (the mechanism of
record; the deferred launcher v2 in §6 would replace it):

1. `_runtime_policy.optimized_forward_policy_universe()` returns a **fixed,
   ordered 11-policy universe** (fallback `64x64x64 / 3 stages / 2x2 warps /
   sw128` first, then the production candidates, all `BLOCK_N ≤ 128`). The
   length is asserted at import in `_backend_optimized_gluon.py`.
2. The kernel signature bakes in **11 descriptor pairs (22 args)**; a constexpr
   `POLICY_ID` selects a pair via an if-chain, so each compiled variant
   dead-codes the other 21 args. At decoration time the universe becomes 11
   `triton.Config`s (`POLICY_ID=i` + block/warp/stage constexprs).
3. At launch, the autotuner (key `B, S, D, V`; tensor dtypes are appended to
   the cache key by Triton ≥3.6) runs `early_config_prune`, which derives the
   **active candidate set** from the real `DeviceProfile` and `ProblemSpec`:
   resource-invalid policies drop; tiny problems (`M<1024 or N<1024 or K<64`)
   collapse to the fallback alone; the rest are ranked (wave fill, occupancy,
   tails). An empty set is a hard error (no silent fallback).
4. The host builds descriptors for **all 11** policies each call
   (`make_descriptor_bank`), because any pruned-in config may win — this is the
   ~0.051 ms/call descriptor-rebuild overhead recorded in §6.

Constraints for any policy added to the universe: `BLOCK_K * itemsize >=
swizzle_byte_width`; stage memory `NUM_STAGES*(BLOCK_M+BLOCK_N)*BLOCK_K*2B +
barriers <= 101376 B`; `BLOCK_N <= 128` in the production universe. Changing the
universe length means changing the kernel signature **and** the if-chain
together.

### 4.4 Retained-reference and decision rules in force

- **`legacy_fused_sparton_bwd` is the test-pinned A/B reference of record** —
  the M11 segmented design, retained with a role comment naming its removal
  condition (it stays until a later milestone supersedes the comparison
  evidence). Retired implementations are either deleted or promoted to an
  explicit, test-pinned reference; never silent dead code, and never two legacy
  copies (when M13 promoted the split, the baton passed: the segmented design
  became the reference and the M2-era kernel was deleted in the same change).
- **D2 — kernel-body duplication** between the optimized forward and the GEMM
  benchmark (`scripts/bench_gluon_gemm.py`): the mainloop and the descriptor
  if-chain are intentionally **not** deduplicated. The two kernels differ in
  epilogue and barrier-reset structure, share all host-side plumbing via
  `_gluon_policy_runtime.py`, and a shared `@gluon.jit` mainloop helper would
  churn validated kernel code for no gain. **Discharged by re-affirmation** at
  the M12 close (no production-mainloop rewrite happened; the cross-reference
  comments were re-affirmed). Re-open only if a future milestone reopens the
  kernel body.
- **Capability dispatch** for fatal-failure APIs (the Gluon MMA families abort
  the process at LLVM selection — §2.2) uses static whitelists probed in
  subprocesses, never try/except fallback.

---

## 5. Kernel designs of record

### 5.1 Optimized forward (default)

One Gluon kernel, `sparton_optimized_forward_kernel`, no logits materialization
(allowed temporaries: stage buffers, accumulator fragments, `[BLOCK_N]`-sized
running state, the `[B, V]` outputs):

```text
grid over (batch b, vocab tile n0)                       # batch_block fixed at 1
running_max[BLOCK_N] = 0.0; running_idx[BLOCK_N] = 0
for s0 in range(0, S, BLOCK_M):
    acc[BLOCK_M, BLOCK_N] = TMA + mma_v2 mainloop over K  # staged A/B tiles, fp32 accumulate
    vals = (acc + bias[n0:n0+BLOCK_N]) * mask[b, s0+row]   # masked rows -> 0
    rows with s0+row >= S contribute 0                     # batch-boundary S-tail
    tile_max, tile_arg = max/argmax over rows (strict >)
    update running state with strict >
scores = log1p(relu(running_max)); store scores, running_idx (int64)
```

Mechanism details that hold:

- **Descriptors with no physical transpose.** `A = hidden.reshape(B*S, D)` is
  consumed via a host `TensorDescriptor` over the row-major `[M, K]` view.
  `B = embed [V, D]` row-major is consumed as conceptual `embed.T`: TMA loads
  `[BLOCK_N, BLOCK_K]` tiles into `NVMMASharedLayout` shared memory, read back
  through `.permute([1, 0]).load(DotOperandLayout(...))` (ldmatrix-transpose).
- **Staging protocol** (validated on production layouts): per-stage mbarrier
  (`init(count=1)` → producer `expect(bar, NBYTES)` → two
  `tma.async_copy_global_to_shared` → consumer `wait(bar, phase)`), circular
  slots `tile % NUM_STAGES`, a CTA `gl.barrier()` between the shared reads and
  the slot refill (WAR hazard). TMA + mbarrier + `mma_v2` compose in one kernel
  on sm_120.
- **Epilogue argmax** uses an explicit `gl.reduce` over `(value, row_index)`
  with a strict-`>`-plus-lowest-index combine (`gl.max` has no `return_indices`).
- **Tails.** K tail: TMA zero-pads out-of-bounds K (zeros contribute nothing).
  V tail: epilogue stores masked on `n < V`. S-tail / batch boundary: the A
  descriptor is over `[B*S, D]`, so a tile starting at `b*S + s0` can cross into
  the next batch's rows when `S % BLOCK_M != 0` — **those are real data, not
  zeros**, and the epilogue zeroes rows with `s0 + row >= S` (correctness-critical;
  unit-tested). All-negative/all-masked columns keep running state `(0.0, 0)`.
- **Policy autotune** over the bounded 11-policy universe (§4.3); accumulation
  fp32, outputs in hidden dtype, indices computed int32 / stored int64.

The selected scheduling variant of record is the non-persistent CTA-barrier
pipeline (the persistent/warp-specialized rewrite was evaluated and **not
shipped** — see §6).

### 5.2 Naive forward (debug baseline)

One Triton kernel, `sparton_naive_forward_kernel`: a program owns a
`batch_block × vocab_block` tile, loops over sequence chunks and over K chunks
with `tl.dot` (fp32 accumulate), applies bias and mask, runs online max/argmax
with the same zero-baseline strict-`>` semantics and tail handling, and stores
scores/indices. No TMA, no Gluon. Bounded Triton autotune (key `(S, D, V)`),
with the original fixed tile retained as a candidate. It does not materialize
logits, and is intended to isolate fused *semantics* from Gluon-specific
failure modes — expected to be slower than hybrid on GEMM-dominated shapes.

### 5.3 Hybrid forward (compatibility)

The compatibility path: a compiled (TorchInductor) tiled matmul/matmul-bias per
vocab tile (`v_tile_from_bs` chooses the tile count), then the Triton reduction
kernel `reduce_seq_max_log1p_relu_kernel_with_indices` for masking, sequence
max/argmax, ReLU, and `log1p`. It materializes per-tile `[B, S, V_tile]` logits
(not the full `[B, S, V]`). Two reduction helpers exist — one returns max
values plus indices (for autograd), one returns only values (skips the `[B, V]`
int64 buffer in inference paths); their memory trade-off is intentional. The
hybrid path must stay behaviorally stable.

### 5.4 Backward — the M13 split segmented backward (shared by all backends)

`fused_sparton_bwd_op` → `split_segmented_sparton_bwd`. It accumulates
`hidden_grad`, `embed_grad`, `bias_grad` in fp32 from the saved `(scores, idx,
hidden, embed, bias)`. Stages:

1. **Prep** (`bwd_prep_kernel`): compute `g = grad_out * exp(-scores)` where
   `scores > 0`, build the flat payload `(g, idx32, keys = b·S + idx)`, and a
   **device-side active count** (`n_active`) via a single counter.
2. **Exclusive-owner embed/bias-grad kernel**: writes `embed_grad` (and
   `bias_grad` when present) with **zero atomics** — each `(v, ·)` is written by
   its exclusive owner — into `torch.empty` outputs.
3. **Sort + payload gather**: `torch.sort` on the keys so equal destination
   rows are contiguous and all active entries precede all sentinels, then
   `bwd_gather_payload_kernel` reorders the gradient/index payload by the sort
   permutation into the sorted streams (`g_sorted`, `v_sorted`) the hidden-grad
   passes consume. (Stages 1–3 are the shared `_bwd_shared_stages` helper, also
   used by the legacy path in §5.5.)
4. **Two complementary hidden-grad passes** at one shared CHUNK granularity:
   - a **vectorized uniform-chunk streaming pass**
     (`uniform_hidden_grad_kernel`): branch-free, pipelined; deposits exactly
     the single-destination chunks. Autotuned, but **CHUNK is pinned to 64**
     across the family.
   - a **mixed-chunk segmented scan** (`mixed_hidden_grad_kernel`): deposits
     exactly the rest (the run-boundary chunks). It is **not** autotuned — it
     runs at the uniform winner's `best_config` CHUNK as its `GRANULE`,
     processing each granule in `SUB`-row segmented-scan tiles.

   `hidden_grad` is zero-filled; the two passes contribute ~two partial-sum
   atomics per destination run (the atomic structure is preserved from the M11
   segmented design).

**Three correctness invariants** — keep them in lockstep with any edit:

1. **Complement at one granularity.** The uniform pass deposits exactly the
   single-destination chunks at the shared CHUNK granularity; the mixed pass
   deposits exactly the rest. The mixed kernel is therefore **not autotuned** —
   it runs at the uniform winner's CHUNK, and `GRANULE % SUB == 0` is
   host-asserted. Independently tuned granularities silently drop contributions.
2. **Sorted-prefix bound.** The prep kernel's device-side active count is a
   valid loop bound only because sorted keys put every active entry strictly
   before every sentinel (`b·S + idx < B·S` for live entries).
3. **Sub-tile composition.** The scan may process a granule in SUB-row tiles
   only because chunk-local partials compose across tile boundaries (forced
   `is_end` at the last lane; a continued run emits its own partial with no
   start-correction).

Compile-time guards: the uniform kernel's maskless-load fast branch is gated on
`hidden_dim % BLOCK_D == 0` (a config breaking divisibility silently takes the
correct-but-slower masked path); a non-multiple-of-SUB CHUNK is caught by the
host assert. CHUNK is pinned to 64 because the mixed fraction scales with the
shared granule (`m ~ runs·CHUNK/N`), so a larger uniform-side CHUNK would
silently multiply the mixed pass's work.

### 5.5 Retained reference: `legacy_fused_sparton_bwd` (M11 segmented)

The M11 segmented backward (`segmented_sparton_bwd`: prep + exclusive-owner
embed/bias kernel + `torch.sort` + payload gather + a single segmented
hidden-grad scan kernel) is retained, reachable only through
`legacy_fused_sparton_bwd`, as the test-pinned A/B reference of record (§4.4).
It is the design the split supersedes and the baseline its speedups are quoted
against.

---

## 6. Measured state & residuals

All numbers are quoted from their runs of record (see
[DEVELOPMENT.md](DEVELOPMENT.md) for provenance and transcripts). The regime
map: `do_bench` (L2-flushed) is the latency of record; ncu durations are
inflated (structure/counters/ratios only); nsys for kernel inventory. Both
kernel performance tracks are **closed** as of 2026-06-13, each with a named
terminal residual; there is no planned milestone beyond M13.

### 6.1 Dev-shape snapshot

Dev shape `B=32, S=128, D=768, V=30522`, fp16, `do_bench`:

| metric | M10 gate run 2 | M11 gate run 2 | M13 |
|---|---:|---:|---:|
| optimized forward + bias | 0.900 ms | 0.880 ms | — |
| optimized fwd+bwd + bias | 2.325 ms | 1.949 ms | — |
| implied optimized backward | 1.425 ms | 1.069 ms | **0.982 ms** |

The dev-shape forward floor ratio of record is **1.02×** (M12 re-measured one
preserved run at **1.029×**) against the 0.880 ms full-V cuBLAS GEMM floor; the
dev forward is essentially done. Hybrid dev-shape reference: forward + bias
1.181 ms (M10), fwd+bwd 2.642 ms; peak extra forward memory 140.50 MiB for
hybrid vs 9.86 MiB for naive/optimized (output-only, ≈14× less).

### 6.2 Grid snapshot

Canonical `naver/splade-code-06B` grid (bf16, `D=1024, V=151936`, all-ones
masks; B∈{4,8,16}, S∈{256,512,768}):

- **Forward floor ratios (M12 run of record): 1.099–1.117×** the per-row cuBLAS
  GEMM floor (preserved same-run; the M8 same-run range was 1.094–1.123×; an
  earlier (v3-era) run's 1.054–1.113× came from a run whose per-row gemm column
  was not preserved and therefore only orients — see [DEVELOPMENT.md](DEVELOPMENT.md)
  M11/M12 for that history).
- Forward absolute gap per grid row (M12 run of record): **0.126–1.628 ms/call,
  a gap of 8.97–10.51% of forward time**.
- M11 implied backward 1.546–3.720 ms per row (−32…−41% vs M10); M13 grid
  implied backward **1.430–3.522 ms** (dev 0.982 ms).
- Derived backward share (re-derived on one provenance at M12): **18.5–52.3%**
  of optimized fwd+bwd across the grid, **54.9%** on the dev shape.
- Captured-real records (V=250002): backward **3.3–4.4 ms** post-M11;
  M13 split takes the captured-real cells to **1.46–1.60×** vs the M11
  segmented design (steps150 docs 1.568–1.596×); every canonical grid row is
  within band or improved.

### 6.3 Host launch overhead

Optimized-forward host overhead is **~0.119 ms/call** wall-minus-GPU at
`8×128×768×1280`, of which **~0.051 ms** is rebuilding the 22-descriptor bank
every call (§4.3). Irrelevant at ≥1 ms GPU times — which is every documented
workload; it would bind only for a small-shape latency-critical caller of the
head itself, which nothing in the repo exercises. This is the deferred F9
overhead (see launcher v2 below).

### 6.4 Determinism band

`embed_grad`/`bias_grad` are structurally deterministic (exclusive-owner plain
stores). `hidden_grad` atomic-accumulation order still varies, but with ~60×
fewer atomics than the pre-M11 kernel. Per-call relative spread: gradient-norm
≤ 1.2e-7, element-sensitive loss-proxy ≤ 4.2e-6 (the M13 split preserves the
M11 atomic structure and band). At **training scale** the chaotic early regime
amplifies this to a measured same-config 150-step loss spread of **16–38%**
depending on backend and statistic (3 seed-matched repeats per backend; this
supersedes the M10-era "~20%" estimate). Establish same-config noise bands
before reading meaning into cross-config training differences; do not promise
bitwise-reproducible training.

### 6.5 Forward residual (M12: tensor-pipe-bound, closed without kernel work)

The production forward kernel was profiled for the first time at M12 and is
**tensor-pipe-bound at 92.3–94.4%** utilization, with the L2 fabric
simultaneously at 89–91% and DRAM at the compulsory byte floor — there are no
scheduling bubbles for a persistent/warp-specialized rewrite to fill. This
**overturned the prior inference** drawn from the GEMM bring-up benchmark kernel
(which sat at 63.9% tensor pipe vs cuBLAS's 86.6% at identical occupancy and
`mma.sync` instruction family — that gap was intra-CTA pipelining quality, but
it was a *benchmark* kernel, not production, and did not transfer). M12 closed
without kernel work: the entry rule's first clause passed (≥10% gap on five grid
rows at 10.16–10.51%, inside the noise band) but the second clause failed
(tensor pipe ~18 points above the 74% binder threshold). **Terminal forward
residual: per-cycle pipe efficiency + L2 pressure at the autotuned 64×64×32 tile
shape — a tile-shape question, not a scheduling one.**

### 6.6 Backward residual (M13)

The M13 split recovered the residual the M11 design left (the embed-kernel
attribution was corrected — its binder is hidden-row gather re-reads, already at
the traffic floor, not g/idx streams). Validated traffic model: per-buffer
residuals ≤1% on the decision counters across four shapes; the uniform pass
attains its floor (production profile 1.43 ms at 56 regs / 73.5% occupancy on
the doc record vs the 1.34 ms modeled conservative floor). **Terminal backward
residual:** the uniform-pass LTS ≈ 61–67% (config-dependent) vs the embed
kernel's 82–104%, plus the short-run regimes capped by the mixed fraction.
Atomics are no longer a bottleneck anywhere (L2 reduction sectors down 27.7×
dev / 51.7× corner / 264× real query record vs pre-M11).

### 6.7 Operative deferral: launcher v2

The scheduled launcher v2 (an owned two-phase selector that builds **one**
descriptor pair per call and deletes the 22-slot bank + `POLICY_ID` if-chain,
fixing the §6.3 overhead) is **deferred by maintainer decision**, with these
still-operative reasons:

1. **No identified latency scenario at this seam.** In every documented
   workload (Trainer training, batched encoding) the head runs behind a backbone
   forward at shapes where its GPU time is ≥1 ms and host launch work overlaps
   it. The 0.119 ms/call binds only for a small-shape latency-critical caller of
   the head itself — nothing in the repo exercises one.
2. **The replacement is unproven against the incumbent.** An owned selector
   still pays the shared wrapper/op/autograd machinery (~0.013 ms/call) plus
   per-call winner-descriptor construction (~0.005 ms) plus a cache lookup; the
   paper estimate (~0.018 ms) lands under the ≤0.02 ms/call target with no
   margin, and no prototype was ever built. The assumed ~6× win could be ~2–3×
   or miss the gate.
3. With (1) and (2) unresolved, churning the validated production selection path
   is speculative-benefit churn.

**Revival triggers:** (a) a real latency/small-shape user appears; (b) a future
forward-kernel rewrite forces the kernel signature open anyway (fold the
single-pair launch in there); (c) a Triton autotune-API change forces the
mechanism open. **Gate design on revival:** prototype and measure the actual
host floor before committing to the target; deterministic candidate-set/ranking
parity vs the `early_config_prune` path on canonical keys (winner-time parity
within a measured A-vs-A band, not exact selection match — autotune-selection
jitter is ±5% between processes on borderline cells); full suite + shape soak;
grid/dev rows within ±5%; keep the multi-slot kernel until parity is proven.
Descriptor-bank *caching* is rejected as an alternative: if the overhead ever
matters, the fix is bank *removal*, not caching it.

### 6.8 Out of scope (hardware/scope)

Excluded from production scope, unchanged across the arc (revisit only on a
hardware or Triton-capability change, re-running the §2.2 probe matrix first):
WGMMA / TCGen05 / TMEM `load_max` (real sm_90a/sm_100 hardware; fatal aborts
here); FP8/FP6/FP4 and block-scaled formats; cluster multicast / multi-CTA
clusters / Cluster Launch Control; native TMA gather/scatter for backward
(Blackwell-datacenter API); CUTLASS/C++ integration; split-K (complicates the
online max); migrating the custom ops to inferred schemas (no `Optional[Tensor]`
return support). Hybrid-side performance fixes (bias fusion, tile-count tuning,
slice-copy elimination) are deferred indefinitely — post-promotion, hybrid is
the compatibility path, so optimizing it double-pays.

---

## 7. Validation gates

The standing gate suite runs under the §2.4 hardened-env prefix, serially. This
section names the gates; the canonical command list and the rules behind them
live in [../AGENTS.md](../AGENTS.md) "Validation" — do not duplicate it.

- **Syntax / import.** `py_compile` over `src/sparton/*.py`, `training/*.py`,
  `tests/*.py`, `scripts/*.py`; `import sparton` must print the export list with
  no other stdout (silent-import regression test).
- **Pytest suite.** Full suite **136 tests** at the M13 close; the documented
  quick loop `-m "not slow"` is **113**. (Pytest ≥9; parametrized cases live in
  named tables with `strict_parametrization_ids`.) The suite must pass under
  `python -m pytest`, the venv `pytest` console script, and `pytest
  /workspace/sparton/tests` from a foreign working directory.
- **Forward correctness soak** (forward-surface changes — kernel, policy
  generation, descriptors, validation): `scripts/soak_optimized_correctness.py`
  (384-case sweep over S/B/D/V/bias/dtype with random masks incl. all-zero
  rows; scores vs reference + the §3.2 index contract). Use `--quick` while
  iterating, the full sweep as the gate.
- **Autograd / AMP / training path:** `scripts/probe_training_smoke.py`
  (synthetic head-only AMP parity, fp16 GradScaler + bf16 autocast). Under fp16
  AMP the GradScaler's 2^16 initial scale legitimately overflows fp16
  score-gradients in early steps — skipped calibration steps are expected, not
  a bug.
- **Backward-kernel changes:** `scripts/bench_backward.py --impls
  current,legacy` over {uniform, zipf} **and** the captured-real bundles in
  `tests/data/bundles/` (regenerable by `scripts/capture_index_distributions.py`,
  which reuses existing files). The harness numerically verifies every cell
  against production before timing it. **Uniform-only evidence is never
  sufficient for a backward change.** The gradient matrix is checked against the
  closed-form expectation from saved `(scores, idx)` and A/B vs the segmented
  reference.
- **Ownership-semantics changes** (atomics→stores, `torch.empty` outputs,
  complementary multi-kernel writers): `compute-sanitizer`
  racecheck/memcheck/initcheck on a small shape through the harness — a
  non-divisible-D small shape (e.g. `B4 S33 D64 V2048`) exercises the masked-tail
  paths, and initcheck mechanically validates the empty-allocation coverage
  claim (§3.4).
- **Validation error templates are part of the API.** Each `_validation.py` rule
  has a `pytest.raises(..., match=...)` test pinning a stable message substring;
  the template shape is `sparton {backend} forward: ...`.
- **Training behavior:** prefer the synthetic smoke probe; full Hub-backed
  `training/train.py` runs are expensive and not casual validation (a
  steady-state 150-step tier-2 run is ~25 s on this host after the cold first
  run). `training/train.py` is smoke-validated against transformers 5.11 only
  (150-step runs); checkpointing/resume/distributed paths are unvalidated.

There is no CI config (no CUDA runner is available to this repo); the documented
command suite is the gate mechanism.
