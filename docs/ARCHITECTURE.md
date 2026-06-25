# Sparton Architecture

This document describes the **built system** as it ships today: the three
kernels, the platform facts they run on, the mathematical and API contract
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
implementation kernels that share one numerics contract and one forward op
schema family; the backward is two kernels selected per-forward:

| Kernel | Role | Forward implementation |
|---|---|---|
| `optimized` | **Default** where available (CUDA sm_90+ with importable `triton.tools.tensor_descriptor`); the production fast path | Single pure-Triton (`@triton.jit`) persistent kernel: host-side TMA descriptors + `tl.dot` mainloop + fused max/argmax/ReLU/log1p epilogue; measured self-tuner; warp specialization as a tuned dimension |
| `hybrid` | Compatibility path; must stay behaviorally stable | Compiled (TorchInductor) tiled matmul per vocab tile + Triton sequence-reduction kernel; materializes per-tile `[B, S, V_tile]` logits |
| `naive` | `tl.dot` fused-forward debug baseline | Single Triton kernel with `tl.dot`, bounded autotune; a one-tile, no-TMA implementation that isolates fused semantics |

The backward is selected per-forward: the optimized forward uses the
**`optimized`** backward (`optimized_bwd_op`, op `sparton::optimized_bwd`) — the
uniform + mixed hidden-grad design; the hybrid and naive forwards use the
**`mono`** backward (`mono_bwd_op`, op `sparton::mono_bwd`) — the restored M2
fully-atomic scatter, which is also the optimized↔mono A/B baseline (§5).
`optimized` was promoted to the default after the M10 gates (forward faster
than hybrid on every measured shape, output-only memory, full correctness
matrix green); promotion gates and evidence are in
[DEVELOPMENT.md](DEVELOPMENT.md) M10. The optimized forward was
**reimplemented in pure Triton post-M13**, replacing the original Gluon kernel
and removing Gluon from the package entirely (§5.1; DEVELOPMENT.md
"Post-M13 — optimized promoted to a pure-Triton TMA forward").

### Module map (`src/sparton/`)

The package is organized by **[direction, variant]**: a `forward/` package and
a `backward/` package, with the facade/router beside them. The only
cross-package edge is `forward/* → backward` (no cycle) — the backward kernels
live in the neutral `backward/` package, not inside the hybrid forward module.

```text
src/sparton/
  __init__.py                  # exports SpartonHead only when CUDA is available
  api.py                       # public facade/router: resolve_kernel, SpartonHead,
                               #   _default_kernel, re-exports, lazy __getattr__ for optimized symbols
  forward/
    __init__.py                # eager hybrid + naive; lazy optimized via __getattr__
    _autograd.py               # shared forward autograd: shared_fwd_fake / shared_setup_context /
                               #   shared_backward / register_shared_forward (registered per op)
    hybrid.py                  # hybrid forward (compiled tiled matmul + Triton reduction):
                               #   hybrid_fwd_op (op sparton::hybrid_fwd), hybrid_forward
    naive.py                   # tl.dot fused-forward debug baseline:
                               #   naive_fwd_op (op sparton::naive_fwd), naive_forward
    optimized.py               # pure-Triton persistent fused forward (host-side TMA + tl.dot),
                               #   self-contained measured tile self-tuner, WS as a tuned dimension;
                               #   optimized_fwd_op (op sparton::optimized_fwd) — the default kernel
  backward/
    __init__.py                # two backward ops, identical schema, one register_fake each:
                               #   optimized_bwd_op (op sparton::optimized_bwd),
                               #   mono_bwd_op (op sparton::mono_bwd)
    optimized.py               # the OPTIMIZED backward (used by the optimized forward): uniform +
                               #   mixed hidden-grad kernels + optimized_bwd, plus the folded shared
                               #   prep stages (_bwd_shared_stages + bwd_prep_kernel + exclusive-owner
                               #   embed_grad_kernel + bwd_gather_payload_kernel)
    mono.py                    # the MONO backward (used by hybrid/naive; the A/B baseline): the
                               #   restored M2 fully-atomic scatter (mono_bwd_kernel + mono_bwd)
  _runtime.py                  # is_optimized_kernel_available(): CUDA + sm_90 capability +
                               #   importable triton.tools.tensor_descriptor (the default-resolution gate)
  _device.py                   # resolves DEVICE (import-time DEBUG log); no longer public
  _validation.py               # autocast canonicalization, prepare_forward_inputs (the shared
                               #   autocast+validate+contiguous helper), validate_forward_inputs(kernel=...)
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
notes, and re-validate the optimized kernel on any GPU or Triton change.

### 2.1 Runtime snapshot (validation platform)

```text
Python 3.12.3
Torch  2.12.0a0+0291f960b6.nv26.04.48445190
Triton 3.7.1                         (target: GPUTarget(backend='cuda', arch=120, warp_size=32))
CUDA   13.2 (torch.version.cuda)
GPU    NVIDIA GeForce RTX 5090, CC (12, 0), 170 SMs
warp size 32, max threads/block 1024, max threads/SM 1536
shared memory: 49152 B/block default, 101376 B/block opt-in, 102400 B/SM
total memory 34190458880 B (~31.8 GiB), L2 cache 100663296 B (96 MiB)
```

The dev environment is the NGC container image
`nvcr.io/nvidia/pytorch:26.05-py3` (torch 2.12 nightly + Triton 3.7.1 + CUDA
13.2 toolchain). The L2 size (96 MiB) is large enough to hold the entire
`hidden` activation for typical dev shapes (`4096 x 768` fp16 = 6 MiB), which
matters for GEMM scheduling policy.

### 2.2 Optimized kernel availability (sm_90+ host-side TMA)

The optimized forward is built on **host-side TMA descriptors**
(`triton.tools.tensor_descriptor.TensorDescriptor`) — a Hopper-class feature — so it is gated at
CUDA capability **sm_90+**. `_runtime.is_optimized_kernel_available(device)` is the cheap
check used by default-kernel resolution: it returns `(False, reason)` if CUDA is unavailable, if
`torch.cuda.get_device_capability(device) < (9, 0)`, or if `triton.tools.tensor_descriptor` does
not import; otherwise `(True, "sm_<major><minor>")`. It imports no kernel code.

This is a deliberate narrowing from the removed Gluon optimized kernel (which ran on sm_80+ via
`mma_v2`): host-side TMA does not exist below Hopper. On an sm_80 (Ampere) device default
resolution falls back to `hybrid` with a one-time `RuntimeWarning`; an explicitly selected
`optimized` still raises with the reason. The validation GPU is sm_120 (Blackwell), where the gate
returns available.

### 2.3 Triton API surface used by the optimized forward

The kernel is stock `@triton.jit` (no Gluon — Gluon was evaluated and removed; see §5.1 and the
CHANGELOG). The Triton features it relies on:

- **Host-side TMA descriptors**: `triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(t,
  block_shape)`, passed as kernel args. Constraints (enforced by `_validation.py` + the launcher):
  base 16-byte aligned, non-last strides 16-byte aligned, last dim contiguous, and
  `D * element_size % 16 == 0` (the inner TMA box; e.g. fp16 needs `D` a multiple of 8). Host
  descriptors are driver-filled (`fill_tma_descriptor_tiled`), leaving `global_scratch_size == 0`,
  so the kernel needs **no `triton.set_allocator`** — the process-global side-effect that
  device-side `tl.make_tensor_descriptor` would require (incompatible with CUDA graphs / the
  caching allocator).
- **`tl.dot`** for the GEMM mainloop (fp32 accumulate; lowers to the `mma.sync` family on sm_120 —
  the same instruction the removed Gluon `mma_v2` path used).
- **`tl.range(..., warp_specialize=WS)`** for the optional producer/consumer split — a *tuned*
  dimension (§5.1). Constraint: Triton 3.7.1's auto-WS pass requires single-result reduces, so the
  argmax is expressed as `tl.max` + masked `tl.min` (not a 2-result `tl.reduce`) when WS is on (§5.1).
- **`triton.testing.do_bench`** as the self-tuner's timer — a measurement caveat: on the validation
  GPU it is clock-boost/thermal/order-biased (DEVELOPMENT.md "Post-M13 …", §do_bench).

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
- The declared `triton>=3.3.1` floor is the **hybrid kernel's** floor. The
  `optimized` kernel additionally requires CUDA **sm_90+** and an importable
  `triton.tools.tensor_descriptor` (host-side TMA; §2.2); the validated Triton is
  3.7.1. Where those are missing, default resolution falls back to `hybrid` and an
  explicitly selected `optimized` raises with the reason (§2.2).
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
strictly exceeds the current running value, so within a kernel ties resolve to
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
is greater than zero** (zero-baseline policy). Within one kernel, ties resolve
to the lowest sequence index.

**Across kernels with different accumulation precision, the winner at a
near-tie is explicitly unspecified.** `naive`/`optimized` accumulate logits in
fp32; `hybrid` produces input-dtype logits. So at near-ties (logit gaps within
input-dtype rounding) different kernels may legitimately return different
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
  intended behavior for all future kernels. Cross-kernel score agreement is
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

| Kernel | Supported `hidden`/`embed`/`bias` dtype |
|---|---|
| `optimized` | fp16, bf16 (descriptor `element_bitwidth=16`; also requires `D * element_size % 16 == 0` for TMA) |
| `naive` | fp16, bf16 |
| `hybrid` | fp16, bf16, fp32 (fp32 is a permitted compatibility path, not benchmark-covered) |

### 3.6 Custom op schemas and autograd saved-tensor set

Op names and schemas are the stable layer. Each kernel registers its own
forward op (so schemas may diverge without touching the others), and each
forward registers autograd that calls the backward op matching its kernel: the
optimized forward → `optimized_bwd_op` (op `sparton::optimized_bwd`); the hybrid
and naive forwards → `mono_bwd_op` (op `sparton::mono_bwd`). The two backward ops
share one saved-tensor schema (one `register_fake` each). The current op names —
`sparton::hybrid_fwd`, `sparton::naive_fwd`, `sparton::optimized_fwd`,
`sparton::optimized_bwd`, and `sparton::mono_bwd` — are the stable layer; the
`hybrid_fwd` name replaced the earlier `fused_sparton_fwd` name in the post-M13
reorganization (a one-time breaking rename; see CHANGELOG 2026-06-25).

```text
sparton::hybrid_fwd    (Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)   # hybrid
sparton::naive_fwd     (Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)   # naive
sparton::optimized_fwd (Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)   # optimized
sparton::optimized_bwd (Tensor grad_out, Tensor max_scores, Tensor max_idx, Tensor hidden,
                        Tensor embed, Tensor? bias, Tensor mask)               -> (Tensor, Tensor, Tensor?)
sparton::mono_bwd      (Tensor grad_out, Tensor max_scores, Tensor max_idx, Tensor hidden,
                        Tensor embed, Tensor? bias, Tensor mask)               -> (Tensor, Tensor, Tensor?)  # identical schema
```

**Autograd saves**: max scores, max indices, hidden states, decoder weights,
bias, and mask. A backward swap that keeps this saved-tensor set is schema-safe
inside the op (proven by M11 and M13). A change to the saved-tensor set
requires a new op name.

---

## 4. Architecture layering & rules

### 4.1 The layering of record

`SpartonHead.forward` resolves a kernel **once at construction** (the instance
binds the per-kernel wrapper; the env var `SPARTON_KERNEL` is read once at
import, never per call) and `forward` just calls the bound wrapper:

```text
SpartonHead.forward
  └─ <kernel>_forward(hidden, embed, bias, mask)       # public per-kernel wrapper
       └─ _validation.prepare_forward_inputs(..., kernel=<kernel>)   # the one shared helper:
            ├─ autocast_canonicalize(...)              # mirrors torch.autocast: casts fp32
            │                                          #   master params to the autocast dtype
            ├─ validate_forward_inputs(..., kernel=<kernel>)  # contract checks; raises named errors
            └─ .contiguous() canonicalization (all inputs)
       └─ sparton::<kernel>_fwd custom op              # stable schema; assumes validated, contiguous inputs
            └─ kernel launch (+ autograd that saves the op's inputs; backward →
               optimized_bwd_op for optimized, mono_bwd_op for hybrid/naive)
```

Wrappers (`hybrid_forward`, `naive_forward`, `optimized_forward`) are the only
public callables; `SpartonHead` binds wrappers, never raw ops. All three
wrappers run the same `prepare_forward_inputs` seam (de-duplicated from the
formerly line-for-line-parallel wrapper bodies). The ops assume
validated, contiguous inputs — **raw-op callers (e.g. profiling targets) bypass
validation by design**; do not "fix" that by validating inside the ops.

`autocast_canonicalize` exists because the custom ops are not autocast-registered:
under an active CUDA autocast region it casts `hidden`/`embed`/`bias` to the
autocast dtype (fp16/bf16) the way `torch.matmul` would, and leaves everything
unchanged outside autocast (`mask` is never cast). This is what makes standard
AMP training with fp32 master parameters work on every kernel.

### 4.2 Cross-kernel rules

- **Symmetry rule.** The Nth implementation of a pattern mirrors the others
  byte-for-byte where semantics allow. The three forward wrappers are
  intentionally parallel — now over the shared `prepare_forward_inputs` seam;
  the historical hybrid no-contiguity bug existed precisely because hybrid
  lacked the canonicalization the others had.
- **One-seam changes.** New cross-kernel behavior is one shared helper called
  at exactly one layer (`prepare_forward_inputs` at the top of each wrapper;
  `_bwd_shared_stages` shared by the optimized backward's two hidden-grad
  passes), never N divergent copies.
- **Kernel isolation.** Forward kernels never import each other; the only
  cross-package edge is `forward/* → backward` (each forward imports its backward
  op — optimized → `optimized_bwd_op`, hybrid/naive → `mono_bwd_op` — from
  `backward`). `_runtime` (the availability gate) imports no kernel code and
  stays importable on any machine.
- **No silent fallbacks.** The single sanctioned exception is *default
  resolution*: with no `kernel` argument and no `SPARTON_KERNEL`, an
  unavailable optimized kernel falls back to hybrid with a one-time
  `RuntimeWarning` (M10). An explicitly selected kernel that is unavailable
  **raises with the reason**. No second exception may be added — data-dependent
  algorithm dispatch inside an op would also need a host sync, which is why the
  backward ships fixed kernel families (§5.4).
- **No data-dependent algorithm dispatch inside an op.** It would need a host
  sync and a second fallback seam. The optimized backward launches its uniform +
  mixed kernels unconditionally with complementary device-side predicates — no
  dispatch. (Kernel selection happens once at forward construction, not per call.)
- **Backward swap is schema-safe.** A backward change that keeps the
  saved-tensor set swaps inside a backward op (`sparton::optimized_bwd` /
  `sparton::mono_bwd`) without touching forward op schemas or autograd wiring
  (proven by M11→M13). A changed saved-tensor set requires a new op name.
- **Diagnostics** go through `logging.getLogger("sparton")` at DEBUG; library
  code never `print`s (a regression test enforces silent import).

### 4.3 Retained-reference and decision rules in force

- **The `mono` backward is the test-pinned original-vs-optimized A/B baseline** —
  the restored M2 fully-atomic scatter (in `backward/mono.py`). Unlike a pure
  reference, it is also a **production** backward: the hybrid and naive forwards
  use it, so `kernel=hybrid`/`naive` runs the original-ish path end-to-end while
  `kernel=optimized` runs the current best — a clean A/B with one switch. Retired
  implementations are either deleted or promoted to an explicit, test-pinned
  reference; never silent dead code, and never two retained references (when the
  backward was reduced to `{mono, optimized}`, the M11 segmented design was
  deleted and the M2 atomic kernel restored as `mono`).

---

## 5. Kernel designs of record

### 5.1 Optimized forward (default)

One pure-Triton (`@triton.jit`) kernel, `optimized_fwd_kernel`, no logits
materialization (allowed temporaries: the `tl.dot` accumulator, `[BLOCK_N]`-sized
running state, the `[B, V]` outputs). It is **persistent**: a fixed CTA count
grid-strides over the `(batch, vocab-tile)` output tiles, S-loop inside.

```text
persistent grid (NUM_CTAS); each CTA grid-strides over tile_id -> (batch b, vocab tile n0):
    running_max[BLOCK_N] = 0.0; running_idx[BLOCK_N] = 0   # reset PER TILE (persistent landmine)
    for s0 in range(0, S, BLOCK_M):
        acc[BLOCK_M, BLOCK_N] = tl.dot over K of hidden_desc[m] x embed_desc[n0].T  # fp32 accumulate
        vals = (acc + bias[n0:n0+BLOCK_N]) * mask[b, s0+row]   # masked rows -> 0
        rows with s0+row >= S contribute 0                     # batch-boundary S-tail
        tile_max, tile_arg = max/argmax over rows (strict >)
        update running state with strict >
    scores = log1p(relu(running_max)); store scores, running_idx (int64)
```

Mechanism details that hold:

- **Host-side TMA descriptors, no physical transpose.** `A = hidden.reshape(B*S, D)`
  and `B = embed [V, D]` are each wrapped in a host
  `triton.tools.tensor_descriptor.TensorDescriptor`; the kernel does
  `hidden_desc.load([m, k])` / `embed_desc.load([n, k])` and `tl.dot(a, tl.trans(b), acc)`.
  Host descriptors are driver-filled, so `global_scratch_size == 0` and the kernel
  needs **no** `triton.set_allocator` (§2.3). The compiler owns SMEM staging /
  pipelining behind the launch `num_stages`; there is no hand-rolled mbarrier protocol.
- **`D`/`V` are `constexpr`** — folds `k_tiles`/`n_col`/the V-tail and
  strength-reduces the persistent `tile_id` div/mod to a constant-divisor
  multiply-shift; B/S stay runtime, so **one compile per (model, tile)** serves all
  batch/sequence shapes. Measured-neutral, kept as the correct structure.
- **Per-tile running-state reset is a load-bearing correctness invariant.** One CTA
  processes many tiles; `running_max`/`running_idx` reset at the *top* of each tile,
  or a prior tile's max leaks (guarded by `test_optimized_forward_persistent_multitile`,
  which forces `NUM_CTAS=1`).
- **Warp specialization is a tuned dimension.** One kernel carries both argmax
  epilogues behind a `WARP_SPECIALIZE` constexpr `if`: the fast 2-result `tl.reduce`
  strict-tie combine for the homogeneous path, and an equivalent `tl.max` + masked
  `tl.min` (two single-result reduces) for the `tl.range(warp_specialize=True)` path
  — the only form Triton 3.7.1's auto-WS pass accepts (it asserts single-result
  reduces). The dead arm is trace-time-eliminated; the two arms are bit-identical
  (`tl.max` returns an actual element, so `vals == tmax` is exact, and
  min-of-qualifying-row-indices == the strict-`>` lowest-index tie-break).
- **Tails.** K tail: TMA zero-pads out-of-bounds K. V tail: epilogue stores masked
  on `n < V`. S-tail / batch boundary: the A descriptor is over `[B*S, D]`, so a tile
  starting at `b*S + s0` can cross into the next batch's rows when `S % BLOCK_M != 0`
  — **those are real data**, and the epilogue zeroes rows with `s0 + row >= S`
  (correctness-critical; unit-tested). All-negative/all-masked columns keep `(0.0, 0)`.
- **Measured self-tuner, not a derived policy bank.** A module-level cache maps
  `(D, V, dtype, arch) -> (tile, warp_specialize)` — keyed on operand size + dtype +
  arch, **NOT** B/S (the persistent grid fills the GPU at any trip count;
  B/S-independence is unit-tested). On a miss it builds a self-contained candidate set
  (`OptimizedTile` / `_CANDIDATE_TILES`, SMEM-validated against the device opt-in
  limit), correctness-gates each tile against the analytic tile's own output, then
  times with `triton.testing.do_bench` (rank homogeneous tiles, then compare
  {top-2} × {homogeneous, WS}). `SPARTON_OPTIMIZED_AUTOTUNE` (default on; off -> the
  small always-valid `_analytic_tile`); `SPARTON_OPTIMIZED_NUM_CTAS` / `_CTAS_PER_SM`
  size the grid; `SPARTON_OPTIMIZED_WARP_SPECIALIZE` forces WS when autotune is off.
  The offline pick was provably non-optimal (the measured winner for some shapes is the
  128-family tile — a ~14% lever the autotuner captures).
- **Numerics.** fp32 `tl.dot` accumulation; outputs in hidden dtype; indices int32
  internally, int64 stored. Forward op schema / saved-tensor set identical to the
  other kernels; the optimized forward's backward is the `optimized` backward
  (§5.4), which shares its saved-tensor schema with `mono` (§5.5).

History: this kernel replaced a Gluon (TMA + `mma_v2`, mbarrier-staged, 11-policy
autotune-bank) optimized forward post-M13; the Gluon path and the intermediate
`experiment` kernels were removed. The convergent arc, the WS blocker/fix, the
`do_bench` measurement caveat, and the re-baseline are in DEVELOPMENT.md "Post-M13 —
optimized promoted to a pure-Triton TMA forward".

### 5.2 Naive forward (debug baseline)

One Triton kernel, `naive_forward_kernel`: a program owns a
`batch_block × vocab_block` tile, loops over sequence chunks and over K chunks
with `tl.dot` (fp32 accumulate), applies bias and mask, runs online max/argmax
with the same zero-baseline strict-`>` semantics and tail handling, and stores
scores/indices. No TMA, one program-owned tile. Bounded Triton autotune (key
`(S, D, V)`), with the original fixed tile retained as a candidate. It does not
materialize logits, and isolates the fused *semantics* in the simplest possible
kernel — expected to be slower than hybrid on GEMM-dominated shapes.

### 5.3 Hybrid forward (compatibility)

The compatibility path: a compiled (TorchInductor) tiled matmul/matmul-bias per
vocab tile (`v_tile_from_bs` chooses the tile count), then the Triton reduction
kernel `reduce_seq_max_log1p_relu_kernel` (host helper
`reduce_seq_max_log1p_relu`) for masking, sequence max/argmax, ReLU, and
`log1p`. It materializes per-tile `[B, S, V_tile]` logits (not the full
`[B, S, V]`). The reduction helper returns max values **plus** indices (the
path autograd needs); the earlier values-only reduction variant was deleted as
dead code (zero callers). The hybrid path must stay behaviorally stable.

### 5.4 The optimized backward (selected by the optimized forward)

`optimized_bwd_op` (op `sparton::optimized_bwd`) → `optimized_bwd`
(`backward/optimized.py`); the uniform + mixed hidden-grad design promoted at M13
(then named `split`). It accumulates `hidden_grad`, `embed_grad`, `bias_grad` in
fp32 from the saved `(scores, idx, hidden, embed, bias)`. Stages:

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
   passes consume. (Stages 1–3 are the `_bwd_shared_stages` helper, folded into
   `backward/optimized.py` alongside the two hidden-grad passes.)
4. **Two complementary hidden-grad passes** at one shared CHUNK granularity:
   - a **uniform-chunk streaming pass** (`uniform_hidden_grad_kernel`):
     branch-free, pipelined; deposits exactly the single-destination chunks.
     Its per-chunk reduce `sum_k g[k]·embed[v[k],:]` is a **dtype-aware
     compensated `tl.dot`** (`g` as a `[MPAD=16, CHUNK]` matrix with only row 0
     live × the gathered `[CHUNK, BLOCK_D]` embed tile): feeding the gathered
     tile to a dot lets the matmul software-pipeliner multi-buffer it via
     cp.async, relieving the L1TEX/load-issue bound (→ L2-BW-bound) and
     overlapping the reduce. `g` is fp32 and enters the model-dtype MMA via a
     hi/lo split for fp32-grade precision; how it is brought into operand range
     is the `COMP_MODE` constexpr, resolved host-side from the dtype
     (`_resolve_uniform_comp_mode`): **fp16/fp32 block-scale** (per-chunk
     `s=max|g|`, `×s`) because g overflows fp16 under AMP loss-scaling, **bf16
     skips the scale** (its 8-bit exponent already spans fp32 range — faster and
     ≤ error). Both range-safe under fp16+bf16 AMP. `SPARTON_BWD_COMPENSATE=0`
     opts into a single non-compensated dot (faster, precision-degraded, unsafe).
     Autotuned (`BLOCK_D` = the dot's N, ≥64), but **CHUNK is pinned to 64** (the
     complement granularity *and* the dot's K).
   - a **mixed-chunk segmented scan** (`mixed_hidden_grad_kernel`): deposits
     exactly the rest (the run-boundary chunks). It is **not** autotuned — it
     runs at the uniform winner's `best_config` CHUNK as its `GRANULE`,
     processing each granule in `SUB`-row segmented-scan tiles (launch retuned
     to BLOCK_D=64 / SUB=32 / num_stages=3).

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

### 5.5 The `mono` backward (restored M2 fully-atomic scatter)

`mono_bwd_op` (op `sparton::mono_bwd`) → `mono_bwd` (`backward/mono.py`), the M2-era
fully-atomic backward restored verbatim from commit `6e19af3`. One kernel
(`mono_bwd_kernel`): it loads the saved argmax index, computes
`g = grad_out · exp(−scores)` where `scores > 0` (the **same exact gradient** as
§5.4), and `atomic_add`s the hidden/embed/bias contributions into
**zero-initialized** fp32 buffers (`torch.zeros`, not `torch.empty` — every active
gradient is atomically accumulated and only argmax-winner rows are touched, so
untouched rows must read 0). A CTA-level `tl.sum(block_max_logits) == 0` early-exit
skips fully-inactive blocks. It is an unoptimized full scatter — non-deterministic
in every buffer and slow by design.

It is **production** for the hybrid and naive forwards (which register
`mono_bwd_op`), and simultaneously the **original-vs-optimized A/B baseline**
(§4.3): the §5.4 optimized backward's speedups are quoted against it. On
captured-real records the optimized backward is ~2–3.7× faster (mono ≈ 0.27–0.48×
of optimized), gradients matching to ~1e-6; on the synthetic `f = 0.10` short-run
regime the two are near-parity (mono's zero-block early-exit). The M11 segmented
design that the optimized backward grew from was removed in the same change that
restored `mono`.

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
- Captured-real records (V=250002): backward **3.3–4.4 ms** post-M11; at M13 the
  optimized backward (then `split`) took the captured-real cells to **1.46–1.60×**
  vs the since-removed M11 segmented design (steps150 docs 1.568–1.596×); every canonical grid row is
  within band or improved.

### 6.3 Host launch overhead

The Gluon optimized forward carried **~0.119 ms/call** wall-minus-GPU host overhead
at `8×128×768×1280`, of which **~0.051 ms** was rebuilding a 22-descriptor bank every
call — the motivation for "launcher v2" (§6.7). The pure-Triton optimized kernel that
replaced it builds **one** descriptor pair per call (no bank, no `POLICY_ID` if-chain),
so that specific bank-rebuild cost is gone; its steady-state host overhead on the new
kernel is unremeasured but bounded below the Gluon figure. Irrelevant at ≥1 ms GPU
times — which is every documented workload; it would bind only for a small-shape
latency-critical caller of the head itself, which nothing in the repo exercises (the
deferred F9 overhead, §6.7).

### 6.4 Determinism band

For the **`optimized`** backward, `embed_grad`/`bias_grad` are structurally
deterministic (exclusive-owner plain stores) and `hidden_grad` atomic-accumulation
order still varies, but with ~60× fewer atomics than the pre-M11 kernel. Per-call
relative spread: gradient-norm ≤ 1.2e-7, element-sensitive loss-proxy ≤ 4.2e-6
(the uniform + mixed passes preserve the M11 atomic structure and band). The
**`mono`** backward `atomic_add`s all three gradients, so none of its buffers is
deterministic (expected for the unoptimized baseline). At **training scale** the chaotic early regime
amplifies this to a measured same-config 150-step loss spread of **16–38%**
depending on kernel and statistic (3 seed-matched repeats per kernel; this
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
shape — a tile-shape question, not a scheduling one.** (That M12 profile was of the
Gluon optimized; the pure-Triton kernel that replaced it re-baselines to **≈parity**
at this large/throughput shape — consistent with the dual-saturation finding, where
beating a saturated forward needs *doing less work* — and **wins** the small/serving
(≈0.68×) and small-V (≈0.65×) regimes via its measured tile + tuned warp
specialization; the small-regime residual is occupancy/register-pressure, not memory.
DEVELOPMENT.md "Post-M13 …".)

### 6.6 Backward residual (M13)

The optimized backward (promoted at M13, then `split`) recovered the residual the
M11 design left (the embed-kernel attribution was corrected — its binder is
hidden-row gather re-reads, already at the traffic floor, not g/idx streams).
Validated traffic model: per-buffer
residuals ≤1% on the decision counters across four shapes; the uniform pass
attains its floor (production profile 1.43 ms at 56 regs / 73.5% occupancy on
the doc record vs the 1.34 ms modeled conservative floor). **Terminal backward
residual:** the uniform-pass LTS ≈ 61–67% (config-dependent) vs the embed
kernel's 82–104%, plus the short-run regimes capped by the mixed fraction.
Atomics are no longer a bottleneck anywhere (L2 reduction sectors down 27.7×
dev / 51.7× corner / 264× real query record vs pre-M11).

**Post-M13 dot landing (2026-06-26).** The uniform pass's per-chunk reduce was
landed as a block-scaled compensated `tl.dot` (the matmul pipeliner multi-buffers
the embed gather via cp.async): it moves the uniform pass off the L1TEX/load-issue
bound onto the L2-BW bound (L1TEX 83→66, L2 61→77, SM 37→79; the landed SASS shows
`LDGSTS` cp.async on the gather). The mixed launch was retuned (BLOCK_D 128→64,
SUB 64→32, num_stages 2→3, exact same cumsum). Net **~1.11–1.13× end-to-end on real
records** (query 2.27→2.04 ms, doc 2.79→2.46 ms), range-safe under fp16+bf16 AMP,
gate-clean (135 tests + mono A/B). An exhaustive joint sweep (warp-specialization ×
num_stages × num_warps × BLOCK_D × CTA-cap) found no further gain on this pass:
auto-WS structurally requires TMA-descriptor loads, which the data-dependent
scattered embed gather cannot be (the same iron-law scatter that blocks load-once;
TMA-gather is Gluon-only), and the pass being L2-BW-bound wants *more* concurrent
CTAs, not larger ones. **New terminal backward residual:** uniform pass L2-BW-bound
(~77%) + the embed kernel at its output-write-DRAM-bound floor (DEVELOPMENT.md
"Post-M13 dot backward"; bound refined from "L2-bound" + guard removed 2026-06-26, see below).

**Post-M13 dtype-aware compensation (2026-06-26).** The single block-scaled reduce above
is now dtype-routed (`COMP_MODE`, host-side from `embed.dtype`): **bf16 uses no-scale
compensated** (its exponent already spans g's fp32 range, so block-scaling only adds a
max-reduce + `1/s` + `×s`, and can *increase* error on representable g) — measured ≤
block-scale error in every regime and **~2.5–4.5% faster bf16 backward** on captured-real;
**fp16/fp32 keep block-scale** (the price of fp16 range safety; the bf16-hi/fp16-lo
alternative was refuted — `tl.dot` operand matching degrades the embed ~25000×). The
mixed pass stays the exact cumsum (the dot-mixed loses on every dtype/regime). Compensation
is a user knob (`SPARTON_BWD_COMPENSATE=0` → non-compensated, faster, unsafe). Gate-clean
(135 tests, mono A/B, training-smoke AMP parity, compute-sanitizer). Evidence:
DEVELOPMENT.md "dtype-aware compensation".

**Post-M13 embed_grad guard removal + bound correction (2026-06-26).** A fresh perturbation
profile (do_bench NO_GATHER / COMPUTE_Nx, ncu corroboration) re-binds embed_grad and **corrects
the "hidden-reuse / L2-bound" note above**: it is **output-write-DRAM-bound** — the V×D fp32 store
is ≈100% of DRAM traffic (ncu 762/768 MB), the hidden gather is L2-served and only 21–39% of time,
and compute is pure slack. (So the `tl.dot`-reduce question is moot — no compute lever exists, and
the reduce is a batched GEMV `bv,bvd→vd`, not a GEMM.) The never-firing `tl.sum(tl.abs(g))` tile-skip
guard was **removed**: it skipped only all-zero `[BLOCK_B,BLOCK_V]` tiles (~never at B≥16: 0% on real,
0.7% at f=0.01) while costing a per-d-tile reduction (+7–33% on the pass). End-to-end **~3–4% faster on
query**, within-noise on doc (query-weighted — embed_grad is a larger fraction of the short query
backward). The per-lane `gather_mask=(g!=0)` is the only suppression. Gate-clean (135 tests, mono A/B,
sanitizer, training smoke). Evidence: DEVELOPMENT.md "embed_grad guard removal".

### 6.7 Operative deferral: host launch overhead

The launcher-v2 *core* — building **one** descriptor pair per call instead of a
22-slot bank, with no `POLICY_ID` if-chain — **shipped** with the pure-Triton
`optimized` kernel (post-M13). So the §6.3 descriptor-bank rebuild cost (~0.051 ms)
is gone from the default path, and the tile comes from a measured self-tuner cache.

What remains deferred (by maintainer decision) is any further trimming of the
*shared* wrapper/op/autograd host machinery (~0.013 ms/call), for unchanged reasons:

1. **No identified latency scenario at this seam.** In every documented workload
   (Trainer training, batched encoding) the head runs behind a backbone forward at
   ≥1 ms GPU times where host launch work overlaps it; the per-call host cost binds
   only for a small-shape latency-critical caller of the head itself, which nothing
   in the repo exercises.
2. **The residual is small and unprototyped on the new kernel** — its steady-state
   host overhead has not been re-measured, and trimming the shared machinery below
   ~0.013 ms/call has no built prototype.

**Revival triggers:** a real latency/small-shape user appears, or a future
forward-kernel rewrite forces the kernel signature open anyway. **On revival:**
re-measure the actual host floor of the pure-Triton kernel before committing to a
target; keep full suite + shape soak parity and grid/dev rows within ±5%.

### 6.8 Out of scope (hardware/scope)

Excluded from production scope, unchanged across the arc (revisit only on a
hardware or Triton-capability change, re-validating availability first):
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
  split,segmented` over {uniform, zipf} **and** the captured-real bundles in
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
  the template shape is `sparton {kernel} forward: ...`.
- **Training behavior:** prefer the synthetic smoke probe; full Hub-backed
  `training/train.py` runs are expensive and not casual validation (a
  steady-state 150-step tier-2 run is ~25 s on this host after the cold first
  run). `training/train.py` is smoke-validated against transformers 5.11 only
  (150-step runs); checkpointing/resume/distributed paths are unvalidated.

There is no CI config (no CUDA runner is available to this repo); the documented
command suite is the gate mechanism.
