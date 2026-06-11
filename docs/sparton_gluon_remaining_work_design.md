# Sparton Gluon Remaining-Work Design

Date: 2026-06-11.
Status: **active guide for subsequent backend-refactor development.**

This document supersedes
[sparton_gluon_current_platform_design.md](sparton_gluon_current_platform_design.md)
as the forward plan. The original design remains useful for its mathematical
contract, benchmark shape matrix, and optimization vocabulary.
[sparton_gluon_design_review.md](sparton_gluon_design_review.md) remains the
audit layer over that design; where this document and the review agree, the
review's corrections are adopted as decisions. Everything below is grounded in
source inspection and runtime probes executed on 2026-06-11 in this workspace;
probe provenance and rerun commands are in Appendices A and B.

---

## 1. Confirmed current state

### 1.1 Repository state

- Branch `codex/milestone2-bias-none-docs`, latest commit
  `48b55fa Fix no-bias backward and add milestone docs`.
- Milestones 1 and 2 of the review's revised order are complete:
  - PyTorch semantic reference and pytest 9 coverage
    (`tests/test_sparton_kernel.py`, 11 tests) matching the kernel's
    zero-baseline, strict-`>` index semantics.
  - `bias=None` backward fixed in the hybrid path; custom-op schemas declare
    optional bias.
- The only implemented backend is the hybrid path:
  `SpartonHead.forward -> fused_sparton_fwd_op ->
  fused_sparton_fwd_with_indices` (compiled `matmul`/`matmul_bias` per vocab
  tile) `-> reduce_seq_max_log1p_relu_with_indices` (Triton reduction), with
  `fused_sparton_bwd_op` -> `fused_sparton_bwd_kernel_with_bias`
  (`HAS_BIAS: tl.constexpr`).
- `SpartonHead` has **no** `backend` argument. There is no backend routing, no
  `_backend_hybrid.py`, no naive Triton fused forward, and no Gluon code in the
  repository.

### 1.2 Validation ledger (exact results, this session)

```bash
PYTHONPATH=src /workspace/venvs/sparton/bin/python -m py_compile \
  src/sparton/__init__.py src/sparton/sparton_kernel.py \
  training/model.py training/train.py tests/conftest.py tests/test_sparton_kernel.py
# -> passed

TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas /workspace/venvs/sparton/bin/python -m pytest -q
# run 1 (concurrent with another Triton-compiling process):
#   FAILED tests/test_sparton_kernel.py::test_fused_forward_matches_reference[bias-bf16]
#   1 failed, 10 passed, 1 warning   (InductorError: gcc failed building cuda_utils.c)
# focused rerun of bias-bf16 alone: 1 passed
# run 2 (serial):                   11 passed, 1 warning in 3.14s
# run 3 (hardened env, see §2.4):   11 passed, 1 warning in 20.47s (cold Inductor cache)

# custom-op schemas:
sparton::fused_sparton_fwd(Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)
sparton::fused_sparton_bwd(Tensor grad_out, Tensor max_scores, Tensor max_idx, Tensor hidden,
                           Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor, Tensor?)
```

The `bias-bf16` failure is **not** a kernel defect and is **not** random; see
§2.4 for the root cause (two deterministic environment defects masked by warm
caches) and the verified fix.

---

## 2. Platform and runtime facts

### 2.1 Runtime snapshot (re-verified)

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

Matches the planning snapshot; the L2 size (96 MiB) is newly recorded and is
large enough to hold the entire `hidden` activation for typical dev shapes
(`4096 x 768` fp16 = 6 MiB), which matters for GEMM scheduling policy.

### 2.2 MMA availability matrix on sm_120 (proven by compile+run probes)

| Gluon MMA family | Local API | Result on RTX 5090 |
|---|---|---|
| `ampere.mma_v2` (mma.sync) | `triton.experimental.gluon.language.nvidia.ampere.mma_v2` | **Works.** Tiny GEMM tile max-abs-err 6e-6 vs fp32 matmul |
| `hopper.warpgroup_mma` (WGMMA) | `...nvidia.hopper.warpgroup_mma` | **Fatal abort**: `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.wgmma.commit_group.sync.aligned` |
| `blackwell.tcgen05_mma` + tensor memory | `...nvidia.blackwell.tcgen05_mma` | **Fatal abort**: `LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.ld` (review saw `...wait.st`) |

Decisive consequences:

1. **`mma_v2` is the only tensor-core path for the optimized backend on this
   GPU.** It is the same `mma.sync` instruction family that `tl.dot` lowers to
   on sm_120, so Gluon's edge over plain Triton here is *manual control of
   staging, layouts, synchronization, and scheduling* — not access to a bigger
   MMA instruction.
2. The Gluon front end accepts all three families; failures occur at LLVM
   instruction selection as **fatal process aborts, not catchable Python
   exceptions**. Any capability dispatch must therefore happen *before*
   compilation from a static whitelist. "Try one, fall back on error" is not
   implementable.
3. Blackwell tensor-memory `load_max` (an N-dimension max fused into the TMEM
   load — a perfect fit for this kernel's epilogue) exists in the local API but
   is datacenter-Blackwell-only; it is recorded as future-hardware material and
   excluded from this plan.

### 2.3 Local Gluon API surface (verified names, Triton 3.6.0)

All under `triton.experimental.gluon`:

- Host-side descriptor: `gluon.nvidia.hopper.TensorDescriptor`
  (re-exported by `gluon.nvidia.blackwell`). Constraints enforced in
  `__post_init__`: base 16-byte aligned; all non-last strides 16-byte aligned;
  last dim contiguous (`strides[-1] == 1`); layout must be `NVMMASharedLayout`;
  element bitwidth in {8, 16, 32}; `block_shape[-1] >= swizzle_byte_width /
  elem_bytes` (for fp16 + 128-byte swizzle this means **BLOCK_K >= 64**,
  confirmed by a probe failure at BLOCK_K=32).
- Device language `gluon.language` (`gl`): `allocate_shared_memory`,
  `shared_memory_descriptor.{load,store,slice,index,permute,reshape}`,
  `warp_specialize`, `arange/load/store/reduce`, `gl.max/sum/min`, atomics,
  layouts (`BlockedLayout`, `SliceLayout`, `DotOperandLayout`,
  `NVMMADistributedLayout`, `NVMMASharedLayout.get_default_for`,
  `SwizzledSharedLayout`, `PaddedSharedLayout`), `gl.barrier()`.
- TMA: `gl.nvidia.hopper.tma.async_copy_global_to_shared(desc, coord, barrier,
  smem)`, `async_copy_shared_to_global`, `store_wait`. (Blackwell adds
  `async_gather/async_scatter`; excluded from scope.)
- mbarrier: `gl.nvidia.ampere.mbarrier.{allocate_mbarrier, init, wait, arrive,
  invalidate, MBarrierLayout}` plus `gl.nvidia.hopper.mbarrier.expect`.
  Barriers are rank-1 `[1]` shared memdescs (a batch allocation is
  `[batch, 1]`, indexed with `.index(i)`); passing a rank-0 view is rejected.
- `hopper.fence_async_shared(cluster=False)`.
- Reductions are available inside Gluon kernels (`gl.max` is re-exported from
  `triton.language.standard`), which the fused epilogue depends on.

The tutorial-era names the original design used (`tma.async_load`, a single
implementation-neutral MMA) do not exist locally. The compatibility shim
(§4.2) is mandatory, exactly as the review concluded.

### 2.4 Environment defects, root-caused and fixed (operational)

The "transient first-run compiler/cache failure" called out in the planning
brief is **deterministic** and fully explained. Two independent defects:

1. **The NVIDIA-built Triton wheel ships no bundled CUDA headers.**
   `triton/backends/nvidia/include/` does not exist in this install, but
   `triton/backends/nvidia/driver.py` hardcodes it as the only include dir for
   building its `cuda_utils` C extension. Any *cold* build fails with
   `fatal error: cuda.h: No such file or directory`.
2. **`/tmp` is mounted `noexec`** (`tmpfs rw,nosuid,nodev,noexec`). The default
   TorchInductor cache root is `/tmp/torchinductor_<user>` (here
   `/tmp/torchinductor_root`); when the Triton cache is redirected there, a
   successfully built `cuda_utils.so` fails to `dlopen` with
   `failed to map segment from shared object`.

The redirect mechanism (verified in the installed torch source): before
compiling any Inductor Triton kernel,
`torch/_inductor/runtime/triton_heuristics.py:394-398` sets
`TRITON_CACHE_DIR = {inductor cache}/triton/<device>` if the variable is
unset, where the inductor cache root comes from `TORCHINDUCTOR_CACHE_DIR` or
the `/tmp` default (`torch/_inductor/runtime/cache_dir_utils.py`);
`torch/_inductor/async_compile.py:405-414` forwards both variables to its
compile-worker subprocesses. Triton's `cuda_utils` C extension is built once
per process, in whatever `TRITON_CACHE_DIR` is active at the **first** Triton
driver initialization. That defines the failure window precisely:

- In a process that touches a plain Triton kernel first (e.g. Sparton's
  autotuned reduction during import/warm-up), the driver initializes under the
  default `~/.triton/cache` (warm, exec filesystem) — later Inductor compiles
  reuse the process-wide driver and nothing fails. This is why most runs pass.
- In a process whose first Triton driver init happens *after* Inductor set the
  variable — async-compile workers on a cache-miss compile, or any
  fresh-cache context — `cuda_utils` rebuilds under the redirected dir: the
  build fails without `CPATH` (defect 1), and even a successful build cannot
  be `dlopen`ed from noexec `/tmp` (defect 2).

This is also exactly why the fix works: `TORCHINDUCTOR_CACHE_DIR` relocates
the derived `triton/<device>` cache onto an executable filesystem, and `CPATH`
lets cold `cuda_utils` builds find `cuda.h`. Reproduction was deterministic
3/3 with `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` (which forces the cold path);
the fix then passed 3/3 and the full pytest suite passes under it.

**Hardened environment prefix for all validation and benchmarking:**

```bash
TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas \
CPATH=/usr/local/cuda-13.2/include \
TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
PYTHONPATH=src /workspace/venvs/sparton/bin/python
```

Additional operational rule: do not run two Triton/Inductor-compiling
processes concurrently when comparing results; compile-time contention makes
failures and timings harder to attribute (first observed failure in this
session was under exactly that condition).

### 2.5 Profiler availability (corrected twice)

- `nsys` 2026.2.1 (`/usr/local/bin/nsys`): **works**. Demonstrated on the GEMM
  microbenchmark; per-kernel summary reported `gemm_abt_kernel` at ~1.008 ms
  mean over 6 back-to-back launches (warm L2; `do_bench` reports 1.125 ms with
  L2 flushed between launches — both numbers are valid in context, quote
  `do_bench` as primary).
- `ncu` 2026.1.1 (`/usr/local/bin/ncu`): initially blocked
  (`ERR_NVGPUCTRPERM`, a host-driver counter restriction). The restriction was
  lifted on the host later the same day; **hardware-counter profiling now
  works locally** and produced the evidence in §3.5. Usage notes learned while
  collecting it:
  - On this GPU the DRAM byte counters are `dram__bytes_op_read.sum` /
    `dram__bytes_op_write.sum` (`dram__bytes_read.sum` reports `n/a`).
  - ncu serializes launches and flushes caches between kernels by default, so
    per-kernel durations are inflated and inter-kernel L2 reuse is hidden; use
    ncu numbers for structure and ratios, `do_bench` for latency.
  - NVTX ranges are thread-local and autograd backward runs on a worker
    thread: `--nvtx --nvtx-include "range/"` around `tensor.backward()`
    captures nothing. Profile backward by invoking
    `sparton::fused_sparton_bwd` (or the backend op) directly inside the
    range on the main thread.

### 2.6 Packaging floors

- `torch>=2.7.1` (declared): `Tensor?` custom-op support assessed in §3.4.
- `triton>=3.3.1` (declared): **`triton.experimental.gluon` does not exist in
  3.3.1**; it first appears in Triton 3.4.0 and the validated version is the
  installed 3.6.0. The declared floor stays valid for the hybrid backend only.
  Gluon code must be imported lazily, only when the optimized/naive-gluon
  backend is requested, and must fail with a clear error naming the required
  Triton version. Given experimental-API churn (§2.3), record "validated with
  Triton 3.6.0" in the shim and re-validate on any Triton bump rather than
  declaring a loose floor.

---

## 3. Feasibility evidence

### 3.1 Standalone Gluon GEMM microbenchmark (new, this session)

A standalone kernel was built against the local API and validated on the
production Sparton layouts — this was the review's gate for any fused
optimized work, and it now **passes**:

- `A = hidden.reshape(B*S, D)` consumed via a host `TensorDescriptor` over the
  row-major `[M, K]` view (no copy).
- `B = embed [V, D]` row-major consumed as conceptual `embed.T` with **no
  physical transpose**: TMA loads `[BLOCK_N, BLOCK_K]` tiles into
  `NVMMASharedLayout` shared memory; the tile is read back through
  `.permute([1, 0]).load(DotOperandLayout(1, mma, 2))` (ldmatrix-transpose
  path). Numerics match cuBLAS at the benchmark's printed precision: both
  report `0.0001` max-abs error vs the fp32 reference (`%.4f` formatting; the
  values are fp16-accumulation-level, not bit-identical).
- Synchronization protocol that works: per-stage mbarrier
  (`init(count=1)` -> producer `expect(bar, NBYTES)` -> two
  `tma.async_copy_global_to_shared` into the stage -> consumer
  `wait(bar, phase)` with `phase = (tile_index // NUM_STAGES) & 1`), circular
  slots `tile % NUM_STAGES`, and a CTA `gl.barrier()` between the shared-memory
  reads and the slot refill (WAR hazard). TMA + mbarrier + `mma_v2` compose in
  one kernel on sm_120.
- Rank-3 shared allocations (`[NUM_STAGES, BLOCK_M, BLOCK_K]`) over a rank-2
  `NVMMASharedLayout` work and are indexable per stage.

Results, dev shape `M=4096 (B=32, S=128), K=768, N=30522`, fp16 in / fp32 acc,
`triton.testing.do_bench` (L2 flushed), RTX 5090:

| BM x BN x BK / stages / warps | Gluon | cuBLAS (`torch.matmul`) | ratio |
|---|---|---|---|
| **128x128x64 / 3 / 4x2** | **1.125 ms — 170.6 TFLOP/s** | 0.850 ms — 225.8 TFLOP/s | **76%** |
| 128x128x64 / 2 / 4x2 | 1.190 ms — 161.4 | — | 71% |
| 128x64x64 / 4 / 4x2 | 1.192 ms — 161.1 | — | 72% |
| 64x128x64 / 3 / 2x4 | 1.211 ms — 158.5 | — | 70% |
| 128x256x64 / 2 / 4x2 | 1.488 ms — 129.0 | — | 57% |
| 128x128x32 / 4 / 4x2 | rejected by descriptor layout: 128-byte swizzle requires `block_shape[-1] >= 64` (fp16) | — | — |

Known, untried headroom: the per-K-iteration CTA barrier (replaceable with
producer/consumer "empty-slot" barriers or warp specialization), L2-aware tile
rasterization / persistent scheduling, instruction-level tuning. A first-cut
kernel at 76% of cuBLAS with this much headroom makes the optimized forward
*plausible*; it does not make it a foregone winner (§3.3).

A real deadlock was hit and fixed while building this (prefetch must target
slot `next_tile % NUM_STAGES`, not the just-consumed slot). Operational rule
adopted from it: **every kernel-bringup probe runs in a subprocess with a
timeout**, because an mbarrier deadlock presents as a permanent 100%-GPU spin
inside `torch.cuda.synchronize()`.

### 3.2 Hybrid baseline measurements (new, this session)

Dev shape `B=32, S=128, D=768, V=30522`, fp16, mask density 75%, after
autotune warm-up; `do_bench`:

```text
hybrid forward (bias):     1.170 ms
hybrid forward (no-bias):  1.050 ms
full-V cuBLAS GEMM only:   0.856 ms  (224.2 TFLOP/s)   <- floor for any fused forward
hybrid overhead over GEMM: 37% (bias case)
peak extra device memory during forward: 140.5 MiB (outputs alone: 9.3 MiB)
hybrid forward+backward (bias): 2.598 ms  (=> backward ~1.43 ms)
v_tile_from_bs(32, 128, 30522) = 8192  (4 vocab tiles)
```

### 3.3 Break-even model for the optimized forward

Per logit element the hybrid path pays ~4 bytes of extra HBM round-trip (tile
write + reduction read) for `2*D` FLOPs of GEMM work; at D=768 and this GPU's
measured compute/bandwidth balance that is roughly a quarter of GEMM time, and
the measured end-to-end overhead including the reduction kernel and tile loop
is 37%. Therefore:

- To beat the hybrid bias-path (1.170 ms), a fused kernel must run at
  `0.856 / 1.170 ≈ 73%` of cuBLAS *including* its fused epilogue.
- The first-cut Gluon GEMM is at 76% **without** the epilogue. The epilogue
  (mask multiply, running max/argmax over the sequence chunk, final
  `log1p(relu)`) adds register pressure and reduction work but zero extra
  global traffic; a 5–15% hit would put the first fused version at ~65–72%,
  i.e. **slightly slower than hybrid** until the known headroom is exploited.
- Conclusion (decision): the Gluon track stays *experimental* exactly as the
  review demanded; hybrid remains default; promotion is gated on measurements
  (§6, §12). The GEMM-throughput gate before fusion work is set at **>= 85% of
  cuBLAS** on dev shapes so the fused version has realistic slack to clear
  hybrid.

Hardware counters (§3.5) later confirmed and sharpened this model: the hybrid
forward's DRAM reads total a measured 583 MB vs the Gluon kernel's 55 MB (the
tile logits are read twice because Inductor leaves the bias add unfused), its
tiled GEMM is ~11% slower than one full-V cuBLAS call (same-regime ncu
comparison), and the Gluon kernel is compute-bound (11.7% DRAM), so fusing the
epilogue spends idle memory headroom rather than scarce tensor-pipe cycles.
The break-even bar (~73% of cuBLAS including epilogue) stands; the measured
route to it is raising tensor-pipe utilization from 63.9% toward cuBLAS's
86.6% (§3.5).

### 3.4 `torch>=2.7.1` and `Tensor?` custom-op schemas

Verified against the v2.7.1 release tree (not just the installed 2.12):

- `torch/_library/infer_schema.py` derives `(Optional[Tensor], "Tensor?")` for
  parameters, and 2.7.1's own test suite exercises `Optional[Tensor]`
  custom-op parameters and explicit `schema=` strings
  (`test_mutated`, `test_manual_schema` in `test/test_custom_ops.py`).
- This repo passes **explicit schema strings**, bypassing inference — the path
  that matters is the TorchScript schema parser plus the custom-op runtime.
  2.7.1's `backend_impl` return handling only iterates tensors for aliasing
  checks and tolerates `None` entries.
- **Residual risk (small):** an end-to-end 2.7.1 test of an op *returning*
  `None` for a declared `Tensor?` slot was not found verbatim, and
  `infer_schema`'s `SUPPORTED_RETURN_TYPES` does not list `Optional[Tensor]`
  even in 2.12 (irrelevant while explicit schemas are used — do not migrate
  these ops to inferred schemas).
- Decision: keep `torch>=2.7.1` declared. Before any release that advertises
  no-bias training support on the floor version, run the existing pytest suite
  once against real torch 2.7.1. Documented fallback if that ever fails:
  split ops (`sparton::fused_sparton_bwd_bias` / `..._nobias`) so no optional
  return is needed; the Python wrapper hides the split. No code change now.

### 3.5 Hardware-counter evidence (ncu, collected after counters were enabled)

All numbers from `ncu` on the dev shape (`M=4096, K=768, N=30522`, fp16),
cache-flushed and serialized (§2.5 caveat); kernel identities and byte counts
are exact, durations are ncu-inflated.

**Gluon `gemm_abt_kernel` (best config 128x128x64 / 3 stages / 8 warps):**

```text
Tensor pipe utilization: 63.9%  (highest pipe; SOL Compute 63.8%, Memory 65.6%)
DRAM throughput:         11.7%  -> not memory-bound
DRAM bytes:              read 55.0 MB (analytic floor A+B = 53.3 MB), write 215.7 MB (~C)
Cache hit rates:         L1/TEX 87.8%, L2 97.3%
Registers:               188/thread; occupancy 16.67% theoretical / 16.29% achieved
Occupancy limiters:      registers -> 1 block/SM, shared memory (96 KiB) -> 1 block/SM
Top stall:               'wait for execution pipe' 10.9 of 28.8 cycles/issue (37.8%)
```

**cuBLAS kernel for the same GEMM** (`torch.matmul` dispatches
`cutlass_80_tensorop_f16_s16816gemm_f16_128x256_32x3_tn_align2`):

```text
Instruction family:      m16n8k16 mma.sync (s16816) -> SAME family as mma_v2
Tensor pipe utilization: 86.6%
DRAM bytes:              read 53.9 MB
Registers:               218/thread; occupancy 16.67% theoretical / 16.65% achieved (1 CTA/SM)
CTA tile:                128x256 (M x N, per the kernel name), BK=32, 3 stages; grid 3840
```

Interpretation (load-bearing for §7 and §9): cuBLAS achieves its 225 TFLOP/s
on this GPU **at the same occupancy, same 1 CTA/SM, similar register count,
and the same MMA instruction family** as the Gluon kernel. The 76% -> 100%
gap is therefore *intra-CTA pipelining quality* (issue scheduling, no CTA-wide
barrier, deeper effective K pipeline), not occupancy and not instruction-set
access. Hitting the M7 85%-of-cuBLAS gate corresponds to raising tensor-pipe
utilization from 63.9% to roughly 74%.

**Hybrid forward, measured kernel inventory.** `v_tile_from_bs` yields 4
vocab tiles: three of 8192 columns and a 5946-column tail. Per tile:

```text
Tiles 1-3 (V_tile = 8192), tile 1 shown (tiles 2-3 within ~1 us / ~0.4 MB):
  cuBLAS GEMM (s16816gemm_relu_f16_256x128_32x3_tn_align8)  288.6 us  read 19.1 MB
  triton_poi_fused__unsafe_view_add_0 (bias add)             44.3 us  read 67.2 MB
  reduce_seq_max_log1p_relu_kernel_with_indices              45.9 us  read 67.1 MB
  2x slice-copy elementwise kernels                          ~7.7 us  read ~2.7 MB

Tail tile 4 (V_tile = 5946):
  cuBLAS GEMM (s16816gemm_f16_128x256_32x3_tn_align2)       219.3 us  read 15.5 MB
  bias add                                                   37.6 us  read 48.7 MB
  reduction                                                  32.9 us  read 48.7 MB
  2x slice-copy kernels                                      ~7.2 us  read ~1.9 MB
```

(The tail tile dispatches a different cuBLAS kernel — the same 128x256 align2
variant the full-V matmul uses — than the 256x128 relu/align8 variant chosen
for the 8192-column tiles.)

Three structural findings, all previously unknown:

1. **Inductor does not fuse the bias into the GEMM epilogue.** `matmul_bias`
   compiles to GEMM + a separate pointwise add that re-reads and re-writes the
   full 64 MiB logits tile, consistent with the measured bias/no-bias forward
   gap (1.170 ms vs 1.050 ms warm; 171 us of bias-add kernels summed
   serialized), and it doubles the logits read traffic.
2. **The tile logits are read twice from DRAM** (bias add + reduction): total
   hybrid forward DRAM reads, summed over all 16 profiled kernels, are
   **583 MB vs the Gluon GEMM's 55 MB** — a >10x read-traffic gap before the
   fused kernel even removes the C write (250 MB -> ~10 MB of `[B, V]`
   outputs).
3. **Tiled GEMM is ~11% slower than one full-V GEMM** in the like-for-like
   ncu-serialized comparison: 288.6 + 289.6 + 287.9 + 219.3 = 1085.4 us tiled
   vs 975.6 us full-V (~178 vs ~197 effective TFLOP/s), from tail-tile
   inefficiency plus wave quantization (1024-CTA tile launches ≈ 6.0 waves on
   170 SMs, 768-CTA tail ≈ 4.5). An earlier draft claimed ~35% by comparing
   serialized ncu tile times against the `do_bench` full-V number — a
   measurement-regime mix-up; the correct penalty is the ~11% above.

**Hybrid backward (`fused_sparton_bwd_kernel_with_bias`, grid (1908,1)x128):**

```text
Kernel:        1.38 ms; SOL Compute 6.0%, DRAM 9.6%
DRAM bytes:    read 167.0 MB, write 65.9 MB
Plus 3 zero-init fill kernels for fp32 grad buffers (~34 us, 26 MB written)
```

The backward kernel is neither compute- nor bandwidth-bound: it is
latency/atomic-bound at ~6% utilization. This quantifies the headroom for the
B2/B3 backward track (§8) — and equally for cheaper Triton-level fixes — but
does not change the ordering: correctness-first, promotion by measurement.

### 3.6 Remaining unresolved checks (entry gates for later milestones)

1. **Fused epilogue probe** (gate for M8): max/argmax reduction over the M axis
   of an `mma_v2`-layout accumulator, mask multiply from a `[BLOCK_M]` vector,
   strict-`>` running update against `[BLOCK_N]` register state, index
   tracking, `log1p`. Expected to work via `gl.reduce`/`gl.max`; not yet
   compiled.
2. **`gl.warp_specialize` probe on sm_120** (gate for M9): the API exists
   locally; producer/consumer register budgets and barrier interplay untested.
3. **BF16 `mma_v2` GEMM probe** (gate for M7 exit): the microbenchmark ran
   fp16 only; hybrid BF16 is covered by pytest. `k_width=2` applies to bf16 as
   well; expected to pass, must be confirmed.
4. **Persistent scheduling** (M9): plain `tl.program_id`-style persistence in
   Gluon, plus L2-aware rasterization order; untested locally.
5. **torch 2.7.1 end-to-end suite run** (release gate, §3.4).
6. ~~ncu metrics blocked locally~~ — resolved; counters were enabled on the
   host and the evidence is in §3.5. Counter-based gates (§6, §11) run
   locally.

---

## 4. Backend architecture and dispatch plan

### 4.1 Target file layout

```text
src/sparton/
  sparton_kernel.py            # import-compatible entry: custom ops, SpartonHead, router
  _backend_hybrid.py           # current implementation, moved mechanically (M3)
  _backend_naive_triton.py     # Triton-only fused forward baseline (M5)
  _backend_optimized_gluon.py  # Gluon fused forward/backward (M8+)
  _gluon_runtime.py            # compatibility shim: ALL gluon imports + capability whitelist (M6)
  _runtime_policy.py           # device/problem profiles, policy generation (M7+)
benchmarks/                    # validated probe/bench scripts (promoted 2026-06-11; extended at M6/M7)
```

`tests/test_sparton_kernel.py` keeps the reference; backend-parity tests are
added per milestone. Nothing moves until M3's no-behavior-change gate passes.

### 4.2 Gluon compatibility shim (`_gluon_runtime.py`)

- The **only** module allowed to import `triton.experimental.gluon.*`.
  Kernels import project-local names
  (`from ._gluon_runtime import gl, gluon, tma, mbarrier, TensorDescriptor,
  fence_async_shared, mma`).
- Lazy import: nothing Gluon-related is imported unless a Gluon backend is
  requested, keeping `triton>=3.3.1` valid for hybrid users. A missing/old
  Gluon namespace raises `RuntimeError` naming the required Triton.
- **Static capability whitelist, no speculative compilation.** Because a wrong
  MMA family is a fatal LLVM abort (§2.2), the shim maps
  `torch.cuda.get_device_capability()` to the MMA implementation:
  `(12, x) -> mma_v2`; `(9, 0) -> mma_v2` (WGMMA only after it is separately
  validated on real sm_90a); `(10, x) -> mma_v2` (TCGen05 likewise);
  anything `< (8, 0)` -> Gluon backends unavailable. This is private,
  architecture-aware dispatch — permitted by the review's correction of the
  no-feature-gate rule — while public names stay architecture-neutral. The
  production `optimized` kernel for this plan is written against `mma_v2`
  only; WGMMA/TCGen05 variants are future work behind the same shim.
- The shim records `VALIDATED_TRITON = "3.6.0"` and warns (once) when running
  against a different Triton, since `experimental` namespaces move.

### 4.3 Routing

- Module-level default backend `"hybrid"`, overridable by env var
  `SPARTON_BACKEND` (read once at import) and later by a constructor argument
  (§5). Resolution happens **once, at `SpartonHead.__init__`**: the instance
  binds the backend-specific wrapper (and op handle) at construction, and
  `forward` just calls the bound wrapper. The env var is consulted at init,
  never per call, and never inside a hot kernel.
- Per-backend Python wrappers around per-backend custom ops. The existing op
  names `sparton::fused_sparton_fwd/bwd` stay bound to the hybrid
  implementation forever (compat with anything that captured those ops). New
  backends register their own ops (`sparton::naive_fwd`,
  `sparton::optimized_fwd/bwd`) with their own fake impls and autograd, so
  schemas may diverge without touching hybrid.
- **No silent fallback.** If a selected backend is unavailable (no CUDA arch
  support, Triton too old, kernel compile failure), raise with the reason.
  The only soft path is the documented env/constructor choice of `"hybrid"`.

### 4.4 Hybrid extraction rules (M3)

- Mechanical move of the current code into `_backend_hybrid.py`; `sparton_kernel.py`
  re-exports `SpartonHead`, the op handles, and the helper functions whose
  names tests/users may rely on (`fused_sparton_fwd_op`, `fused_sparton_bwd_op`,
  `fused_sparton_fwd_with_indices`, `reduce_seq_max_log1p_relu_with_indices`,
  `v_tile_from_bs`, ...).
- Keep autotune config lists, `key=`, `cache_results=True`, custom-op schema
  strings, fake registrations, and autograd setup byte-identical. The
  import-time device print moves with the module (tests and AGENTS.md account
  for it); do not duplicate it.
- Both reduction helpers (with/without indices) move as-is; their memory
  trade-off (no-indices variant exists to skip the `[B, V]` int64 buffer in
  inference paths) is preserved and documented in code.

---

## 5. Public API stance

- **Now (M3–M5):** `SpartonHead(vocab_size, hidden_dim, use_bias=False)` —
  unchanged. No backend argument. Experiments select backends via
  `SPARTON_BACKEND` only. CUDA-only export behavior of
  `src/sparton/__init__.py` unchanged.
- **At M4 (routing proven):** add keyword-only `backend: str = "hybrid"` to
  `SpartonHead.__init__`. Adding the argument with a hybrid default is
  non-breaking; `"naive"`/`"optimized"` raise until their milestones land.
  Public names stay implementation-descriptive (`hybrid`, `naive`,
  `optimized`) and architecture-neutral.
- **Default switch to `optimized`:** only via the promotion gates in §12, as a
  deliberate, changelog-documented release decision. Until then every public
  default remains the hybrid path.
- Mask contract made explicit in docs and asserted cheaply where possible:
  `attention_mask` is a 0/1 (or boolean) `[B, S]` tensor; non-binary values
  produce weighted logits (`logits * mask`), which is defined behavior but not
  the HF attention-mask contract. Indices are `int64 [B, V]`; index values are
  semantically meaningful only where `scores > 0` (zero-baseline contract).

---

## 6. Corrected milestone order

M1 (reference + tests) and M2 (`bias=None` backward) are **done** (commit
`48b55fa`). Remaining order, each with entry/exit gates:

| # | Milestone | Exit gates |
|---|---|---|
| M3 | Extract `_backend_hybrid.py`, no behavior change | pytest 11/11; schema strings identical; dev-shape fwd and fwd+bwd timings within noise (±5%) of §3.2; no new imports at package import time |
| M4 | Router + `SPARTON_BACKEND` + `backend="hybrid"` kwarg | routing unit tests; default proven hybrid; unavailable backend raises with reason |
| M5 | `_backend_naive_triton.py` fused forward (`tl.dot`, no Gluon); backward is NOT reimplemented — the naive backend registers its own autograd that calls the current Triton backward (B1) | matches reference on the full §10 correctness matrix incl. ties/tails/masks/no-bias/BF16; allocator check shows no `[B,S,V_tile]`; measured and recorded vs hybrid (expected slower; no perf gate) |
| M6 | `_gluon_runtime.py` shim + Gluon smoke test (probe/bench scripts already live in `benchmarks/`; extend as needed) | shim imports lazily; capability whitelist unit-tested; `mma_v2` smoke kernel passes on RTX 5090; smoke is skipped cleanly where Gluon/CUDA absent |
| M7 | In-repo Gluon GEMM microbenchmark + policy generator + tuning sweep | correctness vs cuBLAS on dev shapes fp16 **and bf16**; **>= 85% of cuBLAS** on at least `M=4096, K=768, N=30522` and one stress shape; policy generator only emits configs satisfying §7.2 constraints |
| M8 | Gluon fused forward O1 (non-persistent) | epilogue probe (§3.6 item 1) passed first; full correctness matrix vs reference incl. index policy; memory gate: peak extra < 2x outputs (vs hybrid's ~140 MiB); perf recorded, no gate |
| M9 | O2/O3: persistent + warp-specialized + barrier-free pipeline | beats naive fused on all dev shapes; **>= hybrid forward on at least one realistic shape**; nsys shows single kernel launch; ncu shows DRAM reads within 1.3x the analytic A+B floor and no logits-sized write stream (§11) |
| M10 | Gluon backward B2 (direct atomic), then B3 (local `d_bias`/`d_embed` aggregation) | gradient matrix vs reference and vs current backward (bias and no-bias); no NaN under AMP smoke; B3 only after profiling realistic index distributions; perf >= current Triton backward before any default consideration |
| M11 | Promotion decision | §12 gates, explicit changelog + README update |

Naive-forward (M5) intentionally precedes Gluon work: it isolates fused
*semantics* (zero baseline, strict `>`, masking, tails) from Gluon-specific
failure modes, on the same `mma.sync` hardware path the optimized kernel uses.

---

## 7. Optimized Gluon forward design (updated by evidence)

### 7.1 Kernel shape

One kernel, no logits materialization (allowed temporaries: stage buffers,
accumulator fragments, `[BLOCK_N]`-sized running state, outputs):

```text
grid over (batch b, vocab tile n0)          # persistent variant: flat tile id loop
running_max[BLOCK_N] = 0.0; running_idx[BLOCK_N] = 0
for s0 in range(0, S, BLOCK_M):
    acc[BLOCK_M, BLOCK_N] = TMA+mma_v2 mainloop over K          # §3.1 protocol
    vals = acc (+ bias[n0:n0+BLOCK_N]) * mask[b, s0+row]        # masked rows -> 0
    rows with s0+row >= S contribute 0                          # batch-boundary tail, §7.3
    tile_max, tile_arg = max/argmax over rows (strict >)
    update running state with strict >
scores = log1p(relu(running_max)); store scores, running_idx (int64)
```

`batch_block` is fixed at 1 for the first implementation (running state and
mask indexing stay 1-D; the original design's `batch_block>1` variant is a
later policy option).

### 7.2 Policy generator constraints (measured + API-derived)

- `BLOCK_K >= 64` for fp16/bf16 with 128-byte swizzle; `BLOCK_K = 32` requires
  a 64-byte-swizzle descriptor layout (descriptor assertion, §2.3). The
  64-byte-swizzle family is worth sweeping: cuBLAS's winning kernel on the dev
  shape is `128x256 (M x N), BK=32, 3 stages` (§3.5), whose per-stage footprint
  is half of ours — include that exact configuration in the M7 sweep.
- Stage memory `NUM_STAGES * (BLOCK_M*BLOCK_K + BLOCK_N*BLOCK_K) * 2 B` ≤
  ~96 KiB budget (101376 B opt-in, minus barriers/scratch). The measured best
  (3-stage 128x128x64 = 96 KiB) sits at this edge by design.
- `warps_per_cta = [WARPS_M, WARPS_N]`, product = `num_warps`; measured best
  `4x2` at 128x128. `BLOCK_N=256` measurably hurts (57%); cap candidate
  `BLOCK_N` at 128 unless re-measured.
- Persistent grid sizing from `sm_count=170` and per-CTA smem/threads, as in
  the original design §6.5 (`sparton_gluon_current_platform_design.md`) —
  unchanged, but only after the M9 probe.
- Candidate enumeration stays resource-driven (no architecture names in keys);
  cache keys: shape bucket, dtype, has_bias, requires_grad, sm_count,
  smem-opt-in bucket — unchanged from the original design §11.

### 7.3 Tails and edge semantics

- K tail: TMA zero-pads out-of-bounds K reads — zeros contribute nothing to
  the dot product. No masking needed in the mainloop.
- V tail: TMA zero-pads B-tile rows; epilogue stores are masked on `n < V`
  (probed, works).
- S tail / batch boundary: the A descriptor is over `[B*S, D]`, so a
  `[BLOCK_M, BLOCK_K]` tile starting at `b*S + s0` can cross into batch
  `b+1`'s rows when `S % BLOCK_M != 0`. **Those rows are real data, not
  zeros** — the epilogue must zero the contribution of rows with
  `s0 + row >= S` (cheap predicate, same role as `tile_mask` in the current
  reduction kernel). This is a correctness-critical rule for the
  implementation and a required unit test (e.g. `S=17`, `BLOCK_M=64`).
- All-negative / all-masked columns: running state stays `(0.0, 0)` —
  matching the reference and the strict-`>` contract exactly.
- Accumulate fp32; outputs in hidden's dtype; indices computed in int32 and
  stored int64 (public contract unchanged).

### 7.4 Scheduling variants

O1 non-persistent + CTA-barrier pipeline (known-working protocol) for
correctness; O2 persistent; O3 warp-specialized producer/consumer with
empty/full barrier pairs replacing the per-iteration CTA barrier (the main
measured inefficiency); pingpong-like O4 only if O3 still leaves tensor-pipe
utilization clearly below the 86.6% cuBLAS reference under ncu (§3.5).
Schedule names stay architecture-neutral (`cooperative`, `pingpong`,
`persistent`).

---

## 8. Backward plan (unchanged in substance, evidence-annotated)

- B1 (current Triton backward) remains the baseline and default, and it also
  serves the naive backend (M5) through that backend's autograd registration —
  no naive backward kernel exists or is planned. Measured
  context: backward ≈ 1.43 ms on the dev shape, ~55% of fwd+bwd time; the
  kernel runs at 6.0% compute / 9.6% DRAM utilization with 167 MB read /
  66 MB written (§3.5) — latency/atomic-bound, so the headroom is large and
  not bandwidth-limited.
- B2 Gluon direct-atomic backward is correctness-first; it reuses the saved
  `scores`/`indices` contract (`g = grad_out * exp(-scores)` where
  `scores > 0`), `d_bias` and `d_embed` per-CTA aggregation before atomics,
  `d_hidden` as irregular atomic scatter.
- B3 local aggregation and any duplicate-index `d_hidden` aggregation are
  benchmark-gated; realistic index distributions must come from a real SPLADE
  checkpoint or trained-for-a-few-steps model, not uniform random (uniform
  indices understate atomic conflicts on hot tokens).
- TMA gather/scatter stays excluded (datacenter-Blackwell API, out of the
  minimum feature set).
- No Gluon backward becomes default until it matches B1 correctness on the
  full matrix and is not slower on representative shapes (M10 gates).

---

## 9. Optimization candidates and rejection criteria

Accepted for evaluation (in priority order, each requires an A/B measurement
on dev shapes before being kept):

1. Intra-CTA pipelining quality: warp-specialized producer/consumer (or
   barrier-free double-buffer protocol) replacing the per-iteration CTA
   barrier, plus mainloop issue scheduling. Counter-justified (§3.5): cuBLAS
   reaches 86.6% tensor-pipe utilization at the *same* occupancy, CTAs/SM, and
   MMA instruction family — the entire 76%->100% gap lives inside the CTA, so
   this is the primary lever for the 85% M7 gate. Raising occupancy (2 CTAs/SM
   via smaller tiles/stages) is demoted to a sweep dimension, not a thesis:
   cuBLAS does not need it on this shape.
2. Persistent scheduling + L2-aware tile rasterization (96 MiB L2 fits all of
   `hidden`; measured 97.3% L2 hit rate already, so treat as a stress-shape
   lever rather than a dev-shape one).
3. Tile-shape/stage sweep within §7.2 constraints (policy generator),
   including the 64-byte-swizzle `BK=32` family and cuBLAS's `128x256x32/3`
   reference point.
4. Reduced-frequency epilogue: keep running max in registers across sequence
   chunks (already the design); fold bias into the epilogue, not the mainloop.
5. `maxnreg`/register budgeting between producer and consumer partitions
   (only with warp specialization).
6. For backward: local aggregation (B3) and index-distribution-aware variants.
   Counter-sized headroom: the current backward kernel runs at 6.0% compute /
   9.6% DRAM utilization (§3.5) — latency/atomic-bound.

Hybrid-side candidates (measurement-gated, **only after M3/M4**, because they
change hybrid behavior; each directly addresses a §3.5 measured inefficiency):

7. Fuse the bias into the GEMM (e.g. `torch.addmm` on the flattened view, or
   Inductor `mode="max-autotune"` epilogue fusion): removes the
   44 us + 67 MB-read pointwise pass per tile; measured upper bound ~0.12 ms
   of the 1.170 ms forward.
8. Raise the `v_tile_from_bs` temp budget (64 MiB cap) and/or pick tile counts
   that fill whole waves: the 4-tile GEMM is ~11% slower than one full-V call
   in the same-regime ncu comparison (§3.5), an upper bound of roughly 0.1 ms
   serialized — re-measure warm with `do_bench` before adopting, since the
   warm-overlap saving may be smaller. 31.8 GiB VRAM makes a 128-256 MiB temp
   cap plausible on this class of GPU. Must remain shape-aware (the cap exists
   to bound activation memory on smaller cards).
9. Eliminate the per-tile slice-copy kernels: `reduce_seq_max_log1p_relu_with_indices`
   allocates fresh `vals`/`idxs` per tile and the caller then copies them into
   `sparse_reps[:, i:i+C]` / `max_indices[:, i:i+C]` (the two measured
   `direct_copy` kernels). The reduction kernel already takes explicit output
   strides, so it can store straight into the `[B, V]` slice views.

Rejected / out of scope (confirmed against local hardware and API):

- WGMMA and TCGen05 paths on this GPU (fatal LLVM aborts; sm_90a/sm_100a
  features).
- FP8/FP6/FP4, block-scaled formats (excluded from first production scope).
- Cluster multicast, multi-CTA clusters, Cluster Launch Control (excluded;
  consumer Blackwell + complexity).
- Native TMA gather/scatter for backward (Blackwell-datacenter API).
- TMEM `load_max` epilogue (perfect fit, wrong hardware generation —
  revisit only on sm_100-class hardware).
- CUTLASS/C++ integration (Python-first constraint stands).
- Split-K (complicates online max; GEMM-only microbench may use it, the
  production fused kernel does not).
- Migrating custom ops to inferred schemas (no `Optional[Tensor]` return
  support in inference; explicit schema strings stay).

Rejection criteria for any candidate: fails correctness matrix; >5% regression
on any dev shape without >10% gain on a target shape; exceeds smem/register
budgets (compile- or launch-time failure); requires an API absent from the
validated Triton; or its claimed benefit cannot be shown in a counter (§11) or
A/B timing measurement (don't keep unmeasurable "optimizations").

---

## 10. Validation matrix

### 10.1 Correctness (every backend, vs the in-repo reference)

- Shapes: unit `B=1,S=4,D=16,V=32`; `B=2,S=5,D=16,V=19` (current tests);
  tail-stress `B=3,S=17,D=48,V=129`; dev `B=8..32,S=128,D=768,V=30522`;
  stress `B=8,S=512,D=1024,V=250000` (memory permitting).
- Cases: bias/no-bias × fp16/bf16 × {all-positive, all-negative, mixed, tie
  columns, zero-mask rows, partial masks, S-tail, V-tail}; non-contiguous
  inputs made contiguous at the wrapper.
- Tolerances (from the existing suite): scores fp16 `atol=rtol=2e-3`, bf16
  `5e-2`; gradients fp16 `2e-3`. Indices: exact equality where the winning
  logit is positive and non-tied; ambiguous otherwise (zero-baseline policy);
  `torch.equal` on the current seeds is the practiced form.
- Backward: vs PyTorch autograd reference and vs hybrid backward; random
  upstream grads; `scores == 0 -> zero gradient` checks; AMP smoke.

### 10.2 Memory

- Allocator gate per backend/shape: `reset_peak_memory_stats`; peak-minus-base
  compared against outputs + slack. Hybrid reference point: 140.5 MiB extra on
  the dev shape; fused forward gate: < 2x outputs (≈ 19 MiB on dev shape).
- Absence-of-materialization cross-check via nsys: exactly one forward kernel,
  no auxiliary kernel reading a logits-sized buffer, no `[B,S,V_tile]`-sized
  device allocations in the allocator trace.

### 10.3 Performance (recorded at every milestone, gating per §6)

- `do_bench` (L2-flushed) forward and forward+backward on dev shapes, bias and
  no-bias, vs the §3.2 hybrid baselines and the 0.856 ms GEMM floor.
- The mask-density sweep from the original design §14.4
  (`sparton_gluon_current_platform_design.md`) applies to backward
  benchmarks (atomic conflicts) and to fused-forward epilogue cost.

---

## 11. Profiling plan

- Latency of record: `triton.testing.do_bench` (L2-flushed) + CUDA-event
  timing in `benchmarks/`. ncu durations are serialized/cache-flushed and are
  not comparable to `do_bench` (§2.5).
- `nsys profile --stats=true ...` for kernel inventory, launch counts, stream
  overlap, and the no-extra-kernel/no-memcpy structure checks.
- `ncu` (working locally, §2.5) for the gated counter metrics. Standard
  command shapes, all with the §2.4 hardened env prefix, run serially:

```bash
# Single kernel by name (skip warm-up launches):
ncu --launch-skip 4 --launch-count 1 -k "regex:<kernel>" \
    --section SpeedOfLight --section ComputeWorkloadAnalysis \
    --section Occupancy --section LaunchStats --section WarpStateStats \
    --metrics dram__bytes_op_read.sum,dram__bytes_op_write.sum python <bench>.py

# A multi-kernel region (forward path): wrap the call in an NVTX range on the
# main thread and filter. For backward, call the bwd op directly — autograd's
# backward thread does not inherit main-thread NVTX ranges:
ncu --nvtx --nvtx-include "hybrid_fwd/" \
    --metrics dram__bytes_op_read.sum,dram__bytes_op_write.sum,gpu__time_duration.sum python <bench>.py
```

- Metric notes for this GPU: DRAM bytes are `dram__bytes_op_read.sum` /
  `dram__bytes_op_write.sum`; tensor-pipe utilization is read from
  `ComputeWorkloadAnalysis` ("Tensor (FP)" pipe); per-kernel write counts are
  understated for outputs still resident in L2 at kernel end — compare reads,
  or size writes analytically.
- Gate-relevant counter checks: tensor-pipe utilization vs the 86.6% cuBLAS
  reference (§3.5); DRAM reads vs the analytic A+B floor (M9 gate: <= 1.3x);
  absence of a logits-sized write stream; backward atomic behavior via the
  §10.3 mask-density sweep plus SOL utilization.
- Fallbacks (kept for counter-restricted environments): analytic byte model
  (§3.3), A/B `do_bench` deltas, `n_regs`/`shared` from the Triton launch
  metadata.

---

## 12. Assumptions, risks, and promotion gates

### 12.1 Assumptions (each was verified where marked)

- sm_120 keeps `mma_v2` + TMA + mbarrier working under Triton 3.6.0
  (verified by probes).
- `do_bench` numbers on this workstation are stable enough for ±5% gates
  (kernel-time stddev observed ~0.5%, §2.5).
- Gluon API may change on any Triton upgrade (assume churn; shim + recorded
  validated version).

### 12.2 Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Fused epilogue erases the 76%-and-improving GEMM margin | high | M7 85% gate before fusion; epilogue probe before O1; hybrid stays default |
| `gl.warp_specialize` immature on sm_120 | medium | probe before M9; CTA-barrier pipeline is the working fallback |
| Gluon API churn on Triton upgrades | medium | shim, recorded validated version, M6 smoke test in CI/local runs |
| Backward atomics dominate on real index distributions | medium | B1 remains default; distribution-aware benchmarks before B3 promotion |
| `Tensor?` returns on real torch 2.7.1 | low | one suite run on 2.7.1 pre-release; split-op fallback documented |
| Environment regressions (cache wipes resurface §2.4 defects) | low | hardened env prefix is part of every documented command |

### 12.3 Gates for changing the default backend (all required)

1. Full §10.1 correctness matrix green for optimized forward + chosen backward
   on this platform, fp16 and bf16, bias and no-bias.
2. Forward: faster than hybrid on >= 2 representative training shapes and not
   slower than hybrid on any §10.1 dev shape by more than 5%.
3. Forward+backward: net >= hybrid on the same representative shapes.
4. Memory gate (§10.2) passed.
5. Training-integration smoke (`training/model.py` heads aligned; a few
   hundred optimizer steps; finite grads; loss parity with hybrid within
   noise).
6. Documented in CHANGELOG + README with the measured evidence, and
   `backend="hybrid"` remains available unchanged.

---

## Appendix A. Probe provenance

The probe and benchmark scripts were promoted into the repository at
`benchmarks/` on 2026-06-11 (see `benchmarks/README.md`):
`probe_mma_matrix.py` (MMA availability), `bench_gluon_gemm.py` (TMA+mma_v2
GEMM, subprocess-per-config with timeouts), `bench_hybrid_baseline.py`,
`repro_inductor_env_defects.py` (environment-defect reproduction),
`ncu_runner.py` and `ncu_targets.py` (fixed-config launchers and NVTX ranges
for the §3.5 counter profiles). The load-bearing kernel, verbatim
as validated (best config: `BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
NUM_STAGES=3, warps 4x2`, launched with `num_warps=8`):

```python
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia import ampere
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma

@gluon.jit
def gemm_abt_kernel(a_desc, b_desc, c_ptr, M, N, K,
                    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
                    NUM_STAGES: gl.constexpr, WARPS_M: gl.constexpr, WARPS_N: gl.constexpr):
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[WARPS_M, WARPS_N],
                                                  instr_shape=[16, 8])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 2)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 2)

    a_smem = gl.allocate_shared_memory(gl.float16, [NUM_STAGES, BLOCK_M, BLOCK_K], a_desc.layout)
    b_smem = gl.allocate_shared_memory(gl.float16, [NUM_STAGES, BLOCK_N, BLOCK_K], b_desc.layout)
    bars = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(NUM_STAGES):
        mbarrier.init(bars.index(i), count=1)

    k_tiles = gl.cdiv(K, BLOCK_K)
    NBYTES: gl.constexpr = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * 2

    # Prologue: tile t lives in slot t % NUM_STAGES; fill slots 0..NUM_STAGES-2.
    for s in gl.static_range(NUM_STAGES - 1):
        if s < k_tiles:
            bar = bars.index(s)
            mbarrier.expect(bar, NBYTES)
            tma.async_copy_global_to_shared(a_desc, [off_m, s * BLOCK_K], bar, a_smem.index(s))
            tma.async_copy_global_to_shared(b_desc, [off_n, s * BLOCK_K], bar, b_smem.index(s))

    acc = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, mma)
    for kt in range(k_tiles):
        buf = kt % NUM_STAGES
        phase = (kt // NUM_STAGES) & 1
        mbarrier.wait(bars.index(buf), phase)
        a = a_smem.index(buf).load(dot_a)
        b = b_smem.index(buf).permute([1, 0]).load(dot_b)
        # All warps must finish reading before any warp refills a slot.
        gl.barrier()
        nk = kt + (NUM_STAGES - 1)
        if nk < k_tiles:
            nbuf = nk % NUM_STAGES  # slot freed in the previous iteration
            bar = bars.index(nbuf)
            mbarrier.expect(bar, NBYTES)
            tma.async_copy_global_to_shared(a_desc, [off_m, nk * BLOCK_K], bar, a_smem.index(nbuf))
            tma.async_copy_global_to_shared(b_desc, [off_n, nk * BLOCK_K], bar, b_smem.index(nbuf))
        acc = ampere.mma_v2(a, b, acc)

    offs_cm = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, mma))
    offs_cn = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, mma))
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.store(c_ptr + offs_cm[:, None].to(gl.int64) * N + offs_cn[None, :], acc.to(gl.float16), mask)
```

Host side: `TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K], layout)` and
`TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K], layout)` with
`layout = NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16,
rank=2)`, where `a = hidden.reshape(B*S, D)` and `b = embed` (`[V, D]`,
row-major, untransposed); grid `(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))`.

## Appendix B. Rerun commands

All commands run from the repository root. The env prefix is applied with
`env $ENV ...` — do not use `eval`, which re-parses quoting and breaks
commands containing semicolons.

```bash
# Hardened env prefix (use everywhere; see §2.4):
ENV='TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas CPATH=/usr/local/cuda-13.2/include TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor'

# Baseline validation:
env $ENV PYTHONPATH=src /workspace/venvs/sparton/bin/python -m py_compile \
  src/sparton/__init__.py src/sparton/sparton_kernel.py training/model.py \
  training/train.py tests/conftest.py tests/test_sparton_kernel.py
env $ENV /workspace/venvs/sparton/bin/python -m pytest -q
env $ENV PYTHONPATH=src /workspace/venvs/sparton/bin/python -c "import torch; import sparton.sparton_kernel; print(torch.ops.sparton.fused_sparton_fwd.default._schema); print(torch.ops.sparton.fused_sparton_bwd.default._schema)"

# Probes and benchmarks (see benchmarks/README.md):
env $ENV /workspace/venvs/sparton/bin/python -u benchmarks/probe_mma_matrix.py
env $ENV /workspace/venvs/sparton/bin/python -u benchmarks/bench_gluon_gemm.py
env $ENV PYTHONPATH=src /workspace/venvs/sparton/bin/python -u benchmarks/bench_hybrid_baseline.py

# nsys timeline + per-kernel summary:
env $ENV /usr/local/bin/nsys profile --stats=true -o /tmp/trace \
  /workspace/venvs/sparton/bin/python -u <bench>.py

# ncu counter profiles (see §11 for section/metric guidance).
# ncu_runner.py imports bench_gluon_gemm, hence PYTHONPATH=benchmarks:
env $ENV PYTHONPATH=benchmarks /usr/local/bin/ncu --launch-skip 4 --launch-count 1 \
  -k "regex:gemm_abt" \
  --section SpeedOfLight --section ComputeWorkloadAnalysis --section Occupancy \
  --metrics dram__bytes_op_read.sum,dram__bytes_op_write.sum \
  /workspace/venvs/sparton/bin/python -u benchmarks/ncu_runner.py
env $ENV PYTHONPATH=src /usr/local/bin/ncu --nvtx --nvtx-include "hybrid_fwd/" \
  --metrics dram__bytes_op_read.sum,dram__bytes_op_write.sum,gpu__time_duration.sum \
  /workspace/venvs/sparton/bin/python -u benchmarks/ncu_targets.py
```
