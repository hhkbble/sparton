# Kernel Optimization on Triton and Gluon

**Date:** 2026-06-12  
**Scope:** Triton language/compiler kernels and Triton’s experimental Gluon kernel language. This does **not** refer to NVIDIA Triton Inference Server.

## 1. Executive summary

Triton and Gluon share a tile-based GPU programming model and Python/JIT workflow, but they sit at different abstraction levels. Triton hides many layout, memory-allocation, data-movement, and asynchrony details behind the compiler. Gluon exposes more of those details, so it is useful when Triton is close but not quite enough and the bottleneck requires explicit control of register/shared-memory layouts, async pipelines, tensor-core pipelines, or hardware-specific scheduling.

Common optimization work falls into a small number of loops:

1. Build a correct scalar or simple tiled baseline.
2. Benchmark with stable shapes, dtypes, strides, warmups, and fixed hardware/software versions.
3. Classify the bottleneck: memory bandwidth, tensor-core/MFMA utilization, occupancy/register pressure, instruction mix, synchronization, cache locality, load imbalance, or launch overhead.
4. Tune the high-impact kernel knobs first: tile sizes, `num_warps`, `num_stages`, `num_ctas`, register caps, program ordering, accumulation dtype, vectorization/alignment, masks, and fusion boundaries.
5. Use profilers to validate the hypothesis, not just the timing delta.
6. Only then add lower-level techniques: shared-memory staging, swizzled layouts, async copy/TMA, persistent kernels, warp specialization, split-K or multi-CTA, hardware-specific layouts, and low-precision scale pipelines.

## 2. Triton vs. Gluon: when to use which

| Use Triton when... | Use Gluon when... |
|---|---|
| You need a custom operator quickly and the compiler-generated layouts are good enough. | The critical bottleneck is layout, shared memory, async copy, tensor memory, warp specialization, or architecture-specific scheduling. |
| The kernel is mostly elementwise, reduction-like, small GEMM-like, softmax-like, or fusion-heavy. | You need fine control over register/thread/warp/CTA distribution or bank-conflict behavior. |
| Portability and code size matter more than the last few percent. | You are chasing near-peak on a fixed target GPU generation. |
| `triton.autotune` over tile/meta-parameters is enough to expose the performance frontier. | Triton’s abstraction hides the mechanism you need to change. |

Gluon’s own introduction says it shares Triton’s compiler stack and tile-based SPMD model, but exposes details that Triton normally manages automatically: tile layouts, memory allocation, data movement, and asynchrony. The trade-off is explicit responsibility for more GPU-hardware details.

## 3. Baseline optimization methodology

### 3.1 Define the performance contract

Capture these before tuning:

- Shapes, dtypes, strides, alignment, batch sizes, and realistic input distributions.
- Target hardware: NVIDIA Ampere/Hopper/Blackwell, AMD CDNA/gfx target, clocking constraints, shared-memory size, and whether Tensor Cores/MFMA/WMMA are expected.
- Correctness tolerances by dtype and operation order. Keep a simple reference implementation and randomized edge-case tests.
- Throughput metric: GB/s for memory-bound kernels, TFLOP/s or tensor-core utilization for GEMM-like kernels, latency for small kernels, end-to-end model impact for fused kernels.

### 3.2 Use a reproducible benchmark harness

At minimum:

- Warm up JIT compilation separately from timing.
- Use CUDA/ROCm synchronization around timing.
- Use enough repetitions to reduce noise.
- Pin shapes and config keys when autotuning.
- Log Triton version, driver/runtime versions, GPU model, clock/power mode if controlled, environment variables, and selected autotune config.

### 3.3 Classify the bottleneck first

A useful first split:

| Symptom | Likely bottleneck | First actions |
|---|---|---|
| Low arithmetic intensity, high DRAM traffic | Memory-bound | Fuse ops, reduce stores/loads, improve coalescing, use cache-friendly program order, stage reusable data. |
| Low tensor-core/MFMA active cycles | Compute pipeline underfed | Increase tile reuse, tune `BLOCK_M/N/K`, use async copy/TMA, improve layouts, pipeline prologue/steady-state/epilogue. |
| Low occupancy from too many registers | Register pressure | Reduce tile size, split accumulators, use `maxnreg` carefully, reduce live ranges, reconsider unrolling/pipelining. |
| Many shared-memory conflicts | Shared-memory layout | Use swizzled/shared layouts, adjust vector width, alignment, and layout mapping. |
| High synchronization or barrier stalls | Pipeline/scheduling issue | Reduce barriers, use finer staging, consider warp specialization or persistent scheduling. |
| Wide variance across blocks | Load imbalance | Persistent kernels, grouped scheduling, CLC on Blackwell, split work more evenly. |

## 4. Common Triton optimization techniques

### 4.1 Tile shape and program mapping

The most important Triton knobs are usually the meta-parameters defining tile shape: `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, vector width, block size for reductions, and group size for program ordering. Triton’s official matmul tutorial emphasizes block-level matrix multiplication, multidimensional pointer arithmetic, program reordering for L2-cache hit rate, and automatic performance tuning.

Practical rules:

- Start from known-good tile families for the operation class, then tune around them.
- For GEMM-like kernels, tune `BLOCK_M/N/K` jointly with `num_warps` and `num_stages`.
- Increase tile size only while occupancy, register pressure, shared-memory footprint, and masking overhead stay acceptable.
- Prefer program-ordering schemes that reuse one operand in L2 across neighboring CTAs, especially for matmul and grouped GEMM.
- For reductions, tune the reduction tile separately from the output tile; avoid excessive per-program register arrays.

### 4.2 Autotuning

`triton.autotune` evaluates multiple `triton.Config` entries and re-evaluates when selected key arguments change. `triton.Config` controls kernel meta-arguments plus compiler/runtime knobs such as `num_warps`, `num_stages`, `num_ctas`, and `maxnreg`.

Typical search dimensions:

```python
triton.Config(
    {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
    num_warps=4,
    num_stages=3,
)
```

Guidance:

- Use `key=[...]` that captures shape/dtype/stride changes affecting performance.
- Keep the search space small enough to run in CI or precompute per architecture.
- For kernels that write outputs during tuning, use reset/restore hooks or scratch outputs to avoid corrupting data across repeated runs.
- Log selected configs with `TRITON_PRINT_AUTOTUNING=1` when comparing runs.
- Use early pruning or performance models when the Cartesian product grows too large.

### 4.3 Occupancy, register pressure, and software pipelining

`num_warps` changes how many GPU warps execute one Triton program. `num_stages` controls compiler software pipelining and is especially relevant for matmul-style loops on SM80+ NVIDIA GPUs. `maxnreg` can cap registers per thread on supported platforms, but it can also induce spills, so treat it as an experiment to validate with a profiler.

Practical loop:

1. Increase tile reuse until register pressure or shared-memory footprint hurts occupancy.
2. Sweep `num_warps` for each tile family; larger is not automatically better.
3. Sweep `num_stages`; more stages can hide latency but increase shared memory and live ranges.
4. Check generated resource usage and profiler stalls before accepting a timing improvement.

### 4.4 Memory access and fusion

For memory-bound Triton kernels:

- Make loads/stores contiguous and coalesced where possible.
- Minimize rereads and materialized temporaries by fusing elementwise epilogues.
- Use masks only where needed; excessive masked lanes waste bandwidth and instructions.
- Align pointer offsets and vectorization so each program maps cleanly to memory transactions.
- Avoid unnecessary dtype conversions in the hot path.

For compute-bound kernels:

- Keep accumulators in registers and write once.
- Use tensor-core-friendly tile dimensions and dtypes.
- Fuse epilogues only when the added register pressure does not reduce tensor-core utilization.

### 4.5 Persistent kernels and cache-aware grouped scheduling

Persistent kernels keep a bounded set of CTAs resident and iterate over multiple work tiles. They are useful when launch scheduling, tile imbalance, or producer/consumer locality limits performance. Triton’s persistent matmul tutorial includes naive, persistent, and TMA-based approaches, supports FP16 and FP8 paths, and benchmarks against cuBLAS using Proton.

Use persistence when:

- The number of tiles is large and scheduling overhead or load imbalance matters.
- The kernel can benefit from cache-aware grouped work assignment.
- You can keep the persistent loop simple enough to avoid register/synchronization blowups.

Avoid persistence when:

- The simple tiled kernel already saturates the device.
- The persistent loop reduces occupancy or complicates correctness more than it helps.

## 5. Common Gluon-specific techniques

### 5.1 Explicit tensor layouts

In Gluon, tensors require layouts. A layout maps tensor elements to CTAs, warps, lanes, and per-lane registers. `BlockedLayout` is the common baseline layout, parameterized by values such as `size_per_thread`, `threads_per_warp`, `warps_per_cta`, and `order`.

Optimization use cases:

- Choose register ownership to match the operation’s access pattern.
- Avoid unnecessary physical registers for shapes that do not use the full tile.
- Match tensor-core operand layout requirements.
- Convert or compose layouts only when the data movement is paid back by better compute or memory behavior.

### 5.2 Shared-memory layout and bank conflicts

Gluon exposes shared-memory descriptors and layouts. Its async-copy tutorial notes that shared-memory layout is selected to reduce bank conflicts and sometimes to satisfy operation constraints.

Optimization use cases:

- Stage global memory into shared memory when reuse justifies the extra traffic and barriers.
- Use swizzled shared layouts for bank-conflict reduction.
- Validate bank conflicts with hardware profilers; do not assume a layout is better from code shape alone.
- Keep shared-memory footprint compatible with occupancy targets.

### 5.3 Async copy, TMA, and pipeline staging

Gluon exposes lower-level async-copy/TMA machinery. Use it for kernels where data movement must overlap with compute:

- `cp.async`-style global-to-shared staging on NVIDIA Ampere/Hopper paths.
- TMA-based multidimensional copies on Hopper/Blackwell-style kernels.
- Explicit prologue, steady-state, and epilogue pipeline stages.
- Barriers/mbarriers only at required producer-consumer boundaries.

Good signals:

- Higher tensor-core/MFMA utilization.
- Lower memory-dependency stalls.
- Stable or acceptable occupancy after adding stages.

Bad signals:

- More stages increase register/shared-memory pressure enough to lower throughput.
- Barrier overhead or producer/consumer imbalance dominates.

### 5.4 Warp specialization

Warp specialization assigns different warps in a CTA to different roles, for example memory producers and compute consumers. Gluon’s tutorial describes it as a way to overlap independent work and reduce the per-warp critical path, while warning about synchronization overhead, higher shared-memory use, and higher register pressure. In Gluon it is documented as Hopper-or-newer for NVIDIA GPUs.

Use it when:

- Async data movement and compute can genuinely overlap.
- The producer and consumer roles have different instruction streams.
- The extra synchronization and shared-memory traffic are smaller than the hidden latency.

### 5.5 Tensor-core/MFMA pipelines and low precision

For GEMM-like kernels, Gluon makes operand layout, accumulator layout, shared-memory movement, and architecture-specific matrix instructions explicit. The AMD ROCm Gluon GEMM tutorial shows a profiler-driven path from a naive FP16 GEMM to near-peak MFMA utilization, then transfers the same design to BF8 and MXFP4 with a scale pipeline.

Common techniques:

- Use architecture-native matrix instructions: NVIDIA WGMMA/TCGen05 where available, AMD MFMA/WMMA on supported targets.
- Keep accumulator live ranges under control; split or slice when register pressure dominates.
- Specialize low-precision paths around scale/dequantization pipelines rather than bolting scaling onto an FP16 kernel late.
- Remap workgroups for chiplet/XCD locality on AMD when profiler evidence supports it.

### 5.6 Dynamic work distribution on Blackwell

Gluon documents Cluster Launch Control (CLC) for Blackwell/SM100+, where a block that finishes early can cancel a pending cluster and take over its work. The tutorial’s key optimization is issuing CLC during the TMA prologue and checking the result after tile completion to hide CLC latency behind compute.

Use CLC when:

- There is real inter-block load imbalance.
- The target is Blackwell-class hardware.
- The CLC request/check can be overlapped with useful work.

## 6. Tooling checklist

| Tool | Best use | Notes |
|---|---|---|
| `triton.testing.do_bench` | Fast kernel microbenchmarking | Good for local iteration; still log shape/config/version. |
| `triton.autotune` + `triton.Config` | Search tile/compiler knobs | Include tile sizes, `num_warps`, `num_stages`, `num_ctas`, `maxnreg`; use correct keys. |
| `TRITON_PRINT_AUTOTUNING=1` | See selected configs | Useful for regression debugging and CI logs. |
| Triton debugging ops | Compile-time and runtime checks | `static_print`, `static_assert`, `device_print`, `device_assert`; `device_assert` requires `TRITON_DEBUG=1`. |
| `TRITON_INTERPRET=1` | CPU-side functional debugging | Useful before profiling; has documented limitations such as no `bfloat16` support and limited indirect access support. |
| FpSan | Structural equivalence checks for floating-point kernels | Good for comparing optimized vs reference kernels under sanitized semantics, not for IEEE accuracy. |
| Proton | Triton/Gluon profiling and intra-kernel scopes | Triton’s persistent matmul tutorial uses Proton; Triton’s repo includes Proton DSL examples for Triton and Gluon kernels. |
| NVIDIA Nsight Compute / `ncu` | NVIDIA kernel-level performance counters | Use for occupancy, SM/memory throughput, warp stalls, instruction mix, tensor-core activity, replay-aware profiling. |
| NVIDIA Compute Sanitizer | NVIDIA correctness debugging | `memcheck`, `racecheck`, `initcheck`, `synccheck` catch memory, race, uninitialized, and synchronization issues. |
| ROCm Compute Profiler / `rocprof` | AMD profiling | ROCm Compute Profiler collects hardware counters and offers speed-of-light, memory chart, roofline, and baseline comparisons; `rocprof` exposes lower-level raw counters/traces. |
| Triton-Viz | Visualize Triton program behavior | Useful for memory-access visualization and education/debugging; not a replacement for hardware counters. |
| PTXAS/compiler inspection | NVIDIA codegen/resource debugging | Use to inspect register count, spills, generated PTX/SASS, and ptxas-option experiments. |

### 6.1 IR-stage visibility in practice (recipe added at M11)

Triton (and Gluon, which shares the pipeline) keeps every compilation stage
of a launched kernel in memory: walk `jit_fn.device_caches` (the function
under an `@triton.autotune` wrapper is `.fn`) to the `CompiledKernel`s and
read the `.asm` dict — keys `ttir`, `ttgir`, `llir`, `ptx`, `cubin`,
`source`; `nvdisasm -c` on the cubin yields SASS. `TRITON_KERNEL_DUMP=1`
(+`TRITON_DUMP_DIR`) dumps the same to disk; `MLIR_ENABLE_DUMP=1` adds
per-pass IR. What each stage answers:

- **TTGIR** — the optimization-relevant stage: chosen `#blocked`/shared
  layouts (`sizePerThread`, `threadsPerWarp`, `order`), `tt.scan`/reduce
  lowering, pipelining structure, swizzles. For Gluon kernels the layouts
  are user-chosen, so this is where to verify they survived.
- **PTX/SASS** — instruction mix and final forms: load vector widths
  (`LDG.E.128` vs scalar), atomic forms (`REDG.E.ADD.F32x4` = 4-wide
  vectorized reduction; `REDUX.*` = warp-level reduce), spills
  (`ld.local/st.local`), predication, barrier count.

Use it as a zero-GPU-cost "confirm the lowering" step between picking a
config family and benchmarking it, and to settle ncu surprises by reading
the instruction form instead of inferring it from counter arithmetic.
M11 examples: the backward's 4-wide vectorized `REDG` (inferred indirectly
from instruction counts in T1; one SASS grep proves it), and the
`tl.cumsum` shuffle-tree whose cost was discovered by per-kernel timing —
visible immediately as `tt.scan` + SHFL chains in TTGIR/SASS.

### 6.2 Layout and lowering attribution in practice (lessons added at M13)

Triton chooses tensor layouts from the *consuming* operations, and that
choice propagates back into load vector widths. Lessons measured on the
sparton backward (M13 memo §2/§5.4; regenerate the per-config IR/SASS
dumps any time with `scripts/dump_backward_ir.py` → `tests/data/ir_dump/`):

- **Reductions are layout-agnostic; scans are not.** A tile feeding only
  `tl.sum` vectorizes freely (`LDG.E.128` at ordinary register budgets); a
  tile feeding `tl.cumsum` along the row axis anchors a row-per-thread or
  narrow layout — the measured outcome on a (CHUNK, BLOCK_D) gather tile
  was scalar `ld.global.b32` on every config except the one-thread-per-row
  shape, which paid `BLOCK_D` floats of live scan state per thread
  (255 registers → 1 CTA/SM). Vectorize-or-occupancy was a structural
  trade, not a tuning trade.
- **Branch-local SSA separation is not layout separation.** Giving the hot
  branch its own `tl.load` (separate SSA value from the scan branch's
  load) compiled to vectorized *sites* in PTX/SASS, but the executed
  instruction mix stayed scalar-class (runtime bytes-per-instruction
  ≈ unchanged). The working fix classes are structural: a branch-free hot
  kernel (split kernels with complementary predicates), or folding the
  suppression into load *masks* — a per-row mask broadcast over the vector
  axis preserves the wide form, while a branch around the load re-anchors
  it.
- **`while` walks do not software-pipeline; `tl.range` for-loops do.** The
  persistent `while` + sentinel-exit pattern serializes each iteration's
  keys→check→tile round trips; converting the hot pass to a bounded
  for-loop (device-side trip count, e.g. an active-entry counter written
  by an upstream kernel — no host sync) is what unlocked
  pipelined/vectorized execution. Keep sentinel economics with masks, not
  loop breaks.
- **Attribute empirically when static reading stalls.** Two tools settle
  what instruction listings cannot: (1) *differential compilation* —
  compile the suspect region in isolation (a throwaway kernel with only
  the hot branch) and diff its census against the full kernel's; (2)
  *bytes-per-warp-instruction* — `sectors × 32 ÷
  smsp__inst_executed_op_global_ld` from one ncu pass tells you the
  executed width regardless of what the SASS sites suggest (≈3–4 B/lane =
  scalar, ≈16 B/lane = v4). Three rounds of instruction-counting failed to
  explain what these settled in minutes.
- **Measured config equivalences are evidence.** Two autotune configs from
  different occupancy classes (16.6% vs 24.8% warps) timing identically
  eliminates occupancy-alone as the binder; a vectorized-but-1-CTA config
  tying a scalar-but-3-CTA config localizes the constraint to the
  layout coupling above. Read autotuner ties as information about the
  bottleneck, not as jitter to suppress.

## 7. Common failure modes

- **Tuning only wall time.** A faster microbenchmark can still be fragile if it relies on lower clocks, cache luck, or shape-specific artifacts.
- **Over-fusion.** Fusion reduces memory traffic but can increase registers enough to reduce occupancy or tensor-core utilization.
- **Too-large tiles.** Larger tiles improve reuse until they cause spills, shared-memory pressure, masks, or lower occupancy.
- **Autotune key mismatch.** If key arguments omit a performance-relevant stride/shape/dtype, Triton may reuse a bad config.
- **Independently tuned copies of a shared parameter.** When two cooperating kernels must agree on a partitioning parameter (a chunk/granule size whose predicates must complement), letting each autotune its own silently drops or double-counts work at the disagreement boundary — and a small-shape repro can pass because both tuners happen to agree there. Single-source the choice (one kernel tunes, the other reads `best_config` host-side) and assert the compatibility (M13 memo §5.4 item 2).
- **Ignoring tails.** Non-power-of-two dimensions and ragged batches often dominate real workloads.
- **Assuming Gluon layout changes are free.** Layout conversions and explicit shared-memory staging have real cost.
- **Misusing async pipelines.** More stages do not help if barriers, live ranges, or producer/consumer imbalance dominate.
- **Profiler perturbation.** Nsight Compute and ROCm Compute Profiler can replay kernels or collect counters in multiple passes, so benchmark separately from detailed profiling.

## 8. Practical optimization playbooks

### Memory-bound elementwise or reduction kernel

1. Start with a simple Triton implementation.
2. Benchmark GB/s versus expected bandwidth.
3. Tune block size/vector width and `num_warps`.
4. Reduce loads/stores by fusing adjacent ops.
5. Improve coalescing and alignment.
6. Check masks and tail handling.
7. Use profiler memory throughput, cache hit rate, and warp stall metrics.
8. Move to Gluon only if explicit layout/shared-memory control is needed.

### GEMM-like kernel

1. Start from Triton matmul-style tiling.
2. Autotune `BLOCK_M/N/K`, `GROUP_M`, `num_warps`, `num_stages`.
3. Check tensor-core/MFMA utilization and occupancy.
4. Add cache-aware program ordering.
5. Consider persistent scheduling for tile reuse/load balance.
6. Move to Gluon when you need explicit operand layouts, shared-memory swizzles, async/TMA, WGMMA/TCGen05/MFMA control, or warp specialization.
7. For low precision, design the scale/dequant pipeline as part of the main tile pipeline.

### Attention or fused sequence kernel

1. Identify whether the dominant path is memory traffic, softmax/reduction latency, or matmul/tensor-core utilization.
2. Tile by sequence length and head dimension to keep working sets in registers/shared memory.
3. Use numerically stable online reductions.
4. Fuse epilogues only while controlling register pressure.
5. Profile tails, causal masks, variable sequence lengths, and KV-cache layout.
6. Consider Gluon for explicit layout, warp-level reductions, async copy/TMA, and warp specialization.

## 9. Recommended source trail

Primary and high-signal references used for this note:

1. Triton documentation: `triton.autotune` — <https://triton-lang.org/main/python-api/generated/triton.autotune.html>
2. Triton documentation: `triton.Config` — <https://triton-lang.org/main/python-api/generated/triton.Config.html>
3. Triton tutorial: Matrix Multiplication — <https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html>
4. Triton tutorial: Persistent Matmul — <https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html>
5. Triton documentation: Debugging Triton — <https://triton-lang.org/main/programming-guide/chapter-3/debugging.html>
6. Triton documentation: Floating-Point Sanitizer — <https://triton-lang.org/main/programming-guide/chapter-3/fpsan.html>
7. Triton Gluon tutorial: Introduction to Gluon — <https://triton-lang.org/main/getting-started/tutorials/gluon/intro.html>
8. Triton Gluon tutorial: Tensor Layouts — <https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html>
9. Triton Gluon tutorial: Async Copy — <https://triton-lang.org/main/getting-started/tutorials/gluon/async-copy.html>
10. Triton Gluon tutorial: Warp Specialization — <https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html>
11. Triton Gluon tutorial: Cluster Launch Control — <https://triton-lang.org/main/getting-started/tutorials/gluon/cluster-launch-control.html>
12. AMD ROCm blog: From Naive to Near-Peak: Building High-Performance GEMM Kernels with Gluon — <https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html>
13. AMD ROCm workload optimization/profiling docs — <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html>
14. NVIDIA Nsight Compute Profiling Guide — <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
15. NVIDIA Compute Sanitizer documentation — <https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html>
16. Triton Proton intra-kernel profiling example for Triton/Gluon kernels — <https://github.com/triton-lang/triton/blob/main/third_party/proton/tutorials/intra_kernel/example_dsl.py>
17. Triton-Viz GitHub repository — <https://github.com/Deep-Learning-Profiling-Tools/triton-viz>
18. TritonForge paper abstract — <https://arxiv.org/abs/2512.09196>

