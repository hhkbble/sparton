# Sparton Gluon-First Refactor Design

**Goal:** refactor `src/sparton/sparton_kernel.py` into a three-backend implementation where the production fast path is a Gluon optimized fused forward/backward path, while keeping the original Sparton hybrid implementation and a Triton-only naive fused implementation as baselines.

**Scope:** Python-first, no CUTLASS/C++ dependency, no per-architecture source forks, no runtime feature gates. Runtime specialization is allowed only for *hardware parameters* and problem shapes: shared-memory budget, SM count, warp limits, total memory, dtype, and tensor dimensions.

**Non-goal:** support GPUs older than the minimum platform family that provides the selected common feature set. Also out of scope for the first production milestone: FP8/FP6/FP4, cluster multicast, multi-CTA cluster kernels, device-specific tensor-memory paths, native TMA gather/scatter, or architecture-specific GEMM instructions unavailable on the current development GPU.

---

## 1. Naming and backend taxonomy

The project should use three implementation labels consistently.

### 1.1 `hybrid`

`hybrid` means the current Sparton implementation style:

```text
for each vocabulary tile:
    tile_logits = torch.compile(matmul or matmul_bias)(hidden, embed_tile)
    tile_scores, tile_indices = Triton reduction(tile_logits, mask)
```

Characteristics:

- Uses vendor-backed PyTorch matmul for `hidden @ embed_tile.T`.
- Materializes `tile_logits [B, S, V_tile]`.
- Uses a Triton reduction kernel for sequence max, argmax, ReLU, and log1p.
- Provides the current integration contract through `torch.library.custom_op`, fake implementation, and registered autograd.
- Remains the main correctness and regression baseline.

### 1.2 `naive`

`naive` means a **Triton-only fused** baseline, not the current Sparton hybrid.

```text
one Triton kernel:
    compute GEMM tile with tl.dot
    apply bias and mask
    online sequence max/argmax
    store scores and indices
```

Characteristics:

- Does not use Gluon.
- Does not materialize `tile_logits` in production.
- Intended as a readable fused-kernel baseline.
- Expected to be slower than `hybrid` for large GEMM-dominated shapes, but useful for debugging fusion semantics and isolating Gluon-specific bugs.

### 1.3 `optimized`

`optimized` means the **Gluon-first fused** implementation.

```text
one Gluon forward kernel:
    TMA/async staged A/B tiles
    warp-group MMA mainloop
    bias + mask
    online sequence max/argmax
    ReLU + log1p
    store only [B, V] scores and [B, V] indices
```

Backward also gets a Gluon implementation, but it is not expected to look like a dense GEMM. It remains an indexed gather/scatter + atomic accumulation problem driven by the forward `scores` and `indices`.

---

## 2. Design corrections from earlier drafts

The earlier design mixed three different ideas that must now be separated.

### 2.1 Hardware features vs hardware parameters

A hardware **feature** is a capability such as TMA, warp-group MMA, warp specialization, multicast, native TMA gather/scatter, special accumulator storage, or a new low-precision MMA family.

A hardware **parameter** is a quantitative resource such as:

- number of SMs;
- maximum shared memory per block;
- maximum shared memory per SM;
- maximum resident warps or threads;
- warp size;
- total device memory;
- L2 cache size if discoverable;
- dtype size;
- problem dimensions.

This refactor assumes one fixed minimum feature set. It must not contain branches such as:

```python
# forbidden pattern
if device_capability >= ...:
    use_feature_a()
else:
    use_feature_b()
```

It may contain resource-driven policy generation:

```python
profile = get_device_profile()
policies = generate_candidate_policies(problem, profile)
policy = benchmark_or_lookup_best_policy(problem, profile, policies)
```

### 2.2 Removed optimization points

The following optimization points are intentionally removed from the production design because they either are not available on the current development GPU class, are not part of the shared minimum feature set, or belong to a future low-precision/backend experiment:

- cluster multicast;
- cluster shape greater than `1x1x1`;
- multi-CTA cooperative GEMM as a production requirement;
- two-SM or CTA-pair GEMM;
- device-specific tensor-memory accumulator paths;
- architecture-specific fifth-generation MMA instruction paths;
- native TMA gather/scatter for backward;
- Cluster Launch Control as a required scheduler;
- FP8/FP6/FP4 and block-scaled low precision;
- CUTLASS/C++ integration.

The production optimized path uses only the common selected feature set:

- Gluon kernels;
- tensor descriptors;
- TMA/async global-to-shared staging;
- mbarrier/fence synchronization;
- warp-group MMA or Gluon-provided MMA abstraction for FP16/BF16;
- warp specialization;
- persistent scheduling implemented in Python/Gluon style;
- regular global atomics for backward.

### 2.3 Compatibility target

The implementation should be developed on the current RTX development platform and remain compatible with the previous supported platform family by construction, because it uses the shared selected feature set. It should not try to support older GPUs.

This is a **minimum-platform assumption**, not a runtime feature-gate. If the package is installed on an unsupported GPU, tests should fail clearly or the user should explicitly select `backend="hybrid"` for debugging. The production optimized path itself should not contain architecture-named branches.

---

## 3. Mathematical contract

Inputs:

```text
hidden: [B, S, D]
embed:  [V, D]
bias:   [V] or None
mask:   [B, S]
```

Output:

```text
scores:  [B, V]
indices: [B, V]
```

For each batch row `b` and vocabulary id `v`:

```text
raw[b, s, v] = dot(hidden[b, s, :], embed[v, :]) + bias[v]
masked_raw[b, s, v] = raw[b, s, v] * mask[b, s]
running_max[b, v] = max_s masked_raw[b, s, v], with baseline 0
indices[b, v] = first s that strictly improves running_max
scores[b, v] = log(1 + max(0, running_max[b, v]))
```

The baseline `0` is intentional. It matches Sparton semantics because the downstream activation is `ReLU` followed by `log1p`. If all valid logits are negative, the output score is zero and the index is not semantically important for gradient propagation.

Tie-breaking must use strict `>` update, matching the current kernel behavior:

```text
if tile_value > running_value:
    update value and index
else:
    keep previous value and index
```

Backward uses the saved score and index:

```text
g[b, v] = grad_out[b, v] * exp(-scores[b, v]) if scores[b, v] > 0 else 0
```

Then:

```text
d_bias[v] += sum_b g[b, v]
d_embed[v, d] += sum_b g[b, v] * hidden[b, indices[b, v], d]
d_hidden[b, indices[b, v], d] += g[b, v] * embed[v, d]
```

---

## 4. Current file structure and refactor plan

Keep the public file `src/sparton/sparton_kernel.py` as the import-compatible entry point, but split the implementation internally.

Suggested structure:

```text
src/sparton/
  sparton_kernel.py                 # public module, custom ops, SpartonHead
  _backend_hybrid.py                # current Sparton implementation, minimally changed
  _backend_naive_triton.py          # Triton-only fused baseline
  _backend_optimized_gluon.py       # production Gluon implementation
  _runtime_policy.py                # parameter-derived policy generation and autotune cache
  _tensor_layout.py                 # contiguous checks, descriptor creation, weight-layout helpers
  _test_reference.py                # pure PyTorch reference used by tests only
```

Do not put architecture names into backend names, kernel names, policy names, cache keys, or public config values. Backend names must describe implementation strategy, not GPU generation.

### 4.1 Public API

Preserve the current API:

```python
class SpartonHead(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, use_bias: bool = False, backend: str = "optimized"):
        ...

    def forward(self, hidden_states, attention_mask):
        scores, _idx = fused_sparton_fwd_op(hidden_states, self.weight, self.bias, attention_mask)
        return scores
```

Recommended backend selection:

```text
backend="optimized"  # default production path
backend="hybrid"     # current Sparton baseline
backend="naive"      # Triton-only fused baseline
```

The custom op signature should remain compatible:

```python
@torch.library.custom_op("sparton::fused_sparton_fwd", mutates_args=())
def fused_sparton_fwd_op(hidden, embed, bias, mask) -> Tuple[torch.Tensor, torch.Tensor]:
    ...
```

The fake implementation remains:

```python
@fused_sparton_fwd_op.register_fake
def _(hidden, embed, bias, mask):
    B, S, D = hidden.shape
    V, D2 = embed.shape
    return hidden.new_empty((B, V)), torch.empty((B, V), device=hidden.device, dtype=torch.int64)
```

The registered autograd remains:

```python
def _setup_context(ctx, inputs, output):
    hidden, embed, bias, mask = inputs
    scores, idx = output
    ctx.save_for_backward(scores, idx, hidden, embed, bias, mask)

def _backward(ctx, grad_scores, grad_idx):
    scores, idx, hidden, embed, bias, mask = ctx.saved_tensors
    hidden_g, embed_g, bias_g = fused_sparton_bwd_op(grad_scores, scores, idx, hidden, embed, bias, mask)
    return hidden_g, embed_g, bias_g, None
```

### 4.2 Bias handling

The current implementation is most robust when bias exists. The refactor should explicitly support both:

- `bias is None`: use zero bias path, return `None` for bias gradient.
- `bias is Tensor[V]`: compute and return bias gradient.

The custom op schema may need two Python wrappers if `None` is awkward for `torch.library.custom_op`:

```text
sparton::fused_sparton_fwd_bias
sparton::fused_sparton_fwd_nobias
```

The public `SpartonHead` should hide this split.

---

## 5. Tensor layout contract

### 5.1 Hidden layout

Require or create contiguous hidden:

```text
hidden: [B, S, D], row-major contiguous
A view for GEMM: [B*S, D], row-major
```

Do not accept arbitrary non-contiguous hidden in optimized mode unless it is made contiguous before launch. Silent wrong-address behavior is unacceptable.

### 5.2 Embedding layout

Current Sparton stores:

```text
embed: [V, D], row-major contiguous
```

The GEMM is conceptually:

```text
A = hidden.reshape(B*S, D)       # [M, K]
B = embed.T                      # [K, N]
```

No physical transpose is required. Treat `embed` as a descriptor with conceptual shape `[D, V]` and strides:

```text
B[k, v] -> embed[v, k]
stride_k = 1
stride_v = D
```

This gives a row-major A and column-major B conceptual GEMM layout without materializing `embed.T`.

### 5.3 Mask layout

Require or create contiguous mask:

```text
mask: [B, S]
```

The optimized forward only needs mask loads during epilogue/reduction, not during the MMA mainloop.

### 5.4 Output layout

```text
scores:  [B, V], same dtype as hidden unless configured otherwise
indices: [B, V], int64 for API compatibility
```

Internally, the kernel can keep indices as 32-bit while computing, but it should store int64 unless the public contract is changed deliberately.

---

## 6. Runtime policy: parameter-derived, not feature-gated

### 6.1 Device profile

The runtime profile should contain only quantitative hardware parameters.

```python
@dataclass(frozen=True)
class DeviceProfile:
    name: str
    sm_count: int
    warp_size: int
    max_threads_per_block: int
    max_threads_per_multiprocessor: int | None
    shared_memory_per_block: int
    shared_memory_per_block_optin: int | None
    shared_memory_per_multiprocessor: int | None
    total_memory: int
    l2_cache_size: int | None
```

Collection should use PyTorch/CUDA attributes without branching on architecture names:

```python
def get_device_profile(device=None) -> DeviceProfile:
    props = torch.cuda.get_device_properties(device or torch.cuda.current_device())
    return DeviceProfile(
        name=props.name,
        sm_count=props.multi_processor_count,
        warp_size=getattr(props, "warp_size", 32),
        max_threads_per_block=getattr(props, "max_threads_per_block", 1024),
        max_threads_per_multiprocessor=getattr(props, "max_threads_per_multi_processor", None),
        shared_memory_per_block=getattr(props, "shared_memory_per_block", 48 * 1024),
        shared_memory_per_block_optin=getattr(props, "shared_memory_per_block_optin", None),
        shared_memory_per_multiprocessor=getattr(props, "shared_memory_per_multiprocessor", None),
        total_memory=props.total_memory,
        l2_cache_size=getattr(props, "l2_cache_size", None),
    )
```

If a needed attribute is not available from PyTorch, use a small CUDA runtime helper or default to conservative values. This is still parameter-derived, not feature-gated.

### 6.2 Problem profile

```python
@dataclass(frozen=True)
class ProblemProfile:
    B: int
    S: int
    D: int
    V: int
    dtype: torch.dtype
    has_bias: bool
    requires_grad: bool
```

### 6.3 Kernel policy

```python
@dataclass(frozen=True)
class KernelPolicy:
    batch_block: int
    seq_block: int
    vocab_block: int
    k_block: int
    pipeline_stages: int
    num_warps: int
    producer_warps: int
    consumer_warps: int
    persistent: bool
    split_k: int = 1
```

`split_k` should remain `1` for the first production implementation. Split-K complicates online sequence max because partial dot products must be reduced before max. It is only useful for GEMM-only microbenchmarks or future two-kernel fallback variants.

### 6.4 Candidate generation

The policy generator should derive candidates from resource budgets.

```text
M_tile = batch_block * seq_block
N_tile = vocab_block
K_tile = k_block
```

Candidate families:

```text
batch_block: 1, 2, 4
seq_block:   32, 64, 128
vocab_block: 64, 128, 256
k_block:     32, 64, 128
stages:      derived from shared-memory budget, usually 2-4
num_warps:   4, 8, 12, 16, pruned by thread limit and register pressure
```

Pruning rules:

```text
M_tile should be 64 or 128 for the first optimized forward.
N_tile should usually be 128 or 256.
K_tile must divide or cleanly tile D, with tail handling.
shared_memory_total <= opt-in shared-memory budget * safety_factor.
num_warps * warp_size <= max_threads_per_block.
M_tile * N_tile accumulator footprint must fit expected register pressure.
```

Approximate shared memory per pipeline stage:

```text
smem_a = M_tile * K_tile * dtype_bytes
smem_b = K_tile * N_tile * dtype_bytes
smem_per_stage = smem_a + smem_b
smem_total = stages * smem_per_stage + barrier_bytes + scratch_bytes
```

Select the largest `stages` satisfying:

```text
smem_total <= shared_memory_budget * 0.85
```

Do not use a feature flag such as `supports_multicast` or `supports_tensor_memory`. Those optimizations are not part of this design.

### 6.5 Persistent scheduling parameters

A persistent grid should be sized from SM count and desired CTAs per SM:

```text
ctas_per_sm_by_smem = floor(shared_memory_per_multiprocessor / smem_total)
ctas_per_sm_by_threads = floor(max_threads_per_multiprocessor / (num_warps * warp_size))
ctas_per_sm = clamp(min(ctas_per_sm_by_smem, ctas_per_sm_by_threads), 1, max_ctas_per_sm_cap)
grid_ctas = sm_count * ctas_per_sm
```

The kernel uses a persistent tile scheduler over logical tiles:

```text
logical tile id -> (batch_block_id, vocab_block_id)
```

The sequence dimension is not part of the grid. Each CTA scans sequence chunks internally and maintains running max/argmax.

---

## 7. Optimized Gluon forward design

### 7.1 Logical work ownership

Each logical CTA owns:

```text
batch block: batch_block batch rows
vocab block: vocab_block vocabulary columns
```

Inside that CTA:

```text
running_max: [batch_block, vocab_block]
running_idx: [batch_block, vocab_block]
```

It loops over sequence chunks:

```text
for s0 in range(0, S, seq_block):
    compute logits for hidden rows [batch_block * seq_block] and vocab columns [vocab_block]
    reduce over seq_block for each batch row and vocabulary column
    update running_max/running_idx
```

### 7.2 Data descriptors

Create tensor descriptors outside the kernel where Gluon requires descriptor objects.

```text
A_desc: hidden as [B*S, D]
B_desc: embed as conceptual [D, V] with strides [1, D]
Out_desc: scores as [B, V]
Idx_desc: indices as [B, V]
```

Mask and bias may be raw pointers or descriptors depending on which path produces cleaner Gluon code. Bias is one-dimensional and usually does not benefit from TMA.

### 7.3 Mainloop shape

For one sequence chunk:

```text
A_tile: [M_tile, K_tile] where M_tile = batch_block * seq_block
B_tile: [K_tile, N_tile] where N_tile = vocab_block
Accumulator: [M_tile, N_tile]
```

The inner K loop:

```text
for k0 in range(0, D, k_block):
    producer issues async/TMA loads for A_tile and B_tile into shared buffers
    consumer waits on mbarrier
    consumer issues warp-group MMA
```

The accumulator should be FP32 for initial correctness and stability. A BF16/FP16 accumulator option can be benchmarked later, but FP32 is the reference optimized path.

### 7.4 Pipeline

Use multi-buffered staging:

```text
num_buffers = pipeline_stages
shared_A[num_buffers, M_tile, K_tile]
shared_B[num_buffers, K_tile, N_tile]
barriers[num_buffers]
```

Per K iteration:

```text
producer:
    expect bytes for A and B
    issue async loads into buffer i

consumer:
    wait for buffer i
    issue MMA using shared_A[i], shared_B[i]
```

Use `fence_async_shared` whenever data crosses between generic shared-memory accesses and the async proxy. The initial implementation should be conservative with fences. Remove only after instruction-level validation.

### 7.5 Warp specialization

Use at least two partitions:

```text
producer partition:
    descriptor/TMA load issue
    mbarrier bookkeeping

consumer partition:
    MMA
    local epilogue/reduction
```

The producer should use few registers. The consumer holds accumulators and running max state.

The final tuned variants should compare:

```text
cooperative: one consumer group owns one output tile
pingpong-like: two consumer groups alternate chunks or vocab tiles
```

Do not encode device names in schedule names. Use names like:

```text
schedule="cooperative"
schedule="pingpong"
```

### 7.6 Online max epilogue

After finishing the K loop for one sequence chunk, the accumulator has shape:

```text
[batch_block * seq_block, vocab_block]
```

Convert logical row `m` to:

```text
local_b = m // seq_block
local_s = m % seq_block
absolute_s = s0 + local_s
```

For each `local_b` and vocab column:

```text
value = accumulator[m, n] + bias[n]
if mask[b, absolute_s] == 0:
    value = 0
local_max, local_arg = max over local_s
```

Then update running state:

```text
if local_max > running_max[local_b, n]:
    running_max[local_b, n] = local_max
    running_idx[local_b, n] = absolute_s_of_local_arg
```

Final store:

```text
score = log(1 + max(0, running_max))
store score
store running_idx
```

### 7.7 No materialization guarantee

The optimized forward production path must not allocate or write:

```text
[B, S, V]
[B, S, V_tile]
[B*S, V_tile]
partial max workspace [num_tiles, B, V]
```

Allowed outputs:

```text
scores:  [B, V]
indices: [B, V]
```

Allowed temporary storage:

```text
shared-memory A/B stage buffers
register/TMA accumulator fragments
running max and index state inside the CTA
small autotune scratch/cache metadata
```

### 7.8 Forward variants

Start with these optimized variants:

```text
O1: non-persistent Gluon fused forward
O2: persistent Gluon fused forward
O3: persistent + warp-specialized Gluon fused forward
O4: persistent + warp-specialized + pingpong-like scheduling
```

`O1` is for correctness. `O3/O4` are performance candidates. All variants use the same feature set and differ only in parameters/scheduling.

---

## 8. Naive Triton fused baseline

The naive baseline is intentionally simpler:

```text
one Triton program owns batch_block x vocab_block
loops over sequence chunks
loops over K chunks using tl.dot
online max/argmax
stores scores/indices
```

No TMA, no Gluon, no special pipeline.

Purpose:

- Validate fused semantics independent of Gluon.
- Provide a fallback for debugging optimized kernel errors.
- Make it easy to inspect generated Triton IR for the basic algorithm.

Expected limitations:

- lower GEMM throughput than vendor-backed hybrid;
- high register pressure if `seq_block * vocab_block` is too large;
- less control over asynchronous data movement.

---

## 9. Hybrid baseline preservation

The hybrid backend should stay close to the original implementation.

Required cleanups:

- rename comments so `hybrid` is not called fused optimized;
- fix `NEG_INF` naming to `BASELINE_ZERO` or equivalent;
- make bias-none path explicit;
- force or assert contiguous hidden/embed/mask before kernels;
- keep current autotuned reduction configs as baseline;
- keep the current backward baseline.

The hybrid backend remains the primary performance baseline because it uses vendor-backed matmul.

---

## 10. Optimized Gluon backward design

Backward remains based on forward-saved `scores` and `indices`.

### 10.1 Work ownership

Each logical CTA owns:

```text
batch block: B_BLOCK_BWD
vocab block: V_BLOCK_BWD
hidden-dim block: D_BLOCK_BWD
```

It loads:

```text
scores[b, v]
grad_out[b, v]
indices[b, v]
embed[v, d]
hidden[b, indices[b, v], d]
```

Then computes:

```text
g = grad_out * exp(-scores) if scores > 0 else 0
```

### 10.2 Bias gradient

Within one CTA:

```text
d_bias_local[v] = sum over batch block of g[b, v]
```

Use one atomic add per vocab element per batch block:

```text
atomic_add(bias_grad[v], d_bias_local[v])
```

### 10.3 Embedding gradient

For each hidden-dim block:

```text
d_embed_local[v, d] = sum_b g[b, v] * hidden[b, indices[b, v], d]
```

Use one atomic add per `(v, d)` per batch block. This is already much better than atomic per `(b, v, d)`.

### 10.4 Hidden gradient

For each `(b, v, d)`:

```text
d_hidden[b, indices[b, v], d] += g[b, v] * embed[v, d]
```

This is an irregular scatter and remains the hardest part. Use regular global atomics first.

Optional local aggregation variant:

```text
for each b in batch block:
    compare indices across vocab lanes
    for equal sequence indices, sum contributions before atomic_add
```

This variant should be benchmarked. It helps when many vocabulary dimensions choose the same token; it may hurt when indices are uniformly distributed.

### 10.5 Removed backward optimizations

Do not use native TMA gather/scatter in production. It is not part of the selected common feature set and imposes extra descriptor/layout/alignment constraints. Backward should use regular vectorized loads/stores/atomics and only use TMA descriptors for regular coalesced tiles when beneficial.

Do not use multi-CTA cluster cooperation or multicast.

### 10.6 Backward variants

```text
B1: current Triton backward baseline
B2: Gluon direct atomic backward
B3: Gluon direct atomic + local d_embed/d_bias aggregation
B4: Gluon direct atomic + optional d_hidden duplicate-index aggregation
```

The initial optimized release can ship with B2/B3 while keeping B1 selectable.

---

## 11. Autotuning and caching

### 11.1 Cache key

Use a resource-and-shape key, not feature key:

```text
backend = optimized
B, S, D, V
hidden dtype
has_bias
requires_grad
sm_count bucket
shared_memory_per_block_optin bucket
total_memory bucket
```

Avoid architecture strings.

### 11.2 Tuning stages

Forward tuning:

```text
1. generate candidate policies from problem + device profile
2. compile candidates lazily
3. benchmark with warmup and repeat
4. validate correctness against hybrid or PyTorch reference
5. store fastest valid policy
```

Backward tuning:

```text
1. tune B_BLOCK_BWD, V_BLOCK_BWD, D_BLOCK_BWD, num_warps
2. benchmark with representative gradient density
3. check correctness against PyTorch reference and hybrid backward
4. store fastest valid policy
```

### 11.3 Guardrails

A candidate is invalid if:

- shared-memory estimate exceeds budget;
- threads exceed max block threads;
- generated kernel fails compilation;
- output differs from reference beyond tolerance;
- memory allocation indicates logits materialization;
- Nsight shows unexpected global stores proportional to `B*S*V_tile`.

---

## 12. Development guide

### 12.1 Phase 0: clean baseline refactor

Tasks:

- move current code into `_backend_hybrid.py` without semantic changes;
- add backend router;
- add explicit backend argument and environment override;
- add tests that prove `backend="hybrid"` matches current behavior;
- fix bias-none handling or mark it unsupported with a clear test.

Exit criteria:

- all current Sparton tests pass;
- `SpartonHead` works unchanged for existing users;
- hybrid speed is unchanged within noise.

### 12.2 Phase 1: PyTorch reference and semantic tests

Implement a pure reference:

```python
def sparton_reference(hidden, embed, bias, mask):
    logits = hidden @ embed.T
    if bias is not None:
        logits = logits + bias
    logits = logits * mask[:, :, None]
    raw_max, idx = logits.max(dim=1)
    scores = torch.log1p(torch.relu(raw_max))
    return scores, idx
```

Important: this reference materializes logits and is only for tests.

Test cases:

- all positive logits;
- all negative logits;
- mixed logits;
- zero mask rows;
- partial masks;
- tie cases;
- with and without bias;
- non-contiguous input converted to contiguous;
- dtype FP16 and BF16;
- small shapes for exact debug;
- realistic SPLADE-like shapes.

### 12.3 Phase 2: naive Triton fused forward

Implement `backend="naive"`:

- one Triton fused forward kernel;
- no Gluon;
- no materialized logits;
- correctness-first, not performance-first.

Exit criteria:

- matches reference/hybrid within tolerance;
- returns same index semantics for strict ties;
- memory peak is lower than hybrid for large `V_tile` cases.

### 12.4 Phase 3: Gluon GEMM-only microkernel

Build a microbenchmark-only Gluon GEMM that computes:

```text
hidden.reshape(B*S, D) @ embed.T
```

It may store an output matrix in this phase because the goal is to validate mainloop throughput, descriptors, layout, and policy generation. This code must not be used by production `optimized` forward.

Exit criteria:

- confirms A/B descriptor layout;
- validates pipeline and warp-specialization code;
- reaches a reasonable fraction of vendor-backed matmul throughput on representative shapes;
- resource model predicts valid policies.

### 12.5 Phase 4: Gluon fused forward O1

Implement non-persistent Gluon fused forward:

- one CTA per logical batch/vocab tile;
- TMA/async staged A/B tiles;
- online max over sequence;
- final score/idx store;
- no logits materialization.

Exit criteria:

- correctness against reference;
- no allocation of `[B,S,V_tile]`;
- stable for all test masks/tails.

### 12.6 Phase 5: Persistent + warp-specialized forward O3/O4

Add:

- persistent work scheduler;
- producer/consumer partitioning;
- multi-buffer pipeline;
- cooperative and pingpong-like schedules;
- resource-derived policy selection.

Exit criteria:

- beats naive fused clearly;
- approaches or beats hybrid on at least one realistic training shape;
- Nsight confirms lower HBM traffic than hybrid;
- no extra global stores for logits.

### 12.7 Phase 6: Gluon backward B2/B3

Implement Gluon backward direct atomic and local aggregation variants.

Exit criteria:

- matches PyTorch/hybrid gradients within tolerance;
- no NaNs under typical AMP training;
- bias-none path correct;
- performance no worse than current Triton backward on target shapes before making it default.

### 12.8 Phase 7: End-to-end integration

Tasks:

- train a small SPLADE/LSR model step with all backends;
- verify retrieval vector sparsity and top-k vocab overlap;
- verify optimizer state and AMP integration;
- add continuous benchmarks.

---

## 13. Profiling plan

### 13.1 Benchmark harness

Use both simple timing and profiler runs.

Simple timing:

```text
triton.testing.do_bench
CUDA events
warmup iterations
median/p20/p80 latency
```

Profiler runs:

```text
Nsight Systems: kernel launches, overlap, synchronization, allocations
Nsight Compute: memory traffic, tensor-core utilization, occupancy, atomics
PyTorch memory stats: peak allocated and reserved bytes
```

### 13.2 Forward metrics

Collect:

- latency;
- peak allocated memory;
- number of kernels launched;
- DRAM bytes read/write;
- tensor-core utilization;
- L2 hit rate;
- achieved occupancy;
- shared-memory throughput;
- register count;
- shared memory per CTA;
- output correctness;
- index match rate;
- nonzero-score match rate.

Expected forward wins of optimized over hybrid:

```text
less HBM traffic because tile logits are not written and reread
fewer kernel launches per vocabulary tile
lower peak temporary memory
potentially larger effective vocabulary tiles because no logits tile allocation
```

Expected risks:

```text
custom Gluon GEMM mainloop may underperform vendor-backed matmul
online max epilogue may reduce tensor-core utilization
register pressure may reduce occupancy
persistent scheduler may not help small shapes
```

### 13.3 Backward metrics

Collect:

- latency;
- atomic throughput;
- global load efficiency;
- L2 hit rate on embed and hidden;
- hidden-gradient atomic conflict rate;
- embedding-gradient atomic write traffic;
- gradient correctness.

Expected backward wins:

```text
better local aggregation for d_embed and d_bias
better descriptor/layout control for coalesced embed loads
possible reduction in atomic operations
```

Expected backward risks:

```text
hidden-gradient scatter remains irregular
local duplicate-index aggregation may cost more than it saves
Gluon backward may initially be slower than current Triton backward
```

---

## 14. Benchmark shape matrix

### 14.1 Unit shapes

```text
B=1, S=4, D=16, V=32
B=2, S=8, D=32, V=64
B=3, S=17, D=48, V=129
```

Purpose: tails, masks, ties, bias, debug.

### 14.2 Development shapes

```text
B=8,  S=128, D=768,  V=30522
B=16, S=128, D=768,  V=30522
B=32, S=128, D=768,  V=30522
B=8,  S=512, D=768,  V=30522
```

Purpose: SPLADE-like runtime behavior.

### 14.3 Stress shapes

```text
B=64,  S=256, D=768,  V=30522
B=128, S=256, D=768,  V=30522
B=16,  S=512, D=1024, V=50257
B=8,   S=512, D=1024, V=250000
```

Purpose: memory pressure, vocabulary scaling, scheduler behavior.

### 14.4 Mask-density sweep

```text
valid tokens per row: 16, 32, 64, 128, 256, 512
mask density: 5%, 10%, 25%, 50%, 100%
```

Purpose: evaluate whether mask density changes optimal tile size or backward atomic conflicts.

---

## 15. Correctness plan

### 15.1 Forward correctness

Compare all backends to PyTorch reference:

```text
hybrid vs reference
naive vs reference
optimized vs reference
```

Metrics:

```text
max_abs_error(scores)
max_rel_error(scores)
index_match_rate for positive scores
nonzero_score_match_rate
topk_vocab_overlap per batch row
```

Tolerances:

```text
FP32 debug: strict
BF16/FP16: dtype-appropriate tolerance
indices: exact where scores are clearly positive and no near-tie exists
```

For near-tie cases, compare scores and allow index ambiguity only if logits are exactly equal or within a small explicit tolerance.

### 15.2 Backward correctness

Use:

- small-shape finite difference tests;
- PyTorch autograd reference;
- hybrid backward comparison;
- random upstream gradients;
- sparse masks;
- all-negative logits;
- tie cases.

Metrics:

```text
hidden_grad error
embed_grad error
bias_grad error
zero-gradient correctness when scores == 0
```

### 15.3 Integration correctness

Use a minimal model step:

```text
transformer output hidden -> SpartonHead -> sparse reps -> loss -> backward -> optimizer step
```

Check:

- no graph breaks beyond expected custom ops;
- AMP works;
- gradients are finite;
- outputs match backend baselines;
- `torch.compile` compatibility where relevant.

---

## 16. Memory validation

The optimized forward must be validated for no materialization.

Add an allocation monitor around forward:

```text
reset peak memory
run backend
record peak allocated
compare with theoretical outputs + ordinary overhead
```

Theoretical output memory:

```text
scores_bytes  = B * V * sizeof(dtype)
indices_bytes = B * V * sizeof(int64)
```

Hybrid temporary memory lower bound:

```text
tile_logits_bytes = B * S * V_tile * sizeof(dtype)
```

Optimized forward should not show allocations proportional to `B*S*V_tile`.

Nsight validation:

- no global store stream with size proportional to `B*S*V_tile`;
- no second kernel reading logits tiles;
- global writes dominated by final `scores` and `indices`.

---

## 17. Performance targets

### 17.1 Milestone targets

```text
M1: optimized forward correct on all unit/dev shapes
M2: optimized forward uses less peak memory than hybrid
M3: optimized forward beats naive fused on all dev shapes
M4: optimized forward matches hybrid within 20% on GEMM-dominated shapes
M5: optimized forward beats hybrid on at least one memory-bound large-S/V shape
M6: optimized backward matches current Triton backward correctness
M7: optimized backward matches or beats current Triton backward on representative shapes
```

### 17.2 Interpreting failure

If optimized forward is slower than hybrid:

```text
case A: GEMM mainloop is slow
    Focus on tile shape, pipeline stages, warp specialization, descriptor layout.

case B: epilogue/reduction dominates
    Reduce seq_block or vocab_block, optimize local max layout, reduce register pressure.

case C: occupancy too low
    Smaller accumulator tile, fewer stages, fewer warps, smaller running state.

case D: memory traffic remains high
    Verify no materialized logits, check descriptor loads, L2 hit rate, output stores.
```

If optimized backward is slower:

```text
case A: hidden scatter atomics dominate
    Try duplicate-index local aggregation.

case B: embed loads dominate
    Increase D_BLOCK or improve coalescing.

case C: d_embed atomics dominate
    Increase batch aggregation per CTA.
```

---

## 18. Implementation notes for Gluon code organization

### 18.1 Compatibility shim

Keep all Gluon import details behind one module:

```text
_sparton_gluon_runtime.py
```

High-level kernels should import only project-local names:

```python
from ._sparton_gluon_runtime import gluon, gl, TensorDescriptor, tma, mbarrier, fence_async_shared
```

Do not scatter third-party namespace details through kernels. This keeps source code free of architecture-named dispatch and makes it easier to update as Gluon APIs evolve.

### 18.2 Kernel names

Good:

```text
sparton_optimized_forward_kernel
sparton_optimized_backward_kernel
sparton_naive_forward_kernel
```

Bad:

```text
names containing GPU generation or compute capability
```

### 18.3 Policy names

Good:

```text
cooperative
pingpong
persistent
```

Bad:

```text
names containing GPU generation or compute capability
```

### 18.4 Fallback behavior

Do not silently fallback from optimized to hybrid due to hardware feature checks.

Allowed fallback cases:

- user explicitly selects `backend="hybrid"`;
- optimized kernel compilation fails during development and a test intentionally marks the case as xfail;
- debug environment variable explicitly requests a baseline backend.

Production default should fail loudly if optimized mode is selected but unavailable.

---

## 19. Risk register

### 19.1 Gluon API stability

Gluon is lower-level and evolving. Keep the implementation isolated behind `_backend_optimized_gluon.py` and `_sparton_gluon_runtime.py`.

Mitigation:

- pin Triton version in development;
- keep a minimal Gluon smoke test;
- keep hybrid backend functional.

### 19.2 Custom GEMM underperforms vendor matmul

The optimized forward only wins if the cost saved by not materializing logits exceeds custom GEMM throughput loss.

Mitigation:

- build GEMM-only Gluon microbench first;
- compare to PyTorch matmul;
- tune tile shapes before integrating epilogue.

### 19.3 Register pressure from online max/argmax

The fused kernel carries accumulators plus running max and index.

Mitigation:

- start with `batch_block=1`;
- keep `seq_block` moderate;
- benchmark `vocab_block=64/128/256`;
- use Nsight register and occupancy metrics.

### 19.4 Backward irregular scatter

Backward hidden-gradient scatter may remain atomic-limited.

Mitigation:

- keep current Triton backward as baseline;
- implement local aggregation variants;
- benchmark with realistic mask and sparsity distributions.

### 19.5 Descriptor/alignment constraints

Tensor descriptors and TMA paths impose alignment and stride requirements.

Mitigation:

- enforce contiguous hidden/embed/mask;
- pad or tail-handle K/V dimensions;
- validate descriptor block shapes in tests;
- keep a small debug mode with simpler loads.

---

## 20. Deliverables

### 20.1 Code deliverables

```text
D1: backend router and cleaned hybrid backend
D2: PyTorch reference implementation and test helpers
D3: naive Triton fused forward baseline
D4: runtime policy generator and autotune cache
D5: Gluon GEMM-only microbenchmark
D6: Gluon optimized fused forward O1
D7: Gluon optimized fused forward O3/O4
D8: Gluon optimized backward B2/B3
D9: benchmark and profiling scripts
D10: CI smoke tests and local performance dashboard
```

### 20.2 Benchmark scripts

```text
bench_forward.py
bench_backward.py
bench_end_to_end.py
profile_nsys.sh
profile_ncu.sh
```

### 20.3 Test files

```text
test_reference.py
test_hybrid_compat.py
test_naive_fused_forward.py
test_optimized_forward.py
test_backward.py
test_sparton_head_integration.py
test_memory_no_materialization.py
```

---

## 21. Summary

The final architecture is:

```text
hybrid:
    original Sparton implementation
    vendor-backed matmul + Triton reduction
    materializes tile logits

naive:
    Triton-only fused implementation
    no Gluon
    no logits materialization
    debug/performance baseline

optimized:
    Gluon-first fused implementation
    TMA/async staged GEMM mainloop
    warp-specialized producer/consumer pipeline
    online max/argmax epilogue
    no logits materialization
    Gluon backward with current saved-state contract
```

The key design rule is:

```text
Use one selected common feature set.
Derive only numeric parameters at runtime.
Do not branch by hardware feature.
Do not keep unsupported or current-platform-inapplicable optimizations in the production design.
```

This keeps the codebase Python-first, close to the current `sparton_kernel.py` integration model, and still gives the optimized path a real chance to beat the hybrid implementation by eliminating `tile_logits` VRAM traffic.

---

## References

1. Sparton source file, `sparton_kernel.py`: https://github.com/thongnt99/sparton/blob/main/src/sparton/sparton_kernel.py
2. Sparton paper: https://arxiv.org/pdf/2603.25011
3. Triton Gluon tutorials index: https://triton-lang.org/main/getting-started/tutorials/gluon/index.html
4. Triton Gluon TMA tutorial: https://triton-lang.org/main/getting-started/tutorials/gluon/tma.html
5. Triton Gluon warp specialization tutorial: https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html
6. Triton `make_tensor_descriptor` API: https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html
7. NVIDIA CUDA GPU compute capability table: https://developer.nvidia.com/cuda/gpus
8. CUTLASS Blackwell functionality documentation, especially GeForce limitations: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
