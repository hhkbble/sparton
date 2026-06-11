# Review of Sparton Gluon-First Refactor Design

This document reviews `docs/sparton_gluon_current_platform_design.md` against
the current repository source, the local runtime, and targeted CUDA/Gluon smoke
checks. It is intentionally an audit document, not an implementation plan.

## Executive verdict

The design is directionally useful, but it is not decision-safe as written for a
production refactor. The current code is a working hybrid baseline with one
confirmed `bias=None` backward bug. The proposed `optimized` Gluon backend is
plausible as a research track, but should not become the public default until a
known-good Gluon MMA microkernel compiles and runs on the target GPU class.

Recommended correction:

1. Keep the first refactor default behavior equivalent to the current hybrid
   implementation.
2. Add backend routing only after tests prove the moved hybrid backend is
   behaviorally unchanged.
3. Treat Gluon as experimental until forward and backward correctness,
   compilation, and performance are independently proven.

## Source-grounded findings

### Current implementation is hybrid

The current runtime path is:

```text
SpartonHead.forward
  -> fused_sparton_fwd_op
  -> fused_sparton_fwd_with_indices
  -> torch.compile matmul/matmul_bias per vocab tile
  -> Triton reduce_seq_max_log1p_relu_with_indices
```

Relevant source:

- `src/sparton/sparton_kernel.py:204` defines compiled `matmul`.
- `src/sparton/sparton_kernel.py:208` defines compiled `matmul_bias`.
- `src/sparton/sparton_kernel.py:226` starts
  `fused_sparton_fwd_with_indices`.
- `src/sparton/sparton_kernel.py:249` calls the Triton reduction.
- `src/sparton/sparton_kernel.py:643` registers
  `sparton::fused_sparton_fwd`.
- `src/sparton/sparton_kernel.py:707` registers custom autograd.
- `src/sparton/sparton_kernel.py:710` defines `SpartonHead`.

The design document's `hybrid` taxonomy matches the actual source. The current
implementation materializes `[B, S, V_tile]` logits and uses the Triton
reduction to produce `[B, V]` scores and `[B, V]` indices.

### `optimized` as immediate default is unsafe

The design proposes:

```python
def __init__(..., backend: str = "optimized")
```

The current public class has no `backend` argument:

```python
class SpartonHead(nn.Module):
    def __init__(self, vocab_size, hidden_dim, use_bias=False):
```

Changing the default to `optimized` would silently change existing users from
the known hybrid path to an unimplemented experimental path. That is a public
behavior change, even if the constructor remains import-compatible.

Correction:

- Add `backend` only with `backend="hybrid"` as the first shipped default, or
  make `"optimized"` opt-in behind tests and explicit documentation.
- Promote `"optimized"` to default only after it passes the same correctness
  and performance gates as hybrid on target shapes.

### Backend dispatch is underspecified

The design adds a constructor-level backend argument but keeps the custom op
call backend-less:

```python
scores, _idx = fused_sparton_fwd_op(hidden_states, self.weight, self.bias, attention_mask)
```

If implemented literally, the selected backend would not affect execution.

Correction:

- `SpartonHead.forward` must route through a backend-aware Python wrapper, or
  the custom op schema must include backend selection explicitly.
- Prefer backend-specific Python wrappers around custom ops so the public
  module can keep `SpartonHead` simple and so backend-specific custom op schemas
  can diverge if needed.
- Do not pass a string backend into a performance-critical kernel unless it is
  resolved before launch.

### `bias=None` is a confirmed current bug in backward

Current forward handles `bias is None`:

```python
if bias is not None:
    tile_logits = matmul_bias(...)
else:
    tile_logits = matmul(...)
```

Current backward does not:

```python
bias_grad = torch.zeros_like(bias, dtype=torch.float32)
```

Observed behavior in this workspace:

```text
use_bias=True:  forward and backward pass
use_bias=False: forward passes, backward fails
TypeError: zeros_like(): argument 'input' must be Tensor, not NoneType
```

Installed Torch supports optional tensor schemas for custom ops:

```text
Optional[torch.Tensor] -> Tensor?
```

Also, `ctx.save_for_backward(..., None)` works in the installed runtime.

Correction:

- Type the Python custom op argument as `Optional[torch.Tensor]` if the declared
  Torch support matrix permits it.
- Return `None` for the bias gradient when `bias is None`.
- If Torch `2.7.1` lacks optional custom-op support, then split bias and
  no-bias custom ops. This should be a compatibility decision, not the default
  assumption.

### Reference implementation is score-equivalent but not index-equivalent

The design's reference:

```python
logits = hidden @ embed.T
if bias is not None:
    logits = logits + bias
logits = logits * mask[:, :, None]
raw_max, idx = logits.max(dim=1)
scores = torch.log1p(torch.relu(raw_max))
```

This matches scores for normal mask semantics, but it does not fully match
current index semantics. The current Triton reduction initializes running max to
`0.0` and only updates on strict `>`:

```text
running_max = 0
running_idx = 0
if tile_value > running_max:
    update value and index
```

Consequences:

- All-negative valid logits produce score `0` and index `0`, regardless of
  which token has the least-negative logit.
- All-zero logits produce score `0` and index `0`.
- Masked positions can effectively contribute zero to the score baseline, but
  strict `>` prevents equal-zero updates.
- Index equality is only meaningful for positive non-tied winning logits.

Correction:

- Implement a test reference that reproduces baseline-zero, strict-improvement
  index behavior when indices are under test.
- Or compare exact indices only where scores are positive and the winning logit
  is not tied or near-tied.

### Mask contract should be explicit

The implementation multiplies logits by the mask value:

```python
logits_tile = logits_tile * mask_vals[:, None]
```

The design should state that `mask` is expected to be a 0/1 attention mask. If
non-binary mask values are passed, the operation becomes weighted scaling rather
than masking. That behavior may be mathematically valid, but it is not the
usual Hugging Face attention-mask contract and should not be accidental.

### No-feature-gate rule conflicts with Gluon reality

The design says production should use one selected feature set and should not
branch on hardware features. That rule is clean architecturally, but it is the
largest feasibility issue.

Local facts:

```text
GPU: NVIDIA GeForce RTX 5090
CUDA capability: 12.0
Triton target arch: 120
Torch: 2.12.0a0+0291f960b6.nv26.04.48445190
Triton: 3.6.0
CUDA runtime reported by Torch: 13.2
```

Triton Gluon exposes separate NVIDIA namespaces for Hopper and Blackwell:

```text
triton.experimental.gluon.language.nvidia.hopper
triton.experimental.gluon.language.nvidia.blackwell
```

The WGMMA tutorial is Hopper-oriented, while TCGen05 is Blackwell-oriented. In
the local API, Hopper exposes `warpgroup_mma`; Blackwell exposes `tcgen05_mma`
and tensor memory descriptors. That is not a single implementation-neutral MMA
surface.

Correction:

- Either narrow the `optimized` backend to one minimum GPU family and fail
  clearly elsewhere, or allow architecture/capability dispatch inside a private
  `_sparton_gluon_runtime.py` compatibility layer.
- Keep public backend names architecture-neutral, but do not forbid internal
  capability checks needed to call the correct experimental API.
- Do not claim one source path can cover WGMMA and TCGen05 until a shared local
  abstraction is proven by compiling examples for the target GPU classes.

### Gluon API instability is concrete

The design's compatibility shim is necessary, not optional. The local Triton
`3.6.0` API does not exactly match online snippets.

Examples from the installed runtime:

- Local Hopper/Blackwell TMA exposes `tma.async_copy_global_to_shared`.
- Current tutorial snippets use names such as `tma.async_load`.
- Local tensor descriptors can come from
  `triton.experimental.gluon.nvidia.hopper.TensorDescriptor`.
- Local Gluon language symbols live under
  `triton.experimental.gluon.language`, not `triton.language`.

Correction:

- Add `_sparton_gluon_runtime.py` before writing production kernels.
- Pin or record the exact Triton version used for development.
- Keep all direct Gluon imports behind the shim.
- Include a minimal Gluon compile smoke test in CI or local validation.

## Feasibility assessment

### High feasibility: hybrid refactor

Moving the current implementation into `_backend_hybrid.py` is feasible if done
mechanically and validated before behavior changes. This should be the first
phase.

Required gates:

- Current `SpartonHead` behavior unchanged.
- Forward scores match the current implementation.
- Backward gradients match the current implementation for bias and no-bias
  cases after fixing the no-bias bug.
- Kernel autotune configs and custom-op/autograd registration remain stable.

### High feasibility: `bias=None` correction

The bug is localized. The only compatibility uncertainty is whether the minimum
declared Torch version supports optional custom-op schemas. The installed Torch
does.

Recommended implementation shape:

- Use `Optional[torch.Tensor]` if supported across the declared support matrix.
- Allocate `bias_grad` only when `bias is not None`.
- Return `None` for bias gradient in autograd when `bias is None`.
- Add explicit forward/backward tests for both bias and no-bias modes.

### Medium feasibility: naive Triton fused forward

A single Triton kernel that loops over `S` and `D` with `tl.dot` is feasible as
a correctness/debug baseline. It should be expected to underperform the hybrid
path on GEMM-heavy shapes because the hybrid path delegates matmul to
vendor-backed PyTorch/TorchInductor kernels.

Recommended scope:

- Forward only at first.
- Correct score and index semantics before optimization.
- Use it to validate no-logits-materialization behavior and isolate Gluon bugs.

### Medium-to-high risk: optimized Gluon forward

Basic Gluon works in this workspace. A minimal Gluon load/store kernel compiled
and ran. A minimal Gluon TMA copy kernel also compiled and ran.

However, the optimized path depends on a correct high-throughput MMA mainloop.
That remains unverified. A hand-adapted TCGen05 smoke failed during LLVM codegen:

```text
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st
```

This does not prove TCGen05 is impossible. It does prove that the current design
should not treat the optimized MMA path as implementation-ready without a
known-good local Gluon example.

Required gates before integration:

- Compile and validate a standalone Gluon GEMM microkernel on RTX 5090.
- Compare against `torch.matmul` for correctness and throughput.
- Confirm the chosen MMA API, descriptor layout, and synchronization protocol.
- Only then fuse online max/argmax into the epilogue.

### Medium risk: optimized Gluon backward

The backward math is clear and atomics make a direct implementation plausible.
Performance is uncertain because `d_hidden` is an irregular scatter and can be
atomic-conflict limited.

Recommended ordering:

1. Keep current Triton backward as the default baseline.
2. Implement Gluon direct atomic backward as correctness-first.
3. Add local aggregation only after profiling realistic sparsity and index
   distributions.
4. Promote only if it matches correctness and does not regress target-shape
   performance.

## Validation performed

### Static/source inspection

Inspected:

- `docs/sparton_gluon_current_platform_design.md`
- `src/sparton/sparton_kernel.py`
- `src/sparton/__init__.py`
- `training/model.py`
- `pyproject.toml`
- Installed Triton Gluon modules under
  `/usr/local/lib/python3.12/dist-packages/triton/experimental/gluon`

### Runtime environment check

Observed in `/workspace/venvs/sparton`:

```text
Python: 3.12.3
Torch: 2.12.0a0+0291f960b6.nv26.04.48445190
Triton: 3.6.0
CUDA available: True
Torch CUDA: 13.2
GPU: NVIDIA GeForce RTX 5090
CUDA capability: (12, 0)
Triton target: GPUTarget(backend='cuda', arch=120, warp_size=32)
SM count: 170
warp size: 32
max threads per block: 1024
max threads per SM: 1536
shared memory per block: 49152
shared memory per block opt-in: 101376
shared memory per SM: 102400
```

### Current Sparton smoke

Command shape:

```text
B=2, S=4, D=8, V=16
hidden dtype=float16, CUDA
mask dtype=int64, CUDA
```

Result:

```text
use_bias=True:
  forward shape: (2, 16)
  output dtype: torch.float16
  backward: ok

use_bias=False:
  forward shape: (2, 16)
  output dtype: torch.float16
  backward: failed
  error: TypeError from torch.zeros_like(None)
```

### Current Sparton vs PyTorch reference

For a small biased FP16 CUDA case:

```text
score max abs error: 0.0
hidden grad max abs error: 0.00048828125
weight grad max abs error: 0.001953125
bias grad max abs error: 0.0009765625
score allclose: True
```

These gradient differences are consistent with FP16/Triton accumulation and do
not indicate a correctness failure for the tested shape.

### Index semantics probes

Observed:

```text
all_equal_positive:
  scores ~= log1p(1)
  idx = 0

all_negative:
  scores = 0
  idx = 0

first_valid_late with positive logits:
  scores ~= log1p(1)
  idx = first valid positive sequence index
```

This confirms strict `>` tie handling and baseline-zero behavior.

### Custom op optional tensor probe

Installed Torch accepted:

```python
Optional[torch.Tensor]
```

as:

```text
Tensor?
```

for `torch.library.custom_op`. Calling the custom op with `None` succeeded in
the probe. This should still be checked against the declared minimum Torch
version before making it the only implementation.

### Gluon smoke checks

Passed:

- Minimal Gluon global load/store kernel on RTX 5090.
- Minimal Gluon TMA copy kernel using local
  `tma.async_copy_global_to_shared`.

Failed:

- Hand-adapted Blackwell TCGen05 GEMM smoke failed during LLVM codegen:

```text
LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.tcgen05.wait.st
```

Interpretation:

- Basic Gluon compilation and TMA are available on this workspace.
- The optimized MMA path remains unproven.
- The design should require a known-good standalone MMA microbenchmark before
  any fused optimized forward work.

## Required design corrections

1. Change the initial public default from `backend="optimized"` to
   `backend="hybrid"`, or defer adding a default backend argument until routing
   exists.
2. Specify backend dispatch explicitly. A constructor argument alone is not
   enough.
3. Treat `bias=None` backward as a required correctness fix, not a cleanup.
4. Update reference tests to handle baseline-zero index semantics.
5. Make binary mask semantics explicit.
6. Relax or reframe the no-feature-gate rule. Keep public names
   architecture-neutral, but allow private compatibility logic where Gluon APIs
   diverge by GPU family.
7. Make `_sparton_gluon_runtime.py` mandatory and version-pinned.
8. Add a standalone Gluon GEMM milestone before the fused optimized forward.
9. Do not make Gluon backward default until forward is stable and backward has
   been compared against current Triton backward.

## Suggested revised milestone order

1. Add tests and PyTorch reference with correct index policy.
2. Fix `bias=None` backward in current code.
3. Move current implementation into a `hybrid` backend without behavior change.
4. Add explicit backend routing with `hybrid` default.
5. Add `naive` Triton fused forward as a debug baseline.
6. Add `_sparton_gluon_runtime.py` and minimal Gluon CI/local smoke tests.
7. Build and validate standalone Gluon GEMM on target GPU.
8. Implement Gluon fused forward only after GEMM is proven.
9. Implement Gluon backward only after optimized forward is stable.
10. Consider changing the default backend only after correctness and performance
    data justify it.

## References

- [Sparton source file](https://github.com/thongnt99/sparton/blob/main/src/sparton/sparton_kernel.py)
- [Sparton paper](https://arxiv.org/pdf/2603.25011)
- [Triton Gluon tutorials index](https://triton-lang.org/main/getting-started/tutorials/gluon/index.html)
- [Triton Gluon TMA tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/tma.html)
- [Triton Gluon WGMMA tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/wgmma.html)
- [Triton Gluon warp specialization tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/warp-specialization.html)
- [Triton Gluon TCGen05 tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/tcgen05.html)
- [Triton `make_tensor_descriptor` API](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html)
- [NVIDIA CUDA GPU compute capability table](https://developer.nvidia.com/cuda/gpus)
- [CUTLASS Blackwell functionality notes](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
