# Milestone 2 Memo: `bias=None` Backward

## Scope

This task implements milestones 1 and 2 from
`docs/sparton_gluon_design_review.md`:

1. Add a PyTorch reference and semantic tests.
2. Fix `bias=None` backward in the current hybrid code path.

It does not start backend routing, hybrid extraction, naive Triton, or Gluon
implementation work.

## Root Cause

Forward already accepted `bias is None` in
`fused_sparton_fwd_with_indices`, selecting `matmul` instead of
`matmul_bias`.

Backward did not preserve that contract. `fused_sparton_bwd_op` always ran:

```python
bias_grad = torch.zeros_like(bias, dtype=torch.float32)
```

When `SpartonHead(use_bias=False)` passed `None`, backward failed before the
Triton kernel launch with:

```text
TypeError: zeros_like(): argument 'input' must be Tensor, not NoneType
```

The registered custom-op schemas also declared `bias` as `Tensor`, even though
the Python path accepted `None` in eager execution.

## Implementation Notes

- `pyproject.toml` now requires Python `>=3.10`, adds a PEP 735
  `[dependency-groups] test = ["pytest>=9.0,<10"]`, and configures pytest 9
  through native `[tool.pytest]`.
- `tests/test_sparton_kernel.py` adds a PyTorch reference that matches current
  kernel semantics: mask multiplication, zero baseline, strict `>` running-max
  update, `relu`, and `log1p`.
- `sparton::fused_sparton_fwd` now declares `Tensor? bias`.
- `sparton::fused_sparton_bwd` now declares `Tensor? bias` and returns
  `Tensor?` for `bias_grad`.
- The existing Triton backward kernel is reused with a `HAS_BIAS: tl.constexpr`
  branch. Bias-gradient atomics are compiled out for no-bias launches.
- The no-bias backward path passes a scalar float32 dummy pointer to satisfy the
  Triton launch signature, but returns `None` to autograd for the bias gradient.

## Validation

Commands run:

```bash
PYTHONPATH=src /workspace/venvs/sparton/bin/python -m py_compile src/sparton/__init__.py src/sparton/sparton_kernel.py training/model.py training/train.py
TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas /workspace/venvs/sparton/bin/python -m pytest -v
TRITON_PTXAS_PATH=/usr/local/cuda-13.2/bin/ptxas PYTHONPATH=src /workspace/venvs/sparton/bin/python -c "import torch; import sparton.sparton_kernel; print(torch.ops.sparton.fused_sparton_fwd.default._schema); print(torch.ops.sparton.fused_sparton_bwd.default._schema)"
```

Results:

```text
py_compile: passed
pytest: 11 passed, 1 warning
fwd schema: sparton::fused_sparton_fwd(Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor)
bwd schema: sparton::fused_sparton_bwd(Tensor grad_out, Tensor max_scores, Tensor max_idx, Tensor hidden, Tensor embed, Tensor? bias, Tensor mask) -> (Tensor, Tensor, Tensor?)
```

BF16 forward coverage includes both bias and no-bias paths. No forward
implementation change was needed after removing the implementation-related
`bias-bf16` skip; the current `matmul_bias` path passed against the reference.

## Remaining Follow-Ups

- Continue milestone 3 only after preserving current hybrid behavior under the
  new pytest coverage.
