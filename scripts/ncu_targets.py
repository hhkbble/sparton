"""NVTX-wrapped profiling targets for the hybrid path on the dev shape.

Provides ranges for `ncu --nvtx --nvtx-include "<range>/"`:
  cublas/            one full-V torch.matmul
  hybrid_fwd/        one fused_sparton_fwd_op call (all per-tile kernels)
  hybrid_bwd/        tensor.backward() — captures NOTHING under ncu: autograd
                     runs on a worker thread and NVTX ranges are thread-local
                     (kept as a documented negative example; see ARCHITECTURE.md §2.5)
  hybrid_bwd_direct/ fused_sparton_bwd_op called directly on the main thread

Needs PYTHONPATH=src and the hardened env prefix; see
ARCHITECTURE.md §2.5 (profilers and measurement regimes).
"""

import torch
import sparton.sparton_kernel as sk

B, S, D, V = 32, 128, 768, 30522
torch.manual_seed(0)
hidden = torch.randn(B, S, D, device="cuda", dtype=torch.float16) * 0.05
embed = torch.randn(V, D, device="cuda", dtype=torch.float16) * 0.05
bias = torch.randn(V, device="cuda", dtype=torch.float16) * 0.05
mask = (torch.rand(B, S, device="cuda") > 0.25).to(torch.int32)
hf = hidden.reshape(B * S, D)

for _ in range(3):  # warm compile + fwd autotune (cached) + cuBLAS heuristics
    torch.matmul(hf, embed.T)
    sk.fused_sparton_fwd_op(hidden, embed, bias, mask)
h = hidden.detach().requires_grad_(True)
e = embed.detach().requires_grad_(True)
b2 = bias.detach().requires_grad_(True)
s, _ = sk.fused_sparton_fwd_op(h, e, b2, mask)
s.float().sum().backward()  # warm bwd autotune
torch.cuda.synchronize()

torch.cuda.nvtx.range_push("cublas")
torch.matmul(hf, embed.T)
torch.cuda.nvtx.range_pop()

torch.cuda.nvtx.range_push("hybrid_fwd")
sk.fused_sparton_fwd_op(hidden, embed, bias, mask)
torch.cuda.nvtx.range_pop()

h = hidden.detach().requires_grad_(True)
e = embed.detach().requires_grad_(True)
b2 = bias.detach().requires_grad_(True)
s, _ = sk.fused_sparton_fwd_op(h, e, b2, mask)
g = torch.ones_like(s)
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("hybrid_bwd")
s.backward(g)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print("targets done")

scores, idx = sk.fused_sparton_fwd_op(hidden, embed, bias, mask)
g2 = torch.ones_like(scores)
torch.cuda.synchronize()
torch.cuda.nvtx.range_push("hybrid_bwd_direct")
sk.fused_sparton_bwd_op(g2, scores, idx, hidden, embed, bias, mask)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print("bwd_direct done")
