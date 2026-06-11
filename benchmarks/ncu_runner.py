"""Fixed-config Gluon GEMM launcher for ncu/nsys profiling.

Launches CONFIGS[0] of bench_gluon_gemm six times back-to-back so a profiler
can skip warm-up launches (e.g. `ncu --launch-skip 4 --launch-count 1`).
Run from the benchmarks/ directory with the hardened env prefix; see
docs/sparton_gluon_remaining_work_design.md §11 and Appendix B.
"""

import torch, triton
from bench_gluon_gemm import build_kernel, CONFIGS
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language._layouts import NVMMASharedLayout

M, N, K = 4096, 30522, 768
BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, WARPS_M, WARPS_N = CONFIGS[0]
kernel = build_kernel()
torch.manual_seed(0)
a = torch.randn((M, K), device="cuda", dtype=torch.float16) * 0.05
b = torch.randn((N, K), device="cuda", dtype=torch.float16) * 0.05
c = torch.empty((M, N), device="cuda", dtype=torch.float16)
lay = NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K], lay)
b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K], lay)
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
for _ in range(6):
    kernel[grid](a_desc, b_desc, c, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, WARPS_M, WARPS_N, num_warps=WARPS_M*WARPS_N)
torch.cuda.synchronize()
print("done")
