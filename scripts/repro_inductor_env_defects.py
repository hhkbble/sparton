"""Reproducer for the two cache-cold TorchInductor environment defects.

Run with TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 to force fresh compiles:
  - without CPATH=/usr/local/cuda-13.2/include: gcc fails building Triton's
    cuda_utils (`fatal error: cuda.h`) because the NVIDIA Triton wheel ships
    no bundled CUDA headers;
  - with CPATH but the Inductor cache under noexec /tmp: dlopen fails with
    "failed to map segment from shared object".
Both pass with the hardened env prefix. Mechanism and fix:
ARCHITECTURE.md §2.4.
"""

import torch

@torch.compile
def matmul(a, b):
    return a @ b.T

@torch.compile
def matmul_bias(a, b, c):
    return a @ b.T + c

a = torch.randn(32, 128, 768, device="cuda", dtype=torch.float16)
b = torch.randn(8192, 768, device="cuda", dtype=torch.float16)
c = torch.randn(8192, device="cuda", dtype=torch.float16)
matmul(a, b)
matmul_bias(a, b, c)
torch.cuda.synchronize()
print("compile OK")
