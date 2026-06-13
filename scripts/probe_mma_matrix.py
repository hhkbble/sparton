"""MMA availability matrix for Gluon on the local GPU.

Each probe compiles and runs a tiny GEMM tile through one MMA family:
  mma_v2  -> ampere mma.sync path (expected to work on sm_120)
  wgmma   -> hopper warpgroup_mma (sm_90a feature, expected to fail on sm_120)
  tcgen05 -> blackwell tcgen05_mma + tensor memory (sm_100a feature, expected to fail)

LLVM codegen failures abort the process, so the driver runs each probe in a
subprocess and records exit status plus the tail of the output.

Validated results on RTX 5090 / Triton 3.6.0:
ARCHITECTURE.md §2.2.
"""

import sys

M, N, K = 64, 64, 32


def build_kernels():
    import triton  # noqa: F401
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia import ampere, hopper, blackwell
    from triton.experimental.gluon.language.nvidia.hopper import mbarrier

    @gluon.jit
    def mma_v2_kernel(a_ptr, b_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
        blocked: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
        mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[4, 1], instr_shape=[16, 8])

        offs_am = gl.arange(0, M, gl.SliceLayout(1, blocked))
        offs_ak = gl.arange(0, K, gl.SliceLayout(0, blocked))
        a_reg = gl.load(a_ptr + offs_am[:, None] * K + offs_ak[None, :])
        offs_bk = gl.arange(0, K, gl.SliceLayout(1, blocked))
        offs_bn = gl.arange(0, N, gl.SliceLayout(0, blocked))
        b_reg = gl.load(b_ptr + offs_bk[:, None] * N + offs_bn[None, :])

        a_sh = gl.allocate_shared_memory(
            gl.float16, [M, K], gl.NVMMASharedLayout.get_default_for([M, K], gl.float16), a_reg)
        b_sh = gl.allocate_shared_memory(
            gl.float16, [K, N], gl.NVMMASharedLayout.get_default_for([K, N], gl.float16), b_reg)

        a = a_sh.load(gl.DotOperandLayout(0, mma, 2))
        b = b_sh.load(gl.DotOperandLayout(1, mma, 2))
        acc = gl.zeros([M, N], gl.float32, mma)
        acc = ampere.mma_v2(a, b, acc)

        offs_cm = gl.arange(0, M, gl.SliceLayout(1, mma))
        offs_cn = gl.arange(0, N, gl.SliceLayout(0, mma))
        gl.store(c_ptr + offs_cm[:, None] * N + offs_cn[None, :], acc)

    @gluon.jit
    def wgmma_kernel(a_ptr, b_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr):
        blocked: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
        mma: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, 64, 16])

        offs_am = gl.arange(0, M, gl.SliceLayout(1, blocked))
        offs_ak = gl.arange(0, K, gl.SliceLayout(0, blocked))
        a_reg = gl.load(a_ptr + offs_am[:, None] * K + offs_ak[None, :])
        offs_bk = gl.arange(0, K, gl.SliceLayout(1, blocked))
        offs_bn = gl.arange(0, N, gl.SliceLayout(0, blocked))
        b_reg = gl.load(b_ptr + offs_bk[:, None] * N + offs_bn[None, :])

        a_sh = gl.allocate_shared_memory(
            gl.float16, [M, K], gl.NVMMASharedLayout.get_default_for([M, K], gl.float16), a_reg)
        b_sh = gl.allocate_shared_memory(
            gl.float16, [K, N], gl.NVMMASharedLayout.get_default_for([K, N], gl.float16), b_reg)

        acc = gl.zeros([M, N], gl.float32, mma)
        acc = hopper.warpgroup_mma(a_sh, b_sh, acc)
        hopper.warpgroup_mma_wait(0, deps=[acc])

        offs_cm = gl.arange(0, M, gl.SliceLayout(1, mma))
        offs_cn = gl.arange(0, N, gl.SliceLayout(0, mma))
        gl.store(c_ptr + offs_cm[:, None] * N + offs_cn[None, :], acc)

    @gluon.jit
    def tcgen05_kernel(a_ptr, b_ptr, c_ptr, M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                       NUM_WARPS: gl.constexpr):
        blocked: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])

        offs_am = gl.arange(0, M, gl.SliceLayout(1, blocked))
        offs_ak = gl.arange(0, K, gl.SliceLayout(0, blocked))
        a_reg = gl.load(a_ptr + offs_am[:, None] * K + offs_ak[None, :])
        offs_bk = gl.arange(0, K, gl.SliceLayout(1, blocked))
        offs_bn = gl.arange(0, N, gl.SliceLayout(0, blocked))
        b_reg = gl.load(b_ptr + offs_bk[:, None] * N + offs_bn[None, :])

        a_sh = gl.allocate_shared_memory(
            gl.float16, [M, K], gl.NVMMASharedLayout.get_default_for([M, K], gl.float16), a_reg)
        b_sh = gl.allocate_shared_memory(
            gl.float16, [K, N], gl.NVMMASharedLayout.get_default_for([K, N], gl.float16), b_reg)

        tmem_layout: gl.constexpr = blackwell.TensorMemoryLayout((M, N), col_stride=1)
        acc_tmem = blackwell.allocate_tensor_memory(gl.float32, [M, N], tmem_layout)
        bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)
        blackwell.tcgen05_mma(a_sh, b_sh, acc_tmem, use_acc=False, mbarriers=[bar])
        mbarrier.wait(bar, 0)

        reg_layout: gl.constexpr = blackwell.get_tmem_reg_layout(gl.float32, [M, N], tmem_layout, NUM_WARPS)
        acc = acc_tmem.load(reg_layout)

        offs_cm = gl.arange(0, M, gl.SliceLayout(1, reg_layout))
        offs_cn = gl.arange(0, N, gl.SliceLayout(0, reg_layout))
        gl.store(c_ptr + offs_cm[:, None] * N + offs_cn[None, :], acc)

    return {"mma_v2": mma_v2_kernel, "wgmma": wgmma_kernel, "tcgen05": tcgen05_kernel}


def run_probe(name: str) -> None:
    import torch

    kernels = build_kernels()
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    if name == "tcgen05":
        kernels[name][(1, )](a, b, c, M, N, K, 4, num_warps=4)
    else:
        kernels[name][(1, )](a, b, c, M, N, K, num_warps=4)
    torch.cuda.synchronize()

    ref = a.float() @ b.float()
    err = (c - ref).abs().max().item()
    print(f"{name}: ran, max_abs_err vs fp32 matmul = {err:.6f}")
    assert err < 1e-2, f"{name}: numerical mismatch"


def main() -> None:
    if len(sys.argv) > 1:
        run_probe(sys.argv[1])
        return

    import subprocess
    for probe in ["mma_v2", "wgmma", "tcgen05"]:
        result = subprocess.run([sys.executable, __file__, probe], capture_output=True, text=True,
                                timeout=900)
        status = "OK" if result.returncode == 0 else f"FAIL rc={result.returncode}"
        print(f"=== {probe}: {status} ===")
        tail = (result.stdout + result.stderr).strip().splitlines()
        for line in tail[-12:]:
            print("   ", line)


if __name__ == "__main__":
    main()
