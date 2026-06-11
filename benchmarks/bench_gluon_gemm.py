"""Standalone Gluon GEMM microbenchmark for the Sparton optimized-forward feasibility check.

Computes C = A @ B.T with the production Sparton layouts:
  A = hidden.reshape(B*S, D)   row-major [M, K]
  B = embed                    row-major [V, D] = [N, K], used as conceptual [K, N]

Kernel structure (the candidate optimized-forward mainloop):
  - host TensorDescriptor for A tiles [BLOCK_M, BLOCK_K] and B tiles [BLOCK_N, BLOCK_K]
  - TMA async_copy_global_to_shared staged through NUM_STAGES shared buffers
  - mbarrier expect/wait synchronization (tile t lives in slot t % NUM_STAGES)
  - ampere.mma_v2 (the only MMA family that works on sm_120)
  - B tile transposed via shared-memory permute, consumed as DotOperandLayout rhs

Correctness vs fp32 matmul, throughput vs torch.matmul (cuBLAS).
The driver runs each config in a subprocess with a timeout so a deadlocked
kernel is killed and reported rather than hanging the run.

Validated results on RTX 5090 / Triton 3.6.0:
docs/sparton_gluon_remaining_work_design.md §3.1 and Appendix A.
"""

import argparse
import subprocess
import sys

CONFIGS = [
    # BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, WARPS_M, WARPS_N
    (128, 128, 64, 3, 4, 2),
    (128, 128, 64, 2, 4, 2),
    (128, 64, 64, 4, 4, 2),
    (64, 128, 64, 3, 2, 4),
    (128, 256, 64, 2, 4, 2),
    (128, 128, 32, 4, 4, 2),
]


def build_kernel():
    import triton  # noqa: F401
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia import ampere
    from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma

    @gluon.jit
    def gemm_abt_kernel(a_desc, b_desc, c_ptr, M, N, K,  #
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

    return gemm_abt_kernel


def run_config(idx: int, M: int, N: int, K: int) -> None:
    import torch
    import triton
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
    from triton.experimental.gluon.language._layouts import NVMMASharedLayout

    BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, WARPS_M, WARPS_N = CONFIGS[idx]
    kernel = build_kernel()

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16) * 0.05
    b = torch.randn((N, K), device="cuda", dtype=torch.float16) * 0.05
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    smem_layout = NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    a_desc = TensorDescriptor.from_tensor(a, [BLOCK_M, BLOCK_K], smem_layout)
    b_desc = TensorDescriptor.from_tensor(b, [BLOCK_N, BLOCK_K], smem_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    num_warps = WARPS_M * WARPS_N

    def launch():
        kernel[grid](a_desc, b_desc, c, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES,
                     WARPS_M, WARPS_N, num_warps=num_warps)

    launch()
    torch.cuda.synchronize()

    ref32 = a.float() @ b.float().T
    ref16 = torch.matmul(a, b.T)
    err_gluon = (c.float() - ref32).abs().max().item()
    err_cublas = (ref16.float() - ref32).abs().max().item()

    flops = 2.0 * M * N * K
    ms_gluon = triton.testing.do_bench(launch, warmup=25, rep=100)
    ms_cublas = triton.testing.do_bench(lambda: torch.matmul(a, b.T), warmup=25, rep=100)
    tf_gluon = flops / ms_gluon * 1e-9
    tf_cublas = flops / ms_cublas * 1e-9
    print(f"BM={BLOCK_M} BN={BLOCK_N} BK={BLOCK_K} stages={NUM_STAGES} warps={WARPS_M}x{WARPS_N}: "
          f"gluon {ms_gluon:.3f} ms ({tf_gluon:.1f} TFLOP/s) | "
          f"cublas {ms_cublas:.3f} ms ({tf_cublas:.1f} TFLOP/s) | "
          f"ratio {tf_gluon / tf_cublas * 100:.0f}% | "
          f"max_err gluon={err_gluon:.4f} cublas={err_cublas:.4f}",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=int, default=None)
    parser.add_argument("--M", type=int, default=4096)   # B*S = 32*128
    parser.add_argument("--N", type=int, default=30522)  # V (BERT vocab)
    parser.add_argument("--K", type=int, default=768)    # D
    args = parser.parse_args()

    if args.config is not None:
        run_config(args.config, args.M, args.N, args.K)
        return

    print(f"GEMM A[{args.M},{args.K}] @ B[{args.N},{args.K}].T  fp16 in, fp32 acc", flush=True)
    for idx, cfg in enumerate(CONFIGS):
        cmd = [sys.executable, "-u", __file__, "--config", str(idx),
               "--M", str(args.M), "--N", str(args.N), "--K", str(args.K)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=420)
        except subprocess.TimeoutExpired:
            print(f"config {cfg}: TIMEOUT (likely kernel deadlock) — killed", flush=True)
            continue
        out = (result.stdout + result.stderr).strip().splitlines()
        if result.returncode == 0:
            for line in out:
                if line.startswith("BM="):
                    print(line, flush=True)
        else:
            tail = out[-1] if out else "no output"
            print(f"config {cfg}: FAILED rc={result.returncode}: {tail}", flush=True)


if __name__ == "__main__":
    main()
