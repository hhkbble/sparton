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
The driver uses Triton/Gluon autotune over a bounded, runtime-pruned policy
set and gates the selected result against cuBLAS.

Validated results on RTX 5090 / Triton 3.6.0:
docs/sparton_gluon_remaining_work_design.md §3.1 and Appendix A.
"""

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sparton._gluon_runtime import is_gluon_backend_available  # noqa: E402
from sparton._gluon_policy_runtime import (  # noqa: E402
    configs_for_policies,
    element_ty_for_dtype,
    make_descriptor_bank,
    policy_from_config,
    prune_configs_for_policies,
)
from sparton._runtime_policy import (  # noqa: E402
    DeviceProfile,
    ProblemSpec,
    generate_gluon_gemm_policies,
    gluon_gemm_policy_universe,
    torch_device_profile,
)

MAX_GEMM_AUTOTUNE_POLICIES = 12


def require_gluon_backend() -> None:
    available, reason = is_gluon_backend_available()
    if not available:
        raise RuntimeError(
            "bench_gluon_gemm.py requires the optimized Gluon backend "
            f"(CUDA sm_80+ and importable triton.experimental.gluon): {reason}"
        )


def parse_dtype(value: str):
    import torch

    normalized = value.strip().lower()
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise argparse.ArgumentTypeError("dtype must be fp16 or bf16")


def dtype_name(dtype) -> str:
    import torch

    if dtype is torch.float16:
        return "fp16"
    if dtype is torch.bfloat16:
        return "bf16"
    return str(dtype).replace("torch.", "")


def policies_for_args(args):
    problem = ProblemSpec(args.M, args.N, args.K, args.dtype)
    if args.device_profile == "actual":
        device = torch_device_profile()
    else:
        device = DeviceProfile(
            sm_count=170,
            warp_size=32,
            max_threads_per_block=1024,
            max_threads_per_sm=1536,
            shared_memory_per_block_optin=101376,
            shared_memory_per_sm=102400,
            capability_major=12,
            capability_minor=0,
            device_name="NVIDIA GeForce RTX 5090",
            shared_memory_per_block=49152,
            regs_per_sm=65536,
            l2_cache_size=100663296,
            memory_bus_width=512,
            total_memory=34190458880,
        )
    return generate_gluon_gemm_policies(
        problem,
        device,
        include_block_n_256=args.include_block_n_256,
    )


def gemm_autotune_policy_universe():
    """Return the fixed descriptor-slot universe for the GEMM benchmark."""

    return gluon_gemm_policy_universe(include_block_n_256=True)


def gemm_autotune_configs_for_policies(policies):
    if len(policies) > MAX_GEMM_AUTOTUNE_POLICIES:
        raise RuntimeError(
            f"bench_gluon_gemm.py supports at most {MAX_GEMM_AUTOTUNE_POLICIES} "
            f"autotune policies, got {len(policies)}"
        )
    return configs_for_policies(policies)


def policy_from_gemm_autotune_config(config, policies=None):
    policy_bank = gemm_autotune_policy_universe() if policies is None else policies
    return policy_from_config(config, policy_bank)


def prune_gemm_autotune_configs(configs, args):
    active_policies = policies_for_args(args)
    return prune_configs_for_policies(
        configs,
        active_policies,
        gemm_autotune_policy_universe(),
    )


def _autotune_cache_key(kernel, positional_args, keyword_args):
    arg_names = getattr(kernel, "arg_names", None)
    keys = getattr(kernel, "keys", None)
    if arg_names is None or keys is None:
        return None
    named_args = dict(zip(arg_names, positional_args))
    all_args = {**named_args, **keyword_args}
    used_args = {key: value for key, value in all_args.items() if key in arg_names}
    try:
        cache_key = [used_args[key] for key in keys if key in used_args]
        for _name, arg in used_args.items():
            if hasattr(arg, "dtype"):
                cache_key.append(str(arg.dtype))
        return tuple(cache_key)
    except (KeyError, TypeError):
        return None


def build_autotuned_kernel(args):
    import triton  # noqa: F401
    from sparton._gluon_runtime import autotune, gl, gluon, mbarrier, mma_v2, tma

    policy_bank = gemm_autotune_policy_universe()
    configs = gemm_autotune_configs_for_policies(policy_bank)

    def early_config_prune(configs, named_args, **kwargs):
        return prune_gemm_autotune_configs(configs, args)

    # The TMA + mma_v2 mainloop below intentionally mirrors the optimized
    # forward kernel in src/sparton/_backend_optimized_gluon.py; the kernels
    # stay separate because this one materializes C while the production
    # kernel runs the Sparton max/argmax epilogue. See
    # docs/sparton_remaining_work_design_v2.md (D2) before deduplicating.
    @autotune(
        configs=configs,
        key=["M", "N", "K"],
        prune_configs_by={"early_config_prune": early_config_prune},
        cache_results=True,
    )
    @gluon.jit
    def gemm_abt_autotuned_kernel(
        a_desc_0, b_desc_0,
        a_desc_1, b_desc_1,
        a_desc_2, b_desc_2,
        a_desc_3, b_desc_3,
        a_desc_4, b_desc_4,
        a_desc_5, b_desc_5,
        a_desc_6, b_desc_6,
        a_desc_7, b_desc_7,
        a_desc_8, b_desc_8,
        a_desc_9, b_desc_9,
        a_desc_10, b_desc_10,
        a_desc_11, b_desc_11,
        c_ptr,
        M,
        N,
        K,
        ELEMENT_TY: gl.constexpr,
        POLICY_ID: gl.constexpr,
        BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr,
        BLOCK_K: gl.constexpr,
        NUM_STAGES: gl.constexpr,
        WARPS_M: gl.constexpr,
        WARPS_N: gl.constexpr,
    ):
        a_desc = a_desc_0
        b_desc = b_desc_0
        if POLICY_ID == 1:
            a_desc = a_desc_1
            b_desc = b_desc_1
        if POLICY_ID == 2:
            a_desc = a_desc_2
            b_desc = b_desc_2
        if POLICY_ID == 3:
            a_desc = a_desc_3
            b_desc = b_desc_3
        if POLICY_ID == 4:
            a_desc = a_desc_4
            b_desc = b_desc_4
        if POLICY_ID == 5:
            a_desc = a_desc_5
            b_desc = b_desc_5
        if POLICY_ID == 6:
            a_desc = a_desc_6
            b_desc = b_desc_6
        if POLICY_ID == 7:
            a_desc = a_desc_7
            b_desc = b_desc_7
        if POLICY_ID == 8:
            a_desc = a_desc_8
            b_desc = b_desc_8
        if POLICY_ID == 9:
            a_desc = a_desc_9
            b_desc = b_desc_9
        if POLICY_ID == 10:
            a_desc = a_desc_10
            b_desc = b_desc_10
        if POLICY_ID == 11:
            a_desc = a_desc_11
            b_desc = b_desc_11

        pid_m = gl.program_id(0)
        pid_n = gl.program_id(1)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        mma: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[WARPS_M, WARPS_N],
                                                      instr_shape=[16, 8])
        dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 2)
        dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 2)

        a_smem = gl.allocate_shared_memory(ELEMENT_TY, [NUM_STAGES, BLOCK_M, BLOCK_K], a_desc.layout)
        b_smem = gl.allocate_shared_memory(ELEMENT_TY, [NUM_STAGES, BLOCK_N, BLOCK_K], b_desc.layout)
        bars = gl.allocate_shared_memory(gl.int64, [NUM_STAGES, 1], mbarrier.MBarrierLayout())
        for i in gl.static_range(NUM_STAGES):
            mbarrier.init(bars.index(i), count=1)

        k_tiles = gl.cdiv(K, BLOCK_K)
        NBYTES: gl.constexpr = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * 2

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
            gl.barrier()
            nk = kt + (NUM_STAGES - 1)
            if nk < k_tiles:
                nbuf = nk % NUM_STAGES
                bar = bars.index(nbuf)
                mbarrier.expect(bar, NBYTES)
                tma.async_copy_global_to_shared(a_desc, [off_m, nk * BLOCK_K], bar, a_smem.index(nbuf))
                tma.async_copy_global_to_shared(b_desc, [off_n, nk * BLOCK_K], bar, b_smem.index(nbuf))
            acc = mma_v2(a, b, acc)

        offs_cm = off_m + gl.arange(0, BLOCK_M, gl.SliceLayout(1, mma))
        offs_cn = off_n + gl.arange(0, BLOCK_N, gl.SliceLayout(0, mma))
        mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        gl.store(c_ptr + offs_cm[:, None].to(gl.int64) * N + offs_cn[None, :], acc.to(ELEMENT_TY), mask)

    return gemm_abt_autotuned_kernel


def run_benchmark(args) -> None:
    import torch
    import triton

    policy_bank = gemm_autotune_policy_universe()
    active_policies = policies_for_args(args)
    dtype = parse_dtype(args.dtype)
    element_ty = element_ty_for_dtype(dtype)

    torch.manual_seed(0)
    a = torch.randn((args.M, args.K), device="cuda", dtype=dtype) * 0.05
    b = torch.randn((args.N, args.K), device="cuda", dtype=dtype) * 0.05
    c = torch.empty((args.M, args.N), device="cuda", dtype=dtype)
    descriptor_bank = make_descriptor_bank(
        a,
        b,
        policy_bank,
        max_policies=MAX_GEMM_AUTOTUNE_POLICIES,
    )
    kernel = build_autotuned_kernel(args)

    def grid(meta):
        return (triton.cdiv(args.M, meta["BLOCK_M"]), triton.cdiv(args.N, meta["BLOCK_N"]))

    launch_args = (
        *descriptor_bank,
        c,
        args.M,
        args.N,
        args.K,
        element_ty,
    )
    launch_kwargs = {}
    cache_key = _autotune_cache_key(kernel, launch_args, launch_kwargs)
    cache = getattr(kernel, "cache", None)
    in_process_cache_hit = isinstance(cache, dict) and cache_key is not None and cache_key in cache

    def launch():
        kernel[grid](*launch_args, **launch_kwargs)

    print(
        f"GEMM autotune A[{args.M},{args.K}] @ B[{args.N},{args.K}].T  "
        f"{args.dtype} in, fp32 acc active_candidates={len(active_policies)} "
        f"descriptor_slots={len(policy_bank)}",
        flush=True,
    )
    launch()
    torch.cuda.synchronize()

    best_config = kernel.best_config
    best_policy = policy_from_gemm_autotune_config(best_config, policy_bank)
    cache_status = "unavailable"
    if cache_key is not None and isinstance(cache, dict):
        cache_status = "in-process hit" if in_process_cache_hit else "miss, disk hit, or benchmarked"

    ref32 = a.float() @ b.float().T
    ref16 = torch.matmul(a, b.T)
    err_gluon = (c.float() - ref32).abs().max().item()
    err_cublas = (ref16.float() - ref32).abs().max().item()

    ms_gluon = triton.testing.do_bench(launch, warmup=args.warmup, rep=args.rep)
    ms_cublas = triton.testing.do_bench(lambda: torch.matmul(a, b.T), warmup=args.warmup, rep=args.rep)
    flops = 2.0 * args.M * args.N * args.K
    tf_gluon = flops / ms_gluon * 1e-9
    tf_cublas = flops / ms_cublas * 1e-9
    ratio = tf_gluon / tf_cublas * 100.0
    print(
        f"autotune selected POLICY_ID={best_config.kwargs['POLICY_ID']} "
        f"{best_policy.label} cache={cache_status}",
        flush=True,
    )
    print(
        f"autotune dtype={dtype_name(dtype)}: "
        f"gluon {ms_gluon:.3f} ms ({tf_gluon:.1f} TFLOP/s) | "
        f"cublas {ms_cublas:.3f} ms ({tf_cublas:.1f} TFLOP/s) | "
        f"ratio {ratio:.3f}% | "
        f"max_err gluon={err_gluon:.4f} cublas={err_cublas:.4f}",
        flush=True,
    )
    if args.require_ratio is not None and ratio < args.require_ratio:
        raise SystemExit(
            f"autotune ratio {ratio:.3f}% did not meet required {args.require_ratio:.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=4096)   # B*S = 32*128
    parser.add_argument("--N", type=int, default=30522)  # V (BERT vocab)
    parser.add_argument("--K", type=int, default=768)    # D
    parser.add_argument("--dtype", choices=("fp16", "float16", "bf16", "bfloat16"), default="fp16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--device-profile", choices=("actual", "rtx5090"), default="actual")
    parser.add_argument("--include-block-n-256", action="store_true")
    parser.add_argument("--require-ratio", type=float, default=None)
    args = parser.parse_args()
    require_gluon_backend()
    run_benchmark(args)


if __name__ == "__main__":
    main()
