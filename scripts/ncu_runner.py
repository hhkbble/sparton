"""Autotuned Gluon GEMM launcher for ncu/nsys profiling.

Launches the policy selected by the GEMM autotune path six times back-to-back so
a profiler can skip warm-up launches (e.g. `ncu --launch-skip 4 --launch-count 1`).
Run from the scripts/ directory with the hardened env prefix; see
docs/sparton_gluon_remaining_work_design.md §11 and Appendix B.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import triton
from bench_gluon_gemm import (
    MAX_GEMM_AUTOTUNE_POLICIES,
    build_autotuned_kernel,
    gemm_autotune_policy_universe,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sparton._gluon_runtime import is_gluon_backend_available  # noqa: E402
from sparton._gluon_policy_runtime import (  # noqa: E402
    element_ty_for_dtype,
    make_descriptor_bank,
)
from sparton._runtime_policy import (  # noqa: E402
    ProblemSpec,
    generate_gluon_gemm_policies,
    torch_device_profile,
)


M, N, K = 4096, 30522, 768


def main() -> None:
    available, reason = is_gluon_backend_available()
    if not available:
        raise RuntimeError(
            "ncu_runner.py requires the optimized Gluon backend "
            f"(CUDA sm_80+ and importable triton.experimental.gluon): {reason}"
        )

    problem = ProblemSpec(M=M, N=N, K=K, dtype_name="fp16")
    policies = generate_gluon_gemm_policies(problem, torch_device_profile())
    if not policies:
        raise RuntimeError("no valid Gluon GEMM policies for the active CUDA device")
    policy = policies[0]
    args = SimpleNamespace(
        M=M,
        N=N,
        K=K,
        dtype="fp16",
        device_profile="actual",
        include_block_n_256=False,
    )
    policy_bank = gemm_autotune_policy_universe()
    kernel = build_autotuned_kernel(args)

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16) * 0.05
    b = torch.randn((N, K), device="cuda", dtype=torch.float16) * 0.05
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    element_ty = element_ty_for_dtype(torch.float16)
    descriptor_bank = make_descriptor_bank(
        a,
        b,
        policy_bank,
        max_policies=MAX_GEMM_AUTOTUNE_POLICIES,
    )

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    for _ in range(6):
        kernel[grid](
            *descriptor_bank,
            c,
            M,
            N,
            K,
            element_ty,
        )
    torch.cuda.synchronize()
    selected = kernel.best_config
    if selected is not None:
        policy = policy_bank[selected.kwargs["POLICY_ID"]]
    print(f"done: {policy.label}")


if __name__ == "__main__":
    main()
