"""Resource-derived policy generation for experimental Gluon kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DeviceProfile:
    sm_count: int
    warp_size: int
    max_threads_per_block: int
    max_threads_per_sm: int
    shared_memory_per_block_optin: int
    shared_memory_per_sm: int
    capability_major: int = 0
    capability_minor: int = 0
    device_name: str = ""
    shared_memory_per_block: int = 0
    regs_per_sm: int = 0
    l2_cache_size: int = 0
    memory_bus_width: int = 0
    total_memory: int = 0

    @property
    def shared_memory_per_block_limit(self) -> int:
        return max(self.shared_memory_per_block_optin, self.shared_memory_per_block)


@dataclass(frozen=True)
class ProblemSpec:
    M: int
    N: int
    K: int
    dtype_name: str

    @property
    def element_bitwidth(self) -> int:
        if self.dtype_name in {"fp16", "float16", "bf16", "bfloat16"}:
            return 16
        raise ValueError(f"unsupported Gluon GEMM dtype {self.dtype_name!r}")

    @property
    def element_bytes(self) -> int:
        return self.element_bitwidth // 8


@dataclass(frozen=True)
class GluonGemmPolicy:
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    warps_m: int
    warps_n: int
    swizzle_byte_width: int

    @property
    def num_warps(self) -> int:
        return self.warps_m * self.warps_n

    def shared_memory_bytes(self, dtype_bytes: int) -> int:
        stage_bytes = (self.block_m * self.block_k + self.block_n * self.block_k) * dtype_bytes
        # mbarriers are tiny relative to tile storage; include slack so policy
        # pruning matches real launches near the opt-in shared-memory edge.
        return self.num_stages * stage_bytes + self.num_stages * 16

    @property
    def label(self) -> str:
        return (
            f"BM={self.block_m} BN={self.block_n} BK={self.block_k} "
            f"stages={self.num_stages} warps={self.warps_m}x{self.warps_n} "
            f"swizzle={self.swizzle_byte_width}"
        )


_OPTIMIZED_FORWARD_FALLBACK_POLICY = GluonGemmPolicy(
    block_m=64,
    block_n=64,
    block_k=64,
    num_stages=3,
    warps_m=2,
    warps_n=2,
    swizzle_byte_width=128,
)


def torch_device_profile() -> DeviceProfile:
    import torch

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return DeviceProfile(
        sm_count=props.multi_processor_count,
        warp_size=getattr(props, "warp_size", 32),
        max_threads_per_block=props.max_threads_per_block,
        max_threads_per_sm=props.max_threads_per_multi_processor,
        shared_memory_per_block_optin=props.shared_memory_per_block_optin,
        shared_memory_per_sm=props.shared_memory_per_multiprocessor,
        capability_major=props.major,
        capability_minor=props.minor,
        device_name=props.name,
        shared_memory_per_block=getattr(props, "shared_memory_per_block", 0),
        regs_per_sm=getattr(props, "regs_per_multiprocessor", 0),
        l2_cache_size=getattr(props, "L2_cache_size", 0),
        memory_bus_width=getattr(props, "memory_bus_width", 0),
        total_memory=getattr(props, "total_memory", 0),
    )


def _candidate_policies() -> Sequence[GluonGemmPolicy]:
    return (
        # Current validated best.
        GluonGemmPolicy(128, 128, 64, 3, 4, 2, 128),
        GluonGemmPolicy(128, 128, 64, 2, 4, 2, 128),
        GluonGemmPolicy(128, 64, 64, 4, 4, 2, 128),
        GluonGemmPolicy(64, 128, 64, 3, 2, 4, 128),
        # BK=32 family with 64-byte swizzle; mirrors cuBLAS's shallower K tile.
        GluonGemmPolicy(128, 128, 32, 4, 4, 2, 64),
        GluonGemmPolicy(128, 128, 32, 3, 4, 2, 64),
        GluonGemmPolicy(128, 64, 32, 4, 4, 2, 64),
        GluonGemmPolicy(64, 128, 32, 4, 2, 4, 64),
        # Smaller 2-CTA/SM candidates.
        GluonGemmPolicy(64, 64, 64, 3, 2, 2, 128),
        GluonGemmPolicy(64, 64, 32, 4, 2, 2, 64),
        GluonGemmPolicy(64, 64, 32, 3, 2, 2, 64),
        # Kept for benchmark evidence only; production fused forward caps
        # BLOCK_N at 128, so this is filtered by default.
        GluonGemmPolicy(128, 256, 32, 3, 4, 2, 64),
    )


def _valid_policy(
    policy: GluonGemmPolicy,
    problem: ProblemSpec,
    device: DeviceProfile,
    *,
    include_block_n_256: bool,
) -> bool:
    if policy.block_n > 128 and not include_block_n_256:
        return False
    threads_per_cta = policy.num_warps * device.warp_size
    if threads_per_cta > device.max_threads_per_block:
        return False
    if threads_per_cta > device.max_threads_per_sm:
        return False
    if policy.block_k * problem.element_bytes < policy.swizzle_byte_width:
        return False
    if policy.block_k % 16 != 0:
        return False
    if policy.block_m % 16 != 0 or policy.block_n % 8 != 0:
        return False
    smem_bytes = policy.shared_memory_bytes(problem.element_bytes)
    if smem_bytes > device.shared_memory_per_block_limit:
        return False
    if smem_bytes > device.shared_memory_per_sm:
        return False
    if _cta_occupancy(policy, problem, device) < 1:
        return False
    return True


def generate_gluon_gemm_policies(
    problem: ProblemSpec,
    device: DeviceProfile,
    *,
    include_block_n_256: bool = False,
) -> tuple[GluonGemmPolicy, ...]:
    """Return valid policies in deterministic tuning order."""

    seen: set[GluonGemmPolicy] = set()
    policies: list[GluonGemmPolicy] = []
    for policy in _candidate_policies():
        if policy in seen:
            continue
        seen.add(policy)
        if _valid_policy(
            policy,
            problem,
            device,
            include_block_n_256=include_block_n_256,
        ):
            policies.append(policy)
    return tuple(policies)


def gluon_gemm_policy_universe(
    *,
    include_block_n_256: bool = True,
) -> tuple[GluonGemmPolicy, ...]:
    """Return the bounded GEMM config superset without resource assumptions."""

    seen: set[GluonGemmPolicy] = set()
    policies: list[GluonGemmPolicy] = []
    for policy in _candidate_policies():
        if policy in seen:
            continue
        if policy.block_n > 128 and not include_block_n_256:
            continue
        seen.add(policy)
        policies.append(policy)
    return tuple(policies)


def optimized_forward_policy_universe() -> tuple[GluonGemmPolicy, ...]:
    """Return the bounded production config superset for optimized forward."""

    seen: set[GluonGemmPolicy] = set()
    universe = [_OPTIMIZED_FORWARD_FALLBACK_POLICY]
    for policy in _candidate_policies():
        if policy.block_n > 128:
            continue
        universe.append(policy)

    deduped = []
    for policy in universe:
        if policy in seen:
            continue
        seen.add(policy)
        deduped.append(policy)
    return tuple(deduped)


def optimized_forward_fallback_policy() -> GluonGemmPolicy:
    """Return the small-shape fallback policy for optimized forward."""

    return _OPTIMIZED_FORWARD_FALLBACK_POLICY


def policy_config_kwargs(policy: GluonGemmPolicy, policy_id: int) -> dict[str, int]:
    """Return static autotune metadata shared by Gluon policy call sites."""

    return {
        "POLICY_ID": policy_id,
        "BLOCK_M": policy.block_m,
        "BLOCK_N": policy.block_n,
        "BLOCK_K": policy.block_k,
        "NUM_STAGES": policy.num_stages,
        "WARPS_M": policy.warps_m,
        "WARPS_N": policy.warps_n,
    }


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _is_tiny_problem(problem: ProblemSpec) -> bool:
    return problem.M < 1024 or problem.N < 1024 or problem.K < 64


def _cta_occupancy(
    policy: GluonGemmPolicy,
    problem: ProblemSpec,
    device: DeviceProfile,
) -> int:
    threads_per_cta = policy.num_warps * device.warp_size
    smem_bytes = policy.shared_memory_bytes(problem.element_bytes)
    ctas_by_threads = device.max_threads_per_sm // threads_per_cta
    ctas_by_smem = device.shared_memory_per_sm // smem_bytes
    ctas_by_regs = _estimated_ctas_by_registers(policy, device)
    return min(ctas_by_threads, ctas_by_smem, ctas_by_regs)


def _estimated_ctas_by_registers(policy: GluonGemmPolicy, device: DeviceProfile) -> int:
    if device.regs_per_sm <= 0:
        return max(device.max_threads_per_sm // (policy.num_warps * device.warp_size), 1)

    threads_per_cta = policy.num_warps * device.warp_size
    # Use accumulator footprint as a conservative policy-relative pressure
    # estimate. Real register counts remain compiler output, not a Python fact.
    acc_values_per_thread = _ceil_div(policy.block_m * policy.block_n, threads_per_cta)
    estimated_regs_per_thread = 32 + acc_values_per_thread
    estimated_regs_per_cta = estimated_regs_per_thread * threads_per_cta
    return max(device.regs_per_sm // estimated_regs_per_cta, 1)


def _optimized_forward_rank_key(
    policy: GluonGemmPolicy,
    problem: ProblemSpec,
    device: DeviceProfile,
    universe_index: int,
) -> tuple[int, int, int, int, int, int, int]:
    tiles_m = _ceil_div(problem.M, policy.block_m)
    tiles_n = _ceil_div(problem.N, policy.block_n)
    total_ctas = tiles_m * tiles_n
    target_ctas = max(device.sm_count * 2, 1)
    n_tail = tiles_n * policy.block_n - problem.N
    k_tail = _ceil_div(problem.K, policy.block_k) * policy.block_k - problem.K
    occupancy = _cta_occupancy(policy, problem, device)

    return (
        0 if total_ctas >= target_ctas else 1,
        -min(total_ctas, target_ctas),
        -occupancy,
        n_tail,
        k_tail,
        0 if policy == _OPTIMIZED_FORWARD_FALLBACK_POLICY else 1,
        universe_index,
    )


def derive_optimized_forward_policies(
    problem: ProblemSpec,
    device: DeviceProfile,
) -> tuple[GluonGemmPolicy, ...]:
    """Return runtime GPU-derived optimized-forward candidates."""

    universe = optimized_forward_policy_universe()
    valid = [
        (idx, policy)
        for idx, policy in enumerate(universe)
        if _valid_policy(
            policy,
            problem,
            device,
            include_block_n_256=False,
        )
    ]
    if _is_tiny_problem(problem):
        return (
            (_OPTIMIZED_FORWARD_FALLBACK_POLICY,)
            if any(policy == _OPTIMIZED_FORWARD_FALLBACK_POLICY for _, policy in valid)
            else tuple()
        )
    return tuple(
        policy
        for idx, policy in sorted(
            valid,
            key=lambda item: _optimized_forward_rank_key(
                item[1],
                problem,
                device,
                item[0],
            ),
        )
    )


def policy_table_rows(
    policies: Iterable[GluonGemmPolicy],
    *,
    dtype_bytes: int,
) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (policy.label, policy.num_warps, policy.shared_memory_bytes(dtype_bytes))
        for policy in policies
    )


__all__ = [
    "DeviceProfile",
    "GluonGemmPolicy",
    "ProblemSpec",
    "derive_optimized_forward_policies",
    "generate_gluon_gemm_policies",
    "gluon_gemm_policy_universe",
    "optimized_forward_fallback_policy",
    "optimized_forward_policy_universe",
    "policy_config_kwargs",
    "policy_table_rows",
    "torch_device_profile",
]
