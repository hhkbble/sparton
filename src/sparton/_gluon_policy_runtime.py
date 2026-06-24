"""Lazy host-side helpers shared by Gluon policy call sites."""

from __future__ import annotations

from typing import Sequence

from ._runtime_policy import GluonGemmPolicy, policy_config_kwargs


def config_for_policy(policy: GluonGemmPolicy, policy_id: int):
    import triton

    return triton.Config(
        policy_config_kwargs(policy, policy_id),
        num_warps=policy.num_warps,
        num_stages=policy.num_stages,
    )


def configs_for_policies(policies: Sequence[GluonGemmPolicy]):
    return [config_for_policy(policy, idx) for idx, policy in enumerate(policies)]


def policy_from_config(config, policies: Sequence[GluonGemmPolicy]) -> GluonGemmPolicy:
    policy_id = int(config.kwargs["POLICY_ID"])
    if policy_id < 0 or policy_id >= len(policies):
        raise RuntimeError(
            f"autotune config POLICY_ID={policy_id} is outside the "
            f"{len(policies)}-policy bank"
        )
    return policies[policy_id]


def prune_configs_for_policies(
    configs,
    active_policies: Sequence[GluonGemmPolicy],
    policy_bank: Sequence[GluonGemmPolicy],
):
    config_by_policy = {
        policy_from_config(config, policy_bank): config
        for config in configs
    }
    pruned = [
        config_by_policy[policy]
        for policy in active_policies
        if policy in config_by_policy
    ]
    if not pruned:
        raise RuntimeError("no valid Gluon autotune configs")
    return pruned


def element_ty_for_dtype(dtype):
    import torch

    from ._gluon_runtime import gl

    if dtype is torch.bfloat16:
        return gl.bfloat16
    if dtype is torch.float16:
        return gl.float16
    raise RuntimeError(f"Gluon policy kernels only support fp16/bf16, got {dtype}")


def make_descriptor_bank(
    lhs,
    rhs,
    policies: Sequence[GluonGemmPolicy],
    *,
    max_policies: int | None = None,
):
    from ._gluon_runtime import NVMMASharedLayout, TensorDescriptor

    if not policies:
        raise RuntimeError("descriptor-bank construction requires at least one policy")
    if max_policies is not None and len(policies) > max_policies:
        raise RuntimeError(
            f"descriptor-bank signature expects at most {max_policies} policies, "
            f"got {len(policies)}"
        )

    descriptors = []
    for policy in policies:
        smem_layout = NVMMASharedLayout(
            swizzle_byte_width=policy.swizzle_byte_width,
            element_bitwidth=16,
            rank=2,
        )
        descriptors.extend(
            (
                TensorDescriptor.from_tensor(
                    lhs,
                    [policy.block_m, policy.block_k],
                    smem_layout,
                ),
                TensorDescriptor.from_tensor(
                    rhs,
                    [policy.block_n, policy.block_k],
                    smem_layout,
                ),
            )
        )

    if max_policies is not None:
        while len(descriptors) < max_policies * 2:
            descriptors.extend(descriptors[:2])
    return tuple(descriptors)


__all__ = [
    "config_for_policy",
    "configs_for_policies",
    "element_ty_for_dtype",
    "make_descriptor_bank",
    "policy_from_config",
    "prune_configs_for_policies",
]
