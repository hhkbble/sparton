"""Availability gate for the optimized (pure-Triton) forward backend.

The optimized backend (:mod:`sparton._backend_optimized`) is a persistent fused
forward built on host-side TMA descriptors
(``triton.tools.tensor_descriptor``) and ``tl.dot``. Host-side TMA is a
Hopper-class feature, so the backend is gated at CUDA capability sm_90+. This
module performs the cheap capability/import check used by default-backend
resolution; it imports no kernel code.
"""

from __future__ import annotations

import torch


# Host-side TMA descriptors require Hopper-class (sm_90+) tensor-memory support.
# Below this floor the optimized backend is unavailable and default resolution
# falls back to the hybrid backend.
OPTIMIZED_CAPABILITY_FLOOR = (9, 0)


def is_optimized_backend_available(
    device: torch.device | int | None = None,
) -> tuple[bool, str]:
    """Return ``(available, reason)`` for the optimized backend on ``device``.

    Cheap check used by default-backend resolution. ``reason`` carries the
    device arch string (e.g. ``"sm_90"``) when available, or the disqualifying
    cause otherwise. Never imports the kernel module.
    """

    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    major, minor = torch.cuda.get_device_capability(device)
    floor_major, floor_minor = OPTIMIZED_CAPABILITY_FLOOR
    if (major, minor) < OPTIMIZED_CAPABILITY_FLOOR:
        return False, (
            "Sparton optimized backend requires CUDA capability "
            f"sm_{floor_major}{floor_minor} (Hopper) or newer for host-side "
            f"TMA; current device is sm_{major}{minor}."
        )
    try:
        import triton.tools.tensor_descriptor  # noqa: F401
    except ImportError as exc:
        return False, (
            "Sparton optimized backend requires triton.tools.tensor_descriptor "
            f"(Triton with host-side TMA support): {exc}"
        )
    return True, f"sm_{major}{minor}"


__all__ = [
    "OPTIMIZED_CAPABILITY_FLOOR",
    "is_optimized_backend_available",
]
