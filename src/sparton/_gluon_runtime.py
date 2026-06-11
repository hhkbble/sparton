"""Lazy compatibility layer for Triton Gluon.

The hybrid backend must keep working with the package floor of
``triton>=3.3.1``.  Gluon first appears later and its experimental API moves, so
all Gluon imports are centralized here and performed only when a Gluon backend
is explicitly requested.
"""

from __future__ import annotations

import importlib
import warnings
from types import SimpleNamespace
from typing import Any

import torch
import triton


VALIDATED_TRITON = "3.6.0"

_STATE: SimpleNamespace | None = None
_WARNED_VERSION = False
_EXPORTS = {
    "autotune",
    "gl",
    "gluon",
    "tma",
    "mbarrier",
    "TensorDescriptor",
    "NVMMASharedLayout",
    "fence_async_shared",
    "mma_v2",
}


def _warn_if_unvalidated_triton() -> None:
    global _WARNED_VERSION
    if _WARNED_VERSION or triton.__version__ == VALIDATED_TRITON:
        return
    warnings.warn(
        "Sparton optimized Gluon backend is validated with Triton "
        f"{VALIDATED_TRITON}, but this process is running Triton "
        f"{triton.__version__}. Re-run the Gluon validation probes before "
        "relying on optimized-backend results.",
        RuntimeWarning,
        stacklevel=3,
    )
    _WARNED_VERSION = True


def _capability_mma_family(capability: tuple[int, int]) -> str:
    major, _minor = capability
    if major < 8:
        raise RuntimeError(
            "Sparton optimized Gluon backend requires CUDA capability sm_80 "
            f"or newer; current device is sm_{major}{capability[1]}."
        )
    return "mma_v2"


def select_mma_family(device: torch.device | int | None = None) -> str:
    """Return the statically whitelisted MMA family for the current CUDA device."""

    if not torch.cuda.is_available():
        raise RuntimeError("Sparton optimized Gluon backend requires CUDA.")
    capability = torch.cuda.get_device_capability(device)
    return _capability_mma_family(capability)


def is_gluon_available(device: torch.device | int | None = None) -> tuple[bool, str]:
    """Cheap availability check that never imports Gluon."""

    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    try:
        family = select_mma_family(device)
    except RuntimeError as exc:
        return False, str(exc)
    return True, family


def is_gluon_backend_available(device: torch.device | int | None = None) -> tuple[bool, str]:
    """Return whether optimized Gluon symbols are importable for this device."""

    available, reason = is_gluon_available(device)
    if not available:
        return False, reason
    try:
        _load_gluon()
    except (AttributeError, ImportError, RuntimeError) as exc:
        return False, str(exc)
    return True, reason


def _load_gluon() -> SimpleNamespace:
    try:
        gluon = importlib.import_module("triton.experimental.gluon")
        gl = importlib.import_module("triton.experimental.gluon.language")
        ampere = importlib.import_module(
            "triton.experimental.gluon.language.nvidia.ampere"
        )
        hopper_lang = importlib.import_module(
            "triton.experimental.gluon.language.nvidia.hopper"
        )
        hopper_host = importlib.import_module("triton.experimental.gluon.nvidia.hopper")
        layouts = importlib.import_module("triton.experimental.gluon.language._layouts")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Sparton optimized Gluon backend requires "
            "triton.experimental.gluon. Install Triton >=3.4.0; this backend "
            f"is validated with Triton {VALIDATED_TRITON}."
        ) from exc

    return SimpleNamespace(
        autotune=getattr(gluon, "autotune", triton.autotune),
        gl=gl,
        gluon=gluon,
        tma=hopper_lang.tma,
        mbarrier=hopper_lang.mbarrier,
        TensorDescriptor=hopper_host.TensorDescriptor,
        NVMMASharedLayout=layouts.NVMMASharedLayout,
        fence_async_shared=hopper_lang.fence_async_shared,
        mma_v2=ampere.mma_v2,
    )


def ensure_gluon(device: torch.device | int | None = None) -> SimpleNamespace:
    """Load and return Gluon symbols after validating the CUDA capability."""

    global _STATE
    family = select_mma_family(device)
    if family != "mma_v2":
        raise RuntimeError(f"Unsupported Gluon MMA family {family!r}.")
    _warn_if_unvalidated_triton()
    if _STATE is None:
        _STATE = _load_gluon()
    return _STATE


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(ensure_gluon(), name)
    raise AttributeError(name)


__all__ = [
    "VALIDATED_TRITON",
    "ensure_gluon",
    "is_gluon_backend_available",
    "is_gluon_available",
    "select_mma_family",
    *_EXPORTS,
]
