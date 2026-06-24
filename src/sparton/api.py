import logging
import math
import os
import warnings
from typing import Optional

import torch
from torch import nn

from . import _device  # noqa: F401  (import-time "using device" debug log)
from .backward import mono_bwd_op, optimized_bwd_op
from .forward import (
    hybrid_forward,
    hybrid_fwd_op,
    naive_forward,
    naive_fwd_op,
)


logger = logging.getLogger("sparton")

# None when SPARTON_KERNEL is unset or empty; an explicit value otherwise.
_ENV_KERNEL = os.environ.get("SPARTON_KERNEL", "").strip().lower() or None

_DEFAULT_FALLBACK_WARNED = False


def _default_kernel() -> str:
    """Adaptive default (M10 promotion): optimized where available.

    This availability-gated fallback applies ONLY to default resolution (no
    ``kernel`` argument and no ``SPARTON_KERNEL``); an explicitly selected
    kernel that is unavailable still raises with the reason.
    """

    global _DEFAULT_FALLBACK_WARNED
    from . import _runtime

    available, reason = _runtime.is_optimized_kernel_available()
    if available:
        return "optimized"
    if not _DEFAULT_FALLBACK_WARNED:
        warnings.warn(
            "Sparton default kernel 'optimized' is unavailable on this "
            f"platform ({reason}); falling back to 'hybrid'. Select a kernel "
            "explicitly via SPARTON_KERNEL or SpartonHead(kernel=...) to "
            "silence this warning.",
            RuntimeWarning,
            stacklevel=3,
        )
        _DEFAULT_FALLBACK_WARNED = True
    return "hybrid"


def resolve_kernel(kernel: Optional[str]) -> str:
    if kernel is None and _ENV_KERNEL is None:
        return _default_kernel()
    from_env = kernel is None
    selected = _ENV_KERNEL if from_env else kernel.strip().lower()
    if selected in {"hybrid", "naive", "optimized"}:
        return selected
    source = "SPARTON_KERNEL environment variable" if from_env else "kernel argument"
    raise ValueError(
        f"Unknown Sparton kernel {selected!r} from {source}; "
        "expected 'hybrid', 'naive', or 'optimized'."
    )


def _forward_op_for_kernel(kernel: str):
    if kernel == "hybrid":
        return hybrid_forward
    if kernel == "naive":
        return naive_forward
    if kernel == "optimized":
        from .forward import optimized_forward

        return optimized_forward
    raise AssertionError(f"unhandled resolved Sparton kernel {kernel!r}")


def __getattr__(name: str):
    if name in {"optimized_forward", "optimized_fwd_op"}:
        from .forward import optimized

        return getattr(optimized, name)
    raise AttributeError(name)


class SpartonHead(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        use_bias=False,
        *,
        kernel: Optional[str] = None,
    ):
        super().__init__()

        self.kernel = resolve_kernel(kernel)
        self._forward_op = _forward_op_for_kernel(self.kernel)

        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_dim))
        if use_bias:
            self.bias = nn.Parameter(torch.empty(vocab_size))
        else:
            self.register_parameter("bias", None)
        self.init_parameters()

    def tie_weights(self, decoder):
        self.weight = decoder.weight
        self.bias = decoder.bias

    def load(self, weights_dict):
        with torch.no_grad():
            w = weights_dict["weight"]  # (vocab_size, hidden_dim)
            self.weight.copy_(w)

            if self.bias is not None and "bias" in weights_dict:
                self.bias.copy_(weights_dict["bias"])
            else:
                logger.debug("SpartonHead.load: checkpoint has no bias for this head")

    def init_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, hidden_states, attention_mask):
        scores, _idx = self._forward_op(
            hidden_states,
            self.weight,
            self.bias,
            attention_mask,
        )
        return scores


__all__ = [
    "SpartonHead",
    "resolve_kernel",
    "hybrid_forward",
    "naive_forward",
    "optimized_forward",
    "hybrid_fwd_op",
    "naive_fwd_op",
    "optimized_fwd_op",
    "mono_bwd_op",
    "optimized_bwd_op",
]
