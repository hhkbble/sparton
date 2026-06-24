import logging
import math
import os
import warnings
from typing import Optional

import torch
from torch import nn

from ._backend_hybrid import (
    DEVICE,
    fused_sparton_bwd_op,
    fused_sparton_fwd,
    fused_sparton_fwd_op,
    fused_sparton_fwd_with_indices,
    get_embed_grad_bwd_configs,
    get_fast_forward_configs,
    get_segmented_hidden_grad_configs,
    get_slow_forward_configs,
    get_uniform_hidden_grad_configs,
    hybrid_forward,
    legacy_fused_sparton_bwd,
    matmul,
    matmul_bias,
    reduce_seq_max_log1p_relu,
    reduce_seq_max_log1p_relu_kernel,
    reduce_seq_max_log1p_relu_kernel_with_indices,
    reduce_seq_max_log1p_relu_with_indices,
    segmented_sparton_bwd,
    split_segmented_sparton_bwd,
    v_tile_from_bs,
)
from ._backend_naive import naive_forward, naive_fwd_op


logger = logging.getLogger("sparton")

# None when SPARTON_BACKEND is unset or empty; an explicit value otherwise.
_ENV_BACKEND = os.environ.get("SPARTON_BACKEND", "").strip().lower() or None

_DEFAULT_FALLBACK_WARNED = False


def _default_backend() -> str:
    """Adaptive default (M10 promotion): optimized where available.

    This availability-gated fallback applies ONLY to default resolution (no
    ``backend`` argument and no ``SPARTON_BACKEND``); an explicitly selected
    backend that is unavailable still raises with the reason.
    """

    global _DEFAULT_FALLBACK_WARNED
    from . import _backend_runtime

    available, reason = _backend_runtime.is_optimized_backend_available()
    if available:
        return "optimized"
    if not _DEFAULT_FALLBACK_WARNED:
        warnings.warn(
            "Sparton default backend 'optimized' is unavailable on this "
            f"platform ({reason}); falling back to 'hybrid'. Select a backend "
            "explicitly via SPARTON_BACKEND or SpartonHead(backend=...) to "
            "silence this warning.",
            RuntimeWarning,
            stacklevel=3,
        )
        _DEFAULT_FALLBACK_WARNED = True
    return "hybrid"


def resolve_backend(backend: Optional[str]) -> str:
    if backend is None and _ENV_BACKEND is None:
        return _default_backend()
    from_env = backend is None
    selected = _ENV_BACKEND if from_env else backend.strip().lower()
    if selected in {"hybrid", "naive", "optimized"}:
        return selected
    source = "SPARTON_BACKEND environment variable" if from_env else "backend argument"
    raise ValueError(
        f"Unknown Sparton backend {selected!r} from {source}; "
        "expected 'hybrid', 'naive', or 'optimized'."
    )


def _forward_op_for_backend(backend: str):
    if backend == "hybrid":
        return hybrid_forward
    if backend == "naive":
        return naive_forward
    if backend == "optimized":
        from ._backend_optimized import optimized_forward

        return optimized_forward
    raise AssertionError(f"unhandled resolved Sparton backend {backend!r}")


def __getattr__(name: str):
    if name in {"optimized_forward", "optimized_fwd_op"}:
        from ._backend_optimized import optimized_forward, optimized_fwd_op

        return {
            "optimized_forward": optimized_forward,
            "optimized_fwd_op": optimized_fwd_op,
        }[name]
    raise AttributeError(name)


class SpartonHead(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        use_bias=False,
        *,
        backend: Optional[str] = None,
    ):
        super().__init__()

        self.backend = resolve_backend(backend)
        self._forward_op = _forward_op_for_backend(self.backend)

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
    "DEVICE",
    "SpartonHead",
    "fused_sparton_bwd_op",
    "fused_sparton_fwd",
    "fused_sparton_fwd_op",
    "fused_sparton_fwd_with_indices",
    "get_fast_forward_configs",
    "get_slow_forward_configs",
    "get_uniform_hidden_grad_configs",
    "hybrid_forward",
    "legacy_fused_sparton_bwd",
    "matmul",
    "matmul_bias",
    "naive_forward",
    "naive_fwd_op",
    "reduce_seq_max_log1p_relu",
    "reduce_seq_max_log1p_relu_kernel",
    "reduce_seq_max_log1p_relu_kernel_with_indices",
    "reduce_seq_max_log1p_relu_with_indices",
    "resolve_backend",
    "split_segmented_sparton_bwd",
    "v_tile_from_bs",
]
