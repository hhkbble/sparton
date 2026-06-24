"""Forward kernels: hybrid (compatibility), naive (debug), optimized (default).

hybrid and naive import eagerly; optimized is imported lazily (it needs
host-side TMA, gated at sm_90+) so importing this package never pulls the
TMA-only kernel on unsupported hardware.
"""

from .hybrid import hybrid_forward, hybrid_fwd_op
from .naive import naive_forward, naive_fwd_op

__all__ = [
    "hybrid_forward",
    "hybrid_fwd_op",
    "naive_forward",
    "naive_fwd_op",
    "optimized_forward",
    "optimized_fwd_op",
]


def __getattr__(name):
    if name in ("optimized_forward", "optimized_fwd_op"):
        from . import optimized

        return getattr(optimized, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
