import warnings

import torch

if not torch.cuda.is_available():
    warnings.warn(
        "CUDA is not available. Sparton only runs on CUDA devices.",
        stacklevel=2,
    )
    __all__ = []
else:
    __all__ = ["SpartonHead"]
    from .sparton_kernel import SpartonHead