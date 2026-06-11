from __future__ import annotations

import pytest
import torch


@pytest.fixture(scope="session")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("Sparton CUDA kernel tests require a CUDA device")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def sparton_kernel(cuda_device: torch.device):
    pytest.importorskip("triton")
    return pytest.importorskip("sparton.sparton_kernel")
