from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Make `import benchmarks.<script>` resolve as a namespace package regardless
# of how pytest is invoked (`python -m pytest` adds the CWD; the console
# script does not).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


@pytest.fixture(scope="session")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("Sparton CUDA kernel tests require a CUDA device")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def sparton_kernel(cuda_device: torch.device):
    pytest.importorskip("triton")
    return pytest.importorskip("sparton.sparton_kernel")
