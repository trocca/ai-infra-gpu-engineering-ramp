"""Shared test plumbing. COMPLETE — do not edit.

Kernel tests need a CUDA GPU + Triton (run under WSL2). Skeleton code that
still raises NotImplementedError reports as SKIPPED, not ERROR.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU required (run under WSL2)"
)


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")


@pytest.fixture(name="call")
def call_fixture():
    return call
