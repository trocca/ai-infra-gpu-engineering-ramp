"""COMPLETE test harness — do not edit. Oracle: torch.softmax."""

import pytest
import torch

pytest.importorskip("triton", reason="Triton required (ships with torch cu128 under WSL2)")

from conftest import requires_gpu
from src.softmax_triton import softmax

torch.manual_seed(0)


@requires_gpu
@pytest.mark.parametrize("rows,cols", [(1, 8), (64, 128), (128, 1000), (32, 4096)])
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-5), (torch.float16, 1e-2)])
def test_matches_torch(call, rows, cols, dtype, atol):
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    out = call(softmax, x)
    ref = torch.softmax(x.float(), dim=-1).to(dtype)
    torch.testing.assert_close(out, ref, atol=atol, rtol=0)


@requires_gpu
def test_non_power_of_two_columns(call):
    """Masked lanes must not leak into the normalization."""
    x = torch.randn(16, 777, device="cuda", dtype=torch.float32)
    out = call(softmax, x)
    torch.testing.assert_close(out, torch.softmax(x, dim=-1), atol=1e-5, rtol=0)


@requires_gpu
def test_rows_sum_to_one(call):
    x = torch.randn(64, 500, device="cuda", dtype=torch.float16)
    out = call(softmax, x)
    torch.testing.assert_close(out.float().sum(-1),
                               torch.ones(64, device="cuda"), atol=1e-3, rtol=0)


@requires_gpu
def test_numerical_stability_large_logits(call):
    """No max-subtraction => exp overflows to inf at fp16-range logits."""
    x = torch.full((4, 256), 60000.0, device="cuda", dtype=torch.float32)
    x[:, 0] += 1.0
    out = call(softmax, x)
    assert torch.isfinite(out).all(), "overflow: subtract the row max before exp"
    assert out[:, 0].min() > 0.5
