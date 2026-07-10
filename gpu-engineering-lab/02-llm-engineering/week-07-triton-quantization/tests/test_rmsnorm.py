"""COMPLETE test harness — do not edit. Oracle: fp32-accumulating torch RMSNorm."""

import pytest
import torch

pytest.importorskip("triton", reason="Triton required (ships with torch cu128 under WSL2)")

from conftest import requires_gpu
from src.rmsnorm_triton import rmsnorm, rmsnorm_torch

torch.manual_seed(0)


@requires_gpu
@pytest.mark.parametrize("rows,cols", [(1, 64), (128, 512), (64, 4096), (16, 1000)])
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-5), (torch.float16, 1e-2)])
def test_matches_reference(call, rows, cols, dtype, atol):
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    w = torch.randn(cols, device="cuda", dtype=dtype)
    out = call(rmsnorm, x, w)
    torch.testing.assert_close(out, rmsnorm_torch(x, w), atol=atol, rtol=0)


@requires_gpu
def test_non_power_of_two_divides_by_true_n(call):
    """Dividing the sum of squares by BLOCK_SIZE instead of n_cols is the
    classic bug — padded zero lanes deflate the mean. 1000 != 1024 catches it."""
    x = torch.randn(8, 1000, device="cuda", dtype=torch.float32)
    w = torch.ones(1000, device="cuda")
    out = call(rmsnorm, x, w)
    torch.testing.assert_close(out, rmsnorm_torch(x, w), atol=1e-5, rtol=0)


@requires_gpu
def test_weight_is_applied(call):
    x = torch.randn(4, 256, device="cuda", dtype=torch.float32)
    w2 = torch.full((256,), 2.0, device="cuda")
    w1 = torch.ones(256, device="cuda")
    out2 = call(rmsnorm, x, w2)
    out1 = call(rmsnorm, x, w1)
    torch.testing.assert_close(out2, out1 * 2.0, atol=1e-5, rtol=1e-5)


@requires_gpu
def test_fp16_accumulates_in_fp32(call):
    """Large-magnitude fp16 rows overflow if x*x is summed in fp16."""
    x = (torch.randn(4, 2048, device="cuda") * 100).to(torch.float16)
    w = torch.ones(2048, device="cuda", dtype=torch.float16)
    out = call(rmsnorm, x, w)
    assert torch.isfinite(out).all(), "sum x^2 in fp32 (see kernel TODO step 2)"
    torch.testing.assert_close(out, rmsnorm_torch(x, w), atol=1e-2, rtol=0)
