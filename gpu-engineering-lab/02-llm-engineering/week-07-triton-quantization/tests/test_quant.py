"""COMPLETE test harness — do not edit.

Oracles: dequantized fp matmul (for the kernel) and simple algebraic
properties (for the packer). Pack tests run on CPU; kernel tests need CUDA.
"""

import pytest
import torch

from conftest import requires_gpu
from src.quant.pack import dequantize_int8, quantization_error, quantize_int8

torch.manual_seed(0)


# ------------------------------------------------------------------ packing

def test_shapes_and_dtypes(call):
    w = torch.randn(128, 256)
    w_int8, scales = call(quantize_int8, w)
    assert w_int8.shape == w.shape and w_int8.dtype == torch.int8
    assert scales.shape == (128,)
    assert int(w_int8.abs().max()) <= 127


def test_roundtrip_error_is_small(call):
    w = torch.randn(256, 512)
    w_int8, scales = call(quantize_int8, w)
    err = quantization_error(w, w_int8, scales)
    assert err < 0.01, f"relative Frobenius error too high: {err:.4f}"


def test_row_max_maps_to_127(call):
    """Symmetric per-channel: each row's largest |w| should hit the grid edge."""
    w = torch.randn(64, 128)
    w_int8, scales = call(quantize_int8, w)
    row_abs_max = w.abs().max(dim=1).values
    torch.testing.assert_close(scales.float(), row_abs_max / 127.0,
                               atol=1e-6, rtol=1e-4)
    assert (w_int8.abs().max(dim=1).values == 127).all()


def test_per_channel_isolates_outliers(call):
    """One huge row must not destroy the precision of the other rows."""
    w = torch.randn(32, 64)
    w[0] *= 1000.0
    w_int8, scales = call(quantize_int8, w)
    w_hat = call(dequantize_int8, w_int8, scales, torch.float32)
    rest_err = (w[1:] - w_hat[1:]).norm() / w[1:].norm()
    assert rest_err < 0.01, (
        "outlier row leaked into other channels — is your scale per-tensor "
        f"instead of per-channel? err={rest_err:.4f}"
    )


def test_zero_row_does_not_nan(call):
    w = torch.randn(8, 16)
    w[3] = 0.0
    w_int8, scales = call(quantize_int8, w)
    assert torch.isfinite(scales).all()
    assert (w_int8[3] == 0).all()


# ----------------------------------------------------------- fused matmul

@requires_gpu
@pytest.mark.parametrize("M,N,K", [(1, 128, 256), (64, 256, 512), (33, 100, 320)])
def test_matmul_matches_dequant_reference(call, M, N, K):
    pytest.importorskip("triton")
    from src.quant.matmul_w8a16 import matmul_w8a16
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(N, K, device="cuda", dtype=torch.float16)
    w_int8, scales = call(quantize_int8, w)
    y = call(matmul_w8a16, x, w_int8.cuda(), scales.cuda())
    ref = x.float() @ call(dequantize_int8, w_int8, scales, torch.float32).cuda().T
    torch.testing.assert_close(y.float(), ref, atol=5e-2, rtol=1e-2)


@requires_gpu
def test_linear_module_end_to_end(call):
    pytest.importorskip("triton")
    from src.quant.matmul_w8a16 import W8A16Linear
    lin = torch.nn.Linear(256, 128, bias=True, dtype=torch.float16, device="cuda")
    qlin = call(W8A16Linear, lin)
    x = torch.randn(4, 10, 256, device="cuda", dtype=torch.float16)
    y = call(qlin.forward, x)
    ref = lin(x)
    assert y.shape == ref.shape == (4, 10, 128)
    # int8 weights => outputs close but not identical to fp16
    rel = (y.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.02, f"relative output error too high: {rel:.4f}"
