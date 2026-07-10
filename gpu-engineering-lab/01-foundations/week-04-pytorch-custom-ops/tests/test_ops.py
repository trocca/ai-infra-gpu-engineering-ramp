"""Acceptance tests for rusty_kernels (COMPLETE — do not edit).

Oracle: torch.nn.functional. Error bounds (the acceptance criteria):
    fp32: max-abs error <= 1e-6
    fp16: max-abs error <= 1e-3
LayerNorm must additionally pass torch.autograd.gradcheck in float64.

All tests are CUDA-marked and skip cleanly on CPU-only machines (CI runs an
import smoke test only; GPU tests run locally).

Run: pytest tests/test_ops.py -v
"""

import pytest
import torch
import torch.nn.functional as F

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="requires a CUDA device")

# (rows, cols) — transformer-ish: rows = batch * seq, cols = hidden
SHAPES = [
    (8, 128),
    (512, 768),
    (2048, 1024),
    (128, 4096),
    (64, 8192),
    (1000, 777),   # non-power-of-two both dims
    (1, 1),        # degenerate
]

TOL = {torch.float32: 1e-6, torch.float16: 1e-3}


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def test_package_imports():
    """CPU-safe: the package must import without a GPU (CI runs this)."""
    import rusty_kernels  # noqa: F401

    assert hasattr(rusty_kernels, "softmax")
    assert hasattr(rusty_kernels, "layer_norm")


@cuda_only
class TestPassthrough:
    def test_roundtrip(self):
        from rusty_kernels.ops import passthrough

        x = torch.randn(1024, 513, device="cuda")
        y = passthrough(x)
        assert y.data_ptr() != x.data_ptr(), "must be a copy, not the same storage"
        assert torch.equal(y, x)


@cuda_only
class TestFusedSoftmax:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_matches_torch(self, shape, dtype):
        import rusty_kernels

        x = torch.randn(*shape, device="cuda", dtype=dtype)
        got = rusty_kernels.softmax(x, dim=-1)
        want = F.softmax(x, dim=-1)
        err = _max_abs_err(got, want)
        assert err <= TOL[dtype], f"{shape} {dtype}: max abs err {err:.3e}"

    def test_rows_sum_to_one(self):
        import rusty_kernels

        x = torch.randn(256, 1000, device="cuda")
        y = rusty_kernels.softmax(x, dim=-1)
        assert torch.allclose(y.sum(dim=-1), torch.ones(256, device="cuda"),
                              atol=1e-5)

    def test_extreme_logits_stable(self):
        """Large-magnitude logits must not overflow (online rescaling)."""
        import rusty_kernels

        x = torch.randn(64, 512, device="cuda") * 1000.0
        y = rusty_kernels.softmax(x, dim=-1)
        assert torch.isfinite(y).all(), "softmax overflowed on large logits"
        want = F.softmax(x, dim=-1)
        assert _max_abs_err(y, want) <= 1e-6

    def test_3d_input(self):
        """(batch, seq, hidden) — wrapper must handle leading dims."""
        import rusty_kernels

        x = torch.randn(4, 128, 768, device="cuda")
        got = rusty_kernels.softmax(x, dim=-1)
        want = F.softmax(x, dim=-1)
        assert got.shape == want.shape
        assert _max_abs_err(got, want) <= 1e-6


@cuda_only
class TestFusedLayerNorm:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_forward_matches_torch(self, shape, dtype):
        import rusty_kernels

        rows, cols = shape
        x = torch.randn(rows, cols, device="cuda", dtype=dtype)
        w = torch.randn(cols, device="cuda", dtype=dtype)
        b = torch.randn(cols, device="cuda", dtype=dtype)
        got = rusty_kernels.layer_norm(x, w, b)
        want = F.layer_norm(x, (cols,), w, b)
        err = _max_abs_err(got, want)
        assert err <= TOL[dtype], f"{shape} {dtype}: max abs err {err:.3e}"

    def test_backward_matches_torch(self):
        import rusty_kernels

        rows, cols = 128, 768
        x0 = torch.randn(rows, cols, device="cuda")
        w0 = torch.randn(cols, device="cuda")
        b0 = torch.randn(cols, device="cuda")
        dy = torch.randn(rows, cols, device="cuda")

        x1, w1, b1 = (t.clone().requires_grad_(True) for t in (x0, w0, b0))
        rusty_kernels.layer_norm(x1, w1, b1).backward(dy)

        x2, w2, b2 = (t.clone().requires_grad_(True) for t in (x0, w0, b0))
        F.layer_norm(x2, (cols,), w2, b2).backward(dy)

        assert _max_abs_err(x1.grad, x2.grad) <= 1e-4, "dx mismatch"
        assert _max_abs_err(w1.grad, w2.grad) <= 1e-3, "dweight mismatch"
        assert _max_abs_err(b1.grad, b2.grad) <= 1e-3, "dbias mismatch"

    def test_gradcheck_float64(self):
        """The acceptance criterion: analytical == numerical gradients."""
        import rusty_kernels

        rows, cols = 4, 33
        x = torch.randn(rows, cols, device="cuda", dtype=torch.float64,
                        requires_grad=True)
        w = torch.randn(cols, device="cuda", dtype=torch.float64,
                        requires_grad=True)
        b = torch.randn(cols, device="cuda", dtype=torch.float64,
                        requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda x_, w_, b_: rusty_kernels.layer_norm(x_, w_, b_),
            (x, w, b), eps=1e-6, atol=1e-5, rtol=1e-4,
        ), "gradcheck failed"

    def test_3d_input(self):
        import rusty_kernels

        x = torch.randn(2, 64, 1024, device="cuda")
        w = torch.ones(1024, device="cuda")
        b = torch.zeros(1024, device="cuda")
        got = rusty_kernels.layer_norm(x, w, b)
        want = F.layer_norm(x, (1024,), w, b)
        assert got.shape == want.shape
        assert _max_abs_err(got, want) <= 1e-6
