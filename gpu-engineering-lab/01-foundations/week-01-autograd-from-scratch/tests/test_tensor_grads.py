"""Acceptance tests for the Tensor engine, nn, and optim (COMPLETE — do not edit).

Strategy: build the identical computation in your engine and in PyTorch
(float64 both sides), backprop a scalar loss, compare every gradient with
relative error <= 1e-6. Shapes are randomized, including broadcast
patterns, because broadcasting backward is where tensor engines break.

Run: pytest tests/test_tensor_grads.py -v
"""

import numpy as np
import pytest
import torch

from src.nn import Linear, ReLU, Sequential, softmax_cross_entropy
from src.optim import SGD, Adam
from src.tensor import Tensor

REL_TOL = 1e-6
RNG = np.random.default_rng(42)


def _pair(shape):
    """Return (mine, theirs) leaf tensors with identical float64 data."""
    data = RNG.standard_normal(shape)
    mine = Tensor(data.copy(), requires_grad=True)
    theirs = torch.tensor(data.copy(), dtype=torch.float64, requires_grad=True)
    return mine, theirs


def _assert_grads_close(mine: Tensor, theirs: torch.Tensor, what: str):
    ref = theirs.grad.detach().numpy()
    got = mine.grad
    assert got.shape == ref.shape, f"{what}: grad shape {got.shape} != {ref.shape}"
    denom = np.maximum(np.abs(ref), 1e-12)
    rel = np.max(np.abs(got - ref) / denom)
    assert rel <= REL_TOL, f"{what}: max rel grad error {rel:.3e} > {REL_TOL}"


BROADCAST_CASES = [
    ((4, 5), (4, 5)),   # same shape
    ((4, 5), (5,)),     # trailing-dim broadcast (the Linear bias case)
    ((4, 5), (1, 5)),   # keepdim broadcast
    ((4, 1), (1, 5)),   # both broadcast
    ((2, 3, 4), (4,)),  # 3-D vs 1-D
    ((2, 3, 4), (3, 1)),
]


class TestBinaryOpsWithBroadcasting:
    @pytest.mark.parametrize("sa,sb", BROADCAST_CASES)
    def test_add(self, sa, sb):
        a, ta = _pair(sa)
        b, tb = _pair(sb)
        ((a + b) * (a + b)).sum().backward()
        ((ta + tb) * (ta + tb)).sum().backward()
        _assert_grads_close(a, ta, f"add grad a {sa}+{sb}")
        _assert_grads_close(b, tb, f"add grad b {sa}+{sb}")

    @pytest.mark.parametrize("sa,sb", BROADCAST_CASES)
    def test_mul(self, sa, sb):
        a, ta = _pair(sa)
        b, tb = _pair(sb)
        (a * b).sum().backward()
        (ta * tb).sum().backward()
        _assert_grads_close(a, ta, f"mul grad a {sa}*{sb}")
        _assert_grads_close(b, tb, f"mul grad b {sa}*{sb}")

    @pytest.mark.parametrize("sa,sb", [((4, 5), (5,)), ((4, 5), (4, 5)), ((4, 1), (1, 5))])
    def test_sub_div(self, sa, sb):
        a, ta = _pair(sa)
        b, tb = _pair(sb)
        bo = b * 0.1 + 3.0   # keep denominators away from zero
        tbo = tb * 0.1 + 3.0
        ((a - b) / bo).sum().backward()
        ((ta - tb) / tbo).sum().backward()
        _assert_grads_close(a, ta, "sub/div grad a")
        _assert_grads_close(b, tb, "sub/div grad b")


class TestMatmulReductionsUnary:
    def test_matmul(self):
        a, ta = _pair((6, 4))
        b, tb = _pair((4, 3))
        (a @ b).sum().backward()
        (ta @ tb).sum().backward()
        _assert_grads_close(a, ta, "matmul grad A")
        _assert_grads_close(b, tb, "matmul grad B")

    @pytest.mark.parametrize("axis,keepdims", [(None, False), (0, False), (1, True)])
    def test_sum_mean(self, axis, keepdims):
        a, ta = _pair((5, 7))
        (a.sum(axis=axis, keepdims=keepdims) * 2.0).sum().backward()
        (ta.sum(dim=axis, keepdim=keepdims) * 2.0).sum().backward() if axis is not None \
            else ((ta.sum() * 2.0).backward())
        _assert_grads_close(a, ta, f"sum axis={axis}")

        b, tb = _pair((5, 7))
        if axis is None:
            b.mean().backward()
            tb.mean().backward()
        else:
            b.mean(axis=axis, keepdims=keepdims).sum().backward()
            tb.mean(dim=axis, keepdim=keepdims).sum().backward()
        _assert_grads_close(b, tb, f"mean axis={axis}")

    def test_max_rows(self):
        a, ta = _pair((6, 9))
        a.max(axis=1, keepdims=True).sum().backward()
        ta.max(dim=1, keepdim=True).values.sum().backward()
        _assert_grads_close(a, ta, "row max")

    def test_relu_exp_log_reshape_transpose(self):
        a, ta = _pair((3, 8))
        out = (a.relu() + 0.5).log().exp().reshape(8, 3).T.sum()
        tout = (ta.relu() + 0.5).log().exp().reshape(8, 3).T.sum()
        out.backward()
        tout.backward()
        _assert_grads_close(a, ta, "unary/shape chain")

    def test_gather_rows(self):
        a, ta = _pair((5, 10))
        idx = RNG.integers(0, 10, size=5)
        a.gather_rows(idx).sum().backward()
        ta[torch.arange(5), torch.tensor(idx)].sum().backward()
        _assert_grads_close(a, ta, "gather_rows")


class TestSoftmaxCrossEntropy:
    def test_matches_torch(self):
        n, c = 16, 10
        logits_np = RNG.standard_normal((n, c)) * 3.0
        targets = RNG.integers(0, c, size=n)

        logits = Tensor(logits_np.copy(), requires_grad=True)
        loss = softmax_cross_entropy(logits, targets)
        loss.backward()

        tlogits = torch.tensor(logits_np.copy(), dtype=torch.float64, requires_grad=True)
        tloss = torch.nn.functional.cross_entropy(tlogits, torch.tensor(targets))
        tloss.backward()

        assert abs(loss.data - tloss.item()) / abs(tloss.item()) <= REL_TOL
        _assert_grads_close(logits, tlogits, "cross-entropy grad")

    def test_stable_under_large_logits(self):
        logits_np = RNG.standard_normal((4, 6)) * 500.0  # would overflow naive exp
        targets = RNG.integers(0, 6, size=4)
        logits = Tensor(logits_np.copy(), requires_grad=True)
        loss = softmax_cross_entropy(logits, targets)
        assert np.isfinite(loss.data), "cross-entropy overflowed — subtract the row max"
        loss.backward()
        assert np.all(np.isfinite(logits.grad))


def _mirrored_models(rng_seed=7):
    """Build an identical 2-layer MLP in both frameworks (copied weights)."""
    rng = np.random.default_rng(rng_seed)
    mine = Sequential(Linear(20, 32, rng=rng), ReLU(), Linear(32, 5, rng=rng))

    torch_model = torch.nn.Sequential(
        torch.nn.Linear(20, 32), torch.nn.ReLU(), torch.nn.Linear(32, 5)
    ).double()
    my_linears = [l for l in mine.layers if isinstance(l, Linear)]
    torch_linears = [l for l in torch_model if isinstance(l, torch.nn.Linear)]
    with torch.no_grad():
        for ml, tl in zip(my_linears, torch_linears):
            tl.weight.copy_(torch.tensor(ml.W.data.T))  # torch stores (out, in)
            tl.bias.copy_(torch.tensor(ml.b.data))
    return mine, torch_model, my_linears, torch_linears


class TestEndToEndMLP:
    def test_mlp_gradients_match(self):
        mine, torch_model, my_linears, torch_linears = _mirrored_models()
        x_np = RNG.standard_normal((8, 20))
        y_np = RNG.integers(0, 5, size=8)

        loss = softmax_cross_entropy(mine(Tensor(x_np.copy())), y_np)
        loss.backward()

        tx = torch.tensor(x_np.copy(), dtype=torch.float64)
        tloss = torch.nn.functional.cross_entropy(torch_model(tx), torch.tensor(y_np))
        tloss.backward()

        for i, (ml, tl) in enumerate(zip(my_linears, torch_linears)):
            ref_w = tl.weight.grad.detach().numpy().T
            rel = np.max(np.abs(ml.W.grad - ref_w) / np.maximum(np.abs(ref_w), 1e-12))
            assert rel <= REL_TOL, f"layer {i} W grad rel err {rel:.3e}"
            ref_b = tl.bias.grad.detach().numpy()
            rel = np.max(np.abs(ml.b.grad - ref_b) / np.maximum(np.abs(ref_b), 1e-12))
            assert rel <= REL_TOL, f"layer {i} b grad rel err {rel:.3e}"


class TestOptimizers:
    @pytest.mark.parametrize("opt_name", ["sgd", "sgd_momentum", "adam"])
    def test_updates_match_torch(self, opt_name):
        w_np = RNG.standard_normal((4, 3))
        mine = Tensor(w_np.copy(), requires_grad=True)
        theirs = torch.tensor(w_np.copy(), dtype=torch.float64, requires_grad=True)

        if opt_name == "sgd":
            my_opt = SGD([mine], lr=0.1)
            t_opt = torch.optim.SGD([theirs], lr=0.1)
        elif opt_name == "sgd_momentum":
            my_opt = SGD([mine], lr=0.1, momentum=0.9)
            t_opt = torch.optim.SGD([theirs], lr=0.1, momentum=0.9)
        else:
            my_opt = Adam([mine], lr=0.05)
            t_opt = torch.optim.Adam([theirs], lr=0.05)

        for step in range(5):
            my_opt.zero_grad()
            t_opt.zero_grad()
            (mine * mine).sum().backward()
            (theirs * theirs).sum().backward()
            my_opt.step()
            t_opt.step()
            ref = theirs.detach().numpy()
            rel = np.max(np.abs(mine.data - ref) / np.maximum(np.abs(ref), 1e-12))
            assert rel <= 1e-6, f"{opt_name} diverged from torch at step {step}: {rel:.3e}"
