"""Acceptance tests for the scalar Value engine (COMPLETE — do not edit).

Every test builds the same expression twice — once with your Value, once
with torch scalars — runs backward on both, and compares data and grads.
PyTorch is the oracle here, never the implementation.

Run: pytest tests/test_scalar.py -v
"""

import math

import pytest
import torch

from src.scalar_engine import Value

REL_TOL = 1e-6


def _torch_scalar(x: float) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64, requires_grad=True)


def _assert_close(mine: float, ref: float, what: str):
    denom = max(abs(ref), 1e-12)
    rel = abs(mine - ref) / denom
    assert rel <= REL_TOL, f"{what}: mine={mine!r} ref={ref!r} rel_err={rel:.3e}"


class TestForward:
    def test_add_mul(self):
        v = Value(2.0) * Value(3.0) + Value(4.0)
        assert v.data == pytest.approx(10.0)

    def test_reflected_ops_with_python_numbers(self):
        v = Value(3.0)
        assert (2 + v).data == pytest.approx(5.0)
        assert (2 * v).data == pytest.approx(6.0)
        assert (2 - v).data == pytest.approx(-1.0)
        assert (6 / v).data == pytest.approx(2.0)

    def test_pow_neg_div(self):
        v = Value(4.0)
        assert (v ** 0.5).data == pytest.approx(2.0)
        assert (-v).data == pytest.approx(-4.0)
        assert (v / 8.0).data == pytest.approx(0.5)

    def test_unary(self):
        assert Value(-1.5).relu().data == pytest.approx(0.0)
        assert Value(1.5).relu().data == pytest.approx(1.5)
        assert Value(0.5).tanh().data == pytest.approx(math.tanh(0.5))
        assert Value(0.5).exp().data == pytest.approx(math.exp(0.5))
        assert Value(0.5).log().data == pytest.approx(math.log(0.5))


class TestBackward:
    def test_simple_chain(self):
        # f(a, b) = (a * b + a) ** 2
        a, b = Value(1.5), Value(-2.0)
        f = (a * b + a) ** 2
        f.backward()

        ta, tb = _torch_scalar(1.5), _torch_scalar(-2.0)
        tf = (ta * tb + ta) ** 2
        tf.backward()

        _assert_close(f.data, tf.item(), "forward")
        _assert_close(a.grad, ta.grad.item(), "grad a")
        _assert_close(b.grad, tb.grad.item(), "grad b")

    def test_fanout_accumulation(self):
        # The same node used multiple times must accumulate gradient.
        a = Value(3.0)
        f = a * a + a.exp() + a * 2.0
        f.backward()

        ta = _torch_scalar(3.0)
        tf = ta * ta + ta.exp() + ta * 2.0
        tf.backward()

        _assert_close(a.grad, ta.grad.item(), "fan-out grad")

    def test_relu_kink(self):
        for x0 in (-2.0, 0.5, 3.0):
            a = Value(x0)
            f = (a * 2.0 - 1.0).relu() * 3.0
            f.backward()

            ta = _torch_scalar(x0)
            tf = torch.relu(ta * 2.0 - 1.0) * 3.0
            tf.backward()
            _assert_close(a.grad, ta.grad.item(), f"relu grad at x={x0}")

    def test_tanh_and_division(self):
        a, b = Value(0.7), Value(1.3)
        f = (a / b + b.tanh()) * (a - 2.0)
        f.backward()

        ta, tb = _torch_scalar(0.7), _torch_scalar(1.3)
        tf = (ta / tb + tb.tanh()) * (ta - 2.0)
        tf.backward()

        _assert_close(a.grad, ta.grad.item(), "grad a")
        _assert_close(b.grad, tb.grad.item(), "grad b")

    def test_deep_composition_neuron(self):
        # A tiny neuron: tanh(w1*x1 + w2*x2 + b), grads for weights.
        w1, w2, b = Value(-0.5), Value(0.8), Value(0.1)
        x1, x2 = 2.0, -1.0
        out = (w1 * x1 + w2 * x2 + b).tanh()
        out.backward()

        tw1, tw2, tb = _torch_scalar(-0.5), _torch_scalar(0.8), _torch_scalar(0.1)
        tout = (tw1 * x1 + tw2 * x2 + tb).tanh()
        tout.backward()

        _assert_close(w1.grad, tw1.grad.item(), "grad w1")
        _assert_close(w2.grad, tw2.grad.item(), "grad w2")
        _assert_close(b.grad, tb.grad.item(), "grad b")

    def test_log_softmaxish_expression(self):
        # log(exp(a) / (exp(a) + exp(b))) — the scalar seed of cross-entropy.
        a, b = Value(0.3), Value(-1.1)
        f = (a.exp() / (a.exp() + b.exp())).log()
        f.backward()

        ta, tb = _torch_scalar(0.3), _torch_scalar(-1.1)
        tf = (ta.exp() / (ta.exp() + tb.exp())).log()
        tf.backward()

        _assert_close(a.grad, ta.grad.item(), "grad a")
        _assert_close(b.grad, tb.grad.item(), "grad b")


class TestTopologicalOrder:
    def test_diamond_graph(self):
        # a -> (b, c) -> d : _backward on a must run only after b and c.
        a = Value(2.0)
        bb = a * 3.0
        c = a ** 2
        d = bb * c
        d.backward()

        ta = _torch_scalar(2.0)
        td = (ta * 3.0) * (ta ** 2)
        td.backward()
        _assert_close(a.grad, ta.grad.item(), "diamond grad")

    def test_repeated_backward_requires_fresh_graph(self):
        # Grad accumulates across backward calls (PyTorch semantics).
        a = Value(1.0)
        f = a * 5.0
        f.backward()
        first = a.grad
        assert first == pytest.approx(5.0)
