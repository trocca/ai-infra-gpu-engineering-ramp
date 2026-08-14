"""Tests for the exercises of lesson 04: matrix calculus and Jacobians.

Run `pytest` from this folder. Deterministic: fixed seed.
"""

import torch

import exercises

torch.manual_seed(0)
W_RANDOM = torch.randn(3, 2)


def f_esempio(p):
    return torch.stack([p[0] ** 2, p[0] * p[1]])


def f_lineare(x):
    return W_RANDOM @ x


def test_jacobiana_a_mano():
    out = exercises.jacobiana_a_mano(torch.tensor([2.0, 3.0]))
    expected = torch.tensor([[4.0, 0.0], [3.0, 2.0]])
    assert torch.allclose(out, expected)


def test_jacobiana_finita_esempio():
    out = exercises.jacobiana_finita(f_esempio, torch.tensor([2.0, 3.0]))
    expected = torch.tensor([[4.0, 0.0], [3.0, 2.0]])
    assert out.shape == (2, 2)
    assert torch.allclose(out, expected, atol=1e-2)


def test_jacobiana_finita_rettangolare():
    # 3 outputs, 2 inputs: shape must be (3, 2)
    out = exercises.jacobiana_finita(f_lineare, torch.tensor([0.5, -1.0]))
    assert out.shape == (3, 2)
    assert torch.allclose(out, W_RANDOM, atol=1e-2)


def test_jacobiana_autograd():
    out = exercises.jacobiana_autograd(f_esempio, torch.tensor([2.0, 3.0]))
    expected = torch.tensor([[4.0, 0.0], [3.0, 2.0]])
    assert torch.allclose(out, expected, atol=1e-5)


def test_jacobiana_lineare_e_w():
    out = exercises.jacobiana_autograd(f_lineare, torch.tensor([1.0, 1.0]))
    assert torch.allclose(out, W_RANDOM, atol=1e-5)


def test_gradiente_mse():
    X = torch.tensor(
        [
            [0.50, 2.0],
            [0.70, 3.0],
            [0.90, 3.0],
            [1.10, 4.0],
            [1.30, 5.0],
        ]
    )
    y = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])
    w = torch.tensor([100.0, 10.0])
    out = exercises.gradiente_mse(w, X, y)
    assert out.shape == w.shape
    expected = 2 / len(y) * X.T @ (X @ w - y)
    assert torch.allclose(out, expected, atol=1e-3)
