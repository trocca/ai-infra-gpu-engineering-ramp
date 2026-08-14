"""Tests for the exercises of lesson 03: the chain rule.

Run `pytest` from this folder. Deterministic: no random numbers.
"""

import torch

import exercises


def test_catena_due_anelli_esempio():
    out = exercises.catena_due_anelli(2.0)
    assert isinstance(out, float)
    assert abs(out - 42.0) < 1e-6


def test_catena_due_anelli_altri_punti():
    # y = (3x + 1)^2 expands to 9x^2 + 6x + 1, derivative 18x + 6
    for x in [-1.0, 0.0, 0.5, 3.0]:
        assert abs(exercises.catena_due_anelli(x) - (18 * x + 6)) < 1e-6


def test_catena_loss_a_mano():
    # w=200, x=0.9, y=250: pred=180, err=-70, dloss/dw = 2*(-70)*0.9 = -126
    out = exercises.catena_loss_a_mano(200.0, 0.9, 250.0)
    assert abs(out - (-126.0)) < 1e-6


def test_catena_loss_autograd():
    out = exercises.catena_loss_autograd(200.0, 0.9, 250.0)
    assert abs(out - (-126.0)) < 1e-3


def test_mano_uguale_autograd():
    for w, x, y in [(100.0, 0.5, 150.0), (300.0, 1.3, 350.0), (0.0, 0.7, 200.0)]:
        a_mano = exercises.catena_loss_a_mano(w, x, y)
        auto = exercises.catena_loss_autograd(w, x, y)
        assert abs(a_mano - auto) < 1e-2


def test_catena_tre_anelli():
    # At x = 1: u=2, v=5, dt/dx = 3*25 * 2*2 * 2 = 600
    out = exercises.catena_tre_anelli(1.0)
    assert abs(out - 600.0) < 1e-6


def test_catena_tre_anelli_contro_autograd():
    for x in [0.5, 1.5, -1.0]:
        xt = torch.tensor(x, requires_grad=True)
        t = ((2 * xt) ** 2 + 1) ** 3
        t.backward()
        assert abs(exercises.catena_tre_anelli(x) - xt.grad.item()) < 1e-2
