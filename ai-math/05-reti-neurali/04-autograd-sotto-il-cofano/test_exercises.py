"""Test per gli esercizi della lezione 04: autograd sotto il cofano.

Esegui `pytest` da questa cartella. Deterministici: seed fissato.
"""

import torch

import exercises


def test_grad_locale_prodotto():
    da, db = exercises.grad_locale_prodotto(3.0, 5.0)
    assert da == 5.0 and db == 3.0


def test_grad_attraverso_relu():
    z = torch.tensor([2.0, -1.0, 0.0, 3.0])
    g = torch.tensor([10.0, 10.0, 10.0, 10.0])
    out = exercises.grad_attraverso_relu(z, g)
    assert torch.allclose(out, torch.tensor([10.0, 0.0, 0.0, 10.0]))


def test_grad_attraverso_relu_contro_autograd():
    torch.manual_seed(0)
    z0 = torch.randn(6)
    z = z0.clone().requires_grad_(True)
    torch.relu(z).sum().backward()  # incoming gradient of ones
    out = exercises.grad_attraverso_relu(z0, torch.ones(6))
    assert torch.allclose(out, z.grad)


def test_backprop_lineare_valori_noti():
    # x=2, w=3, b=0, t=10: out=6, err=-4, dW = 2*(-4)*2 = -16, db = -8
    dw, db = exercises.backprop_lineare(2.0, 3.0, 0.0, 10.0)
    assert abs(dw - (-16.0)) < 1e-9
    assert abs(db - (-8.0)) < 1e-9


def test_backprop_lineare_contro_autograd():
    for x, w0, b0, t in [(0.9, 200.0, 10.0, 250.0), (-1.5, 2.0, 1.0, 0.0)]:
        w = torch.tensor(w0, requires_grad=True)
        b = torch.tensor(b0, requires_grad=True)
        ((w * x + b - t) ** 2).backward()
        dw, db = exercises.backprop_lineare(x, w0, b0, t)
        assert abs(dw - w.grad.item()) < 1e-3
        assert abs(db - b.grad.item()) < 1e-3


def test_backprop_due_strati_contro_autograd():
    for x, w1v, w2v, t in [(2.0, 0.5, -1.5, 3.0), (1.0, 2.0, 2.0, 10.0)]:
        w1 = torch.tensor(w1v, requires_grad=True)
        w2 = torch.tensor(w2v, requires_grad=True)
        ((w2 * (w1 * x) - t) ** 2).backward()
        d1, d2 = exercises.backprop_due_strati(x, w1v, w2v, t)
        assert abs(d1 - w1.grad.item()) < 1e-4
        assert abs(d2 - w2.grad.item()) < 1e-4
