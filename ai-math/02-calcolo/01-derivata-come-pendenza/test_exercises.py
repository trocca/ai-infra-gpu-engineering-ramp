"""Test per gli esercizi della lezione 01: derivata come pendenza.

Esegui `pytest` da questa cartella. Deterministici: nessun numero casuale.
"""

import torch

import exercises

SQM100 = torch.tensor([0.50, 0.70, 0.90, 1.10, 1.30])
PRICES = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])


def test_derivata_finita_x_quadro():
    out = exercises.derivata_finita(lambda x: x**2, 3.0)
    assert abs(out - 6.0) < 0.01


def test_derivata_finita_cubica():
    # derivative of x^3 is 3x^2, at x = 2 it is 12
    out = exercises.derivata_finita(lambda x: x**3, 2.0)
    assert abs(out - 12.0) < 0.05


def test_derivata_autograd_x_quadro():
    out = exercises.derivata_autograd(lambda t: t**2, 3.0)
    assert isinstance(out, float)
    assert abs(out - 6.0) < 1e-5


def test_derivata_autograd_polinomio():
    # derivative of x^2 + 5x is 2x + 5, at x = -1 it is 3
    out = exercises.derivata_autograd(lambda t: t**2 + 5 * t, -1.0)
    assert abs(out - 3.0) < 1e-5


def test_pendenza_in_piu_punti():
    xs = torch.tensor([1.0, 2.0, 3.0])
    out = exercises.pendenza_in_piu_punti(lambda t: t**2, xs)
    assert torch.allclose(out, torch.tensor([2.0, 4.0, 6.0]), atol=1e-4)


def test_loss_una_feature():
    out = exercises.loss_una_feature(torch.tensor(0.0), SQM100, PRICES)
    expected = (PRICES**2).mean()
    assert torch.allclose(torch.as_tensor(out), expected)


def test_loss_una_feature_peso_buono():
    bad = exercises.loss_una_feature(torch.tensor(0.0), SQM100, PRICES)
    good = exercises.loss_una_feature(torch.tensor(265.0), SQM100, PRICES)
    assert good.item() < bad.item()


def test_pendenza_della_loss_segno():
    # At w = 200 the slope must be negative: w should grow.
    out = exercises.pendenza_della_loss(200.0, SQM100, PRICES)
    assert isinstance(out, float)
    assert out < 0


def test_pendenza_della_loss_valore():
    # Closed form: dloss/dw = mean(2 * (w*x - y) * x)
    w = 200.0
    expected = (2 * (w * SQM100 - PRICES) * SQM100).mean().item()
    out = exercises.pendenza_della_loss(w, SQM100, PRICES)
    assert abs(out - expected) < 1e-3
