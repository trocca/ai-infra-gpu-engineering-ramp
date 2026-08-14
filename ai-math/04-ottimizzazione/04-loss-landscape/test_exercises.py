"""Tests for the exercises of lesson 04: the loss landscape.

Run `pytest` from this folder. Deterministic: no random numbers.
"""

import torch

import exercises

SQM = torch.tensor([50.0, 70.0, 90.0, 110.0, 130.0])
X = (SQM - SQM.mean()) / SQM.std()
Y = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])

# Exact least squares solution, used as ground truth in several tests.
W_STAR = (((X - X.mean()) * (Y - Y.mean())).sum() / ((X - X.mean()) ** 2).sum()).item()
B_STAR = (Y.mean() - W_STAR * X.mean()).item()


def test_loss_mse_zero_sul_minimo():
    out = exercises.loss_mse(W_STAR, B_STAR, X, Y)
    assert out.item() < 1e-3


def test_loss_mse_valore_noto():
    # w=0, b=0: loss = mean(y^2)
    out = exercises.loss_mse(0.0, 0.0, X, Y)
    assert torch.allclose(torch.as_tensor(out), (Y**2).mean())


def test_loss_su_griglia_forma_e_valori():
    ws = torch.tensor([0.0, 50.0, 100.0])
    bs = torch.tensor([200.0, 250.0])
    L = exercises.loss_su_griglia(ws, bs, X, Y)
    assert L.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            expected = exercises.loss_mse(ws[j], bs[i], X, Y)
            assert torch.allclose(L[i, j], torch.as_tensor(expected))


def test_minimo_della_griglia():
    ws = torch.linspace(-20, 180, 101)
    bs = torch.linspace(130, 370, 121)
    L = exercises.loss_su_griglia(ws, bs, X, Y)
    w_best, b_best = exercises.minimo_della_griglia(L, ws, bs)
    assert abs(w_best.item() - W_STAR) < 3
    assert abs(b_best.item() - B_STAR) < 3


def test_traiettoria_gd_forma():
    path = exercises.traiettoria_gd(X, Y, w0=0.0, b0=0.0, lr=0.3, passi=10)
    assert path.shape == (11, 2)
    assert torch.allclose(path[0], torch.tensor([0.0, 0.0]))


def test_traiettoria_gd_scende():
    path = exercises.traiettoria_gd(X, Y, w0=-10.0, b0=150.0, lr=0.3, passi=50)
    prima = exercises.loss_mse(path[0, 0], path[0, 1], X, Y)
    dopo = exercises.loss_mse(path[-1, 0], path[-1, 1], X, Y)
    assert dopo.item() < prima.item() / 100


def test_traiettoria_gd_arriva_al_fondo():
    path = exercises.traiettoria_gd(X, Y, w0=0.0, b0=0.0, lr=0.4, passi=80)
    assert abs(path[-1, 0].item() - W_STAR) < 1.0
    assert abs(path[-1, 1].item() - B_STAR) < 1.0
