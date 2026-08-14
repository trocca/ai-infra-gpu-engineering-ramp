"""Tests for the lesson 01 exercises: linear regression from scratch.

Run `pytest` from this folder. Deterministic: no random numbers.
"""

import torch

import exercises

X_RAW = torch.tensor(
    [
        [50.0, 2.0],
        [70.0, 3.0],
        [90.0, 3.0],
        [110.0, 4.0],
        [130.0, 5.0],
    ]
)
Y = torch.tensor([150.0, 200.0, 250.0, 300.0, 350.0])
X = (X_RAW - X_RAW.mean(dim=0)) / X_RAW.std(dim=0)


def test_forward():
    w = torch.tensor([2.0, 10.0])
    out = exercises.forward(X_RAW, w, 20.0)
    expected = torch.tensor([140.0, 190.0, 230.0, 280.0, 330.0])
    assert torch.allclose(out, expected)


def test_mse():
    preds = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 6.0])
    out = exercises.mse(preds, target)
    assert torch.allclose(torch.as_tensor(out), torch.tensor(3.0))


def test_mse_perfetta_e_zero():
    out = exercises.mse(Y, Y)
    assert torch.allclose(torch.as_tensor(out), torch.tensor(0.0))


def test_passo_di_training_riduce_la_loss():
    w = torch.zeros(2)
    b = torch.zeros(1)
    w1, b1, loss0 = exercises.passo_di_training(X, Y, w, b, lr=0.1)
    _, _, loss1 = exercises.passo_di_training(X, Y, w1, b1, lr=0.1)
    assert loss1 < loss0
    assert not w1.requires_grad and not b1.requires_grad


def test_passo_di_training_loss_iniziale():
    w = torch.zeros(2)
    b = torch.zeros(1)
    _, _, loss0 = exercises.passo_di_training(X, Y, w, b, lr=0.1)
    assert abs(loss0 - (Y**2).mean().item()) < 1e-3


def test_allena_converge_alla_soluzione_esatta():
    w, b, storia = exercises.allena(X, Y, lr=0.1, epoche=2000)
    assert len(storia) == 2000
    assert storia[-1] < 0.5
    A = torch.cat([X, torch.ones(5, 1)], dim=1)
    exact = torch.linalg.lstsq(A, Y.unsqueeze(1)).solution.squeeze()
    assert torch.allclose(w, exact[:2], atol=0.5)
    assert torch.allclose(b, exact[2:], atol=0.5)


def test_allena_storia_decrescente():
    _, _, storia = exercises.allena(X, Y, lr=0.1, epoche=50)
    assert storia[-1] < storia[0] / 10
