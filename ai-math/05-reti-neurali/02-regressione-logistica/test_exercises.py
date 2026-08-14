"""Tests for the lesson 02 exercises: logistic regression.

Run `pytest` from this folder. Deterministic: no random numbers.
"""

import torch
import torch.nn.functional as F

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
X = (X_RAW - X_RAW.mean(dim=0)) / X_RAW.std(dim=0)
Y = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])


def test_sigmoid_a_mano_punti_noti():
    assert torch.allclose(
        torch.as_tensor(exercises.sigmoid_a_mano(0.0)), torch.tensor(0.5)
    )
    z = torch.tensor([-2.0, 0.0, 2.0])
    assert torch.allclose(exercises.sigmoid_a_mano(z), torch.sigmoid(z), atol=1e-6)


def test_sigmoid_simmetrica():
    z = torch.tensor(1.7)
    assert torch.allclose(
        exercises.sigmoid_a_mano(z) + exercises.sigmoid_a_mano(-z),
        torch.tensor(1.0),
        atol=1e-6,
    )


def test_bce_a_mano_come_torch():
    p = torch.tensor([0.88, 0.12, 0.5])
    t = torch.tensor([1.0, 0.0, 1.0])
    out = exercises.bce_a_mano(p, t)
    expected = F.binary_cross_entropy(p, t)
    assert torch.allclose(torch.as_tensor(out), expected, atol=1e-6)


def test_bce_punisce_la_sicurezza_sbagliata():
    giusto = exercises.bce_a_mano(torch.tensor([0.99]), torch.tensor([1.0]))
    sbagliato = exercises.bce_a_mano(torch.tensor([0.01]), torch.tensor([1.0]))
    assert sbagliato.item() > 100 * giusto.item()


def test_probabilita_case_range():
    w = torch.tensor([1.0, 1.0])
    out = exercises.probabilita_case(X, w, 0.0)
    assert out.shape == (5,)
    assert (out > 0).all() and (out < 1).all()


def test_classifica():
    w = torch.tensor([5.0, 5.0])  # big weights: confident on standardized X
    out = exercises.classifica(X, w, 0.0)
    assert set(out.unique().tolist()).issubset({0.0, 1.0})
    # Standardized features: biggest house positive score, smallest negative.
    assert out[4].item() == 1.0
    assert out[0].item() == 0.0


def test_allena_logistica_impara():
    w, b, storia = exercises.allena_logistica(X, Y, lr=0.5, epoche=400)
    assert storia[-1] < storia[0]
    predizioni = exercises.classifica(X, w, b)
    assert torch.equal(predizioni, Y), "on the toy dataset it must score 5 out of 5"
    assert not w.requires_grad
