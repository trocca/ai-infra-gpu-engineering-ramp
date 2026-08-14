"""Test per gli esercizi della lezione 01: gradient descent a mano.

Esegui `pytest` da questa cartella. Deterministici: nessun numero casuale.
"""

import torch

import exercises


def f_ciotola(p):
    return (p**2).sum()


def test_passo_gd_scalare():
    out = exercises.passo_gd(0.0, -6.0, 0.25)
    assert abs(out - 1.5) < 1e-9


def test_passo_gd_tensor():
    w = torch.tensor([1.0, 2.0])
    grad = torch.tensor([0.5, -0.5])
    out = exercises.passo_gd(w, grad, 0.1)
    assert torch.allclose(out, torch.tensor([0.95, 2.05]))


def test_gd_su_parabola_primi_passi():
    # From the SPIEGAZIONE: after 2 steps with lr 0.25, w = 2.25
    out = exercises.gd_su_parabola(0.0, 0.25, 2)
    assert abs(out - 2.25) < 1e-6


def test_gd_su_parabola_convergenza():
    out = exercises.gd_su_parabola(0.0, 0.25, 30)
    assert abs(out - 3.0) < 1e-3


def test_gd_su_parabola_divergenza():
    out = exercises.gd_su_parabola(0.0, 1.1, 20)
    assert abs(out - 3.0) > 10, "con lr 1.1 deve divergere"


def test_gd_autograd_ciotola():
    out = exercises.gd_autograd(f_ciotola, torch.tensor([4.0, -2.0]), 0.1, 50)
    assert torch.allclose(out, torch.zeros(2), atol=1e-3)
    assert not out.requires_grad


def test_gd_autograd_non_modifica_partenza():
    w0 = torch.tensor([4.0, -2.0])
    exercises.gd_autograd(f_ciotola, w0, 0.1, 5)
    assert torch.equal(w0, torch.tensor([4.0, -2.0])), "w0 non va modificato"


def test_gd_con_storia_decrescente():
    _, storia = exercises.gd_con_storia(f_ciotola, torch.tensor([4.0, -2.0]), 0.1, 20)
    assert len(storia) == 20
    assert storia[-1] < storia[0]
    # Every step downhill on a convex bowl with a sane lr.
    for prima, dopo in zip(storia, storia[1:]):
        assert dopo <= prima + 1e-9
