"""Test per gli esercizi della lezione 02: derivate parziali e gradiente.

Esegui `pytest` da questa cartella. Deterministici: nessun numero casuale.
"""

import torch

import exercises


def f_esempio(p):
    # f(x, y) = x^2 + 3y, the SPIEGAZIONE example
    return p[0] ** 2 + 3 * p[1]


def f_ciotola(p):
    return (p**2).sum()


def test_parziale_finita_x():
    out = exercises.parziale_finita(f_esempio, torch.tensor([2.0, 1.0]), 0)
    assert abs(out - 4.0) < 1e-2


def test_parziale_finita_y():
    out = exercises.parziale_finita(f_esempio, torch.tensor([2.0, 1.0]), 1)
    assert abs(out - 3.0) < 1e-2


def test_parziale_non_modifica_il_punto():
    punto = torch.tensor([2.0, 1.0])
    exercises.parziale_finita(f_esempio, punto, 0)
    assert torch.equal(punto, torch.tensor([2.0, 1.0])), "il punto originale e' stato modificato"


def test_gradiente_autograd():
    out = exercises.gradiente_autograd(f_esempio, torch.tensor([2.0, 1.0]))
    assert torch.allclose(out, torch.tensor([4.0, 3.0]), atol=1e-5)


def test_gradiente_autograd_ciotola():
    out = exercises.gradiente_autograd(f_ciotola, torch.tensor([1.0, -2.0, 0.5]))
    assert torch.allclose(out, torch.tensor([2.0, -4.0, 1.0]), atol=1e-5)


def test_gradiente_ciotola_formula():
    punto = torch.tensor([3.0, -1.5])
    out = exercises.gradiente_ciotola(punto)
    assert torch.allclose(out, torch.tensor([6.0, -3.0]))


def test_formula_uguale_autograd():
    punto = torch.tensor([0.7, 2.2])
    a_mano = exercises.gradiente_ciotola(punto)
    con_autograd = exercises.gradiente_autograd(f_ciotola, punto)
    assert torch.allclose(a_mano, con_autograd, atol=1e-5)


def test_passo_in_discesa_riduce_f():
    punto = torch.tensor([2.0, 2.0])
    nuovo = exercises.passo_in_discesa(f_ciotola, punto, lr=0.1)
    assert f_ciotola(nuovo).item() < f_ciotola(punto).item()
    assert not nuovo.requires_grad


def test_passo_in_discesa_valore():
    # From [2, 2], gradient [4, 4], lr 0.1: new point [1.6, 1.6]
    nuovo = exercises.passo_in_discesa(f_ciotola, torch.tensor([2.0, 2.0]), lr=0.1)
    assert torch.allclose(nuovo, torch.tensor([1.6, 1.6]), atol=1e-5)
