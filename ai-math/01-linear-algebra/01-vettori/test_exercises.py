"""Test per gli esercizi della lezione 01: vettori.

Esegui `pytest` da questa cartella. Deterministici: nessun numero casuale.
"""

import torch

import exercises


def test_crea_vettore():
    v = exercises.crea_vettore([1, 2, 3])
    assert isinstance(v, torch.Tensor)
    assert v.dtype == torch.float32
    assert torch.equal(v, torch.tensor([1.0, 2.0, 3.0]))


def test_scala_vettore():
    v = torch.tensor([2.0, 1.0, -4.0])
    out = exercises.scala_vettore(v, 3)
    assert torch.allclose(out, torch.tensor([6.0, 3.0, -12.0]))


def test_scala_vettore_per_zero():
    v = torch.tensor([5.0, -7.0])
    out = exercises.scala_vettore(v, 0)
    assert torch.allclose(out, torch.zeros(2))


def test_combinazione_lineare():
    u = torch.tensor([1.0, 0.0])
    v = torch.tensor([0.0, 1.0])
    out = exercises.combinazione_lineare(u, v, 2, 3)
    assert torch.allclose(out, torch.tensor([2.0, 3.0]))


def test_combinazione_lineare_generale():
    u = torch.tensor([1.0, 2.0, 3.0])
    v = torch.tensor([-1.0, 0.5, 2.0])
    out = exercises.combinazione_lineare(u, v, 2.0, -2.0)
    assert torch.allclose(out, torch.tensor([4.0, 3.0, 2.0]))


def test_dot_manuale():
    u = torch.tensor([2.0, 1.0])
    v = torch.tensor([1.0, 3.0])
    out = exercises.dot_manuale(u, v)
    assert isinstance(out, float)
    assert abs(out - 5.0) < 1e-6


def test_dot_manuale_come_torch():
    u = torch.tensor([1.5, -2.0, 0.5, 4.0])
    v = torch.tensor([2.0, 1.0, -3.0, 0.5])
    out = exercises.dot_manuale(u, v)
    expected = torch.dot(u, v).item()
    assert abs(out - expected) < 1e-5


def test_predici_prezzo():
    x = torch.tensor([50.0, 2.0])
    w = torch.tensor([2.0, 10.0])
    out = exercises.predici_prezzo(x, w, 20.0)
    assert torch.allclose(torch.as_tensor(out), torch.tensor(140.0))


def test_predici_prezzo_altra_casa():
    x = torch.tensor([110.0, 4.0])
    w = torch.tensor([2.0, 10.0])
    out = exercises.predici_prezzo(x, w, 20.0)
    assert torch.allclose(torch.as_tensor(out), torch.tensor(280.0))
