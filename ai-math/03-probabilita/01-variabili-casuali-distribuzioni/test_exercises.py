"""Test per gli esercizi della lezione 01: variabili casuali.

Esegui `pytest` da questa cartella. Deterministici: seed fissati.
"""

import torch

import exercises


def test_valore_atteso_moneta():
    out = exercises.valore_atteso(torch.tensor([0.0, 1.0]), torch.tensor([0.3, 0.7]))
    assert torch.allclose(torch.as_tensor(out), torch.tensor(0.7))


def test_valore_atteso_dado():
    valori = torch.arange(1.0, 7.0)
    probs = torch.full((6,), 1 / 6)
    out = exercises.valore_atteso(valori, probs)
    assert torch.allclose(torch.as_tensor(out), torch.tensor(3.5))


def test_varianza_bernoulli():
    out = exercises.varianza(torch.tensor([0.0, 1.0]), torch.tensor([0.3, 0.7]))
    assert torch.allclose(torch.as_tensor(out), torch.tensor(0.21), atol=1e-6)


def test_varianza_costante_e_zero():
    # A "random" variable that always outputs 5 has zero variance.
    out = exercises.varianza(torch.tensor([5.0]), torch.tensor([1.0]))
    assert torch.allclose(torch.as_tensor(out), torch.tensor(0.0), atol=1e-6)


def test_simula_moneta_deterministica():
    a = exercises.simula_moneta(0.7, 100, seed=42)
    b = exercises.simula_moneta(0.7, 100, seed=42)
    assert torch.equal(a, b), "stesso seed deve dare stessi lanci"
    assert a.shape == (100,)
    assert set(a.unique().tolist()).issubset({0.0, 1.0})


def test_simula_moneta_frequenza():
    lanci = exercises.simula_moneta(0.7, 50_000, seed=0)
    assert abs(lanci.mean().item() - 0.7) < 0.02


def test_probabilita_empirica():
    campioni = torch.tensor([1.0, 0.0, 1.0, 1.0])
    out = exercises.probabilita_empirica(campioni, 1.0)
    assert torch.allclose(torch.as_tensor(out), torch.tensor(0.75))


def test_entro_k_sigma_gaussiana():
    torch.manual_seed(0)
    campioni = torch.distributions.Normal(0.0, 1.0).sample((50_000,))
    out = exercises.entro_k_sigma(campioni, 0.0, 1.0, 2)
    assert abs(out.item() - 0.954) < 0.01


def test_entro_k_sigma_tutti_dentro():
    campioni = torch.tensor([1.0, 2.0, 3.0])
    out = exercises.entro_k_sigma(campioni, 2.0, 10.0, 1)
    assert torch.allclose(torch.as_tensor(out), torch.tensor(1.0))
