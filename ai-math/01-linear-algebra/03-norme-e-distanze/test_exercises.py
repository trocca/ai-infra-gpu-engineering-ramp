"""Test per gli esercizi della lezione 03: norme e distanze.

Esegui `pytest` da questa cartella. Deterministici: seed fissato.
"""

import torch

import exercises

torch.manual_seed(0)
V_RANDOM = torch.randn(6)


def test_norma_l1():
    v = torch.tensor([3.0, -4.0])
    assert torch.allclose(torch.as_tensor(exercises.norma_l1(v)), torch.tensor(7.0))


def test_norma_l1_come_torch():
    out = torch.as_tensor(exercises.norma_l1(V_RANDOM))
    assert torch.allclose(out, torch.linalg.norm(V_RANDOM, ord=1))


def test_norma_l2():
    v = torch.tensor([3.0, 4.0])
    assert torch.allclose(torch.as_tensor(exercises.norma_l2(v)), torch.tensor(5.0))


def test_norma_l2_come_torch():
    out = torch.as_tensor(exercises.norma_l2(V_RANDOM))
    assert torch.allclose(out, torch.linalg.norm(V_RANDOM, ord=2))


def test_norma_linf():
    v = torch.tensor([3.0, -4.0])
    assert torch.allclose(torch.as_tensor(exercises.norma_linf(v)), torch.tensor(4.0))


def test_norma_linf_come_torch():
    out = torch.as_tensor(exercises.norma_linf(V_RANDOM))
    assert torch.allclose(out, torch.linalg.norm(V_RANDOM, ord=float("inf")))


def test_distanza_euclidea():
    u = torch.tensor([1.0, 1.0])
    v = torch.tensor([4.0, 5.0])
    assert torch.allclose(torch.as_tensor(exercises.distanza_euclidea(u, v)), torch.tensor(5.0))


def test_distanza_da_se_stessi_e_zero():
    u = torch.tensor([2.0, -3.0, 1.0])
    assert torch.allclose(torch.as_tensor(exercises.distanza_euclidea(u, u)), torch.tensor(0.0))


def test_normalizza_colonne():
    X = torch.tensor(
        [
            [50.0, 2.0],
            [70.0, 3.0],
            [90.0, 3.0],
            [110.0, 4.0],
            [130.0, 5.0],
        ]
    )
    Xn = exercises.normalizza_colonne(X)
    assert Xn.shape == X.shape
    assert torch.allclose(Xn.mean(dim=0), torch.zeros(2), atol=1e-5)
    assert torch.allclose(Xn.std(dim=0), torch.ones(2), atol=1e-5)


def test_cosine_sim_perpendicolari():
    u = torch.tensor([1.0, 0.0])
    v = torch.tensor([0.0, 1.0])
    assert torch.allclose(torch.as_tensor(exercises.cosine_sim(u, v)), torch.tensor(0.0), atol=1e-6)


def test_cosine_sim_ignora_la_lunghezza():
    u = torch.tensor([1.0, 1.0])
    v = torch.tensor([3.0, 3.0])
    assert torch.allclose(torch.as_tensor(exercises.cosine_sim(u, v)), torch.tensor(1.0), atol=1e-6)


def test_cosine_sim_come_torch():
    u = V_RANDOM
    v = torch.flip(V_RANDOM, dims=[0])
    out = torch.as_tensor(exercises.cosine_sim(u, v))
    expected = torch.nn.functional.cosine_similarity(u, v, dim=0)
    assert torch.allclose(out, expected, atol=1e-6)
