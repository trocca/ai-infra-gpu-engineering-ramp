"""Test per gli esercizi della lezione 02: matrici e matmul.

Esegui `pytest` da questa cartella. Deterministici: seed fissato.
"""

import torch

import exercises

torch.manual_seed(0)
A_RANDOM = torch.randn(3, 4)
B_RANDOM = torch.randn(4, 2)


def test_estrai_diagonale():
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = exercises.estrai_diagonale(A)
    assert torch.allclose(out, torch.tensor([1.0, 4.0]))


def test_estrai_diagonale_3x3():
    A = torch.tensor([[5.0, 0.0, 1.0], [2.0, -3.0, 0.0], [7.0, 1.0, 9.0]])
    out = exercises.estrai_diagonale(A)
    assert torch.allclose(out, torch.tensor([5.0, -3.0, 9.0]))


def test_trasposta_manuale_quadrata():
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    out = exercises.trasposta_manuale(A)
    assert torch.allclose(out, A.T)


def test_trasposta_manuale_rettangolare():
    out = exercises.trasposta_manuale(A_RANDOM)
    assert out.shape == (4, 3)
    assert torch.allclose(out, A_RANDOM.T)


def test_predizioni_case():
    X = torch.tensor(
        [
            [50.0, 2.0],
            [70.0, 3.0],
            [90.0, 3.0],
            [110.0, 4.0],
            [130.0, 5.0],
        ]
    )
    w = torch.tensor([2.0, 10.0])
    out = exercises.predizioni_case(X, w, 20.0)
    expected = torch.tensor([140.0, 190.0, 230.0, 280.0, 330.0])
    assert torch.allclose(out, expected)


def test_matrice_identita():
    out = exercises.matrice_identita(4)
    assert torch.allclose(out, torch.eye(4))


def test_matrice_identita_non_cambia_nulla():
    I = exercises.matrice_identita(3)
    M = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    assert torch.allclose(M @ I, M)


def test_matmul_manuale_esempio_a_mano():
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    out = exercises.matmul_manuale(A, B)
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    assert torch.allclose(out, expected)


def test_matmul_manuale_rettangolare():
    out = exercises.matmul_manuale(A_RANDOM, B_RANDOM)
    assert out.shape == (3, 2)
    assert torch.allclose(out, A_RANDOM @ B_RANDOM, atol=1e-5)
