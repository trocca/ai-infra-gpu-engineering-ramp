"""Tests for the lesson 04 exercises: eigenvalues and SVD.

Run `pytest` from this folder. Deterministic: fixed seed.
"""

import torch

import exercises

torch.manual_seed(0)
_R = torch.randn(4, 4)
SYM_RANDOM = _R + _R.T  # symmetric matrix, safe for eigh
A = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
X_CASE = torch.tensor(
    [
        [50.0, 2.0],
        [70.0, 3.0],
        [90.0, 3.0],
        [110.0, 4.0],
        [130.0, 5.0],
    ]
)


def test_applica_matrice():
    v = torch.tensor([1.0, 0.0])
    out = exercises.applica_matrice(A, v)
    assert torch.allclose(out, torch.tensor([2.0, 1.0]))


def test_e_autovettore_vero():
    assert exercises.e_autovettore(A, torch.tensor([1.0, 1.0]), 3.0) is True
    assert exercises.e_autovettore(A, torch.tensor([1.0, -1.0]), 1.0) is True


def test_e_autovettore_falso():
    assert exercises.e_autovettore(A, torch.tensor([1.0, 0.0]), 2.0) is False


def test_quoziente_rayleigh_su_autovettore():
    out = exercises.quoziente_rayleigh(A, torch.tensor([1.0, 1.0]))
    assert torch.allclose(torch.as_tensor(out), torch.tensor(3.0), atol=1e-5)


def test_quoziente_rayleigh_scala_invariante():
    v = torch.tensor([2.0, -1.0, 0.5, 3.0])
    q1 = torch.as_tensor(exercises.quoziente_rayleigh(SYM_RANDOM, v))
    q2 = torch.as_tensor(exercises.quoziente_rayleigh(SYM_RANDOM, 10 * v))
    assert torch.allclose(q1, q2, atol=1e-4)


def test_ricostruisci_da_eigh():
    eigvals, Q = torch.linalg.eigh(SYM_RANDOM)
    out = exercises.ricostruisci_da_eigh(eigvals, Q)
    assert torch.allclose(out, SYM_RANDOM, atol=1e-4)


def test_ricostruisci_da_svd():
    U, S, Vh = torch.linalg.svd(X_CASE, full_matrices=False)
    out = exercises.ricostruisci_da_svd(U, S, Vh)
    assert torch.allclose(out, X_CASE, atol=1e-3)


def test_approssima_rango_pieno():
    out = exercises.approssima_rango_k(X_CASE, 2)
    assert torch.allclose(out, X_CASE, atol=1e-3)


def test_approssima_rango_1():
    out = exercises.approssima_rango_k(X_CASE, 1)
    assert out.shape == X_CASE.shape
    # Rank 1: every column must be a multiple of the same direction,
    # so the 2x2 minors are all zero (approximately).
    det = out[0, 0] * out[1, 1] - out[0, 1] * out[1, 0]
    assert abs(det.item()) < 1e-2
    # And it must be close to X, since the dataset is almost rank 1.
    rel_err = torch.linalg.norm(X_CASE - out) / torch.linalg.norm(X_CASE)
    assert rel_err.item() < 0.01
