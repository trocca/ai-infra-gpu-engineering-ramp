"""Soluzioni complete degli esercizi della lezione 04: autovalori e SVD.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def applica_matrice(A, v):
    """Applica la matrice A al vettore v, cioe' calcola A @ v."""
    return A @ v


def e_autovettore(A, v, lam):
    """Verifica se v e' un autovettore di A con autovalore lam."""
    return torch.allclose(A @ v, lam * v, atol=1e-5)


def quoziente_rayleigh(A, v):
    """Calcola il quoziente di Rayleigh: (v . (A @ v)) / (v . v)."""
    return torch.dot(v, A @ v) / torch.dot(v, v)


def ricostruisci_da_eigh(eigvals, Q):
    """Ricostruisci la matrice originale dai pezzi di torch.linalg.eigh."""
    return Q @ torch.diag(eigvals) @ Q.T


def ricostruisci_da_svd(U, S, Vh):
    """Ricostruisci la matrice originale dai tre pezzi della SVD."""
    return U @ torch.diag(S) @ Vh


def approssima_rango_k(X, k):
    """Restituisci la migliore approssimazione di rango k di X via SVD."""
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    return U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
