"""Complete solutions for the lesson 04 exercises: eigenvalues and SVD.

Look at these only after a genuine attempt at exercises.py.
"""

import torch


def applica_matrice(A, v):
    """Apply the matrix A to the vector v, i.e. compute A @ v."""
    return A @ v


def e_autovettore(A, v, lam):
    """Check whether v is an eigenvector of A with eigenvalue lam."""
    return torch.allclose(A @ v, lam * v, atol=1e-5)


def quoziente_rayleigh(A, v):
    """Compute the Rayleigh quotient: (v . (A @ v)) / (v . v)."""
    return torch.dot(v, A @ v) / torch.dot(v, v)


def ricostruisci_da_eigh(eigvals, Q):
    """Rebuild the original matrix from the pieces of torch.linalg.eigh."""
    return Q @ torch.diag(eigvals) @ Q.T


def ricostruisci_da_svd(U, S, Vh):
    """Rebuild the original matrix from the three pieces of the SVD."""
    return U @ torch.diag(S) @ Vh


def approssima_rango_k(X, k):
    """Return the best rank-k approximation of X via SVD."""
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    return U[:, :k] @ torch.diag(S[:k]) @ Vh[:k, :]
