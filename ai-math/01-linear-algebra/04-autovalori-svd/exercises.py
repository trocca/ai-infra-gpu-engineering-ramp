"""Exercises for lesson 04: eigenvalues, eigenvectors and SVD.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you move down the file.
"""

import torch


def applica_matrice(A, v):
    """Apply the matrix A to the vector v, i.e. compute A @ v.

    Warm-up: it's a single line, but it's the operation at the heart
    of this whole lesson.
    """
    # TODO
    raise NotImplementedError


def e_autovettore(A, v, lam):
    """Check whether v is an eigenvector of A with eigenvalue lam.

    It must hold that A @ v = lam * v. Compare the two sides with
    torch.allclose (use atol=1e-5) and return True or False.
    """
    # TODO
    raise NotImplementedError


def quoziente_rayleigh(A, v):
    """Compute the Rayleigh quotient: (v . (A @ v)) / (v . v).

    If v is an eigenvector, this number is exactly its eigenvalue.
    If it isn't, it's an estimate. Return a scalar tensor.
    Use torch.dot and the @ operator.
    """
    # TODO
    raise NotImplementedError


def ricostruisci_da_eigh(eigvals, Q):
    """Rebuild the original matrix from the pieces of torch.linalg.eigh.

    eigvals is the vector of eigenvalues, Q the matrix with the
    eigenvectors in its columns. The formula is Q @ diag(eigvals) @ Q.T.
    Hint: torch.diag turns a vector into a diagonal matrix.
    """
    # TODO
    raise NotImplementedError


def ricostruisci_da_svd(U, S, Vh):
    """Rebuild the original matrix from the three pieces of the SVD.

    U, S, Vh are the output of torch.linalg.svd(X, full_matrices=False).
    The formula is U @ diag(S) @ Vh.
    """
    # TODO
    raise NotImplementedError


def approssima_rango_k(X, k):
    """Return the best rank-k approximation of X via SVD.

    Steps: compute the compact SVD of X, then keep only the first k
    columns of U, the first k values of S and the first k rows of Vh,
    and recombine with the same formula as the reconstruction.
    With k equal to the full rank it must return X (up to rounding
    errors).
    """
    # TODO: torch.linalg.svd(X, full_matrices=False), then slice and rebuild
    raise NotImplementedError
