"""Complete solutions for the lesson 02 exercises: matrices and matmul.

Look at these only after a genuine attempt at exercises.py.
"""

import torch


def estrai_diagonale(A):
    """Return the diagonal of the square matrix A as a vector."""
    n = A.shape[0]
    out = torch.zeros(n)
    for i in range(n):
        out[i] = A[i, i]
    return out


def trasposta_manuale(A):
    """Return the transpose of A without using A.T or torch.transpose."""
    rows, cols = A.shape
    out = torch.zeros(cols, rows)
    for i in range(rows):
        for j in range(cols):
            out[j, i] = A[i, j]
    return out


def predizioni_case(X, w, b):
    """Compute the predictions for all the houses in one shot."""
    return X @ w + b


def matrice_identita(n):
    """Build the n x n identity matrix without using torch.eye."""
    out = torch.zeros(n, n)
    for i in range(n):
        out[i, i] = 1.0
    return out


def matmul_manuale(A, B):
    """Multiply A by B with three explicit for loops."""
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "inner dimensions must match"
    C = torch.zeros(m, p)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C
