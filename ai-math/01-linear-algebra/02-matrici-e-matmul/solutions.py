"""Soluzioni complete degli esercizi della lezione 02: matrici e matmul.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def estrai_diagonale(A):
    """Restituisci la diagonale della matrice quadrata A come vettore."""
    n = A.shape[0]
    out = torch.zeros(n)
    for i in range(n):
        out[i] = A[i, i]
    return out


def trasposta_manuale(A):
    """Restituisci la trasposta di A senza usare A.T ne' torch.transpose."""
    rows, cols = A.shape
    out = torch.zeros(cols, rows)
    for i in range(rows):
        for j in range(cols):
            out[j, i] = A[i, j]
    return out


def predizioni_case(X, w, b):
    """Calcola le predizioni per tutte le case in un colpo solo."""
    return X @ w + b


def matrice_identita(n):
    """Costruisci la matrice identita' n x n senza usare torch.eye."""
    out = torch.zeros(n, n)
    for i in range(n):
        out[i, i] = 1.0
    return out


def matmul_manuale(A, B):
    """Moltiplica A per B con tre cicli for espliciti."""
    m, n = A.shape
    n2, p = B.shape
    assert n == n2, "inner dimensions must match"
    C = torch.zeros(m, p)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C
