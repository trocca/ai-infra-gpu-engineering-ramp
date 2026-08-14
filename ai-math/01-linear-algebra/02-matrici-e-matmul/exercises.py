"""Exercises for lesson 02: matrices and matmul.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you move down the file.
"""

import torch


def estrai_diagonale(A):
    """Return the diagonal of the square matrix A as a vector.

    The diagonal is the elements A[0,0], A[1,1], A[2,2] and so on.
    Do not use torch.diag: write it with a loop, or with indexing.
    Example: for [[1., 2.], [3., 4.]] return tensor([1., 4.]).
    """
    # TODO
    raise NotImplementedError


def trasposta_manuale(A):
    """Return the transpose of A without using A.T or torch.transpose.

    Build a new matrix where row and column are swapped:
    the result at position [j, i] equals A[i, j].
    Hint: first create a zeros matrix of the right shape.
    """
    # TODO: build a zeros matrix of shape (cols, rows), then fill it
    raise NotImplementedError


def predizioni_case(X, w, b):
    """Compute the predictions for all the houses in one shot.

    X is the dataset matrix (one row per house), w is the weight
    vector, b is the bias. Return the vector of predictions using a
    single matmul (the @ operator), with no for loops.
    """
    # TODO
    raise NotImplementedError


def matrice_identita(n):
    """Build the n x n identity matrix without using torch.eye.

    It must have 1.0 on the diagonal and 0.0 everywhere else.
    Hint: start from torch.zeros(n, n) and fill the diagonal.
    """
    # TODO
    raise NotImplementedError


def matmul_manuale(A, B):
    """Multiply A by B with three explicit for loops.

    Do not use @, torch.matmul, torch.mm, or even torch.dot.
    Rule: the result C has one row for each row of A and one column
    for each column of B. The cell C[i, j] is the sum of the products
    A[i, k] * B[k, j] over every k.
    It also works with rectangular matrices: (2x3) @ (3x4) gives (2x4).
    """
    # TODO: three nested loops over i, j, k
    raise NotImplementedError
