"""Exercises for lesson 04: matrix calculus and Jacobians.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you move down the file.
"""

import torch


def jacobiana_a_mano(punto):
    """Write the EXACT Jacobian of f(x, y) = [x^2, x*y] at the point.

    No autograd: use the formulas from EXPLANATION.md.
    punto is a tensor [x, y]. Return the 2x2 matrix:
    row 0: [2x, 0], row 1: [y, x].
    """
    # TODO
    raise NotImplementedError


def jacobiana_finita(f, punto, h=1e-4):
    """Build the Jacobian of f by finite differences.

    f takes a 1D tensor of size n and returns a 1D tensor
    of size m. Fill the matrix one COLUMN at a time: nudge
    input j by h, watch how all the outputs move, divide
    by h. Return an m x n matrix.
    Hint: call f(punto) once to find out m.
    """
    # TODO
    raise NotImplementedError


def jacobiana_autograd(f, punto):
    """Compute the Jacobian with torch.

    A single line: use torch.autograd.functional.jacobian.
    """
    # TODO
    raise NotImplementedError


def gradiente_mse(w, X, y):
    """Compute the gradient of the MSE with respect to the weight vector w.

    The model is preds = X @ w, the loss is mean((preds - y)^2).
    Use autograd: clone w with requires_grad, forward, backward,
    return the gradient tensor. It must have the same shape as w.
    The test also compares against the closed form 2/n * X.T @ (X@w - y).
    """
    # TODO
    raise NotImplementedError
