"""Exercises for lesson 02: partial derivatives and the gradient.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you move down the file.
"""

import torch


def parziale_finita(f, punto, indice, h=1e-4):
    """Compute the partial derivative of f with respect to a single variable.

    f takes a tensor (the point) and returns a scalar tensor.
    punto is a 1D tensor. indice says which component to differentiate
    with respect to. Recipe: copy the point (punto.clone()), add h only
    to the component at indice, and use the finite difference.
    Return a float.
    """
    # TODO
    raise NotImplementedError


def gradiente_autograd(f, punto):
    """Compute the full gradient of f at the given point, with autograd.

    punto is a 1D tensor WITHOUT requires_grad. Steps: make a copy
    with requires_grad=True (punto.clone().requires_grad_(True)),
    compute f, call backward, return the .grad tensor.
    """
    # TODO
    raise NotImplementedError


def gradiente_ciotola(punto):
    """Return the EXACT gradient of f(x, y) = x^2 + y^2 at the point.

    No autograd here: write the formula by hand. The partial derivative
    with respect to x is 2x, the one with respect to y is 2y.
    punto is a tensor [x, y]. Return a tensor [2x, 2y].
    """
    # TODO
    raise NotImplementedError


def passo_in_discesa(f, punto, lr):
    """Take one step in the direction of steepest descent.

    Compute the gradient of f at the point (reuse gradiente_autograd) and
    return the new point: punto - lr * gradient.
    lr (learning rate) is the step length. The new point must be
    a tensor without requires_grad (use .detach() if needed).
    Preview of module 04: this IS gradient descent.
    """
    # TODO
    raise NotImplementedError
