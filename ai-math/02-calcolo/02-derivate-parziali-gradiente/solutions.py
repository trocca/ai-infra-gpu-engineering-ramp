"""Complete solutions to the exercises of lesson 02: partial derivatives.

Look at them only after a serious attempt at exercises.py.
"""

import torch


def parziale_finita(f, punto, indice, h=1e-4):
    """Compute the partial derivative of f with respect to a single variable."""
    spostato = punto.clone()
    spostato[indice] += h
    return ((f(spostato) - f(punto)) / h).item()


def gradiente_autograd(f, punto):
    """Compute the full gradient of f at the given point, with autograd."""
    p = punto.clone().requires_grad_(True)
    out = f(p)
    out.backward()
    return p.grad


def gradiente_ciotola(punto):
    """Return the EXACT gradient of f(x, y) = x^2 + y^2 at the point."""
    return 2 * punto


def passo_in_discesa(f, punto, lr):
    """Take one step in the direction of steepest descent."""
    grad = gradiente_autograd(f, punto)
    return (punto - lr * grad).detach()
