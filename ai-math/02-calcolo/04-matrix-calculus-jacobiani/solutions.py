"""Complete solutions to the exercises of lesson 04: Jacobians.

Look at them only after a serious attempt at exercises.py.
"""

import torch
from torch.autograd.functional import jacobian


def jacobiana_a_mano(punto):
    """Write the EXACT Jacobian of f(x, y) = [x^2, x*y] at the point."""
    x, y = punto[0], punto[1]
    return torch.stack(
        [
            torch.stack([2 * x, torch.zeros(())]),
            torch.stack([y, x]),
        ]
    )


def jacobiana_finita(f, punto, h=1e-4):
    """Build the Jacobian of f by finite differences."""
    base = f(punto)
    m, n = len(base), len(punto)
    J = torch.zeros(m, n)
    for j in range(n):
        moved = punto.clone()
        moved[j] += h
        J[:, j] = (f(moved) - base) / h
    return J


def jacobiana_autograd(f, punto):
    """Compute the Jacobian with torch."""
    return jacobian(f, punto)


def gradiente_mse(w, X, y):
    """Compute the gradient of the MSE with respect to the weight vector w."""
    wt = w.clone().requires_grad_(True)
    preds = X @ wt
    loss = ((preds - y) ** 2).mean()
    loss.backward()
    return wt.grad
