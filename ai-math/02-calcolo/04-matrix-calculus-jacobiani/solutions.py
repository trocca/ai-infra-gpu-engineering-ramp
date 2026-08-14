"""Soluzioni complete degli esercizi della lezione 04: Jacobiane.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch
from torch.autograd.functional import jacobian


def jacobiana_a_mano(punto):
    """Scrivi la Jacobiana ESATTA di f(x, y) = [x^2, x*y] nel punto."""
    x, y = punto[0], punto[1]
    return torch.stack(
        [
            torch.stack([2 * x, torch.zeros(())]),
            torch.stack([y, x]),
        ]
    )


def jacobiana_finita(f, punto, h=1e-4):
    """Costruisci la Jacobiana di f per differenze finite."""
    base = f(punto)
    m, n = len(base), len(punto)
    J = torch.zeros(m, n)
    for j in range(n):
        moved = punto.clone()
        moved[j] += h
        J[:, j] = (f(moved) - base) / h
    return J


def jacobiana_autograd(f, punto):
    """Calcola la Jacobiana con torch."""
    return jacobian(f, punto)


def gradiente_mse(w, X, y):
    """Calcola il gradiente della MSE rispetto al vettore dei pesi w."""
    wt = w.clone().requires_grad_(True)
    preds = X @ wt
    loss = ((preds - y) ** 2).mean()
    loss.backward()
    return wt.grad
