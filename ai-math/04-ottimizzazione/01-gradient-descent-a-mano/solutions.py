"""Complete solutions to the exercises of lesson 01: gradient descent.

Look at them only after a serious attempt at exercises.py.
"""

import torch


def passo_gd(w, grad, lr):
    """Apply a single gradient descent step."""
    return w - lr * grad


def gd_su_parabola(w0, lr, passi):
    """Run gradient descent on the parabola loss(w) = (w - 3)^2."""
    w = float(w0)
    for _ in range(passi):
        grad = 2 * (w - 3)
        w = passo_gd(w, grad, lr)
    return w


def gd_autograd(f, w0, lr, passi):
    """Generic gradient descent on any function, with autograd."""
    punto = w0.clone()
    for _ in range(passi):
        p = punto.clone().requires_grad_(True)
        loss = f(p)
        loss.backward()
        punto = (p - lr * p.grad).detach()
    return punto


def gd_con_storia(f, w0, lr, passi):
    """Like gd_autograd, but also record the loss history."""
    punto = w0.clone()
    storia = []
    for _ in range(passi):
        p = punto.clone().requires_grad_(True)
        loss = f(p)
        storia.append(loss.item())
        loss.backward()
        punto = (p - lr * p.grad).detach()
    return punto, storia
