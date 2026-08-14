"""Soluzioni complete degli esercizi della lezione 01: gradient descent.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def passo_gd(w, grad, lr):
    """Applica un singolo passo di gradient descent."""
    return w - lr * grad


def gd_su_parabola(w0, lr, passi):
    """Esegui gradient descent sulla parabola loss(w) = (w - 3)^2."""
    w = float(w0)
    for _ in range(passi):
        grad = 2 * (w - 3)
        w = passo_gd(w, grad, lr)
    return w


def gd_autograd(f, w0, lr, passi):
    """Gradient descent generico su una funzione qualsiasi, con autograd."""
    punto = w0.clone()
    for _ in range(passi):
        p = punto.clone().requires_grad_(True)
        loss = f(p)
        loss.backward()
        punto = (p - lr * p.grad).detach()
    return punto


def gd_con_storia(f, w0, lr, passi):
    """Come gd_autograd, ma registra anche la storia della loss."""
    punto = w0.clone()
    storia = []
    for _ in range(passi):
        p = punto.clone().requires_grad_(True)
        loss = f(p)
        storia.append(loss.item())
        loss.backward()
        punto = (p - lr * p.grad).detach()
    return punto, storia
