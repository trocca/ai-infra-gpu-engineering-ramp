"""Soluzioni complete degli esercizi della lezione 03: momentum e Adam.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def passo_momentum(w, v, grad, lr, beta):
    """Applica un passo di gradient descent con momentum."""
    v_nuovo = beta * v + grad
    w_nuovo = w - lr * v_nuovo
    return w_nuovo, v_nuovo


def velocita_a_regime(lr, beta):
    """Calcola il passo effettivo a regime con gradiente costante 1."""
    return lr / (1 - beta)


def passo_adam(w, m, v, grad, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Applica un passo di Adam. t e' il numero del passo, da 1 in su."""
    m_nuovo = beta1 * m + (1 - beta1) * grad
    v_nuovo = beta2 * v + (1 - beta2) * grad**2
    m_hat = m_nuovo / (1 - beta1**t)
    v_hat = v_nuovo / (1 - beta2**t)
    w_nuovo = w - lr * m_hat / (torch.sqrt(v_hat) + eps)
    return w_nuovo, m_nuovo, v_nuovo


def allena_con_adam(f, w0, lr, passi):
    """Minimizza f partendo da w0 usando il TUO passo_adam."""
    w = w0.clone()
    m = torch.zeros_like(w0)
    v = torch.zeros_like(w0)
    for t in range(1, passi + 1):
        p = w.clone().requires_grad_(True)
        f(p).backward()
        w, m, v = passo_adam(w, m, v, p.grad, t, lr=lr)
        w = w.detach()
    return w
