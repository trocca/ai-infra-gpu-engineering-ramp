"""Soluzioni complete degli esercizi della lezione 04: autograd sotto il cofano.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def grad_locale_prodotto(a, b):
    """Le derivate locali del prodotto f = a * b."""
    return float(b), float(a)


def grad_attraverso_relu(z, grad_in_arrivo):
    """Propaga il gradiente attraverso la ReLU."""
    return grad_in_arrivo * (z > 0).float()


def backprop_lineare(x, w, b, t):
    """Backprop A MANO sul modello scalare: loss = (w*x + b - t)^2."""
    out = w * x + b
    err = out - t
    dloss_dout = 2 * err
    dloss_dw = dloss_dout * x
    dloss_db = dloss_dout * 1.0
    return float(dloss_dw), float(dloss_db)


def backprop_due_strati(x, w1, w2, t):
    """Backprop A MANO su due strati scalari SENZA attivazione."""
    h = w1 * x
    out = w2 * h
    err = out - t
    dloss_dout = 2 * err
    dloss_dw2 = dloss_dout * h
    dloss_dh = dloss_dout * w2
    dloss_dw1 = dloss_dh * x
    return float(dloss_dw1), float(dloss_dw2)
