"""Soluzioni complete degli esercizi della lezione 01: derivata come pendenza.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def derivata_finita(f, x, h=1e-3):
    """Calcola la derivata approssimata di f nel punto x con passo h."""
    return (f(x + h) - f(x)) / h


def derivata_autograd(f, x):
    """Calcola la derivata esatta di f nel punto x usando autograd."""
    t = torch.tensor(float(x), requires_grad=True)
    y = f(t)
    y.backward()
    return t.grad.item()


def pendenza_in_piu_punti(f, xs):
    """Calcola la derivata di f in ognuno dei punti del tensor xs."""
    out = torch.zeros(len(xs))
    for i in range(len(xs)):
        out[i] = derivata_autograd(f, xs[i].item())
    return out


def loss_una_feature(w, sqm100, prices):
    """Calcola la MSE del modello a un peso: pred = w * sqm100."""
    preds = w * sqm100
    return ((preds - prices) ** 2).mean()


def pendenza_della_loss(w0, sqm100, prices):
    """Calcola dloss/dw nel punto w0 usando autograd."""
    w = torch.tensor(float(w0), requires_grad=True)
    loss = loss_una_feature(w, sqm100, prices)
    loss.backward()
    return w.grad.item()
