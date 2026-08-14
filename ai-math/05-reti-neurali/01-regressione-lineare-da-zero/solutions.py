"""Soluzioni complete degli esercizi della lezione 01: regressione lineare.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def forward(X, w, b):
    """Il passo forward: le predizioni del modello lineare."""
    return X @ w + b


def mse(preds, y):
    """Il punteggio di errore: mean squared error."""
    return ((preds - y) ** 2).mean()


def passo_di_training(X, y, w, b, lr):
    """Un giro completo del training loop: forward, loss, backward, update."""
    wt = w.clone().requires_grad_(True)
    bt = b.clone().requires_grad_(True)
    loss = mse(forward(X, wt, bt), y)
    loss.backward()
    w_nuovo = (wt - lr * wt.grad).detach()
    b_nuovo = (bt - lr * bt.grad).detach()
    return w_nuovo, b_nuovo, loss.item()


def allena(X, y, lr, epoche):
    """Il training loop completo: ripeti passo_di_training per ogni epoca."""
    w = torch.zeros(X.shape[1])
    b = torch.zeros(1)
    storia = []
    for _ in range(epoche):
        w, b, loss = passo_di_training(X, y, w, b, lr)
        storia.append(loss)
    return w, b, storia
