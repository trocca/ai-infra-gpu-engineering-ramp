"""Soluzioni complete degli esercizi della lezione 04: loss landscape.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def loss_mse(w, b, x, y):
    """Calcola la MSE del modello a una feature: pred = w * x + b."""
    return ((w * x + b - y) ** 2).mean()


def loss_su_griglia(ws, bs, x, y):
    """Valuta la loss su tutte le combinazioni della griglia."""
    L = torch.zeros(len(bs), len(ws))
    for i in range(len(bs)):
        for j in range(len(ws)):
            L[i, j] = loss_mse(ws[j], bs[i], x, y)
    return L


def minimo_della_griglia(L, ws, bs):
    """Trova la cella piu' profonda della mappa."""
    flat = L.argmin()
    riga = flat // L.shape[1]
    colonna = flat % L.shape[1]
    return ws[colonna], bs[riga]


def traiettoria_gd(x, y, w0, b0, lr, passi):
    """Esegui gradient descent registrando l'intero percorso."""
    w = torch.as_tensor(float(w0))
    b = torch.as_tensor(float(b0))
    path = [(w.item(), b.item())]
    for _ in range(passi):
        err = w * x + b - y
        gw = 2 * (err * x).mean()
        gb = 2 * err.mean()
        w = w - lr * gw
        b = b - lr * gb
        path.append((w.item(), b.item()))
    return torch.tensor(path)
