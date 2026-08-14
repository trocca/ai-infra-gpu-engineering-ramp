"""Complete solutions to the exercises of lesson 04: loss landscape.

Look at them only after a serious attempt at exercises.py.
"""

import torch


def loss_mse(w, b, x, y):
    """Compute the MSE of the one-feature model: pred = w * x + b."""
    return ((w * x + b - y) ** 2).mean()


def loss_su_griglia(ws, bs, x, y):
    """Evaluate the loss on every combination of the grid."""
    L = torch.zeros(len(bs), len(ws))
    for i in range(len(bs)):
        for j in range(len(ws)):
            L[i, j] = loss_mse(ws[j], bs[i], x, y)
    return L


def minimo_della_griglia(L, ws, bs):
    """Find the deepest cell of the map."""
    flat = L.argmin()
    riga = flat // L.shape[1]
    colonna = flat % L.shape[1]
    return ws[colonna], bs[riga]


def traiettoria_gd(x, y, w0, b0, lr, passi):
    """Run gradient descent while recording the entire path."""
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
