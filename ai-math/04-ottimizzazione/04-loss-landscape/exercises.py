"""Exercises for lesson 04: the loss landscape.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you go down the file.
"""

import torch


def loss_mse(w, b, x, y):
    """Compute the MSE of the one-feature model: pred = w * x + b.

    x and y are tensors of the same length, w and b are numbers or
    scalar tensors. Return a scalar tensor: mean((pred - y)^2).
    """
    # TODO
    raise NotImplementedError


def loss_su_griglia(ws, bs, x, y):
    """Evaluate the loss on every combination of the grid.

    ws and bs are 1D tensors of candidates. Return a matrix L of
    shape (len(bs), len(ws)) where L[i, j] = loss with b = bs[i] and
    w = ws[j]. Two explicit for loops are perfectly fine.
    This matrix IS the loss landscape: the map of the valley.
    """
    # TODO
    raise NotImplementedError


def minimo_della_griglia(L, ws, bs):
    """Find the deepest cell of the map.

    L is the matrix from loss_su_griglia. Return the tuple
    (best_w, best_b) as scalar tensors.
    Hint: L.argmin() gives the index into the flattened matrix;
    row = index // number of columns, column = index % number of columns.
    """
    # TODO
    raise NotImplementedError


def traiettoria_gd(x, y, w0, b0, lr, passi):
    """Run gradient descent while recording the entire path.

    The gradient of the MSE for this model, written by hand:
    dloss/dw = 2 * mean(err * x),  dloss/db = 2 * mean(err)
    where err = w * x + b - y.
    Start from (w0, b0) and return a tensor of shape (passi + 1, 2)
    with the (w, b) position before every step and after the last one.
    It's the trail of the ball you'll see drawn on the map.
    """
    # TODO
    raise NotImplementedError
