"""Complete solutions for the lesson 04 exercises: autograd under the hood.

Look at these only after a serious attempt at exercises.py.
"""

import torch


def grad_locale_prodotto(a, b):
    """The local derivatives of the product f = a * b."""
    return float(b), float(a)


def grad_attraverso_relu(z, grad_in_arrivo):
    """Propagate the gradient through the ReLU."""
    return grad_in_arrivo * (z > 0).float()


def backprop_lineare(x, w, b, t):
    """Backprop BY HAND on the scalar model: loss = (w*x + b - t)^2."""
    out = w * x + b
    err = out - t
    dloss_dout = 2 * err
    dloss_dw = dloss_dout * x
    dloss_db = dloss_dout * 1.0
    return float(dloss_dw), float(dloss_db)


def backprop_due_strati(x, w1, w2, t):
    """Backprop BY HAND on two scalar layers WITHOUT activation."""
    h = w1 * x
    out = w2 * h
    err = out - t
    dloss_dout = 2 * err
    dloss_dw2 = dloss_dout * h
    dloss_dh = dloss_dout * w2
    dloss_dw1 = dloss_dh * x
    return float(dloss_dw1), float(dloss_dw2)
