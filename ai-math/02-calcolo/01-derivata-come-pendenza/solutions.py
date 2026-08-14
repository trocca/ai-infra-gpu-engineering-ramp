"""Complete solutions to the exercises of lesson 01: derivative as a slope.

Look at them only after a serious attempt at exercises.py.
"""

import torch


def derivata_finita(f, x, h=1e-3):
    """Compute the approximate derivative of f at the point x with step h."""
    return (f(x + h) - f(x)) / h


def derivata_autograd(f, x):
    """Compute the exact derivative of f at the point x using autograd."""
    t = torch.tensor(float(x), requires_grad=True)
    y = f(t)
    y.backward()
    return t.grad.item()


def pendenza_in_piu_punti(f, xs):
    """Compute the derivative of f at each of the points in the tensor xs."""
    out = torch.zeros(len(xs))
    for i in range(len(xs)):
        out[i] = derivata_autograd(f, xs[i].item())
    return out


def loss_una_feature(w, sqm100, prices):
    """Compute the MSE of the one-weight model: pred = w * sqm100."""
    preds = w * sqm100
    return ((preds - prices) ** 2).mean()


def pendenza_della_loss(w0, sqm100, prices):
    """Compute dloss/dw at the point w0 using autograd."""
    w = torch.tensor(float(w0), requires_grad=True)
    loss = loss_una_feature(w, sqm100, prices)
    loss.backward()
    return w.grad.item()
