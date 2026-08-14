"""Complete solutions to the exercises of lesson 03: the chain rule.

Look at them only after a serious attempt at exercises.py.
"""

import torch


def catena_due_anelli(x):
    """Compute BY HAND the derivative of y = (3x + 1)^2 at the point x."""
    u = 3 * x + 1
    dy_du = 2 * u
    du_dx = 3.0
    return float(dy_du * du_dx)


def catena_loss_a_mano(w, x, y):
    """Compute BY HAND dloss/dw for the single-house model."""
    pred = w * x
    err = pred - y
    dloss_derr = 2 * err
    derr_dpred = 1.0
    dpred_dw = x
    return float(dloss_derr * derr_dpred * dpred_dw)


def catena_loss_autograd(w, x, y):
    """Compute dloss/dw with autograd, same chain as the exercise above."""
    wt = torch.tensor(float(w), requires_grad=True)
    pred = wt * x
    err = pred - y
    loss = err**2
    loss.backward()
    return wt.grad.item()


def catena_tre_anelli(x):
    """Compute BY HAND the derivative of t = ((2x)^2 + 1)^3 at the point x."""
    u = 2 * x
    v = u**2 + 1
    dt_dv = 3 * v**2
    dv_du = 2 * u
    du_dx = 2.0
    return float(dt_dv * dv_du * du_dx)
