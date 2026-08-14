"""Exercises for lesson 03: the chain rule.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you move down the file.
"""

import torch


def catena_due_anelli(x):
    """Compute BY HAND the derivative of y = (3x + 1)^2 at the point x.

    No autograd: use the chain rule as in EXPLANATION.md.
    Inner link u = 3x + 1 with du/dx = 3, outer link y = u^2
    with dy/du = 2u. Multiply. Return a float.
    Example: catena_due_anelli(2.0) returns 42.0.
    """
    # TODO
    raise NotImplementedError


def catena_loss_a_mano(w, x, y):
    """Compute BY HAND dloss/dw for the single-house model.

    The chain: pred = w * x, err = pred - y, loss = err^2.
    The slopes: dloss/derr = 2*err, derr/dpred = 1, dpred/dw = x.
    w, x, y are floats. Return dloss/dw as a float.
    """
    # TODO: forward first (compute err), then multiply the three links
    raise NotImplementedError


def catena_loss_autograd(w, x, y):
    """Compute dloss/dw with autograd, same chain as the exercise above.

    Steps: tensor for w with requires_grad=True, forward, backward,
    return the gradient as a float. The tests check that your
    hand-computed result and this one agree.
    """
    # TODO
    raise NotImplementedError


def catena_tre_anelli(x):
    """Compute BY HAND the derivative of t = ((2x)^2 + 1)^3 at the point x.

    Three links: u = 2x, v = u^2 + 1, t = v^3.
    The slopes: du/dx = 2, dv/du = 2u, dt/dv = 3v^2.
    First compute the values u and v (forward), then multiply the three
    slopes (backward). Return a float.
    """
    # TODO
    raise NotImplementedError
