"""Exercises for lesson 01: the derivative as a slope.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you move down the file.
"""

import torch


def derivata_finita(f, x, h=1e-3):
    """Compute the approximate derivative of f at the point x with step h.

    f is an ordinary Python function that takes a float and returns
    a float. Use the finite difference formula:
    (f(x + h) - f(x)) / h. Return a float.
    Example: derivata_finita(lambda x: x**2, 3.0) returns about 6.
    """
    # TODO
    raise NotImplementedError


def derivata_autograd(f, x):
    """Compute the exact derivative of f at the point x using autograd.

    Steps: build a tensor from x with requires_grad=True, compute y = f(t),
    call y.backward(), and return t.grad as a float (.item()).
    """
    # TODO
    raise NotImplementedError


def pendenza_in_piu_punti(f, xs):
    """Compute the derivative of f at each of the points in the tensor xs.

    Return a tensor with one derivative per point, in the same
    order. You can reuse your derivata_autograd in a loop.
    Example: for f = x**2 and xs = [1., 2., 3.] it returns [2., 4., 6.].
    """
    # TODO
    raise NotImplementedError


def loss_una_feature(w, sqm100, prices):
    """Compute the MSE of the one-weight model: pred = w * sqm100.

    MSE (mean squared error) is the mean of the squared errors:
    mean((pred - prices)^2). It is the model's error score.
    w is a scalar tensor, sqm100 and prices are tensors of the same
    length. Return a scalar tensor.
    """
    # TODO
    raise NotImplementedError


def pendenza_della_loss(w0, sqm100, prices):
    """Compute dloss/dw at the point w0 using autograd.

    w0 is a float. Steps: tensor with requires_grad, loss via your
    loss_una_feature, backward, return the gradient as a float.
    The sign of the result tells you which way to move w to improve.
    """
    # TODO
    raise NotImplementedError
