"""Exercises for lesson 04: autograd under the hood.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you go down the file.
"""

import torch


def grad_locale_prodotto(a, b):
    """The local derivatives of the product f = a * b.

    The product rule, the most used rule in all of backward: the
    derivative with respect to one factor is THE OTHER factor.
    a and b are floats. Return the tuple (df/da, df/db) as floats.
    """
    # TODO
    raise NotImplementedError


def grad_attraverso_relu(z, grad_in_arrivo):
    """Propagate the gradient through the ReLU.

    z is the value BEFORE the ReLU (tensor), grad_in_arrivo is the
    gradient arriving from above (tensor, same shape).
    Rule: where z > 0 the ReLU is a straight wire (the gradient passes
    through untouched), where z <= 0 the ReLU cut it off (the gradient dies: 0).
    Return the propagated gradient.
    """
    # TODO
    raise NotImplementedError


def backprop_lineare(x, w, b, t):
    """Backprop BY HAND on the scalar model: loss = (w*x + b - t)^2.

    All arguments are floats. The chain, as in EXPLANATION.md:
    out = w*x + b, err = out - t, loss = err^2.
    Derivatives: dloss/derr = 2*err, then derr/dout = 1,
    dout/dw = x, dout/db = 1.
    Return the tuple (dloss_dw, dloss_db) as floats. No
    autograd: the test uses it to check you.
    """
    # TODO
    raise NotImplementedError


def backprop_due_strati(x, w1, w2, t):
    """Backprop BY HAND on two scalar layers WITHOUT activation.

    Model: h = w1 * x, out = w2 * h, loss = (out - t)^2.
    All floats. Compute the forward first (h, out, err), then descend:
    dloss/dout = 2*err
    dloss/dw2 = dloss/dout * h        (product rule)
    dloss/dh  = dloss/dout * w2       (the gradient keeps descending)
    dloss/dw1 = dloss/dh * x
    Return the tuple (dloss_dw1, dloss_dw2) as floats.
    This is the backward pass of a two layer "network", stripped bare.
    """
    # TODO
    raise NotImplementedError
