"""Exercises for lesson 01: vectors.

Complete the functions marked with # TODO.
Then run `pytest` from this folder: when all tests pass, you're done.
The difficulty grows as you move down the file.
"""

import torch


def crea_vettore(numeri):
    """Create a PyTorch tensor from a list of Python numbers.

    The tensor must have dtype torch.float32.
    Example: crea_vettore([1, 2, 3]) returns tensor([1., 2., 3.]).
    """
    # TODO: use torch.tensor with an explicit dtype
    raise NotImplementedError


def scala_vettore(v, c):
    """Multiply the vector v by the scalar c.

    Example: scala_vettore(tensor([2., 1.]), 3) returns tensor([6., 3.]).
    """
    # TODO
    raise NotImplementedError


def combinazione_lineare(u, v, a, b):
    """Return the linear combination a*u + b*v.

    A linear combination is a sum of vectors, each scaled by its own
    weight. It's the operation a neural network performs all the time.
    Example: combinazione_lineare([1., 0.], [0., 1.], 2, 3) returns [2., 3.].
    """
    # TODO
    raise NotImplementedError


def dot_manuale(u, v):
    """Compute the dot product of u and v with an explicit for loop.

    Do not use torch.dot or the @ operator. The whole point of the
    exercise is writing the loop by hand: multiply position by position
    and sum. Return a Python float (use .item() at the end).
    Example: dot_manuale(tensor([2., 1.]), tensor([1., 3.])) returns 5.0.
    """
    # TODO: accumulate the products in a loop, then return a Python float
    raise NotImplementedError


def predici_prezzo(x, w, b):
    """Predict the price of a house with a linear model.

    x is the house vector (square meters, rooms), w is the weight
    vector, b is the bias (a single number). The prediction is the dot
    product of x and w, plus b. Return a scalar tensor (the direct
    result of the torch operations is fine, no .item() needed).
    Example: predici_prezzo([50., 2.], [2., 10.], 20.) returns 140.
    """
    # TODO: this time torch.dot is allowed
    raise NotImplementedError
