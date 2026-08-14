"""Exercises for lesson 02: logistic regression.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you go down the file.
"""

import torch


def sigmoid_a_mano(z):
    """The squashing function: 1 / (1 + e^(-z)).

    Do not use torch.sigmoid. It must work both on scalars and on
    tensors (torch operations already handle that for you).
    """
    # TODO
    raise NotImplementedError


def bce_a_mano(p, target):
    """The average binary cross entropy.

    p are the model's probabilities (between 0 and 1), target the truths
    (0 or 1). Per-example formula: -[t*log(p) + (1-t)*log(1-p)], then
    average over all examples. Do not use F.binary_cross_entropy.
    Return a scalar tensor.
    """
    # TODO
    raise NotImplementedError


def probabilita_case(X, w, b):
    """The classifier pipeline: linear score, then sigmoid.

    Return the vector of probabilities P(class 1), one per row
    of X. Reuse your sigmoid_a_mano.
    """
    # TODO
    raise NotImplementedError


def classifica(X, w, b, soglia=0.5):
    """Turn probabilities into 0/1 decisions.

    Probability above the threshold: class 1, otherwise 0.
    Return a tensor of 0.0 and 1.0 (use .float()).
    """
    # TODO
    raise NotImplementedError


def allena_logistica(X, y, lr, epoche):
    """The classifier's training loop.

    Identical to the one from lesson 01, with two differences: the forward
    goes through the sigmoid (reuse probabilita_case) and the loss is the BCE
    (reuse bce_a_mano). Start from w and b at zero.
    Return the tuple (w, b, loss_history).
    """
    # TODO
    raise NotImplementedError
