"""Exercises for lesson 03: entropy, KL and cross entropy.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you go down the file.
"""

import torch


def sorpresa(p):
    """Compute the surprise of an event with probability p: -log p.

    p is a float or a scalar tensor. Use the natural log (torch.log).
    Return a scalar tensor.
    """
    # TODO
    raise NotImplementedError


def entropia(p):
    """Compute the entropy H(P) of a discrete distribution.

    p is a tensor of probabilities (positive, summing to 1, no zeros).
    Formula: sum of p * (-log p). Return a scalar tensor.
    Example: entropia([0.5, 0.5]) returns log 2, about 0.693.
    """
    # TODO
    raise NotImplementedError


def kl_divergence(p, q):
    """Compute KL(P || Q) between two discrete distributions.

    Formula: sum of p * (log p - log q). It's 0 only if p and q are
    identical, otherwise it's positive. Return a scalar tensor.
    """
    # TODO
    raise NotImplementedError


def softmax_a_mano(logits):
    """Turn a vector of raw scores into probabilities.

    Numerically stable version: subtract the max from the logits
    before exponentiating (it doesn't change the result, it avoids
    overflow), then exp and normalize by dividing by the sum.
    Do not use F.softmax. Return a tensor shaped like the logits.
    """
    # TODO
    raise NotImplementedError


def cross_entropy_a_mano(logits, classe_vera):
    """Compute the cross entropy loss for a single example.

    logits is the model's score vector, classe_vera an integer.
    The full recipe: softmax of the logits (reuse your
    softmax_a_mano), then minus the log of the true class's probability.
    Do not use F.cross_entropy: the test uses it to check on you.
    Return a scalar tensor.
    """
    # TODO
    raise NotImplementedError
