"""Exercises for lesson 02: likelihood and MLE.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you go down the file.
"""

import torch


def likelihood_moneta(p, lanci):
    """Compute the likelihood of the sequence of flips for parameter p.

    lanci is a tensor of 0s and 1s (1 = heads). If k is the number of
    heads and n the total, the likelihood is p^k * (1-p)^(n-k).
    p can be a float or a tensor. Return a scalar tensor.
    """
    # TODO
    raise NotImplementedError


def log_likelihood_moneta(p, lanci):
    """Compute the logarithm of the coin's likelihood.

    Use the sum form of the formula directly:
    k * log(p) + (n - k) * log(1 - p).
    Do not compute the likelihood first and then take the log: the whole
    point of the log form is to avoid the tiny numbers.
    """
    # TODO
    raise NotImplementedError


def mle_moneta(lanci):
    """Return the maximum likelihood estimate for the coin.

    For the Bernoulli the answer has a closed formula: the observed
    frequency, heads divided by total flips. Return a scalar tensor.
    """
    # TODO
    raise NotImplementedError


def mle_su_griglia(lanci, griglia):
    """Find the MLE by trying every value of p in the grid.

    griglia is a tensor of candidate p values. Compute the log likelihood
    for each candidate (reuse log_likelihood_moneta) and return the
    candidate with the highest value.
    The result must (almost) coincide with mle_moneta: two roads,
    same answer.
    """
    # TODO
    raise NotImplementedError


def nll_gaussiana(mu, dati, sigma):
    """Compute the negative log likelihood of Normal(mu, sigma) on the data.

    Formula (constants included, so the test is precise):
    nll = n * log(sigma * sqrt(2*pi)) + sum((dati - mu)^2) / (2*sigma^2)
    where n is the number of data points. Use torch.log, torch.sqrt and
    torch.tensor(torch.pi) where needed. Return a scalar tensor.
    Minimizing this function with respect to mu is equivalent to
    minimizing the sum of squares: this is where MSE comes from.
    """
    # TODO
    raise NotImplementedError
