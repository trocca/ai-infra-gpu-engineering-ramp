"""Exercises for lesson 01: random variables and distributions.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you go down the file.
"""

import torch


def valore_atteso(valori, probabilita):
    """Compute the expected value E[X] of a discrete random variable.

    valori and probabilita are tensors of the same length. The expected
    value is the weighted average: sum of value * probability.
    Return a scalar tensor.
    Example: for values [0., 1.] and probabilities [0.3, 0.7] return 0.7.
    """
    # TODO
    raise NotImplementedError


def varianza(valori, probabilita):
    """Compute the variance of a discrete random variable.

    The variance is the expected value of the squared deviation:
    E[(X - mu)^2], where mu is the expected value of X.
    Steps: compute mu (reuse valore_atteso), then the weighted average of
    (valori - mu)^2. Return a scalar tensor.
    Example: Bernoulli(0.7) has variance 0.7 * 0.3 = 0.21.
    """
    # TODO
    raise NotImplementedError


def simula_moneta(p, n, seed):
    """Simulate n flips of a biased Bernoulli(p) coin.

    Set the seed with torch.manual_seed(seed) BEFORE sampling, so the
    result is reproducible. Use torch.distributions.Bernoulli and its
    .sample((n,)) method. Return the tensor of flips (0s and 1s).
    """
    # TODO
    raise NotImplementedError


def probabilita_empirica(campioni, valore):
    """Estimate the probability of a value from its frequency in the samples.

    Return the fraction of elements of campioni equal to valore,
    as a scalar tensor. Hint: the == comparison produces a tensor
    of True and False, and .float().mean() turns it into a frequency.
    """
    # TODO
    raise NotImplementedError


def entro_k_sigma(campioni, mu, sigma, k):
    """Compute the fraction of samples within mu plus or minus k sigma.

    Count how many samples fall in the open interval
    (mu - k*sigma, mu + k*sigma) and divide by the total.
    Return a scalar tensor. For a Gaussian with k=2 the result
    should come out close to 0.95.
    """
    # TODO
    raise NotImplementedError
