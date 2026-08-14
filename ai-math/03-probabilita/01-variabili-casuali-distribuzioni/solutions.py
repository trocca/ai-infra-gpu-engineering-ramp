"""Complete solutions for the lesson 01 exercises: random variables.

Look at them only after giving exercises.py a serious try.
"""

import torch


def valore_atteso(valori, probabilita):
    """Compute the expected value E[X] of a discrete random variable."""
    return (valori * probabilita).sum()


def varianza(valori, probabilita):
    """Compute the variance of a discrete random variable."""
    mu = valore_atteso(valori, probabilita)
    return ((valori - mu) ** 2 * probabilita).sum()


def simula_moneta(p, n, seed):
    """Simulate n flips of a biased Bernoulli(p) coin."""
    torch.manual_seed(seed)
    return torch.distributions.Bernoulli(probs=p).sample((n,))


def probabilita_empirica(campioni, valore):
    """Estimate the probability of a value from its frequency in the samples."""
    return (campioni == valore).float().mean()


def entro_k_sigma(campioni, mu, sigma, k):
    """Compute the fraction of samples within mu plus or minus k sigma."""
    dentro = (campioni > mu - k * sigma) & (campioni < mu + k * sigma)
    return dentro.float().mean()
