"""Complete solutions for the lesson 02 exercises: likelihood and MLE.

Look at them only after giving exercises.py a serious try.
"""

import torch


def likelihood_moneta(p, lanci):
    """Compute the likelihood of the sequence of flips for parameter p."""
    p = torch.as_tensor(p)
    k = lanci.sum()
    n = len(lanci)
    return p**k * (1 - p) ** (n - k)


def log_likelihood_moneta(p, lanci):
    """Compute the logarithm of the coin's likelihood."""
    p = torch.as_tensor(p)
    k = lanci.sum()
    n = len(lanci)
    return k * torch.log(p) + (n - k) * torch.log(1 - p)


def mle_moneta(lanci):
    """Return the maximum likelihood estimate for the coin."""
    return lanci.sum() / len(lanci)


def mle_su_griglia(lanci, griglia):
    """Find the MLE by trying every value of p in the grid."""
    scores = torch.stack([log_likelihood_moneta(p, lanci) for p in griglia])
    return griglia[scores.argmax()]


def nll_gaussiana(mu, dati, sigma):
    """Compute the negative log likelihood of Normal(mu, sigma) on the data."""
    n = len(dati)
    sigma = torch.as_tensor(sigma)
    const = n * torch.log(sigma * torch.sqrt(torch.tensor(2 * torch.pi)))
    return const + ((dati - mu) ** 2).sum() / (2 * sigma**2)
