"""Complete solutions for the lesson 03 exercises: norms and distances.

Look at these only after a genuine attempt at exercises.py.
"""

import torch


def norma_l1(v):
    """Compute the L1 norm of v: sum of the absolute values."""
    return torch.abs(v).sum()


def norma_l2(v):
    """Compute the L2 norm of v: square root of the sum of squares."""
    return torch.sqrt((v * v).sum())


def norma_linf(v):
    """Compute the L-infinity norm of v: the maximum of the absolute values."""
    return torch.abs(v).max()


def distanza_euclidea(u, v):
    """Compute the L2 distance between u and v."""
    return norma_l2(u - v)


def normalizza_colonne(X):
    """Standardize every column of X: subtract the mean, divide by the std."""
    return (X - X.mean(dim=0)) / X.std(dim=0)


def cosine_sim(u, v):
    """Compute the cosine similarity between u and v."""
    return torch.dot(u, v) / (norma_l2(u) * norma_l2(v))
