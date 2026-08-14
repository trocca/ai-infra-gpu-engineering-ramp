"""Exercises for lesson 03: norms and distances.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you move down the file.
"""

import torch


def norma_l1(v):
    """Compute the L1 norm of v: sum of the absolute values.

    Do not use torch.linalg.norm: build it from torch.abs and sum.
    Return a scalar tensor.
    Example: norma_l1(tensor([3., -4.])) returns 7.
    """
    # TODO
    raise NotImplementedError


def norma_l2(v):
    """Compute the L2 norm of v: square root of the sum of squares.

    Do not use torch.linalg.norm: build it from the basic operations
    (multiplication, sum, torch.sqrt). Return a scalar tensor.
    Example: norma_l2(tensor([3., 4.])) returns 5.
    """
    # TODO
    raise NotImplementedError


def norma_linf(v):
    """Compute the L-infinity norm of v: the maximum of the absolute values.

    Do not use torch.linalg.norm. Return a scalar tensor.
    Example: norma_linf(tensor([3., -4.])) returns 4.
    """
    # TODO
    raise NotImplementedError


def distanza_euclidea(u, v):
    """Compute the L2 distance between u and v.

    Remember the rule: first the difference, then the norm of the
    difference. You can reuse your norma_l2.
    Example: distanza_euclidea([1., 1.], [4., 5.]) returns 5.
    """
    # TODO
    raise NotImplementedError


def normalizza_colonne(X):
    """Standardize every column of X: subtract the mean, divide by the std.

    X is a matrix (rows = examples, columns = features). Every column
    of the result must have mean 0 and standard deviation 1.
    Hint: X.mean(dim=0) and X.std(dim=0) work column by column.
    """
    # TODO
    raise NotImplementedError


def cosine_sim(u, v):
    """Compute the cosine similarity between u and v.

    Formula: dot product divided by the product of the two L2 norms.
    Do not use F.cosine_similarity: build it from torch.dot and your
    norma_l2. Return a scalar tensor.
    Example: cosine_sim([1., 0.], [0., 1.]) returns 0.
    """
    # TODO
    raise NotImplementedError
