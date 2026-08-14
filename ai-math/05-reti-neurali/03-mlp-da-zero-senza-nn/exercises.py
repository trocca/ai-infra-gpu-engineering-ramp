"""Exercises for lesson 03: an MLP from scratch, no nn.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you go down the file.
"""

import torch


def relu_a_mano(x):
    """The ReLU: max(0, x), element by element.

    Do not use torch.relu or F.relu. Possible routes:
    torch.clamp, or torch.where, or x * (x > 0).
    """
    # TODO
    raise NotImplementedError


def forward_mlp(X, W1, b1, W2, b2):
    """The forward pass of a single-hidden-layer MLP.

    X has one row per example. The two lines from EXPLANATION.md:
    h = ReLU(X @ W1 + b1), out = h @ W2 + b2.
    Reuse your relu_a_mano. Return out.
    """
    # TODO
    raise NotImplementedError


def conta_parametri(W1, b1, W2, b2):
    """Count the network's total number of parameters.

    Sum the number of elements of each tensor (the .numel() method).
    Return an int. The numbers like '7B' in model names are
    exactly this count, in billions.
    """
    # TODO
    raise NotImplementedError


def xor_a_mano():
    """Build BY HAND the weights that solve XOR, no training.

    Return the tuple (W1, b1, W2, b2) with shapes:
    W1 (2, 2), b1 (2,), W2 (2,), b2 scalar.
    The recipe from EXPLANATION.md: the first hidden neuron computes
    ReLU(x1 + x2), the second ReLU(x1 + x2 - 1), and the output combines
    h1 - 2*h2. The test checks the exact XOR table using YOUR
    forward_mlp.
    """
    # TODO
    raise NotImplementedError
