"""Complete solutions for the lesson 03 exercises: an MLP from scratch.

Look at these only after a serious attempt at exercises.py.
"""

import torch


def relu_a_mano(x):
    """The ReLU: max(0, x), element by element."""
    return torch.clamp(x, min=0)


def forward_mlp(X, W1, b1, W2, b2):
    """The forward pass of a single-hidden-layer MLP."""
    h = relu_a_mano(X @ W1 + b1)
    return h @ W2 + b2


def conta_parametri(W1, b1, W2, b2):
    """Count the network's total number of parameters."""
    return W1.numel() + b1.numel() + W2.numel() + b2.numel()


def xor_a_mano():
    """Build BY HAND the weights that solve XOR, no training."""
    W1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    b1 = torch.tensor([0.0, -1.0])
    W2 = torch.tensor([1.0, -2.0])
    b2 = torch.tensor(0.0)
    return W1, b1, W2, b2
