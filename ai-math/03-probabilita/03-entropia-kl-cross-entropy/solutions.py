"""Complete solutions for the lesson 03 exercises: entropy and cross entropy.

Look at them only after giving exercises.py a serious try.
"""

import torch


def sorpresa(p):
    """Compute the surprise of an event with probability p: -log p."""
    return -torch.log(torch.as_tensor(p))


def entropia(p):
    """Compute the entropy H(P) of a discrete distribution."""
    return -(p * torch.log(p)).sum()


def kl_divergence(p, q):
    """Compute KL(P || Q) between two discrete distributions."""
    return (p * (torch.log(p) - torch.log(q))).sum()


def softmax_a_mano(logits):
    """Turn a vector of raw scores into probabilities."""
    shifted = logits - logits.max()
    e = torch.exp(shifted)
    return e / e.sum()


def cross_entropy_a_mano(logits, classe_vera):
    """Compute the cross entropy loss for a single example."""
    probs = softmax_a_mano(logits)
    return -torch.log(probs[classe_vera])
