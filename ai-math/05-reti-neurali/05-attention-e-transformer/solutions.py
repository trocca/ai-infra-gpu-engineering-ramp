"""Complete solutions for the lesson 05 exercises: attention.

Look at these only after a serious attempt at exercises.py.
"""

import math

import torch


def softmax_per_righe(M):
    """Softmax applied to every ROW of the matrix M."""
    shifted = M - M.max(dim=-1, keepdim=True).values
    e = torch.exp(shifted)
    return e / e.sum(dim=-1, keepdim=True)


def punteggi_attenzione(Q, K):
    """The matrix of affinity scores, scaled."""
    return Q @ K.T / math.sqrt(Q.shape[-1])


def attenzione(Q, K, V):
    """Complete scaled dot product attention."""
    return softmax_per_righe(punteggi_attenzione(Q, K)) @ V


def maschera_causale(n):
    """Build the n x n additive causal mask."""
    futuro = torch.triu(torch.ones(n, n), diagonal=1).bool()
    mask = torch.zeros(n, n)
    mask[futuro] = float("-inf")
    return mask


def attenzione_causale(Q, K, V):
    """Attention with a causal mask: no peeking at the future."""
    scores = punteggi_attenzione(Q, K) + maschera_causale(len(Q))
    return softmax_per_righe(scores) @ V
