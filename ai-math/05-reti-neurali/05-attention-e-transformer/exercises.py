"""Exercises for lesson 05: attention and transformers.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you go down the file.
"""

import math

import torch


def softmax_per_righe(M):
    """Softmax applied to every ROW of the matrix M.

    Stable version: subtract each row's maximum before
    exponentiating. Hint: M.max(dim=-1, keepdim=True).values and
    sums with dim=-1, keepdim=True, so the shapes line up.
    Do not use F.softmax. Every row of the result must sum to 1.
    """
    # TODO
    raise NotImplementedError


def punteggi_attenzione(Q, K):
    """The matrix of affinity scores, scaled.

    Formula: Q @ K transposed, divided by the square root of the
    vector dimension (the last dimension of Q).
    Return the n x n matrix of scores.
    """
    # TODO
    raise NotImplementedError


def attenzione(Q, K, V):
    """Complete scaled dot product attention.

    The three steps: scores (reuse punteggi_attenzione), row-wise
    softmax (reuse softmax_per_righe), weighted average of the values (a
    matmul with V). The test compares against F.scaled_dot_product_attention.
    """
    # TODO
    raise NotImplementedError


def maschera_causale(n):
    """Build the n x n additive causal mask.

    It must be 0 where looking is allowed (diagonal included) and
    minus infinity (float('-inf')) above the diagonal, where the
    future lives. Added to the scores before the softmax, it switches the future off.
    Hint: torch.triu with diagonal=1 selects the future.
    """
    # TODO
    raise NotImplementedError


def attenzione_causale(Q, K, V):
    """Attention with a causal mask: no peeking at the future.

    Like attenzione, but add your maschera_causale to the scores
    before the softmax. The test compares against
    F.scaled_dot_product_attention(is_causal=True).
    """
    # TODO
    raise NotImplementedError
