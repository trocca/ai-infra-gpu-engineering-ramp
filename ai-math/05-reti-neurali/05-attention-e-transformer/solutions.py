"""Soluzioni complete degli esercizi della lezione 05: attention.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import math

import torch


def softmax_per_righe(M):
    """Softmax applicata a ogni RIGA della matrice M."""
    shifted = M - M.max(dim=-1, keepdim=True).values
    e = torch.exp(shifted)
    return e / e.sum(dim=-1, keepdim=True)


def punteggi_attenzione(Q, K):
    """La matrice dei punteggi di affinita', scalata."""
    return Q @ K.T / math.sqrt(Q.shape[-1])


def attenzione(Q, K, V):
    """Scaled dot product attention completa."""
    return softmax_per_righe(punteggi_attenzione(Q, K)) @ V


def maschera_causale(n):
    """Costruisci la maschera additiva causale n x n."""
    futuro = torch.triu(torch.ones(n, n), diagonal=1).bool()
    mask = torch.zeros(n, n)
    mask[futuro] = float("-inf")
    return mask


def attenzione_causale(Q, K, V):
    """Attention con maschera causale: nessuno sguardo al futuro."""
    scores = punteggi_attenzione(Q, K) + maschera_causale(len(Q))
    return softmax_per_righe(scores) @ V
