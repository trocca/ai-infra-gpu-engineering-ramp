"""Soluzioni complete degli esercizi della lezione 03: norme e distanze.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def norma_l1(v):
    """Calcola la norma L1 di v: somma dei valori assoluti."""
    return torch.abs(v).sum()


def norma_l2(v):
    """Calcola la norma L2 di v: radice della somma dei quadrati."""
    return torch.sqrt((v * v).sum())


def norma_linf(v):
    """Calcola la norma L infinito di v: il massimo dei valori assoluti."""
    return torch.abs(v).max()


def distanza_euclidea(u, v):
    """Calcola la distanza L2 tra u e v."""
    return norma_l2(u - v)


def normalizza_colonne(X):
    """Standardizza ogni colonna di X: sottrai la media, dividi per la std."""
    return (X - X.mean(dim=0)) / X.std(dim=0)


def cosine_sim(u, v):
    """Calcola la similarita' coseno tra u e v."""
    return torch.dot(u, v) / (norma_l2(u) * norma_l2(v))
