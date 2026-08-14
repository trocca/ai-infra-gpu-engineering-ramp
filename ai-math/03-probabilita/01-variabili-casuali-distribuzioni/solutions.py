"""Soluzioni complete degli esercizi della lezione 01: variabili casuali.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def valore_atteso(valori, probabilita):
    """Calcola il valore atteso E[X] di una variabile casuale discreta."""
    return (valori * probabilita).sum()


def varianza(valori, probabilita):
    """Calcola la varianza di una variabile casuale discreta."""
    mu = valore_atteso(valori, probabilita)
    return ((valori - mu) ** 2 * probabilita).sum()


def simula_moneta(p, n, seed):
    """Simula n lanci di una moneta truccata Bernoulli(p)."""
    torch.manual_seed(seed)
    return torch.distributions.Bernoulli(probs=p).sample((n,))


def probabilita_empirica(campioni, valore):
    """Stima la probabilita' di un valore dalla frequenza nei campioni."""
    return (campioni == valore).float().mean()


def entro_k_sigma(campioni, mu, sigma, k):
    """Calcola la frazione di campioni dentro mu piu' o meno k sigma."""
    dentro = (campioni > mu - k * sigma) & (campioni < mu + k * sigma)
    return dentro.float().mean()
