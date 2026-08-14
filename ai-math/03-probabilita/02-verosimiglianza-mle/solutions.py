"""Soluzioni complete degli esercizi della lezione 02: verosimiglianza e MLE.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def likelihood_moneta(p, lanci):
    """Calcola la likelihood della sequenza di lanci per il parametro p."""
    p = torch.as_tensor(p)
    k = lanci.sum()
    n = len(lanci)
    return p**k * (1 - p) ** (n - k)


def log_likelihood_moneta(p, lanci):
    """Calcola il logaritmo della likelihood della moneta."""
    p = torch.as_tensor(p)
    k = lanci.sum()
    n = len(lanci)
    return k * torch.log(p) + (n - k) * torch.log(1 - p)


def mle_moneta(lanci):
    """Restituisci la stima di massima verosimiglianza per la moneta."""
    return lanci.sum() / len(lanci)


def mle_su_griglia(lanci, griglia):
    """Trova la MLE provando tutti i valori di p nella griglia."""
    scores = torch.stack([log_likelihood_moneta(p, lanci) for p in griglia])
    return griglia[scores.argmax()]


def nll_gaussiana(mu, dati, sigma):
    """Calcola la negative log likelihood di Normal(mu, sigma) sui dati."""
    n = len(dati)
    sigma = torch.as_tensor(sigma)
    const = n * torch.log(sigma * torch.sqrt(torch.tensor(2 * torch.pi)))
    return const + ((dati - mu) ** 2).sum() / (2 * sigma**2)
