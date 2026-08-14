"""Soluzioni complete degli esercizi della lezione 03: entropia e cross entropy.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def sorpresa(p):
    """Calcola la sorpresa di un evento di probabilita' p: -log p."""
    return -torch.log(torch.as_tensor(p))


def entropia(p):
    """Calcola l'entropia H(P) di una distribuzione discreta."""
    return -(p * torch.log(p)).sum()


def kl_divergence(p, q):
    """Calcola KL(P || Q) tra due distribuzioni discrete."""
    return (p * (torch.log(p) - torch.log(q))).sum()


def softmax_a_mano(logits):
    """Trasforma un vettore di punteggi grezzi in probabilita'."""
    shifted = logits - logits.max()
    e = torch.exp(shifted)
    return e / e.sum()


def cross_entropy_a_mano(logits, classe_vera):
    """Calcola la cross entropy loss per un singolo esempio."""
    probs = softmax_a_mano(logits)
    return -torch.log(probs[classe_vera])
