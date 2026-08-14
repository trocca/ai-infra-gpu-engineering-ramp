"""Soluzioni complete degli esercizi della lezione 02: derivate parziali.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def parziale_finita(f, punto, indice, h=1e-4):
    """Calcola la derivata parziale di f rispetto a una sola variabile."""
    spostato = punto.clone()
    spostato[indice] += h
    return ((f(spostato) - f(punto)) / h).item()


def gradiente_autograd(f, punto):
    """Calcola il gradiente completo di f nel punto dato, con autograd."""
    p = punto.clone().requires_grad_(True)
    out = f(p)
    out.backward()
    return p.grad


def gradiente_ciotola(punto):
    """Restituisci il gradiente ESATTO di f(x, y) = x^2 + y^2 nel punto."""
    return 2 * punto


def passo_in_discesa(f, punto, lr):
    """Fai un passo nella direzione di discesa piu' ripida."""
    grad = gradiente_autograd(f, punto)
    return (punto - lr * grad).detach()
