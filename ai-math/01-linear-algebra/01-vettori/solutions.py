"""Soluzioni complete degli esercizi della lezione 01: vettori.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def crea_vettore(numeri):
    """Crea un tensor PyTorch a partire da una lista di numeri Python."""
    return torch.tensor(numeri, dtype=torch.float32)


def scala_vettore(v, c):
    """Moltiplica il vettore v per lo scalare c."""
    return c * v


def combinazione_lineare(u, v, a, b):
    """Restituisci la combinazione lineare a*u + b*v."""
    return a * u + b * v


def dot_manuale(u, v):
    """Calcola il prodotto scalare tra u e v con un ciclo for esplicito."""
    total = 0.0
    for i in range(len(u)):
        total += (u[i] * v[i]).item()
    return total


def predici_prezzo(x, w, b):
    """Predici il prezzo di una casa con un modello lineare."""
    return torch.dot(x, w) + b
