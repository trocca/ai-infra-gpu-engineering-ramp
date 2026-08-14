"""Complete solutions for the lesson 01 exercises: vectors.

Look at these only after a genuine attempt at exercises.py.
"""

import torch


def crea_vettore(numeri):
    """Create a PyTorch tensor from a list of Python numbers."""
    return torch.tensor(numeri, dtype=torch.float32)


def scala_vettore(v, c):
    """Multiply the vector v by the scalar c."""
    return c * v


def combinazione_lineare(u, v, a, b):
    """Return the linear combination a*u + b*v."""
    return a * u + b * v


def dot_manuale(u, v):
    """Compute the dot product of u and v with an explicit for loop."""
    total = 0.0
    for i in range(len(u)):
        total += (u[i] * v[i]).item()
    return total


def predici_prezzo(x, w, b):
    """Predict the price of a house with a linear model."""
    return torch.dot(x, w) + b
