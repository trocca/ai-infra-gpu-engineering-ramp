"""Soluzioni complete degli esercizi della lezione 02: regressione logistica.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def sigmoid_a_mano(z):
    """La funzione di schiacciamento: 1 / (1 + e^(-z))."""
    return 1 / (1 + torch.exp(-torch.as_tensor(z)))


def bce_a_mano(p, target):
    """La binary cross entropy media."""
    return -(target * torch.log(p) + (1 - target) * torch.log(1 - p)).mean()


def probabilita_case(X, w, b):
    """La pipeline del classificatore: punteggio lineare, poi sigmoid."""
    return sigmoid_a_mano(X @ w + b)


def classifica(X, w, b, soglia=0.5):
    """Trasforma le probabilita' in decisioni 0/1."""
    return (probabilita_case(X, w, b) > soglia).float()


def allena_logistica(X, y, lr, epoche):
    """Il training loop del classificatore."""
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    storia = []
    for _ in range(epoche):
        loss = bce_a_mano(probabilita_case(X, w, b), y)
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
        storia.append(loss.item())
    return w.detach(), b.detach(), storia
