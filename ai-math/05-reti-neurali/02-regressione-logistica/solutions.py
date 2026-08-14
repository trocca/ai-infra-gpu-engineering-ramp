"""Complete solutions for the lesson 02 exercises: logistic regression.

Look at these only after a serious attempt at exercises.py.
"""

import torch


def sigmoid_a_mano(z):
    """The squashing function: 1 / (1 + e^(-z))."""
    return 1 / (1 + torch.exp(-torch.as_tensor(z)))


def bce_a_mano(p, target):
    """The average binary cross entropy."""
    return -(target * torch.log(p) + (1 - target) * torch.log(1 - p)).mean()


def probabilita_case(X, w, b):
    """The classifier pipeline: linear score, then sigmoid."""
    return sigmoid_a_mano(X @ w + b)


def classifica(X, w, b, soglia=0.5):
    """Turn probabilities into 0/1 decisions."""
    return (probabilita_case(X, w, b) > soglia).float()


def allena_logistica(X, y, lr, epoche):
    """The classifier's training loop."""
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
