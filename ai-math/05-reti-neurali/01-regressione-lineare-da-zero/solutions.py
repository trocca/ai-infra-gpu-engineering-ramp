"""Complete solutions for the lesson 01 exercises: linear regression.

Look at these only after a serious attempt at exercises.py.
"""

import torch


def forward(X, w, b):
    """The forward pass: the linear model's predictions."""
    return X @ w + b


def mse(preds, y):
    """The error score: mean squared error."""
    return ((preds - y) ** 2).mean()


def passo_di_training(X, y, w, b, lr):
    """One complete turn of the training loop: forward, loss, backward, update."""
    wt = w.clone().requires_grad_(True)
    bt = b.clone().requires_grad_(True)
    loss = mse(forward(X, wt, bt), y)
    loss.backward()
    w_nuovo = (wt - lr * wt.grad).detach()
    b_nuovo = (bt - lr * bt.grad).detach()
    return w_nuovo, b_nuovo, loss.item()


def allena(X, y, lr, epoche):
    """The complete training loop: repeat passo_di_training for each epoch."""
    w = torch.zeros(X.shape[1])
    b = torch.zeros(1)
    storia = []
    for _ in range(epoche):
        w, b, loss = passo_di_training(X, y, w, b, lr)
        storia.append(loss)
    return w, b, storia
