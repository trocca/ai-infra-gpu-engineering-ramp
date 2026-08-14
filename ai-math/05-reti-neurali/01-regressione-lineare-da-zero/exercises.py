"""Exercises for lesson 01: linear regression from scratch.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you go down the file.
"""

import torch


def forward(X, w, b):
    """The forward pass: the linear model's predictions.

    A matmul plus the bias: X @ w + b. Return the vector of
    predictions, one per row of X.
    """
    # TODO
    raise NotImplementedError


def mse(preds, y):
    """The error score: mean squared error.

    The mean of the squared errors: mean((preds - y)^2).
    Return a scalar tensor.
    """
    # TODO
    raise NotImplementedError


def passo_di_training(X, y, w, b, lr):
    """One complete turn of the training loop: forward, loss, backward, update.

    w and b arrive WITHOUT requires_grad. Steps:
    1. clone them with requires_grad_(True)
    2. forward and loss (reuse your forward and mse)
    3. backward
    4. compute the new w and b with the gradient descent rule,
       detaching them from the graph with .detach()
    Return the tuple (w_new, b_new, loss_value) where
    loss_value is a float (the loss BEFORE the update).
    """
    # TODO
    raise NotImplementedError


def allena(X, y, lr, epoche):
    """The complete training loop: repeat passo_di_training for each epoch.

    Start from w = zeros (one component per column of X) and b = zero.
    Record the loss of every epoch in a list.
    Return the tuple (w, b, loss_history).
    On sensible data the history must go down: if it goes up, the lr is insane.
    """
    # TODO
    raise NotImplementedError
