"""Exercises for lesson 02: SGD and minibatches.

Complete the functions marked with # TODO.
Then run `pytest` from this folder.
The difficulty grows as you go down the file.
"""

import torch


def dividi_in_minibatch(X, y, batch_size):
    """Cut the dataset into consecutive minibatches, without shuffling.

    Return a list of (Xb, yb) tuples, in the original order.
    The last batch may be shorter if the division is not exact.
    Example: 5 examples with batch_size 2 give batches of size 2, 2, 1.
    """
    # TODO: slice X and y in steps of batch_size
    raise NotImplementedError


def mescola_dataset(X, y, seed):
    """Shuffle X and y WITH THE SAME permutation, reproducibly.

    Fix the seed with torch.manual_seed(seed), generate a permutation
    with torch.randperm, and apply it to both X and y. If X and y
    didn't stay aligned, every house would end up with another
    house's price. Return the tuple (X_shuffled, y_shuffled).
    """
    # TODO
    raise NotImplementedError


def gradiente_minibatch(w, b, Xb, yb):
    """Compute the gradient of the MSE on the minibatch alone, with autograd.

    The model is preds = Xb @ w + b, the loss is mean((preds - yb)^2).
    w and b arrive without requires_grad: clone them with
    requires_grad_(True), compute the loss, backward, and return the
    tuple (grad_w, grad_b).
    """
    # TODO
    raise NotImplementedError


def epoca_sgd(w, b, X, y, lr, batch_size, seed):
    """Run ONE full epoch of SGD and return the updated (w, b).

    Steps: shuffle the dataset (reuse mescola_dataset with the seed),
    cut it into minibatches (reuse dividi_in_minibatch), and for each
    minibatch compute the gradient (reuse gradiente_minibatch) and
    update w and b with the gradient descent rule.
    Return the final (w, b) as tensors without requires_grad.
    """
    # TODO
    raise NotImplementedError
