"""Complete solutions to the exercises of lesson 02: SGD and minibatches.

Look at them only after a serious attempt at exercises.py.
"""

import torch


def dividi_in_minibatch(X, y, batch_size):
    """Cut the dataset into consecutive minibatches, without shuffling."""
    batches = []
    for start in range(0, len(X), batch_size):
        batches.append((X[start : start + batch_size], y[start : start + batch_size]))
    return batches


def mescola_dataset(X, y, seed):
    """Shuffle X and y WITH THE SAME permutation, reproducibly."""
    torch.manual_seed(seed)
    perm = torch.randperm(len(X))
    return X[perm], y[perm]


def gradiente_minibatch(w, b, Xb, yb):
    """Compute the gradient of the MSE on the minibatch alone, with autograd."""
    wt = w.clone().requires_grad_(True)
    bt = b.clone().requires_grad_(True)
    loss = ((Xb @ wt + bt - yb) ** 2).mean()
    loss.backward()
    return wt.grad, bt.grad


def epoca_sgd(w, b, X, y, lr, batch_size, seed):
    """Run ONE full epoch of SGD and return the updated (w, b)."""
    Xs, ys = mescola_dataset(X, y, seed)
    w = w.clone()
    b = b.clone()
    for Xb, yb in dividi_in_minibatch(Xs, ys, batch_size):
        gw, gb = gradiente_minibatch(w, b, Xb, yb)
        w = w - lr * gw
        b = b - lr * gb
    return w, b
