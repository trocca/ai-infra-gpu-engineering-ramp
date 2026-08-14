"""Soluzioni complete degli esercizi della lezione 02: SGD e minibatch.

Guardale solo dopo aver provato sul serio con exercises.py.
"""

import torch


def dividi_in_minibatch(X, y, batch_size):
    """Taglia il dataset in minibatch consecutivi, senza mescolare."""
    batches = []
    for start in range(0, len(X), batch_size):
        batches.append((X[start : start + batch_size], y[start : start + batch_size]))
    return batches


def mescola_dataset(X, y, seed):
    """Rimescola X e y CON LA STESSA permutazione, in modo riproducibile."""
    torch.manual_seed(seed)
    perm = torch.randperm(len(X))
    return X[perm], y[perm]


def gradiente_minibatch(w, b, Xb, yb):
    """Calcola il gradiente della MSE sul solo minibatch, con autograd."""
    wt = w.clone().requires_grad_(True)
    bt = b.clone().requires_grad_(True)
    loss = ((Xb @ wt + bt - yb) ** 2).mean()
    loss.backward()
    return wt.grad, bt.grad


def epoca_sgd(w, b, X, y, lr, batch_size, seed):
    """Esegui UNA epoca completa di SGD e restituisci (w, b) aggiornati."""
    Xs, ys = mescola_dataset(X, y, seed)
    w = w.clone()
    b = b.clone()
    for Xb, yb in dividi_in_minibatch(Xs, ys, batch_size):
        gw, gb = gradiente_minibatch(w, b, Xb, yb)
        w = w - lr * gw
        b = b - lr * gb
    return w, b
