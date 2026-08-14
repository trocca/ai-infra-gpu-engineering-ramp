"""Esercizi sulla lezione 02: SGD e minibatch.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def dividi_in_minibatch(X, y, batch_size):
    """Taglia il dataset in minibatch consecutivi, senza mescolare.

    Restituisci una lista di tuple (Xb, yb), nell'ordine originale.
    L'ultimo batch puo' essere piu' corto se la divisione non e' esatta.
    Esempio: 5 esempi con batch_size 2 danno batch di taglia 2, 2, 1.
    """
    # TODO: slice X and y in steps of batch_size
    raise NotImplementedError


def mescola_dataset(X, y, seed):
    """Rimescola X e y CON LA STESSA permutazione, in modo riproducibile.

    Fissa il seed con torch.manual_seed(seed), genera una permutazione
    con torch.randperm, e applicala sia a X sia a y. Se X e y non
    restassero allineati, ogni casa finirebbe col prezzo di un'altra.
    Restituisci la tupla (X_mescolato, y_mescolato).
    """
    # TODO
    raise NotImplementedError


def gradiente_minibatch(w, b, Xb, yb):
    """Calcola il gradiente della MSE sul solo minibatch, con autograd.

    Il modello e' preds = Xb @ w + b, la loss e' mean((preds - yb)^2).
    w e b arrivano senza requires_grad: clonali con requires_grad_(True),
    calcola la loss, backward, e restituisci la tupla (grad_w, grad_b).
    """
    # TODO
    raise NotImplementedError


def epoca_sgd(w, b, X, y, lr, batch_size, seed):
    """Esegui UNA epoca completa di SGD e restituisci (w, b) aggiornati.

    Passi: mescola il dataset (riusa mescola_dataset con il seed),
    taglialo in minibatch (riusa dividi_in_minibatch), e per ogni
    minibatch calcola il gradiente (riusa gradiente_minibatch) e
    aggiorna w e b con la regola del gradient descent.
    Restituisci (w, b) finali come tensor senza requires_grad.
    """
    # TODO
    raise NotImplementedError
