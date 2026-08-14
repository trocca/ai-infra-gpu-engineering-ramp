"""Esercizi sulla lezione 01: la derivata come pendenza.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def derivata_finita(f, x, h=1e-3):
    """Calcola la derivata approssimata di f nel punto x con passo h.

    f e' una normale funzione Python che prende un float e restituisce
    un float. Usa la formula della differenza finita:
    (f(x + h) - f(x)) / h. Restituisci un float.
    Esempio: derivata_finita(lambda x: x**2, 3.0) restituisce circa 6.
    """
    # TODO
    raise NotImplementedError


def derivata_autograd(f, x):
    """Calcola la derivata esatta di f nel punto x usando autograd.

    Passi: crea un tensor da x con requires_grad=True, calcola y = f(t),
    chiama y.backward(), e restituisci t.grad come float (.item()).
    """
    # TODO
    raise NotImplementedError


def pendenza_in_piu_punti(f, xs):
    """Calcola la derivata di f in ognuno dei punti del tensor xs.

    Restituisci un tensor con una derivata per punto, nello stesso
    ordine. Puoi riusare la tua derivata_autograd in un ciclo.
    Esempio: per f = x**2 e xs = [1., 2., 3.] restituisce [2., 4., 6.].
    """
    # TODO
    raise NotImplementedError


def loss_una_feature(w, sqm100, prices):
    """Calcola la MSE del modello a un peso: pred = w * sqm100.

    MSE (mean squared error) e' la media dei quadrati degli errori:
    mean((pred - prices)^2). E' il punteggio di errore del modello.
    w e' un tensor scalare, sqm100 e prices sono tensor della stessa
    lunghezza. Restituisci un tensor scalare.
    """
    # TODO
    raise NotImplementedError


def pendenza_della_loss(w0, sqm100, prices):
    """Calcola dloss/dw nel punto w0 usando autograd.

    w0 e' un float. Passi: tensor con requires_grad, loss con la tua
    loss_una_feature, backward, restituisci il gradiente come float.
    Il segno del risultato dice da che parte muovere w per migliorare.
    """
    # TODO
    raise NotImplementedError
