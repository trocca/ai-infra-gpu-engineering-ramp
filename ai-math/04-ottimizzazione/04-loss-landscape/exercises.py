"""Esercizi sulla lezione 04: il loss landscape.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def loss_mse(w, b, x, y):
    """Calcola la MSE del modello a una feature: pred = w * x + b.

    x e y sono tensor della stessa lunghezza, w e b numeri o tensor
    scalari. Restituisci un tensor scalare: mean((pred - y)^2).
    """
    # TODO
    raise NotImplementedError


def loss_su_griglia(ws, bs, x, y):
    """Valuta la loss su tutte le combinazioni della griglia.

    ws e bs sono tensor 1D di candidati. Restituisci una matrice L di
    forma (len(bs), len(ws)) dove L[i, j] = loss con b = bs[i] e
    w = ws[j]. Due cicli for espliciti vanno benissimo.
    Questa matrice E' il loss landscape: la mappa della valle.
    """
    # TODO
    raise NotImplementedError


def minimo_della_griglia(L, ws, bs):
    """Trova la cella piu' profonda della mappa.

    L e' la matrice di loss_su_griglia. Restituisci la tupla
    (w_migliore, b_migliore) come tensor scalari.
    Suggerimento: L.argmin() da' l'indice nella matrice appiattita;
    riga = indice // numero colonne, colonna = indice % numero colonne.
    """
    # TODO
    raise NotImplementedError


def traiettoria_gd(x, y, w0, b0, lr, passi):
    """Esegui gradient descent registrando l'intero percorso.

    Il gradiente della MSE per questo modello, scritto a mano:
    dloss/dw = 2 * mean(err * x),  dloss/db = 2 * mean(err)
    dove err = w * x + b - y.
    Parti da (w0, b0) e restituisci un tensor di forma (passi + 1, 2)
    con la posizione (w, b) prima di ogni passo e dopo l'ultimo.
    E' la scia della pallina che vedrai disegnata sulla mappa.
    """
    # TODO
    raise NotImplementedError
