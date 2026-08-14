"""Esercizi sulla lezione 04: matrix calculus e Jacobiane.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def jacobiana_a_mano(punto):
    """Scrivi la Jacobiana ESATTA di f(x, y) = [x^2, x*y] nel punto.

    Niente autograd: usa le formule della SPIEGAZIONE.
    punto e' un tensor [x, y]. Restituisci la matrice 2x2:
    riga 0: [2x, 0], riga 1: [y, x].
    """
    # TODO
    raise NotImplementedError


def jacobiana_finita(f, punto, h=1e-4):
    """Costruisci la Jacobiana di f per differenze finite.

    f prende un tensor 1D di dimensione n e restituisce un tensor 1D
    di dimensione m. Riempi la matrice una COLONNA alla volta: sposta
    l'ingresso j di h, guarda come si muovono tutte le uscite, dividi
    per h. Restituisci una matrice m x n.
    Suggerimento: chiama f(punto) una volta per scoprire m.
    """
    # TODO
    raise NotImplementedError


def jacobiana_autograd(f, punto):
    """Calcola la Jacobiana con torch.

    Una riga sola: usa torch.autograd.functional.jacobian.
    """
    # TODO
    raise NotImplementedError


def gradiente_mse(w, X, y):
    """Calcola il gradiente della MSE rispetto al vettore dei pesi w.

    Il modello e' preds = X @ w, la loss e' mean((preds - y)^2).
    Usa autograd: clona w con requires_grad, forward, backward,
    restituisci il tensor gradiente. Deve avere la stessa forma di w.
    Il test confronta anche con la formula chiusa 2/n * X.T @ (X@w - y).
    """
    # TODO
    raise NotImplementedError
