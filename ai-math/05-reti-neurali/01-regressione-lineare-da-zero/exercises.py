"""Esercizi sulla lezione 01: regressione lineare da zero.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def forward(X, w, b):
    """Il passo forward: le predizioni del modello lineare.

    Una matmul piu' il bias: X @ w + b. Restituisci il vettore delle
    predizioni, una per riga di X.
    """
    # TODO
    raise NotImplementedError


def mse(preds, y):
    """Il punteggio di errore: mean squared error.

    La media dei quadrati degli errori: mean((preds - y)^2).
    Restituisci un tensor scalare.
    """
    # TODO
    raise NotImplementedError


def passo_di_training(X, y, w, b, lr):
    """Un giro completo del training loop: forward, loss, backward, update.

    w e b arrivano SENZA requires_grad. Passi:
    1. clonali con requires_grad_(True)
    2. forward e loss (riusa le tue forward e mse)
    3. backward
    4. calcola i nuovi w e b con la regola del gradient descent,
       staccandoli dal grafo con .detach()
    Restituisci la tupla (w_nuovo, b_nuovo, loss_valore) dove
    loss_valore e' un float (la loss PRIMA dell'update).
    """
    # TODO
    raise NotImplementedError


def allena(X, y, lr, epoche):
    """Il training loop completo: ripeti passo_di_training per ogni epoca.

    Parti da w = zeri (una componente per colonna di X) e b = zero.
    Registra la loss di ogni epoca in una lista.
    Restituisci la tupla (w, b, storia_loss).
    Su dati sensati la storia deve scendere: se sale, il lr e' folle.
    """
    # TODO
    raise NotImplementedError
