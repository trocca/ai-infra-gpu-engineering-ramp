"""Esercizi sulla lezione 01: gradient descent a mano.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def passo_gd(w, grad, lr):
    """Applica un singolo passo di gradient descent.

    La regola di aggiornamento: w nuovo = w - lr * grad.
    w e grad possono essere float o tensor della stessa forma.
    Restituisci il nuovo w.
    """
    # TODO
    raise NotImplementedError


def gd_su_parabola(w0, lr, passi):
    """Esegui gradient descent sulla parabola loss(w) = (w - 3)^2.

    Il gradiente esatto e' 2 * (w - 3): scrivilo a mano, niente
    autograd. Parti da w0 (float) e applica la regola per il numero
    di passi richiesto. Restituisci il w finale come float.
    Con lr = 0.25 e abbastanza passi deve avvicinarsi a 3.
    """
    # TODO
    raise NotImplementedError


def gd_autograd(f, w0, lr, passi):
    """Gradient descent generico su una funzione qualsiasi, con autograd.

    f prende un tensor e restituisce un tensor scalare. w0 e' il tensor
    di partenza (senza requires_grad). Il ciclo per ogni passo:
    1. clona il punto corrente con requires_grad_(True)
    2. calcola f e chiama backward
    3. aggiorna: nuovo punto = punto - lr * gradiente (usa .detach())
    Restituisci il punto finale (tensor senza requires_grad).
    """
    # TODO
    raise NotImplementedError


def gd_con_storia(f, w0, lr, passi):
    """Come gd_autograd, ma registra anche la storia della loss.

    Restituisci una tupla (punto_finale, storia) dove storia e' la
    lista dei valori di f (float) a ogni passo, PRIMA di aggiornare.
    La storia e' quello che si guarda per capire se il training va:
    deve scendere. Se sale, il learning rate e' troppo grande.
    """
    # TODO
    raise NotImplementedError
