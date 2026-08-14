"""Esercizi sulla lezione 02: regressione logistica.

Completa le funzioni marcate con # TODO.
Poi esegui `pytest` da questa cartella.
La difficolta' cresce man mano che scendi nel file.
"""

import torch


def sigmoid_a_mano(z):
    """La funzione di schiacciamento: 1 / (1 + e^(-z)).

    Non usare torch.sigmoid. Deve funzionare sia su scalari sia su
    tensor (le operazioni di torch lo fanno gia' da sole).
    """
    # TODO
    raise NotImplementedError


def bce_a_mano(p, target):
    """La binary cross entropy media.

    p sono le probabilita' del modello (tra 0 e 1), target le verita'
    (0 o 1). Formula per esempio: -[t*log(p) + (1-t)*log(1-p)], poi
    media su tutti gli esempi. Non usare F.binary_cross_entropy.
    Restituisci un tensor scalare.
    """
    # TODO
    raise NotImplementedError


def probabilita_case(X, w, b):
    """La pipeline del classificatore: punteggio lineare, poi sigmoid.

    Restituisci il vettore delle probabilita' P(classe 1), una per riga
    di X. Riusa la tua sigmoid_a_mano.
    """
    # TODO
    raise NotImplementedError


def classifica(X, w, b, soglia=0.5):
    """Trasforma le probabilita' in decisioni 0/1.

    Probabilita' sopra la soglia: classe 1, altrimenti 0.
    Restituisci un tensor di 0.0 e 1.0 (usa .float()).
    """
    # TODO
    raise NotImplementedError


def allena_logistica(X, y, lr, epoche):
    """Il training loop del classificatore.

    Identico a quello della lezione 01, con due differenze: il forward
    passa dalla sigmoid (riusa probabilita_case) e la loss e' la BCE
    (riusa bce_a_mano). Parti da w e b a zero.
    Restituisci la tupla (w, b, storia_loss).
    """
    # TODO
    raise NotImplementedError
